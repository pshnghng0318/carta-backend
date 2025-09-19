/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"
#include "Logger/Logger.h"

#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/StokesCoordinate.h>
#include <casacore/coordinates/Coordinates/Projection.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/Stokes.h>
#include <casacore/lattices/Lattices/ArrayLattice.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>

// TensorStore includes
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr/driver_impl.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"

using namespace casacore;

namespace carta {

CartaZarrImage::CartaZarrImage(const std::string& filename) : ImageInterface<float>(), _name(filename) {
    std::cout << "[DEBUG] CartaZarrImage constructor called with filename: " << filename << std::endl;
    // Initialize TensorStore context
    _context = tensorstore::Context::Default();
    
    // Try to read actual Zarr metadata
    try {
        std::filesystem::path zarr_path(filename);
        std::filesystem::path zarray_path = zarr_path / ".zarray";
        
        if (std::filesystem::exists(zarray_path)) {
            std::ifstream zarray_file(zarray_path);
            nlohmann::json zarray_json;
            zarray_file >> zarray_json;
            
            // Read shape from .zarray metadata
            if (zarray_json.contains("shape")) {
                auto shape_array = zarray_json["shape"];
                std::vector<int> shape_vec;
                for (auto& dim : shape_array) {
                    shape_vec.push_back(dim.get<int>());
                }
                if (!shape_vec.empty()) {
                    std::vector<int> reordered_shape;
                    
                    // Check if we have 5D ZARR with [time, frequency, polarization, l, m]
                    if (shape_vec.size() == 5) {
                        // Convert 5D ZARR to 4D CARTA by dropping dimensions with size 1
                        // ZARR: [1, 128, 1, 7763, 4742] -> CARTA 4D: [7763, 4742, 1, 1] = [x, y, freq, stokes]
                        // TEMPORARILY: Force channel dimensions to 1 (only show first channel)
                        reordered_shape.push_back(shape_vec[3]); // l (x) - spatial
                        reordered_shape.push_back(shape_vec[4]); // m (y) - spatial 
                        reordered_shape.push_back(1);           // frequency (forced to 1 for first channel only)
                        reordered_shape.push_back(1);           // polarization (forced to 1 for first channel only)
                        std::cout << "[DEBUG] Converted 5D ZARR [time,freq,pol,l,m] to 4D CARTA [l,m,freq,pol]: " 
                                  << "[" << shape_vec[3] << "," << shape_vec[4] << ",1,1] (channels forced to 1)" << std::endl;
                    } else {
                        // Keep original order for non-5D cases
                        reordered_shape = shape_vec;
                    }
                    
                    _shape = IPosition(reordered_shape);
                    _original_zarr_shape = IPosition(shape_vec); // Keep original for TensorStore access
                    spdlog::info("Zarr image shape (CARTA order): {}", _shape.toString());
                    spdlog::info("Original ZARR shape: {}", _original_zarr_shape.toString());
                } else {
                    _shape = IPosition(2, 1, 1);  // Default fallback
                }
            } else {
                _shape = IPosition(2, 1, 1);  // Default fallback
            }
        } else {
            _shape = IPosition(2, 1, 1);  // Default fallback
        }
        
        // Set _ndim based on shape
        _ndim = _shape.size();
        std::cout << "[DEBUG] Image dimensions: " << _ndim << ", shape: " << _shape.toString() << std::endl;
        
        // Initialize TensorStore for this Zarr file
        initializeTensorStore();
        
        // After TensorStore initialization, we might have updated _shape and _ndim
        // so we need to setup coordinate system based on actual data dimensions
        setupCoordinateSystem();
        
    } catch (std::exception& e) {
        spdlog::warn("Failed to read Zarr metadata for {}: {}", filename, e.what());
        _shape = IPosition(2, 1, 1);  // Default fallback
        _ndim = _shape.size();  // Set _ndim based on default shape
        
        // Still try to setup coordinate system with default values
        setupCoordinateSystem();
    }
    
    // Set coordinate system in ImageInterface base class
    setCoordinateInfo(_coord_sys);
    
    std::cout << "[DEBUG] CartaZarrImage constructor completed" << std::endl;
}

void CartaZarrImage::setupCoordinateSystem() {
    std::cout << "[DEBUG] setupCoordinateSystem() called" << std::endl;
    // Try to read coordinate system from .zattrs file
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        std::cout << "[DEBUG] Checking for .zattrs file at: " << zattrs_path << std::endl;
        
        if (std::filesystem::exists(zattrs_path)) {
            std::cout << "[DEBUG] Reading .zattrs file" << std::endl;
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
            spdlog::info("Found .zattrs file for Zarr image: {}", _name);
            
            // Parse WCS-like coordinate information
            std::cout << "[DEBUG] Parsing WCS from .zattrs" << std::endl;
            if (parseWCSFromZattrs(zattrs_json)) {
                std::cout << "[DEBUG] Successfully parsed coordinate system from .zattrs" << std::endl;
                spdlog::info("Successfully parsed coordinate system from .zattrs");
                return;
            }
        }
    } catch (std::exception& e) {
        std::cout << "[DEBUG] Exception in setupCoordinateSystem: " << e.what() << std::endl;
        spdlog::warn("Failed to read .zattrs for {}: {}", _name, e.what());
    }
    
    // Fallback to minimal coordinate system
    std::cout << "[DEBUG] Creating minimal coordinate system" << std::endl;
    createMinimalCoordinateSystem();
    std::cout << "[DEBUG] setupCoordinateSystem() completed" << std::endl;
}

bool CartaZarrImage::parseWCSFromZattrs(const nlohmann::json& zattrs) {
    try {
        std::cout << "[DEBUG] parseWCSFromZattrs() called" << std::endl;
        
        // Check for ZARR-style coordinate information
        bool has_array_dimensions = zattrs.contains("_ARRAY_DIMENSIONS");
        bool has_direction_info = zattrs.contains("direction") || zattrs.contains("pointing_center");
        
        if (!has_array_dimensions && !has_direction_info) {
            std::cout << "[DEBUG] No ZARR coordinate information found" << std::endl;
            return false;
        }
        
        // Get dimension names if available
        std::vector<std::string> axis_names;
        if (has_array_dimensions) {
            for (const auto& dim : zattrs["_ARRAY_DIMENSIONS"]) {
                axis_names.push_back(dim.get<std::string>());
            }
            std::cout << "[DEBUG] Found array dimensions: ";
            for (const auto& name : axis_names) {
                std::cout << name << " ";
            }
            std::cout << std::endl;
        }
        
        // Create coordinate system based on array dimensions
        if (axis_names.size() == 5) {
            // 5D ZARR: Original was [time, frequency, polarization, l, m]
            // But we converted to CARTA 4D format: [l, m, frequency, polarization] = [x, y, freq, stokes]
            std::cout << "[DEBUG] Creating 4D coordinate system for converted ZARR metadata" << std::endl;
            
            // Create DirectionCoordinate for l,m (spatial) axes - now first in CARTA order
            DirectionCoordinate dir_coord;
            
            // Parse direction coordinate parameters from ZARR metadata
            if (zattrs.contains("direction") || zattrs.contains("pointing_center")) {
                try {
                    double ra_rad = 0.0, dec_rad = 0.0;
                    
                    // Try to get reference coordinates
                    if (zattrs.contains("direction") && zattrs["direction"].contains("reference")) {
                        auto ref_data = zattrs["direction"]["reference"]["data"];
                        if (ref_data.is_array() && ref_data.size() >= 2) {
                            ra_rad = ref_data[0].get<double>();
                            dec_rad = ref_data[1].get<double>();
                        }
                    } else if (zattrs.contains("pointing_center")) {
                        auto center_data = zattrs["pointing_center"]["data"];
                        if (center_data.is_array() && center_data.size() >= 2) {
                            ra_rad = center_data[0].get<double>();
                            dec_rad = center_data[1].get<double>();
                        }
                    }
                    
                    // Create direction coordinate with actual reference values
                    if (ra_rad != 0.0 || dec_rad != 0.0) {
                        casacore::Vector<double> ref_val(2);
                        ref_val(0) = ra_rad;   // RA in radians
                        ref_val(1) = dec_rad;  // DEC in radians
                        
                        casacore::Vector<double> inc(2);
                        inc(0) = -1.0 * M_PI / 180.0 / 3600.0;  // Default pixel increment
                        inc(1) = 1.0 * M_PI / 180.0 / 3600.0;   // Default pixel increment
                        
                        casacore::Matrix<double> xform(2, 2);
                        xform = 0.0;
                        xform.diagonal() = 1.0;
                        
                        casacore::Vector<double> ref_pix(2);
                        ref_pix(0) = _shape(0) / 2.0;  // Center of x axis (l) - now first dimension
                        ref_pix(1) = _shape(1) / 2.0;  // Center of y axis (m) - now second dimension
                        
                        dir_coord = DirectionCoordinate(MDirection::J2000, 
                                                      Projection::SIN,
                                                      ref_val(0), ref_val(1),
                                                      inc(0), inc(1),
                                                      xform,
                                                      ref_pix(0), ref_pix(1));
                        std::cout << "[DEBUG] Created DirectionCoordinate with RA=" << ra_rad << " DEC=" << dec_rad << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << "[DEBUG] Error parsing direction coordinates: " << e.what() << std::endl;
                    // Fall back to default DirectionCoordinate
                    dir_coord = DirectionCoordinate();
                }
            }
            
            // Create SpectralCoordinate for frequency axis (now 3rd dimension in 4D)
            SpectralCoordinate spec_coord;
            
            // Create StokesCoordinate for polarization axis (now 4th dimension in 4D)
            casacore::Vector<int> stokes_types(_shape(3)); // Use actual polarization dimension size
            for (int i = 0; i < _shape(3); ++i) {
                stokes_types(i) = casacore::Stokes::I;  // Default all to Stokes I for now
            }
            StokesCoordinate stokes_coord(stokes_types);
            
            // Add coordinates in the CARTA 4D order: [x, y, freq, stokes]
            _coord_sys.addCoordinate(dir_coord);       // axes 0,1: direction (x, y)
            _coord_sys.addCoordinate(spec_coord);      // axis 2: frequency
            _coord_sys.addCoordinate(stokes_coord);    // axis 3: polarization
            
            std::cout << "[DEBUG] Successfully created 4D coordinate system from 5D ZARR metadata" << std::endl;
            return true;
        }
        
        // Fallback for other dimension counts or missing info
        std::cout << "[DEBUG] Using fallback coordinate parsing" << std::endl;
        return false;
        
    } catch (std::exception& e) {
        std::cout << "[DEBUG] Exception in parseWCSFromZattrs: " << e.what() << std::endl;
        spdlog::warn("Error parsing WCS from .zattrs: {}", e.what());
        return false;
    }
}

void CartaZarrImage::createMinimalCoordinateSystem() {
    std::cout << "[DEBUG] createMinimalCoordinateSystem() called with " << _ndim << " dimensions" << std::endl;
    try {
        // Create coordinate system based on actual image dimensions
        // Similar to CartaHdf5Image approach, but simplified for ZARR
        
        if (_ndim == 2) {
            // 2D image: only DirectionCoordinate (RA/DEC)
            std::cout << "[DEBUG] Creating DirectionCoordinate for 2D image" << std::endl;
            DirectionCoordinate dir_coord;
            _coord_sys.addCoordinate(dir_coord);
            std::cout << "[DEBUG] DirectionCoordinate added successfully for 2D image" << std::endl;
            
        } else if (_ndim == 3) {
            // 3D image: DirectionCoordinate + SpectralCoordinate
            std::cout << "[DEBUG] Creating coordinates for 3D image" << std::endl;
            DirectionCoordinate dir_coord;
            SpectralCoordinate spec_coord;
            
            _coord_sys.addCoordinate(dir_coord);
            _coord_sys.addCoordinate(spec_coord);
            std::cout << "[DEBUG] DirectionCoordinate and SpectralCoordinate added for 3D image" << std::endl;
            
        } else if (_ndim == 4) {
            // 4D image: DirectionCoordinate + SpectralCoordinate + StokesCoordinate
            std::cout << "[DEBUG] Creating coordinates for 4D image" << std::endl;
            DirectionCoordinate dir_coord;
            SpectralCoordinate spec_coord;
            
            // Create StokesCoordinate with appropriate size
            int stokes_size = (_ndim >= 4) ? _shape(3) : 1;
            casacore::Vector<int> stokes_types(stokes_size);
            for (int i = 0; i < stokes_size; ++i) {
                stokes_types(i) = casacore::Stokes::I;  // Default to Stokes I
            }
            StokesCoordinate stokes_coord(stokes_types);
            
            _coord_sys.addCoordinate(dir_coord);
            _coord_sys.addCoordinate(spec_coord);
            _coord_sys.addCoordinate(stokes_coord);
            std::cout << "[DEBUG] All coordinates added for 4D image" << std::endl;
            
        } else {
            // For other dimensions, create a basic system with linear coordinates
            std::cout << "[DEBUG] Creating basic coordinate system for " << _ndim << "D image" << std::endl;
            
            // Always start with DirectionCoordinate for the last 2 axes
            DirectionCoordinate dir_coord;
            
            // Add LinearCoordinates for the first axes
            for (int i = 0; i < _ndim - 2; ++i) {
                LinearCoordinate lin_coord;
                _coord_sys.addCoordinate(lin_coord);
            }
            
            // Add DirectionCoordinate for the last 2 axes
            _coord_sys.addCoordinate(dir_coord);
            
            std::cout << "[DEBUG] Basic coordinate system created with LinearCoordinates + DirectionCoordinate" << std::endl;
        }
        
        std::cout << "[DEBUG] createMinimalCoordinateSystem() completed with " << _coord_sys.nPixelAxes() 
                  << " pixel axes for " << _ndim << "D image" << std::endl;
                  
    } catch (std::exception& e) {
        std::cout << "[DEBUG] Exception in createMinimalCoordinateSystem: " << e.what() << std::endl;
        spdlog::error("Exception in createMinimalCoordinateSystem: {}", e.what());
        
        // Emergency fallback: just create a basic 2D system
        try {
            _coord_sys = CoordinateSystem();  // Reset
            DirectionCoordinate dir_coord;
            _coord_sys.addCoordinate(dir_coord);
            std::cout << "[DEBUG] Emergency fallback: created 2D DirectionCoordinate system" << std::endl;
        } catch (...) {
            std::cout << "[DEBUG] Even emergency fallback failed!" << std::endl;
        }
    }
}

void CartaZarrImage::initializeTensorStore() {
    std::cout << "[DEBUG] initializeTensorStore() called for: " << _name << std::endl;
    try {
        // Create TensorStore spec for Zarr using the correct format
        // Handle hierarchical zarr files (check for subdirectories with .zarray)
        std::string zarr_path = _name;
        std::cout << "[DEBUG] Initial zarr_path: " << zarr_path << std::endl;
        
        // Check if this is a hierarchical zarr (has subdirectories with .zarray)
        std::filesystem::path base_path(zarr_path);
        std::filesystem::path potential_array_path;
        
        // Look for common array subdirectories like SKY, DATA, etc.
        std::vector<std::string> common_array_names = {"SKY", "DATA", "ARRAY", "0"};
        bool found_array = false;
        std::cout << "[DEBUG] Checking for array subdirectories..." << std::endl;
        
        for (const auto& array_name : common_array_names) {
            potential_array_path = base_path / array_name;
            std::cout << "[DEBUG] Checking path: " << potential_array_path.string() << std::endl;
            if (std::filesystem::exists(potential_array_path / ".zarray")) {
                zarr_path = potential_array_path.string();
                found_array = true;
                std::cout << "[DEBUG] Found Zarr array in subdirectory: " << zarr_path << std::endl;
                spdlog::info("Found Zarr array in subdirectory: {}", zarr_path);
                break;
            }
        }
        
        // If no subdirectory found, check if base path has .zarray directly
        if (!found_array && !std::filesystem::exists(base_path / ".zarray")) {
            std::cout << "[DEBUG] ERROR: No .zarray file found in " << _name << " or its subdirectories" << std::endl;
            spdlog::error("No .zarray file found in {} or its subdirectories", _name);
            return;
        }
        
        if (!found_array) {
            zarr_path = _name;  // Use original path
            std::cout << "[DEBUG] Using direct Zarr path: " << zarr_path << std::endl;
            spdlog::info("Using direct Zarr path: {}", zarr_path);
        }
        
        // Create TensorStore spec using the format from extract_slice.cc example
        std::cout << "[DEBUG] Creating TensorStore spec for path: " << zarr_path << std::endl;
        nlohmann::json spec_json = {
            {"driver", "zarr2"},
            {"kvstore", {
                {"driver", "file"},
                {"path", zarr_path}
            }}
        };
        std::cout << "[DEBUG] TensorStore spec JSON: " << spec_json.dump(2) << std::endl;
        
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            std::cout << "[DEBUG] ERROR: Failed to create TensorStore spec: " << spec_result.status().ToString() << std::endl;
            spdlog::error("Failed to create TensorStore spec for {}: {}", zarr_path, spec_result.status().ToString());
            return;
        }
        std::cout << "[DEBUG] Successfully created TensorStore spec" << std::endl;
        
        auto input_spec = spec_result.value();
        
        // Open input tensorstore and resolve the bounds using the pattern from extract_slice.cc
        std::cout << "[DEBUG] Attempting to open TensorStore..." << std::endl;
        auto open_future = tensorstore::Open(
            input_spec, 
            _context, 
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        );
        
        auto open_result = open_future.result();
        if (!open_result.ok()) {
            std::cout << "[DEBUG] ERROR: Failed to open TensorStore: " << open_result.status().ToString() << std::endl;
            spdlog::error("Failed to open TensorStore for {}: {}", _name, open_result.status().ToString());
            return;
        }
        
        std::cout << "[DEBUG] Successfully opened TensorStore!" << std::endl;
        _tensorstore = std::move(open_result).value();
        _tensorstore_initialized = true;
        
        // Verify that the data type is float32 as expected
        auto ts_dtype = _tensorstore.dtype();
        std::cout << "[DEBUG] TensorStore data type: " << ts_dtype.name() << std::endl;
        spdlog::info("TensorStore data type: {}", ts_dtype.name());
        
        // Check if data type is float32 (TensorStore uses "float32" as the name)
        if (ts_dtype.name() != "float32") {
            std::cout << "[DEBUG] WARNING: TensorStore data type is not float32, got: " << ts_dtype.name() << std::endl;
            spdlog::warn("TensorStore data type is not float32, got: {}", ts_dtype.name());
        }
        
        std::cout << "[DEBUG] TensorStore initialized successfully for: " << _name << std::endl;
        spdlog::info("Successfully initialized TensorStore for {}", _name);
        
        // Verify shape matches what we read from .zarray
        auto ts_domain = _tensorstore.domain();
        auto ts_shape = ts_domain.shape();
        std::cout << "[DEBUG] TensorStore rank: " << ts_domain.rank() << std::endl;
        std::cout << "[DEBUG] TensorStore shape: [";
        for (size_t i = 0; i < ts_shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << ts_shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Keep TensorStore shape as original ZARR shape but maintain our reordered shape for CARTA
        std::vector<int> actual_shape_vec;
        for (size_t i = 0; i < ts_shape.size(); ++i) {
            actual_shape_vec.push_back(static_cast<int>(ts_shape[i]));
        }
        _original_zarr_shape = IPosition(actual_shape_vec);
        
        // If we already have a reordered shape from .zarray parsing, keep it
        // Otherwise, use the TensorStore shape as-is
        if (_shape.size() == 0 || _shape.product() == 1) {
            if (actual_shape_vec.size() == 5) {
                // Convert 5D ZARR to 4D CARTA: [time, freq, pol, l, m] -> [l, m, freq, pol]
                std::vector<int> reordered_shape;
                reordered_shape.push_back(actual_shape_vec[3]); // l (x) - spatial
                reordered_shape.push_back(actual_shape_vec[4]); // m (y) - spatial 
                reordered_shape.push_back(actual_shape_vec[1]); // frequency
                reordered_shape.push_back(actual_shape_vec[2]); // polarization as stokes
                _shape = IPosition(reordered_shape);
                std::cout << "[DEBUG] TensorStore: Converted 5D to 4D CARTA shape: " << _shape.toString() << std::endl;
            } else {
                _shape = _original_zarr_shape;
            }
        }
        _ndim = _shape.size();
        
        std::cout << "[DEBUG] Updated image dimensions: " << _ndim << ", shape: " << _shape.toString() << std::endl;
        spdlog::info("Updated Zarr image shape from TensorStore: {}", _shape.toString());
        
        spdlog::info("TensorStore rank: {}", ts_domain.rank());
        spdlog::info("TensorStore shape: [{}]", 
                    [&ts_shape]() {
                        std::string result;
                        for (size_t i = 0; i < ts_shape.size(); ++i) {
                            if (i > 0) result += ", ";
                            result += std::to_string(ts_shape[i]);
                        }
                        return result;
                    }());
        
    } catch (std::exception& e) {
        std::cout << "[DEBUG] Exception in initializeTensorStore: " << e.what() << std::endl;
        spdlog::error("Exception in initializeTensorStore for {}: {}", _name, e.what());
        _tensorstore_initialized = false;
    }
}

String CartaZarrImage::imageType() const {
    return "zarr";
}

DataType CartaZarrImage::dataType() const {
    return TpFloat;
}

Bool CartaZarrImage::doGetSlice(Array<float>& buffer, const Slicer& section) {
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for {}", _name);
        return false;
    }
    
    try {
        // Convert casacore Slicer to TensorStore transform
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        const IPosition& stride = section.stride();
        
        std::cout << "[DEBUG] doGetSlice: slice start=" << start.toString() 
                  << ", length=" << length.toString() 
                  << ", stride=" << stride.toString() << std::endl;
        
        // Map CARTA dimensions back to original ZARR dimensions for 5D case
        IPosition zarr_start = start;
        IPosition zarr_length = length;
        
        if (_original_zarr_shape.size() == 5 && start.size() >= 2) {
            // CARTA 4D format: [l, m, freq, pol] = [x, y, freq, stokes]
            // Original ZARR 5D format: [time, freq, pol, l, m]
            zarr_start.resize(5);
            zarr_length.resize(5);
            
            // Read only the requested region from ZARR, one row at a time if needed
            std::cout << "[DEBUG] Reading requested region directly from ZARR" << std::endl;
            
            zarr_start[0] = 0;                                      // ZARR[0]=time (always 0)
            zarr_start[1] = 0;                                      // ZARR[1]=freq (forced to first channel)
            zarr_start[2] = 0;                                      // ZARR[2]=pol (forced to first channel)
            zarr_start[3] = start[0];                               // ZARR[3]=l (exact x start)
            zarr_start[4] = start[1];                               // ZARR[4]=m (exact y start)
            
            zarr_length[0] = 1;                                     // ZARR[0]=time (always 1)
            zarr_length[1] = 1;                                     // ZARR[1]=freq (forced to 1 for first channel only)
            zarr_length[2] = 1;                                     // ZARR[2]=pol (forced to 1 for first channel only)
            zarr_length[3] = length[0];                             // ZARR[3]=l (exact width requested)
            zarr_length[4] = length[1];                             // ZARR[4]=m (exact height requested)
            
            std::cout << "[DEBUG] CARTA 4D request: [x=" << start[0] << ":" << (start[0] + length[0] - 1) 
                      << ", y=" << start[1] << ":" << (start[1] + length[1] - 1)
                      << ", freq=" << (start.size() > 2 ? start[2] : 0) << ":" << (start.size() > 2 ? start[2] + length[2] - 1 : 0)
                      << ", stokes=" << (start.size() > 3 ? start[3] : 0) << ":" << (start.size() > 3 ? start[3] + length[3] - 1 : 0) << "]" << std::endl;
            std::cout << "[DEBUG] Reading ZARR 5D: [time=" << zarr_start[0] << ":" << (zarr_start[0] + zarr_length[0] - 1)
                      << ", freq=" << zarr_start[1] << ":" << (zarr_start[1] + zarr_length[1] - 1)
                      << ", pol=" << zarr_start[2] << ":" << (zarr_start[2] + zarr_length[2] - 1)
                      << ", l=" << zarr_start[3] << ":" << (zarr_start[3] + zarr_length[3] - 1)
                      << ", m=" << zarr_start[4] << ":" << (zarr_start[4] + zarr_length[4] - 1) << "]" << std::endl;
        }
        
        // Create a Box for slicing all dimensions at once
        std::vector<tensorstore::Index> box_origin(zarr_start.size());
        std::vector<tensorstore::Index> box_shape(zarr_start.size());
        
        for (size_t i = 0; i < zarr_start.size(); ++i) {
            box_origin[i] = zarr_start[i];
            box_shape[i] = zarr_length[i];
        }
        
        tensorstore::Box<> slice_box(box_origin, box_shape);
        std::cout << "[DEBUG] Created slice box with " << slice_box.rank() << " dimensions" << std::endl;
        
        // Apply the box slice to the TensorStore
        auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(slice_box);
        if (!constrained_store.ok()) {
            spdlog::error("Failed to apply transform: {}", constrained_store.status().ToString());
            return false;
        }
        
        // Read data from TensorStore
        auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
        if (!read_result.ok()) {
            spdlog::error("Failed to read data: {}", read_result.status().ToString());
            return false;
        }
        
        auto zarr_array = std::move(read_result.value());
        printf("[DEBUG] Read TensorStore array with shape: [");
        for (size_t i = 0; i < zarr_array.rank(); ++i) {
            if (i > 0) printf(", ");
            printf("%lld", zarr_array.shape()[i]);
        }
        printf("]\n");        
        
        // Convert TensorStore array to casacore Array with correct shape
        IPosition buffer_shape = length;
        buffer.resize(buffer_shape);
        
        // Verify data type matches our expectation (float32)
        auto zarr_dtype = zarr_array.dtype();
        if (zarr_dtype.name() != "float32") {
            spdlog::error("ZARR data type mismatch: expected float32, got {}", zarr_dtype.name());
            return false;
        }
        
        // Copy data from TensorStore array to casacore array
        const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
        float* dest_data = buffer.data();
        
        size_t num_elements = buffer.nelements();
        
        // Since we now read exactly the requested region, we can copy directly
        if (_original_zarr_shape.size() == 5) {
            // Direct copy since we read exactly what was requested
            std::copy(src_data, src_data + num_elements, dest_data);
            
            size_t req_width = length[0];
            size_t req_height = length[1];
            std::cout << "[DEBUG] Direct copy of " << num_elements << " elements (" << req_width << "x" << req_height << ") - exact region read" << std::endl;
        } else {
            // Fallback: direct copy for non-5D cases
            std::copy(src_data, src_data + num_elements, dest_data);
            std::cout << "[DEBUG] Copied " << num_elements << " elements directly (non-5D case)" << std::endl;
        }
        
        spdlog::debug("Successfully read {} elements from Zarr file (dtype: {})", num_elements, zarr_dtype.name());
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in doGetSlice for {}: {}", _name, e.what());
        return false;
    }
}

void CartaZarrImage::doPutSlice(const Array<float>& buffer, const IPosition& where, const IPosition& stride) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::doPutSlice not implemented");
}

Bool CartaZarrImage::doGetMaskSlice(Array<Bool>& buffer, const Slicer& section) {
    // Stub implementation - not needed for file browser
    return false;
}

Bool CartaZarrImage::isMasked() const {
    return false;
}

Bool CartaZarrImage::isPersistent() const {
    return true;
}

Bool CartaZarrImage::isWritable() const {
    return false;
}

String CartaZarrImage::name(Bool stripPath) const {
    if (stripPath) {
        auto pos = _name.find_last_of('/');
        if (pos != std::string::npos) {
            return _name.substr(pos + 1);
        }
    }
    return _name;
}

IPosition CartaZarrImage::shape() const {
    return _shape;
}

void CartaZarrImage::resize(const TiledShape& newShape) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::resize not implemented");
}

Bool CartaZarrImage::ok() const {
    return _tensorstore_initialized;  // Only return true if TensorStore is properly initialized
}

void CartaZarrImage::flush() {
    // Stub implementation - not needed for file browser
}

void CartaZarrImage::tempClose() {
    // Stub implementation - not needed for file browser
}

void CartaZarrImage::reopen() {
    // Stub implementation - not needed for file browser
}

Bool CartaZarrImage::hasPixelMask() const {
    return false;
}

const Lattice<Bool>& CartaZarrImage::pixelMask() const {
    // Return a dummy lattice - use static to avoid recreation
    static ArrayLattice<Bool> dummy_lattice(Array<Bool>(IPosition(2, 1, 1)));
    return dummy_lattice;
}

Lattice<Bool>& CartaZarrImage::pixelMask() {
    // Return a dummy lattice - use static to avoid recreation
    static ArrayLattice<Bool> dummy_lattice(Array<Bool>(IPosition(2, 1, 1)));
    return dummy_lattice;
}

const LatticeRegion* CartaZarrImage::getRegionPtr() const {
    return nullptr;
}

ImageInterface<float>* CartaZarrImage::cloneII() const {
    return new CartaZarrImage(_name);
}

const CoordinateSystem& CartaZarrImage::coordinates() const {
    return _coord_sys;
}

} // namespace carta
