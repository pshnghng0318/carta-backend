/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"
#include "Logger/Logger.h"

#include <limits>
#include <cmath>
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
}

void CartaZarrImage::setupCoordinateSystem() {
    // Try to read coordinate system from .zattrs file
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        if (std::filesystem::exists(zattrs_path)) {
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
            spdlog::info("Found .zattrs file for Zarr image: {}", _name);
            spdlog::debug("ZARR WCS: About to call parseWCSFromZattrs");
            
            // Parse WCS-like coordinate information
            if (parseWCSFromZattrs(zattrs_json)) {
                spdlog::info("Successfully parsed coordinate system from .zattrs");
                return;
            } else {
                spdlog::warn("Failed to parse coordinate system from .zattrs");
            }
        }

        // // Frequency
        // std::filesystem::path zattrs_freq_path = zarr_path / "frequency/.zattrs";

        // if (std::filesystem::exists(zattrs_freq_path)) {
        //     std::ifstream zattrs_freq_file(zattrs_freq_path);
        //     nlohmann::json zattrs_freq_json;
        //     zattrs_freq_file >> zattrs_freq_json;

        //     spdlog::info("Found frequency/.zattrs file for Zarr image: {}", _name);
        //     spdlog::debug("ZARR WCS: About to call parseFreqFromZattrs");

        //     // Parse frequency information
        //     if (parseFreqFromZattrs(zattrs_freq_json)) {
        //         spdlog::info("Successfully parsed frequency information from .zattrs");
        //         return;
        //     } else {
        //         spdlog::warn("Failed to parse frequency information from .zattrs");
        //     }
        // }
    } catch (std::exception& e) {
        spdlog::warn("Failed to read .zattrs for {}: {}", _name, e.what());
    }
    
    // Fallback to minimal coordinate system
    createMinimalCoordinateSystem();
}

bool CartaZarrImage::parseWCSFromZattrs(const nlohmann::json& zattrs) {
    try {
        spdlog::debug("ZARR WCS: Starting parseWCSFromZattrs");
        
        // First, check if we have precise coordinate arrays
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path ra_path = zarr_path / "right_ascension";
        std::filesystem::path dec_path = zarr_path / "declination";
        std::filesystem::path freq_path = zarr_path / "frequency";
        
        bool has_precise_coords = std::filesystem::exists(ra_path) && std::filesystem::exists(dec_path);
        spdlog::debug("ZARR WCS: Precise coordinate arrays available: {}", has_precise_coords);
        
        if (has_precise_coords) {
            return parseWCSFromCoordinateArrays(ra_path, dec_path, freq_path);
        }
        
        // Fallback to metadata-based parsing
        return parseWCSFromMetadata(zattrs);
    } catch (std::exception& e) {
        spdlog::error("ZARR WCS: Exception in parseWCSFromZattrs: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::parseWCSFromCoordinateArrays(const std::filesystem::path& ra_path, const std::filesystem::path& dec_path, const std::filesystem::path& freq_path) {
    try {
        spdlog::debug("ZARR WCS: Reading precise coordinate arrays");
        
        // TODO: Implement TensorStore reading of coordinate arrays
        // For now, use a representative sample to build the coordinate system
        
        // Read array metadata to understand structure
        std::ifstream ra_zarray(ra_path / ".zarray");
        nlohmann::json ra_meta;
        ra_zarray >> ra_meta;
        
        auto shape = ra_meta["shape"];
        size_t height = shape[0].get<size_t>();  // l dimension
        size_t width = shape[1].get<size_t>();   // m dimension
        
        spdlog::debug("ZARR WCS: Coordinate arrays shape: {}x{}", height, width);
        
        // For now, calculate approximate reference coordinates from center positions
        // Later we can read actual data using TensorStore
        size_t center_l = height / 2;
        size_t center_m = width / 2;
        
        // Try to get pointing center from main .zattrs
        double ra_rad = 0.0, dec_rad = 0.0;
        std::filesystem::path main_zattrs = ra_path.parent_path() / ".zattrs";
        std::ifstream main_file(main_zattrs);
        if (main_file.is_open()) {
            nlohmann::json main_json;
            main_file >> main_json;
            
            // Try different sources for reference coordinates
            if (main_json.contains("pointing_center")) {
                auto center_data = main_json["pointing_center"]["data"];
                if (center_data.is_array() && center_data.size() >= 2) {
                    ra_rad = center_data[0].get<double>();
                    dec_rad = center_data[1].get<double>();
                    spdlog::debug("ZARR WCS: Using pointing_center from main .zattrs: RA={:.6f} rad, DEC={:.6f} rad", 
                                ra_rad, dec_rad);
                }
            } else if (main_json.contains("direction") && main_json["direction"].contains("reference")) {
                auto ref_data = main_json["direction"]["reference"]["data"];
                if (ref_data.is_array() && ref_data.size() >= 2) {
                    ra_rad = ref_data[0].get<double>();
                    dec_rad = ref_data[1].get<double>();
                    spdlog::debug("ZARR WCS: Using direction.reference from main .zattrs: RA={:.6f} rad, DEC={:.6f} rad", 
                                ra_rad, dec_rad);
                }
            }
        }
        
        // If no pointing center found, calculate from coordinate array center
        if (ra_rad == 0.0 && dec_rad == 0.0) {
            // TODO: Read actual coordinate values from center pixels
            // For now, use a default reasonable center for ASKAP data
            ra_rad = 5.5;  // ~315 degrees, typical for Hydra field
            dec_rad = -0.65; // ~-37 degrees, typical for southern sky
            spdlog::warn("ZARR WCS: No pointing center found, using default center: RA={:.6f} rad, DEC={:.6f} rad", 
                        ra_rad, dec_rad);
        }

        // Frequency handling - assume single channel for now
        std::ifstream freq_zarray(freq_path / ".zarray");
        nlohmann::json freq_meta;
        freq_zarray >> freq_meta;

        auto freq_shape = freq_meta["shape"];
        size_t depth = freq_shape[0].get<size_t>();  // channel numbers

        spdlog::debug("ZARR FREQ COOR: Frequency arrays shape: {}x{}", depth);

        // For now, calculate approximate reference frequencies from center positions
        // Later we can read actual data using TensorStore
        size_t center_freq_channel = depth / 2;

        // Try to get frequency center from main .zattrs
        double freq_hz = 0.0;
        std::filesystem::path main_freq_zattrs = freq_path.parent_path() / ".zattrs";
        std::ifstream main_freq_file(main_freq_zattrs);
        if (main_freq_file.is_open()) {
            nlohmann::json main_freq_json;
            main_freq_file >> main_freq_json;

            // Try different sources for reference frequencies
            if (main_freq_json.contains("frequency")) {
                auto freq_data = main_freq_json["frequency"]["data"];
                if (freq_data.is_array() && freq_data.size() >= 1) {
                    freq_hz = freq_data[0].get<double>();
                    spdlog::debug("ZARR WCS: Using frequency from main .zattrs: FREQ={:.6f} Hz",
                                freq_hz);
                }
            }
        }

        // If no frequency center found, calculate from frequency array center
        if (freq_hz == 0.0) {
            // TODO: Read actual frequency values from center pixels
            // For now, use a default reasonable center for ASKAP data
            freq_hz = 1.0e9;  // ~1 GHz, typical for ASKAP data
            spdlog::warn("ZARR WCS: No frequency center found, using default center: FREQ={:.6f} Hz",
                        freq_hz);
        }

        return buildDirectionCoordinateFromArrays(ra_rad, dec_rad, freq_hz, height, width, depth);

    } catch (std::exception& e) {
        spdlog::error("ZARR WCS: Exception reading coordinate arrays: {}", e.what());
        return false;
    }
}

// bool parseFreqFromZattrs(const nlohmann::json& zattrs) {
//     try {
//         spdlog::debug("ZARR FREQ: Starting parseFreqFromZattrs");

//         if (zattrs.contains("reference") && zattrs["reference"]["units"].is_string()) {
//             std::string refFreq_unit = zattrs["reference"]["units"].get<std::string>();
//             spdlog::debug("ZARR FREQ: Frequency unit of Reference from .zattrs: {}", refFreq_unit);
//             std::string refFreq_value = zattrs["reference"]["data"].get<std::string>();
//             spdlog::debug("ZARR FREQ: Frequency value of Reference from .zattrs: {}", refFreq_value);
//         }

//         if (zattrs.contains("rest") && zattrs["rest"]["units"].is_string()) {
//             std::string restFreq_unit = zattrs["rest"]["units"].get<std::string>();
//             spdlog::debug("ZARR FREQ: Frequency unit of Rest from .zattrs: {}", restFreq_unit);
//             std::string restFreq_value = zattrs["rest"]["data"].get<std::string>();
//             spdlog::debug("ZARR FREQ: Frequency value of Rest from .zattrs: {}", restFreq_value);
//         }

        
        
//         spdlog::warn("ZARR FREQ: No valid frequency data found in .zattrs");
//         return false;
        
//     } catch (std::exception& e) {
//         spdlog::error("ZARR FREQ: Exception in parseFreqFromZattrs: {}", e.what());
//         return false;
//     }
// }

bool CartaZarrImage::buildDirectionCoordinateFromArrays(double ra_rad, double dec_rad, double freq_hz, size_t height, size_t width, size_t depth) {
    try {
        spdlog::debug("ZARR WCS: Building DirectionCoordinate from coordinate arrays");
        spdlog::debug("ZARR WCS: Reference RA={:.6f} rad ({:.6f}°), DEC={:.6f} rad ({:.6f}°)", 
                     ra_rad, ra_rad * 180.0 / M_PI, dec_rad, dec_rad * 180.0 / M_PI);
        spdlog::debug("ZARR WCS: Array dimensions: {}x{}", height, width);
        
        // Create DirectionCoordinate using available information
        casacore::Vector<double> ref_val(2);
        ref_val(0) = ra_rad;   // RA in radians
        ref_val(1) = dec_rad;  // DEC in radians
        
        // Calculate approximate pixel increments 
        // For ASKAP data, typical pixel scale is around 2.5 arcseconds
        // But we should calculate this from the actual coordinate arrays eventually
        casacore::Vector<double> inc(2);
        inc(0) = -2.5 * M_PI / (180.0 * 3600.0);  // -2.5 arcsec in radians (negative for RA)
        inc(1) = 2.5 * M_PI / (180.0 * 3600.0);   // +2.5 arcsec in radians
        
        // Reference pixel (center of image)
        casacore::Vector<double> ref_pix(2);
        ref_pix(0) = (width - 1) / 2.0;   // Center of x axis (m)
        ref_pix(1) = (height - 1) / 2.0;  // Center of y axis (l)
        
        // Linear transformation matrix (identity for now)
        casacore::Matrix<double> xform(2, 2);
        xform = 0.0;
        xform(0, 0) = 1.0;
        xform(1, 1) = 1.0;
        
        spdlog::debug("ZARR WCS: From Arrays Creating DirectionCoordinate with:");
        spdlog::debug("  Reference value: RA={:.6f}° DEC={:.6f}°", 
                     ra_rad * 180.0 / M_PI, dec_rad * 180.0 / M_PI);
        spdlog::debug("  Reference pixel: ({:.1f}, {:.1f})", ref_pix(0), ref_pix(1));
        spdlog::debug("  Pixel increment: ({:.3f} arcsec, {:.3f} arcsec)", 
                     inc(0) * 180.0 * 3600.0 / M_PI, inc(1) * 180.0 * 3600.0 / M_PI);
        
        DirectionCoordinate dir_coord;
        SpectralCoordinate spec_coord;
        try {
            // Use CAR projection for radio astronomy data
            dir_coord = DirectionCoordinate(MDirection::J2000, 
                                          Projection::CAR,
                                          ref_val(0), ref_val(1),
                                          inc(0), inc(1),
                                          xform,
                                          ref_pix(0), ref_pix(1));
            
            spdlog::debug("ZARR WCS: DirectionCoordinate created successfully with CAR projection");
            
            // Test coordinate conversion
            casacore::Vector<double> world_coord(2);
            casacore::Vector<double> pixel_coord(2);
            pixel_coord(0) = ref_pix(0);
            pixel_coord(1) = ref_pix(1);
            
            if (dir_coord.toWorld(world_coord, pixel_coord)) {
                spdlog::debug("ZARR WCS: Reference pixel ({:.1f}, {:.1f}) -> World ({:.6f}, {:.6f}) radians",
                            pixel_coord(0), pixel_coord(1),
                            world_coord(0), world_coord(1));
                spdlog::debug("ZARR WCS: World coordinates: RA={:.6f}° DEC={:.6f}°",
                            world_coord(0) * 180.0 / M_PI, 
                            world_coord(1) * 180.0 / M_PI);
            } else {
                spdlog::warn("ZARR WCS: Failed to convert reference pixel to world coordinates");
            }
            
        } catch (const std::exception& coord_e) {
            spdlog::error("ZARR WCS: Failed to create DirectionCoordinate: {}", coord_e.what());
            dir_coord = DirectionCoordinate(); // Fallback to default
        }

        // Create SpectralCoordinate

        try {
            // Try default values for now
            double rest_freq = 1420405751.786; // in Hz
            double spectral_crval = freq_hz; // Reference frequency in Hz
            double spectral_cdelt = 0.1e6; // 0.1 MHz channel width (placeholder)
            double spectral_crpix = (depth - 1) / 2.0 + 1; // 1-based pixel
            // Use casacore MFrequency type for the SpectralCoordinate constructor
            casacore::MFrequency::Types frequency_type = casacore::MFrequency::TOPO; // default to TOPO

            spec_coord = SpectralCoordinate(frequency_type, spectral_crval, spectral_cdelt, spectral_crpix, rest_freq);

            spdlog::debug("ZARR FREQ COOR: SpectralCoordinate created successfully");
            
        } catch (const std::exception& coord_e) {
            spdlog::error("ZARR FREQ COOR: Failed to create SpectralCoordinate: {}", coord_e.what());
            spec_coord = SpectralCoordinate(); // Fallback to default
        }
        
        // Create StokesCoordinate - assume single polarization (I) for now
        casacore::Vector<int> stokes_types(_shape(3)); 
        for (int i = 0; i < _shape(3); ++i) {
            stokes_types(i) = casacore::Stokes::I;
        }
        StokesCoordinate stokes_coord(stokes_types);
        
        // Add coordinates in CARTA 4D order: [x, y, freq, stokes]
        try {
            _coord_sys.addCoordinate(dir_coord);       
            spdlog::debug("ZARR WCS: Successfully added DirectionCoordinate from arrays");
            
            _coord_sys.addCoordinate(spec_coord);      
            spdlog::debug("ZARR WCS: Successfully added SpectralCoordinate");
            
            _coord_sys.addCoordinate(stokes_coord);    
            spdlog::debug("ZARR WCS: Successfully added StokesCoordinate");
            
            spdlog::debug("ZARR WCS: CoordinateSystem has {} coordinates", _coord_sys.nCoordinates());
            return true;
            
        } catch (const std::exception& e) {
            spdlog::error("ZARR WCS: Failed to add coordinates to CoordinateSystem: {}", e.what());
            return false;
        }
        
    } catch (std::exception& e) {
        spdlog::error("ZARR WCS: Exception in buildDirectionCoordinateFromArrays: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::parseWCSFromMetadata(const nlohmann::json& zattrs) {
    try {
        spdlog::debug("ZARR WCS: Starting parseWCSFromZattrs");
        
        // Check for ZARR-style coordinate information
        bool has_array_dimensions = zattrs.contains("_ARRAY_DIMENSIONS");
        bool has_direction_info = zattrs.contains("direction") || zattrs.contains("pointing_center");
        
        spdlog::debug("ZARR WCS: has_array_dimensions = {}, has_direction_info = {}", 
                     has_array_dimensions, has_direction_info);
        
        if (!has_array_dimensions && !has_direction_info) {
            spdlog::debug("ZARR WCS: No required info found, returning false");
            return false;
        }
        
        // Get dimension names if available
        std::vector<std::string> axis_names;
        if (has_array_dimensions) {
            for (const auto& dim : zattrs["_ARRAY_DIMENSIONS"]) {
                axis_names.push_back(dim.get<std::string>());
            }
        }
        
        // Create coordinate system based on array dimensions
        if (axis_names.size() == 5) {
            spdlog::debug("ZARR WCS: Processing 5D ZARR file with dimensions: [{}]", 
                         fmt::join(axis_names, ", "));
            
            // 5D ZARR: Original was [time, frequency, polarization, l, m]
            // But we converted to CARTA 4D format: [l, m, frequency, polarization] = [x, y, freq, stokes]
            
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
                            spdlog::debug("ZARR WCS: pointing_center RA={} rad ({}°), DEC={} rad ({}°)", 
                                        ra_rad, ra_rad * 180.0 / M_PI, 
                                        dec_rad, dec_rad * 180.0 / M_PI);
                        }
                    }
                    
                    // Create direction coordinate with actual reference values
                    if (ra_rad != 0.0 || dec_rad != 0.0) {
                        casacore::Vector<double> ref_val(2);
                        ref_val(0) = ra_rad;   // RA in radians
                        ref_val(1) = dec_rad;  // DEC in radians
                        
                        // Use more reasonable pixel increments (about 1 arcsecond)
                        casacore::Vector<double> inc(2);
                        inc(0) = -1.0 * M_PI / 180.0 / 3600.0;  // -1 arcsec in radians for RA (negative for standard orientation)
                        inc(1) = 1.0 * M_PI / 180.0 / 3600.0;   // +1 arcsec in radians for DEC
                        
                        casacore::Matrix<double> xform(2, 2);
                        xform = 0.0;
                        xform.diagonal() = 1.0;  // Identity matrix
                        
                        // Reference pixel at image center (0-based indexing)
                        casacore::Vector<double> ref_pix(2);
                        ref_pix(0) = (_shape(0) - 1) / 2.0;  // Center of x axis (l)
                        ref_pix(1) = (_shape(1) - 1) / 2.0;  // Center of y axis (m)
                        
                        spdlog::debug("ZARR WCS: From Meta Data Creating DirectionCoordinate with:");
                        spdlog::debug("  Reference value: RA={}° DEC={}°", 
                                    ra_rad * 180.0 / M_PI, dec_rad * 180.0 / M_PI);
                        spdlog::debug("  Reference pixel: ({}, {})", ref_pix(0), ref_pix(1));
                        spdlog::debug("  Pixel increment: ({} arcsec, {} arcsec)", 
                                    inc(0) * 180.0 * 3600.0 / M_PI, inc(1) * 180.0 * 3600.0 / M_PI);
                        spdlog::debug("  Image shape: ({}, {})", _shape(0), _shape(1));
                        
                        try {
                            // Try CAR projection first (common for radio astronomy)
                            dir_coord = DirectionCoordinate(MDirection::J2000, 
                                                          Projection::CAR,
                                                          ref_val(0), ref_val(1),
                                                          inc(0), inc(1),
                                                          xform,
                                                          ref_pix(0), ref_pix(1));
                            
                            spdlog::debug("ZARR WCS: DirectionCoordinate created successfully with CAR projection");
                            
                            // Test coordinate conversion
                            casacore::Vector<double> world_coord(2);
                            casacore::Vector<double> pixel_coord(2);
                            pixel_coord(0) = ref_pix(0);
                            pixel_coord(1) = ref_pix(1);
                            
                            if (dir_coord.toWorld(world_coord, pixel_coord)) {
                                spdlog::debug("ZARR WCS: Reference pixel ({}, {}) -> World ({}, {}) radians",
                                            pixel_coord(0), pixel_coord(1),
                                            world_coord(0), world_coord(1));
                                spdlog::debug("ZARR WCS: World coordinates: RA={}° DEC={}°",
                                            world_coord(0) * 180.0 / M_PI, 
                                            world_coord(1) * 180.0 / M_PI);
                            } else {
                                spdlog::warn("ZARR WCS: Failed to convert reference pixel to world coordinates");
                            }
                            
                        } catch (const std::exception& coord_e) {
                            spdlog::error("ZARR WCS: Failed to create DirectionCoordinate: {}", coord_e.what());
                            dir_coord = DirectionCoordinate(); // Fallback to default
                        }
                    } else {
                        spdlog::debug("ZARR WCS: No valid pointing center found, using default DirectionCoordinate");
                        dir_coord = DirectionCoordinate();
                    }
                } catch (const std::exception& e) {
                    // Fall back to default DirectionCoordinate
                    dir_coord = DirectionCoordinate();
                }
            }
            
            // Create SpectralCoordinate for frequency axis (now 3rd dimension in 4D)
            SpectralCoordinate spec_coord;
            bool spec_built = false;
            std::string spec_unit = "Hz";
            double spec_rest = 0.0;
            size_t nchan = (_shape.size() > 2) ? static_cast<size_t>(_shape(2)) : 1;

            // Prefer reading frequency metadata from frequency/.zattrs (FITS-like handling)
            try {
                std::filesystem::path zarr_base(_name.c_str());
                std::filesystem::path freq_zattrs = zarr_base / "frequency" / ".zattrs";
                if (std::filesystem::exists(freq_zattrs)) {
                    spdlog::debug("ZARR WCS: Found frequency/.zattrs at {}", freq_zattrs.string());
                    std::ifstream f(freq_zattrs);
                    if (f.is_open()) {
                        nlohmann::json freq_json;
                        f >> freq_json;

                        // If explicit frequency array provided
                        if (freq_json.contains("data") && freq_json["data"].is_array()) {
                            try {
                                auto arr = freq_json["data"];
                                size_t m = arr.size();
                                casacore::Vector<casacore::Double> vals(m);
                                for (size_t i = 0; i < m; ++i) vals(i) = arr[i].get<double>();
                                spec_unit = freq_json.contains("unit") ? freq_json["unit"].get<std::string>() : spec_unit;
                                if (freq_json.contains("rest_frequency")) spec_rest = freq_json["rest_frequency"].get<double>();
                                spec_coord = SpectralCoordinate(MFrequency::TOPO, vals, spec_unit, spec_rest);
                                spec_built = true;
                                spdlog::info("ZARR WCS: Built SpectralCoordinate from frequency/.zattrs (data) nchan={}", m);
                            } catch (const std::exception& e) {
                                spdlog::warn("ZARR WCS: Failed to build spectral coord from frequency.data: {}", e.what());
                            }
                        }

                        // If not explicit, try reference + increment form (linear axis)
                        if (!spec_built && freq_json.contains("reference") && freq_json.contains("increment")) {
                            try {
                                double refval = 0.0, cdelt = 0.0;
                                auto refj = freq_json["reference"]; 
                                auto incj = freq_json["increment"];
                                // Accept either scalar or array forms
                                if (refj.is_object() && refj.contains("data")) {
                                    auto rdata = refj["data"];
                                    refval = rdata.is_array() && rdata.size() > 0 ? rdata[0].get<double>() : rdata.get<double>();
                                } else if (refj.is_number()) {
                                    refval = refj.get<double>();
                                }
                                if (incj.is_object() && incj.contains("data")) {
                                    auto idata = incj["data"];
                                    cdelt = idata.is_array() && idata.size() > 0 ? idata[0].get<double>() : idata.get<double>();
                                } else if (incj.is_number()) {
                                    cdelt = incj.get<double>();
                                }

                                casacore::Vector<casacore::Double> vals(static_cast<int>(nchan));
                                for (size_t i = 0; i < nchan; ++i) vals(i) = refval + static_cast<double>(i) * cdelt;
                                spec_unit = freq_json.contains("unit") ? freq_json["unit"].get<std::string>() : spec_unit;
                                if (freq_json.contains("rest_frequency")) spec_rest = freq_json["rest_frequency"].get<double>();
                                spec_coord = SpectralCoordinate(MFrequency::TOPO, vals, spec_unit, spec_rest);
                                spec_built = true;
                                spdlog::info("ZARR WCS: Built SpectralCoordinate from frequency/.zattrs (reference+increment) nchan={}", nchan);
                            } catch (const std::exception& e) {
                                spdlog::warn("ZARR WCS: Failed to build spectral coord from frequency.reference/increment: {}", e.what());
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                spdlog::warn("ZARR WCS: Error reading frequency/.zattrs: {}", e.what());
            }

            // If not built, leave spec_coord default and fallback will be applied later

            // Create StokesCoordinate for polarization axis (now 4th dimension in 4D)
            casacore::Vector<int> stokes_types(_shape(3)); // Use actual polarization dimension size
            for (int i = 0; i < _shape(3); ++i) {
                stokes_types(i) = casacore::Stokes::I;  // Default all to Stokes I for now
            }
            StokesCoordinate stokes_coord(stokes_types);
            
            // Add coordinates in the CARTA 4D order: [x, y, freq, stokes]
            try {
                _coord_sys.addCoordinate(dir_coord);       // axes 0,1: direction (x, y)
                spdlog::debug("ZARR WCS: Successfully added DirectionCoordinate");
                
                _coord_sys.addCoordinate(spec_coord);      // axis 2: frequency
                spdlog::debug("ZARR WCS: Successfully added SpectralCoordinate");
                
                _coord_sys.addCoordinate(stokes_coord);    // axis 3: polarization
                spdlog::debug("ZARR WCS: Successfully added StokesCoordinate");
                
                spdlog::debug("ZARR WCS: Successfully added all coordinates to CoordinateSystem");
                spdlog::debug("ZARR WCS: CoordinateSystem has {} coordinates", _coord_sys.nCoordinates());
                
                return true;
                
            } catch (const std::exception& e) {
                spdlog::error("ZARR WCS: Failed to add coordinates to CoordinateSystem: {}", e.what());
                return false;
            }
        }
        
        // Fallback for other dimension counts or missing info
        return false;
        
    } catch (std::exception& e) {
        spdlog::warn("Error parsing WCS from .zattrs: {}", e.what());
        return false;
    }
}

void CartaZarrImage::createMinimalCoordinateSystem() {
    try {
        // Create coordinate system based on actual image dimensions
        // Similar to CartaHdf5Image approach, but simplified for ZARR
        
        if (_ndim == 2) {
            // 2D image: only DirectionCoordinate (RA/DEC)
            DirectionCoordinate dir_coord;
            _coord_sys.addCoordinate(dir_coord);
            
        } else if (_ndim == 3) {
            // 3D image: DirectionCoordinate + SpectralCoordinate
            DirectionCoordinate dir_coord;
            SpectralCoordinate spec_coord;
            
            _coord_sys.addCoordinate(dir_coord);
            _coord_sys.addCoordinate(spec_coord);
            
        } else if (_ndim == 4) {
            // 4D image: DirectionCoordinate + SpectralCoordinate + StokesCoordinate
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
            
        } else {
            // For other dimensions, create a basic system with linear coordinates
            
            // Always start with DirectionCoordinate for the last 2 axes
            DirectionCoordinate dir_coord;
            
            // Add LinearCoordinates for the first axes
            for (int i = 0; i < _ndim - 2; ++i) {
                LinearCoordinate lin_coord;
                _coord_sys.addCoordinate(lin_coord);
            }
            
            // Add DirectionCoordinate for the last 2 axes
            _coord_sys.addCoordinate(dir_coord);
        }
                  
    } catch (std::exception& e) {
        spdlog::error("Exception in createMinimalCoordinateSystem: {}", e.what());
        
        // Emergency fallback: just create a basic 2D system
        try {
            _coord_sys = CoordinateSystem();  // Reset
            DirectionCoordinate dir_coord;
            _coord_sys.addCoordinate(dir_coord);
        } catch (...) {
            // Even emergency fallback failed
        }
    }
}

void CartaZarrImage::initializeTensorStore() {
    try {
        // Create TensorStore spec for Zarr using the correct format
        // Handle hierarchical zarr files (check for subdirectories with .zarray)
        std::string zarr_path = _name;
        
        // Check if this is a hierarchical zarr (has subdirectories with .zarray)
        std::filesystem::path base_path(zarr_path);
        std::filesystem::path potential_array_path;
        
        // Look for common array subdirectories like SKY, DATA, etc.
        std::vector<std::string> common_array_names = {"SKY", "DATA", "ARRAY", "0"};
        bool found_array = false;
        
        for (const auto& array_name : common_array_names) {
            potential_array_path = base_path / array_name;
            if (std::filesystem::exists(potential_array_path / ".zarray")) {
                zarr_path = potential_array_path.string();
                found_array = true;
                spdlog::info("Found Zarr array in subdirectory: {}", zarr_path);
                break;
            }
        }
        
        // If no subdirectory found, check if base path has .zarray directly
        if (!found_array && !std::filesystem::exists(base_path / ".zarray")) {
            spdlog::error("No .zarray file found in {} or its subdirectories", _name);
            return;
        }
        
        if (!found_array) {
            zarr_path = _name;  // Use original path
            spdlog::info("Using direct Zarr path: {}", zarr_path);
        }
        
        // Create TensorStore spec using the format from extract_slice.cc example
        nlohmann::json spec_json = {
            {"driver", "zarr2"},
            {"kvstore", {
                {"driver", "file"},
                {"path", zarr_path}
            }}
        };
        
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            spdlog::error("Failed to create TensorStore spec for {}: {}", zarr_path, spec_result.status().ToString());
            return;
        }
        
        auto input_spec = spec_result.value();
        
        // Open input tensorstore and resolve the bounds using the pattern from extract_slice.cc
        auto open_future = tensorstore::Open(
            input_spec, 
            _context, 
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        );
        
        auto open_result = open_future.result();
        if (!open_result.ok()) {
            spdlog::error("Failed to open TensorStore for {}: {}", _name, open_result.status().ToString());
            return;
        }
        
        _tensorstore = std::move(open_result).value();
        _tensorstore_initialized = true;
        
        // Verify that the data type is float32 as expected
        auto ts_dtype = _tensorstore.dtype();
        spdlog::info("TensorStore data type: {}", ts_dtype.name());
        
        // Check if data type is float32 (TensorStore uses "float32" as the name)
        if (ts_dtype.name() != "float32") {
            spdlog::warn("TensorStore data type is not float32, got: {}", ts_dtype.name());
        }
        
        spdlog::info("Successfully initialized TensorStore for {}", _name);
        
        // Verify shape matches what we read from .zarray
        auto ts_domain = _tensorstore.domain();
        auto ts_shape = ts_domain.shape();
        
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
            } else {
                _shape = _original_zarr_shape;
            }
        }
        _ndim = _shape.size();
        
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
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        const IPosition& stride = section.stride();
        
        // Extract channel information for caching
        int freq_index = (start.size() > 2) ? start[2] : 0;
        int stokes_index = (start.size() > 3) ? start[3] : 0;
        
        // Create a unique channel identifier combining freq and stokes
        int current_channel = freq_index * 1000 + stokes_index;  // Assuming max 1000 stokes per freq
        
        // Decide caching strategy based on request size
        int req_width = length[0];
        int req_height = length[1];
        int full_width = _shape[0];  // CARTA width
        int full_height = _shape[1]; // CARTA height
        
        // Handle special cases for 1D slices (profiles) and single point requests
        bool is_horizontal_profile = (req_height == 1);  // Horizontal line (y-profile)
        bool is_vertical_profile = (req_width == 1);     // Vertical line (x-profile)
        bool is_single_point = (req_width == 1 && req_height == 1); // Single pixel request
        bool is_1d_profile = (is_horizontal_profile || is_vertical_profile) && !is_single_point;
        
        // Use region cache for small requests (e.g., z-profile regions, small tiles)
        // For 1D profiles, always use full channel cache to avoid complexity
        // For single point requests, use direct read to avoid loading entire channel
        bool use_region_cache = !is_1d_profile && (req_width < full_width / 4 && req_height < full_height / 4);
        bool use_direct_read = is_single_point;  // Skip all caching for single point requests
        
        spdlog::debug("CACHE STRATEGY DECISION:");
        spdlog::debug("  Request size: {}x{}, Full size: {}x{}", req_width, req_height, full_width, full_height);
        spdlog::debug("  1D Profile detected: {} (horizontal={}, vertical={})", is_1d_profile ? "YES" : "NO", is_horizontal_profile, is_vertical_profile);
        spdlog::debug("  Single point request: {}", is_single_point ? "YES" : "NO");
        spdlog::debug("  Use region cache: {}", use_region_cache ? "YES" : "NO");
        spdlog::debug("  Use direct read: {}", use_direct_read ? "YES" : "NO");
        spdlog::debug("  Current cache status: loaded={}, channel={}, is_full={}", _channel_cache_loaded, _cached_channel, _is_full_channel_cache);
        spdlog::debug("  Target channel: {}", current_channel);
        
        // For single point requests, skip all caching and read directly from TensorStore
        if (use_direct_read) {
            spdlog::debug("SINGLE POINT OPTIMIZATION: Reading directly from TensorStore without caching");
            return readDirectFromTensorStore(buffer, section);
        }
        
        // Check if we need to load a different channel into cache or switch cache type
        if (!_channel_cache_loaded || _cached_channel != current_channel || 
            (use_region_cache && _is_full_channel_cache) ||     // Want region but have full
            (!use_region_cache && !_is_full_channel_cache)) {   // Want full but have region
            
            if (use_region_cache) {
                // For small requests, load only the required region with some padding
                int padding = std::min(100, std::min(req_width, req_height));  // Add padding for future nearby requests
                int region_start_x = std::max(0, static_cast<int>(start[0]) - padding);
                int region_start_y = std::max(0, static_cast<int>(start[1]) - padding);
                int region_width = std::min(full_width - region_start_x, req_width + 2 * padding);
                int region_height = std::min(full_height - region_start_y, req_height + 2 * padding);
                
                spdlog::info("Loading region cache [freq={}, stokes={}]: {}x{} at ({},{}) with padding {}", 
                            freq_index, stokes_index, region_width, region_height, region_start_x, region_start_y, padding);
                
                if (!loadRegionCache(freq_index, stokes_index, region_start_x, region_start_y, region_width, region_height)) {
                    spdlog::warn("Failed to load region cache, trying full channel cache");
                    use_region_cache = false;  // Fall back to full channel cache
                }
            }
            
            if (!use_region_cache) {
                spdlog::info("Loading full channel [freq={}, stokes={}] into cache for faster access", freq_index, stokes_index);
                if (!loadChannelCache(freq_index, stokes_index)) {
                    spdlog::warn("Failed to load channel cache, falling back to direct read");
                    // Fall through to direct read
                } else {
                    // Try to get slice from cache
                    if (getSliceFromCache(buffer, section)) {
                        return true;
                    }
                    // If cache extraction failed, fall through to direct read
                }
            } else {
                // Try to get slice from region cache
                if (getSliceFromCache(buffer, section)) {
                    return true;
                }
                // If cache extraction failed, fall through to direct read
            }
        } else {
            // Cache is already loaded for this channel, use it
            if (getSliceFromCache(buffer, section)) {
                return true;
            }
            // If cache extraction failed, fall through to direct read
        }
        
        // Direct read from TensorStore (fallback or non-first-channel)
        // Map CARTA dimensions back to original ZARR dimensions for 5D case
        IPosition zarr_start = start;
        IPosition zarr_length = length;
        
        spdlog::debug("COORDINATE MAPPING DEBUG:");
        spdlog::debug("  Original CARTA request: start={}, length={}", start.toString(), length.toString());
        spdlog::debug("  CARTA shape: {}", _shape.toString());
        spdlog::debug("  ZARR original shape: {}", _original_zarr_shape.toString());
        
        if (_original_zarr_shape.size() == 5 && start.size() >= 2) {
            // CARTA 4D format: [l, m, freq, pol] = [x, y, freq, stokes]
            // Original ZARR 5D format: [time, freq, pol, l, m]
            zarr_start.resize(5);
            zarr_length.resize(5);
            
            // Use actual channel indices from the request
            int freq_index = (start.size() > 2) ? start[2] : 0;
            int stokes_index = (start.size() > 3) ? start[3] : 0;
            
            zarr_start[0] = 0;                                      // ZARR[0]=time (always 0)
            zarr_start[1] = freq_index;                             // ZARR[1]=freq (actual requested frequency)
            zarr_start[2] = stokes_index;                           // ZARR[2]=pol (actual requested stokes)
            zarr_start[3] = start[0];                               // ZARR[3]=l (exact x start)
            zarr_start[4] = start[1];                               // ZARR[4]=m (exact y start)
            
            zarr_length[0] = 1;                                     // ZARR[0]=time (always 1)
            zarr_length[1] = 1;                                     // ZARR[1]=freq (single frequency)
            zarr_length[2] = 1;                                     // ZARR[2]=pol (single polarization)
            zarr_length[3] = length[0];                             // ZARR[3]=l (exact width requested)
            zarr_length[4] = length[1];                             // ZARR[4]=m (exact height requested)
            
            spdlog::debug("  Mapped to ZARR 5D: start={}, length={}", zarr_start.toString(), zarr_length.toString());
            spdlog::debug("  ZARR coordinate check: time={}, freq={}, pol={}, l=[{},{}), m=[{},{})", 
                         zarr_start[0], zarr_start[1], zarr_start[2], 
                         zarr_start[3], zarr_start[3] + zarr_length[3],
                         zarr_start[4], zarr_start[4] + zarr_length[4]);
        }
        
        // Create a Box for slicing all dimensions at once
        std::vector<tensorstore::Index> box_origin(zarr_start.size());
        std::vector<tensorstore::Index> box_shape(zarr_start.size());
        
        for (size_t i = 0; i < zarr_start.size(); ++i) {
            box_origin[i] = zarr_start[i];
            box_shape[i] = zarr_length[i];
        }
        
        tensorstore::Box<> slice_box(box_origin, box_shape);
        
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
        std::copy(src_data, src_data + num_elements, dest_data);
        
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

bool CartaZarrImage::loadChannelCache(int freq_channel, int stokes_channel) {
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for channel cache loading");
        return false;
    }
    
    try {
        // For 5D ZARR, load the entire specified channel [time=0, freq=freq_channel, pol=stokes_channel, l=all, m=all]
        if (_original_zarr_shape.size() == 5) {
            _cache_width = _original_zarr_shape[3];   // l dimension (width)
            _cache_height = _original_zarr_shape[4];  // m dimension (height)
            
            spdlog::info("Cache dimensions from ZARR shape: width={}, height={}", _cache_width, _cache_height);
            spdlog::info("CARTA shape: [{}, {}]", _shape[0], _shape[1]);
            spdlog::info("ZARR original shape: [{}]", fmt::join(_original_zarr_shape, ", "));
            
            // Create box for entire channel
            std::vector<tensorstore::Index> box_origin = {0, freq_channel, stokes_channel, 0, 0};
            std::vector<tensorstore::Index> box_shape = {1, 1, 1, _cache_width, _cache_height};
            
            spdlog::debug("CACHE LOADING COORDINATES:");
            spdlog::debug("  ZARR 5D cache box: origin=[{},{},{},{},{}], shape=[{},{},{},{},{}]", 
                         box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4],
                         box_shape[0], box_shape[1], box_shape[2], box_shape[3], box_shape[4]);
            spdlog::debug("  This reads: time={}, freq={}, pol={}, l=[0,{}), m=[0,{})", 
                         freq_channel, freq_channel, stokes_channel, _cache_width, _cache_height);
            
            tensorstore::Box<> cache_box(box_origin, box_shape);
            
            // Apply the box slice to the TensorStore
            auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(cache_box);
            if (!constrained_store.ok()) {
                spdlog::error("Failed to create cache slice: {}", constrained_store.status().ToString());
                return false;
            }
            
            // Read entire channel data
            auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
            if (!read_result.ok()) {
                spdlog::error("Failed to read channel cache: {}", read_result.status().ToString());
                return false;
            }
            
            auto zarr_array = std::move(read_result.value());
            
            // Verify data type
            if (zarr_array.dtype().name() != "float32") {
                spdlog::error("Unexpected data type in cache: {}", zarr_array.dtype().name());
                return false;
            }
            
            // Copy data to cache
            size_t total_elements = _cache_width * _cache_height;
            _channel_cache.resize(total_elements);
            
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            std::copy(src_data, src_data + total_elements, _channel_cache.data());
            
            // Store the cached channel identifier
            _cached_channel = freq_channel * 1000 + stokes_channel;
            _cache_start_x = 0;  // Full channel cache starts at origin
            _cache_start_y = 0;
            _channel_cache_loaded = true;
            _is_full_channel_cache = true;  // This is a full channel cache
            
            spdlog::info("Loaded channel [freq={}, stokes={}] cache: {}x{} pixels ({} MB)", 
                        freq_channel, stokes_channel, _cache_width, _cache_height, 
                        (total_elements * sizeof(float)) / (1024 * 1024));
            
            // DEBUG: Check TensorStore data layout and verify expected NaN boundaries
            spdlog::debug("TENSORSTORE DATA LAYOUT DEBUG:");
            spdlog::debug("  TensorStore array shape: {}", zarr_array.shape().size());
            for (size_t i = 0; i < zarr_array.shape().size(); ++i) {
                spdlog::debug("    Dimension {}: {}", i, zarr_array.shape()[i]);
            }
            
            // Check first few and expected boundary rows specifically 
            std::vector<int> test_rows = {0, 1, 2, 183, 184, 249, 250, 251, 252, 253};
            for (int row : test_rows) {
                if (row >= _cache_height) continue;
                
                // Sample a few columns in this row
                std::vector<int> sample_cols = {100, 1000, 3000, 5000, 7000};
                int nan_count = 0, finite_count = 0;
                
                for (int col : sample_cols) {
                    if (col >= _cache_width) continue;
                    
                    size_t cache_idx = row * _cache_width + col;
                    if (cache_idx < _channel_cache.size()) {
                        float val = _channel_cache[cache_idx];
                        if (std::isnan(val)) {
                            nan_count++;
                        } else if (std::isfinite(val)) {
                            finite_count++;
                        }
                    }
                }
                
                spdlog::debug("  Row {}: {}/{} NaN, {}/{} finite (expected: row<250 should be NaN)", 
                             row, nan_count, sample_cols.size(), finite_count, sample_cols.size());
            }
            
            // Sample a few key positions to understand data layout
            spdlog::debug("CACHE DATA SAMPLING:");
            for (int y = 0; y < std::min(5, _cache_height); y++) {
                int x = 0;
                int idx = y * _cache_width + x;
                float val = _channel_cache[idx];
                spdlog::debug("  row={}, col={}, idx={}, value={}, isNaN={}", y, x, idx, val, std::isnan(val));
            }
            
            // Precise boundary analysis based on image data
            if (_cache_height > 250) {
                spdlog::debug("PRECISE BOUNDARY ANALYSIS:");
                
                // Sample multiple columns to get accurate boundary
                std::vector<int> sample_columns = {100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000};
                int consensus_first_row = -1;
                int consensus_last_row = -1;
                
                // Check each sample column
                for (int col : sample_columns) {
                    if (col >= _cache_width) continue;
                    
                    int col_first_row = -1;
                    int col_last_row = -1;
                    
                    for (int y = 0; y < _cache_height; y++) {
                        int idx = y * _cache_width + col;
                        float val = _channel_cache[idx];
                        if (!std::isnan(val) && std::isfinite(val)) {
                            if (col_first_row == -1) col_first_row = y;
                            col_last_row = y;
                        }
                    }
                    
                    spdlog::debug("  Column {}: first_row={}, last_row={}", col, col_first_row, col_last_row);
                    
                    // Update consensus (use most restrictive bounds)
                    if (col_first_row >= 0) {
                        if (consensus_first_row == -1 || col_first_row < consensus_first_row) {
                            consensus_first_row = col_first_row;
                        }
                    }
                    if (col_last_row >= 0) {
                        if (consensus_last_row == -1 || col_last_row > consensus_last_row) {
                            consensus_last_row = col_last_row;
                        }
                    }
                }
                
                // Additional detailed analysis around discovered boundaries
                if (consensus_first_row >= 0) {
                    spdlog::debug("DETAILED BOUNDARY EXAMINATION:");
                    int check_start = std::max(0, consensus_first_row - 5);
                    int check_end = std::min(_cache_height - 1, consensus_first_row + 5);
                    
                    for (int y = check_start; y <= check_end; y++) {
                        // Count valid pixels in this row
                        int valid_pixels = 0;
                        int total_pixels = 0;
                        for (int x = 0; x < _cache_width; x += 100) { // Sample every 100 pixels
                            int idx = y * _cache_width + x;
                            float val = _channel_cache[idx];
                            if (!std::isnan(val) && std::isfinite(val)) {
                                valid_pixels++;
                            }
                            total_pixels++;
                        }
                        float valid_ratio = (float)valid_pixels / total_pixels;
                        spdlog::debug("    Row {}: {}/{} valid pixels ({:.1f}%)", 
                                     y, valid_pixels, total_pixels, valid_ratio * 100);
                    }
                }
                
                spdlog::info("FINAL BOUNDARY ANALYSIS:");
                spdlog::info("  Consensus first data row: {}", consensus_first_row);
                spdlog::info("  Consensus last data row: {}", consensus_last_row);
                spdlog::info("  Total data rows: {}", (consensus_first_row >= 0 && consensus_last_row >= 0) ? (consensus_last_row - consensus_first_row + 1) : 0);
                spdlog::info("  NaN header rows: {}", consensus_first_row >= 0 ? consensus_first_row : _cache_height);
                spdlog::info("  NaN footer rows: {}", consensus_last_row >= 0 ? (_cache_height - consensus_last_row - 1) : 0);
                spdlog::info("  Data coverage: {:.1f}% of image height", 
                            consensus_first_row >= 0 && consensus_last_row >= 0 ? 
                            ((float)(consensus_last_row - consensus_first_row + 1) / _cache_height * 100) : 0.0);
            }
            
            return true;
        } else {
            // For non-5D arrays, load the entire 2D image (ignore freq/stokes parameters)
            _cache_width = _shape[0];
            _cache_height = _shape[1];
            
            // Create box for entire image
            std::vector<tensorstore::Index> box_origin(_original_zarr_shape.size(), 0);
            std::vector<tensorstore::Index> box_shape;
            for (size_t i = 0; i < _original_zarr_shape.size(); ++i) {
                box_shape.push_back(_original_zarr_shape[i]);
            }
            
            tensorstore::Box<> cache_box(box_origin, box_shape);
            
            auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(cache_box);
            if (!constrained_store.ok()) {
                spdlog::error("Failed to create cache slice: {}", constrained_store.status().ToString());
                return false;
            }
            
            auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
            if (!read_result.ok()) {
                spdlog::error("Failed to read channel cache: {}", read_result.status().ToString());
                return false;
            }
            
            auto zarr_array = std::move(read_result.value());
            
            // Debug: Check array properties
            spdlog::debug("ZARR array dtype: {}", zarr_array.dtype().name());
            spdlog::debug("ZARR array shape: [{}]", fmt::join(zarr_array.shape(), ", "));
            
            size_t total_elements = _cache_width * _cache_height;
            _channel_cache.resize(total_elements);
            
            // Check if data type is float32
            if (zarr_array.dtype() != tensorstore::dtype_v<float>) {
                spdlog::error("ZARR array dtype is not float32: {}", zarr_array.dtype().name());
                return false;
            }
            
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            
            // Debug: Check first few values
            spdlog::debug("First 5 ZARR values: [{}, {}, {}, {}, {}]", 
                         src_data[0], src_data[1], src_data[2], src_data[3], src_data[4]);
            
            std::copy(src_data, src_data + total_elements, _channel_cache.data());
            
            // Debug: Check first few cached values
            spdlog::debug("First 5 cached values: [{}, {}, {}, {}, {}]", 
                         _channel_cache[0], _channel_cache[1], _channel_cache[2], _channel_cache[3], _channel_cache[4]);
            
            _cached_channel = freq_channel * 1000 + stokes_channel;
            _channel_cache_loaded = true;
            
            spdlog::info("Loaded image cache: {}x{} pixels", _cache_width, _cache_height);
            return true;
        }
        
    } catch (std::exception& e) {
        spdlog::error("Exception loading channel cache: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::getSliceFromCache(casacore::Array<float>& buffer, const casacore::Slicer& section) {
    if (!_channel_cache_loaded) {
        return false;
    }
    
    try {
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        
        // Extract region parameters
        int start_x = start[0];  // CARTA x = ZARR l dimension
        int start_y = start[1];  // CARTA y = ZARR m dimension  
        int req_width = length[0];   // Requested slice width
        int req_height = length[1];
        
        // ENHANCED PARAMETER VALIDATION
        if (req_width <= 0 || req_height <= 0) {
            spdlog::error("Invalid request dimensions: width={}, height={}", req_width, req_height);
            return false;
        }
        
        if (_cache_width <= 0 || _cache_height <= 0) {
            spdlog::error("Invalid cache dimensions: width={}, height={}", _cache_width, _cache_height);
            return false;
        }
        
        if (_channel_cache.empty()) {
            spdlog::error("Cache is empty but marked as loaded");
            return false;
        }
        
        spdlog::debug("CACHE SLICE REQUEST:");
        spdlog::debug("  Request: start_x={}, start_y={}, width={}, height={}", start_x, start_y, req_width, req_height);
        spdlog::debug("  Cache size: {} elements, expected: {}", _channel_cache.size(), _cache_width * _cache_height);
        
        // Handle region cache vs full channel cache differently
        if (_is_full_channel_cache) {
            spdlog::debug("  Using full channel cache: {}x{}", _cache_width, _cache_height);
        } else {
            spdlog::debug("  Using region cache: {}x{} at ({},{}) in full image", _cache_width, _cache_height, _cache_start_x, _cache_start_y);
            
            // For region cache, adjust coordinates relative to cached region
            start_x -= _cache_start_x;
            start_y -= _cache_start_y;
            
            spdlog::debug("  Adjusted coordinates for region cache: start_x={}, start_y={}", start_x, start_y);
            
            // Check if request is completely outside region cache bounds
            if (start_x >= _cache_width || start_y >= _cache_height || 
                start_x + req_width <= 0 || start_y + req_height <= 0) {
                spdlog::debug("  Request completely outside region cache bounds");
                return false;  // Request is outside our cached region
            }
        }
        
        spdlog::debug("  Request end coordinates: end_x={}, end_y={}", start_x + req_width - 1, start_y + req_height - 1);
        
        // Calculate downsampling parameters
        int full_width = _cache_width;
        float downsample_factor = (float)full_width / req_width;
        
        spdlog::debug("DOWNSAMPLING PARAMETERS:");
        spdlog::debug("  Full width: {}, Requested width: {}", full_width, req_width);
        spdlog::debug("  Downsample factor: {:.2f}", downsample_factor);
        spdlog::debug("  Pixels per bin: {:.2f}", downsample_factor);
        
        // Check if this is a problematic area - updated based on actual discovered data boundaries
        if (start_y <= 200 && start_y + req_height > 180) {
            spdlog::debug("  *** TRANSITION AREA: Request spans rows {}-{} which includes data transition around row 183 ***", 
                         start_y, start_y + req_height - 1);
        }
        if (start_y < 183) {
            spdlog::debug("  Header region: Request starts before row 183 (expected NaN area)");
        }
        if (start_y >= 183) {
            spdlog::debug("  Data region: Request starts at/after row 183 (actual data area)");
        }
        
        spdlog::debug("Boundary check: start_x={} vs cache_width={}, start_y={} vs cache_height={}", 
                     start_x, _cache_width, start_y, _cache_height);
        
        // Check if request is completely outside cache bounds
        if (start_y >= _cache_height || start_y + req_height <= 0) {
            spdlog::warn("Cache slice completely outside bounds vertically: start_y={}, height={}, cache_height={}", 
                        start_y, req_height, _cache_height);
            return false;
        }
        
        // Calculate valid intersection with cache bounds (vertical only, we'll handle horizontal downsampling)
        int cache_start_y = std::max(0, start_y);
        int cache_end_y = std::min(_cache_height, start_y + req_height);
        int valid_height = cache_end_y - cache_start_y;
        
        if (valid_height <= 0) {
            spdlog::warn("No valid intersection with cache bounds");
            return false;
        }
        
        spdlog::debug("Valid cache region: start_y={}, height={}", cache_start_y, valid_height);
        
        // Resize output buffer
        buffer.resize(length);
        float* dest_data = buffer.data();
        
        // Initialize entire buffer with NaN for out-of-bounds regions
        std::fill(dest_data, dest_data + (req_width * req_height), std::numeric_limits<float>::quiet_NaN());
        
        // Process each row with downsampling
        for (int y = 0; y < req_height; ++y) {
            int src_y = start_y + y;
            if (src_y < 0 || src_y >= _cache_height) continue; // Skip out-of-bounds rows
            
            // For each output column, calculate average from corresponding input pixels
            for (int out_x = 0; out_x < req_width; ++out_x) {
                // Calculate the range of source pixels for this output pixel
                float src_x_start = out_x * downsample_factor;
                float src_x_end = (out_x + 1) * downsample_factor;
                
                int src_x_start_int = (int)std::floor(src_x_start);
                int src_x_end_int = (int)std::ceil(src_x_end);
                
                // Clamp to valid range
                src_x_start_int = std::max(0, src_x_start_int);
                src_x_end_int = std::min(full_width, src_x_end_int);
                
                float sum = 0.0f;
                float weight_sum = 0.0f;
                int valid_pixels = 0;
                
                // Average pixels in the range
                for (int src_x = src_x_start_int; src_x < src_x_end_int; ++src_x) {
                    // Additional safety check: ensure src_x is within cache width bounds
                    if (src_x < 0 || src_x >= _cache_width) {
                        spdlog::warn("src_x {} out of bounds [0, {})", src_x, _cache_width);
                        continue;
                    }
                    
                    // Calculate weight for this pixel based on overlap
                    float weight = 1.0f;
                    if (src_x == src_x_start_int && src_x_start > src_x) {
                        weight = 1.0f - (src_x_start - src_x);
                    }
                    if (src_x == src_x_end_int - 1 && src_x_end < src_x + 1) {
                        weight = src_x_end - src_x;
                    }
                    
                    // CORRECTED: Based on actual testing, ZARR data layout requires column-major indexing
                    // For ZARR [time, freq, pol, l, m], the cache indexing follows ZARR's internal layout
                    // So: cache_idx = x * height + y (column-major order)
                    size_t cache_idx = src_x * _cache_height + src_y;
                    
                    // Enhanced boundary validation
                    if (cache_idx >= _channel_cache.size()) {
                        spdlog::error("BOUNDARY VIOLATION: cache_idx {} >= cache_size {}, src_x={}, src_y={}, cache_width={}, cache_height={}", 
                                     cache_idx, _channel_cache.size(), src_x, src_y, _cache_width, _cache_height);
                        continue;
                    }
                    
                    float val = _channel_cache[cache_idx];
                    if (!std::isnan(val) && std::isfinite(val)) {
                        sum += val * weight;
                        weight_sum += weight;
                        valid_pixels++;
                    }
                }
                
                size_t dest_idx = y * req_width + out_x;
                
                // Enhanced boundary validation for destination buffer
                size_t buffer_size = req_width * req_height;
                if (dest_idx >= buffer_size) {
                    spdlog::error("DEST BUFFER OVERFLOW: dest_idx {} >= buffer_size {}, y={}, out_x={}, req_width={}, req_height={}", 
                                 dest_idx, buffer_size, y, out_x, req_width, req_height);
                    continue;
                }
                
                // Set output value
                if (weight_sum > 0.0f && valid_pixels > 0) {
                    dest_data[dest_idx] = sum / weight_sum;
                } else {
                    dest_data[dest_idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        
        // Debug: Check extracted data
        if (req_width > 0 && req_height > 0) {
            size_t valid_count = 0;
            size_t nan_count = 0;
            for (int i = 0; i < req_width * req_height; ++i) {
                if (std::isnan(dest_data[i])) {
                    nan_count++;
                } else {
                    valid_count++;
                }
            }
            spdlog::debug("Downsampled data: {} valid, {} NaN pixels (factor: {:.2f})", 
                         valid_count, nan_count, downsample_factor);
            
            // Special debug check for downsampled matrices
            if (req_width > 100) {  // Only for significant width requests
                bool first_is_nan = std::isnan(dest_data[0]);
                bool last_is_nan = std::isnan(dest_data[req_width - 1]);
                
                spdlog::debug("Downsampled matrix debug: first_pixel={}, last_pixel={}", 
                             first_is_nan ? "NaN" : std::to_string(dest_data[0]),
                             last_is_nan ? "NaN" : std::to_string(dest_data[req_width - 1]));
                
                // Additional check for row position and data transition
                spdlog::debug("Row position: y={} out of {} total rows (0-indexed)", start_y, _cache_height);
                if (start_y >= _cache_height - 10) {
                    spdlog::debug("→ This is near the bottom edge, NaN values are expected");
                } else if (start_y < 183) {  // Actual data transition boundary discovered
                    if (valid_count > 0) {
                        spdlog::info("✓ DATA TRANSITION: Row {} has {} valid pixels - data region starting!", 
                                    start_y, valid_count);
                    } else {
                        spdlog::debug("→ Row {} in header region, all NaN expected", start_y);
                    }
                } else {
                    spdlog::debug("→ Row {} in main data region, expecting valid pixels", start_y);
                    if (valid_count == 0) {
                        spdlog::warn("Unexpected: Row {} in data region has no valid pixels", start_y);
                    }
                }
                
                // Show sample values for rows with mixed data
                if (valid_count > 0 && nan_count > 0) {
                    spdlog::debug("Mixed data row - showing first few valid values:");
                    int shown = 0;
                    for (int i = 0; i < req_width && shown < 5; ++i) {
                        if (!std::isnan(dest_data[i])) {
                            spdlog::debug("  pixel[{}] = {}", i, dest_data[i]);
                            shown++;
                        }
                    }
                }
            }
        }
        
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception extracting slice from cache: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::loadRegionCache(int freq_channel, int stokes_channel, int start_x, int start_y, int width, int height) {
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for region cache loading");
        return false;
    }
    
    // Safety checks for region dimensions
    if (width <= 0 || height <= 0) {
        spdlog::error("Invalid region dimensions: {}x{} at ({},{})", width, height, start_x, start_y);
        return false;
    }
    
    try {
        // For 5D ZARR, load only the specified region [time=0, freq=freq_channel, pol=stokes_channel, l=start_x:end_x, m=start_y:end_y]
        if (_original_zarr_shape.size() == 5) {
            // Clamp region to valid bounds
            int max_width = _original_zarr_shape[3];   // l dimension (width)
            int max_height = _original_zarr_shape[4];  // m dimension (height)
            
            start_x = std::max(0, std::min(start_x, max_width - 1));
            start_y = std::max(0, std::min(start_y, max_height - 1));
            width = std::min(width, max_width - start_x);
            height = std::min(height, max_height - start_y);
            
            if (width <= 0 || height <= 0) {
                spdlog::warn("Invalid region dimensions: {}x{} at ({},{})", width, height, start_x, start_y);
                return false;
            }
            
            spdlog::info("Loading region cache [freq={}, stokes={}]: {}x{} region at ({},{}) from full {}x{}", 
                        freq_channel, stokes_channel, width, height, start_x, start_y, max_width, max_height);
            
            // Create box for specific region
            std::vector<tensorstore::Index> box_origin = {0, freq_channel, stokes_channel, start_x, start_y};
            std::vector<tensorstore::Index> box_shape = {1, 1, 1, width, height};
            
            spdlog::debug("REGION CACHE LOADING COORDINATES:");
            spdlog::debug("  ZARR 5D region box: origin=[{},{},{},{},{}], shape=[{},{},{},{},{}]", 
                         box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4],
                         box_shape[0], box_shape[1], box_shape[2], box_shape[3], box_shape[4]);
            
            tensorstore::Box<> cache_box(box_origin, box_shape);
            
            // Apply the box slice to the TensorStore
            auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(cache_box);
            if (!constrained_store.ok()) {
                spdlog::error("Failed to create region cache slice: {}", constrained_store.status().ToString());
                return false;
            }
            
            // Read region data
            auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
            if (!read_result.ok()) {
                spdlog::error("Failed to read region cache: {}", read_result.status().ToString());
                return false;
            }
            
            auto zarr_array = std::move(read_result.value());
            
            // Verify data type
            if (zarr_array.dtype().name() != "float32") {
                spdlog::error("Unexpected data type in region cache: {}", zarr_array.dtype().name());
                return false;
            }
            
            // Copy data to cache
            size_t total_elements = width * height;
            _channel_cache.resize(total_elements);
            
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            std::copy(src_data, src_data + total_elements, _channel_cache.data());
            
            // Store the cached region information
            _cached_channel = freq_channel * 1000 + stokes_channel;
            _cache_width = width;
            _cache_height = height;
            _cache_start_x = start_x;
            _cache_start_y = start_y;
            _channel_cache_loaded = true;
            _is_full_channel_cache = false;  // This is a region cache, not full channel
            
            spdlog::info("Loaded region [freq={}, stokes={}] cache: {}x{} pixels at ({},{}) ({} KB)", 
                        freq_channel, stokes_channel, width, height, start_x, start_y,
                        (total_elements * sizeof(float)) / 1024);
            
            return true;
        }
        
        spdlog::warn("Region cache only supported for 5D ZARR files");
        return false;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in loadRegionCache: {}", e.what());
        return false;
    }
}

Bool CartaZarrImage::readDirectFromTensorStore(Array<float>& buffer, const Slicer& section) {
    try {
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        
        // CRITICAL FIX: Handle coordinate mapping based on actual section dimensions
        // The issue is that ZarrLoader is passing coordinates in ZARR order already!
        // We need to map the incoming slicer coordinates correctly
        
        spdlog::debug("readDirectFromTensorStore: section start={}, length={}, shape={}", 
                     start.toString(), length.toString(), _original_zarr_shape.toString());
        
        std::vector<tensorstore::Index> box_origin;
        std::vector<tensorstore::Index> box_shape;
        
        if (start.size() == 5) {
            // 5D case: ZarrLoader already provides ZARR order [time, freq, ?, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            for (int i = 0; i < 5; ++i) {
                box_origin[i] = start[i];
                box_shape[i] = length[i];
            }
        } else if (start.size() == 4) {
            // 4D case: Map to 5D ZARR [time, freq, ?, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;           // time = 0
            box_origin[1] = start[0];    // freq from ZarrLoader
            box_origin[2] = start[1];    // ? (usually 0)
            box_origin[3] = start[2];    // y 
            box_origin[4] = start[3];    // x
            box_shape[0] = 1;            // time = 1
            box_shape[1] = length[0];    // freq length
            box_shape[2] = length[1];    // ? length
            box_shape[3] = length[2];    // y length
            box_shape[4] = length[3];    // x length
        } else if (start.size() == 3) {
            // 3D case: Map to 5D ZARR [time, freq, ?, y, x]  
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;           // time = 0
            box_origin[1] = start[0];    // freq from ZarrLoader
            box_origin[2] = 0;           // ? = 0
            box_origin[3] = start[1];    // y
            box_origin[4] = start[2];    // x
            box_shape[0] = 1;            // time = 1
            box_shape[1] = length[0];    // freq length
            box_shape[2] = 1;            // ? = 1
            box_shape[3] = length[1];    // y length
            box_shape[4] = length[2];    // x length
        } else if (start.size() == 2) {
            // 2D case: Map to 5D ZARR [time, freq, ?, y, x] with single channel
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;           // time = 0
            box_origin[1] = 0;           // freq = 0 (first channel)
            box_origin[2] = 0;           // stokes = 0
            box_origin[3] = start[0];    // y
            box_origin[4] = start[1];    // x
            box_shape[0] = 1;            // time = 1
            box_shape[1] = 1;            // freq = 1 (single channel)
            box_shape[2] = 1;            // ? = 1
            box_shape[3] = length[0];    // y length
            box_shape[4] = length[1];    // x length
        } else {
            spdlog::error("readDirectFromTensorStore: Unsupported section dimensions: {}", start.size());
            return false;
        }
        
        // Validate coordinates against ZARR bounds
        for (size_t i = 0; i < box_origin.size(); ++i) {
            if (box_origin[i] < 0 || 
                (i < _original_zarr_shape.size() && box_origin[i] + box_shape[i] > _original_zarr_shape[i])) {
                spdlog::error("readDirectFromTensorStore: Coordinate {} out of bounds: origin={}, shape={}, max={}", 
                             i, box_origin[i], box_shape[i], 
                             i < _original_zarr_shape.size() ? _original_zarr_shape[i] : -1);
                return false;
            }
        }
        
        spdlog::debug("DIRECT READ: TensorStore slice [time={}, freq={}, ?={}, y={}:{}, x={}:{}]",
                     box_origin[0], box_origin[1], box_origin[2], 
                     box_origin[3], box_origin[3] + box_shape[3] - 1,
                     box_origin[4], box_origin[4] + box_shape[4] - 1);
        
        tensorstore::Box<> cache_box(box_origin, box_shape);
        
        // Apply the box slice to the TensorStore
        auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(cache_box);
        if (!constrained_store.ok()) {
            spdlog::error("Failed to create direct read slice: {}", constrained_store.status().ToString());
            return false;
        }
        
        // Read data directly
        auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
        if (!read_result.ok()) {
            spdlog::error("Failed to read direct data: {}", read_result.status().ToString());
            return false;
        }
        
        auto zarr_array = std::move(read_result.value());
        
        // Verify data type
        if (zarr_array.dtype().name() != "float32") {
            spdlog::error("Unexpected data type in direct read: {}", zarr_array.dtype().name());
            return false;
        }
        
        // Copy data to output buffer
        size_t total_elements = 1;
        for (int i = 0; i < length.size(); ++i) {
            total_elements *= length[i];
        }
        buffer.resize(length);
        
        const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
        std::copy(src_data, src_data + total_elements, buffer.data());
        
        spdlog::debug("DIRECT READ: Successfully read {} elements (shape={})", total_elements, length.toString());
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in readDirectFromTensorStore: {}", e.what());
        return false;
    }
}

} // namespace carta
