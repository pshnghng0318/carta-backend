/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"
#include "Logger/Logger.h"

#include "Util/Casacore.h"

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
            
            // Read dtype from .zarray metadata
            if (zarray_json.contains("dtype")) {
                std::string dtype_str = zarray_json["dtype"].get<std::string>();
                spdlog::info("ZARR dtype: {}", dtype_str);
                
                // Parse ZARR dtype to casacore DataType
                if (dtype_str == "<f4" || dtype_str == ">f4" || dtype_str == "float32") {
                    _actual_data_type = casacore::DataType::TpFloat;
                } else if (dtype_str == "<f8" || dtype_str == ">f8" || dtype_str == "float64") {
                    _actual_data_type = casacore::DataType::TpDouble;
                } else if (dtype_str == "<i4" || dtype_str == ">i4" || dtype_str == "int32") {
                    _actual_data_type = casacore::DataType::TpInt;
                } else if (dtype_str == "<i8" || dtype_str == ">i8" || dtype_str == "int64") {
                    _actual_data_type = casacore::DataType::TpInt64;
                } else {
                    spdlog::warn("Unknown ZARR dtype {}, defaulting to float32", dtype_str);
                    _actual_data_type = casacore::DataType::TpFloat;
                }
                spdlog::info("Mapped ZARR dtype {} to casacore DataType", dtype_str);
            }
            
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
            
            // spdlog::info("Found .zattrs file for Zarr image: {}", _name);
            // spdlog::debug("ZARR WCS: About to call parseWCSFromZattrs");

            // Try to read brightness unit from .zattrs (top-level or SKY/.zattrs)
            try {
                std::string found_bunit;

                // Common keys: BUNIT, bunit, units, brightness_unit
                if (zattrs_json.contains("BUNIT") && zattrs_json["BUNIT"].is_string()) {
                    found_bunit = zattrs_json["BUNIT"].get<std::string>();
                } else if (zattrs_json.contains("bunit") && zattrs_json["bunit"].is_string()) {
                    found_bunit = zattrs_json["bunit"].get<std::string>();
                } else if (zattrs_json.contains("units") && zattrs_json["units"].is_string()) {
                    found_bunit = zattrs_json["units"].get<std::string>();
                } else if (zattrs_json.contains("brightness_unit") && zattrs_json["brightness_unit"].is_string()) {
                    found_bunit = zattrs_json["brightness_unit"].get<std::string>();
                }

                // If not found at top-level, check SKY/.zattrs (common convention for derived arrays)
                if (found_bunit.empty()) {
                    std::filesystem::path zarr_path(_name.c_str());
                    std::filesystem::path sky_zattrs = zarr_path / "SKY" / ".zattrs";
                    if (std::filesystem::exists(sky_zattrs)) {
                        try {
                            std::ifstream sky_file(sky_zattrs);
                            nlohmann::json sky_json;
                            sky_file >> sky_json;
                            if (sky_json.contains("units") && sky_json["units"].is_string()) {
                                found_bunit = sky_json["units"].get<std::string>();
                            } else if (sky_json.contains("BUNIT") && sky_json["BUNIT"].is_string()) {
                                found_bunit = sky_json["BUNIT"].get<std::string>();
                            } else if (sky_json.contains("bunit") && sky_json["bunit"].is_string()) {
                                found_bunit = sky_json["bunit"].get<std::string>();
                            }
                        } catch (const std::exception& e) {
                            spdlog::debug("ZARR UNIT: Failed to read SKY/.zattrs: {}", e.what());
                        }
                    }
                }

                if (!found_bunit.empty()) {
                    casacore::String bunit_str(found_bunit);
                    NormalizeUnit(bunit_str);
                    if (casacore::UnitVal::check(bunit_str)) {
                        // Set image brightness unit
                        setUnits(casacore::Unit(bunit_str));
                        spdlog::info("ZARR UNIT: Set image brightness unit from .zattrs: {}", bunit_str);
                    } else {
                        spdlog::warn("ZARR UNIT: Found brightness unit '{}' in .zattrs but failed to normalize/check", found_bunit);
                    }
                }
            } catch (const std::exception& e) {
                spdlog::warn("ZARR UNIT: Exception while parsing .zattrs for brightness unit: {}", e.what());
            }

            // Try to read restoring beam from BEAM zarr array if present
            try {
                std::filesystem::path beam_array = zarr_path / "BEAM";
                std::filesystem::path beam_zarray = beam_array / ".zarray";
                if (std::filesystem::exists(beam_zarray)) {
                    spdlog::info("Found BEAM array for Zarr image: {}", beam_array.string());

                    // Build TensorStore spec for BEAM array
                    nlohmann::json beam_spec_json = {
                        {"driver", "zarr2"},
                        {"kvstore", { {"driver", "file"}, {"path", beam_array.string()} }}
                    };

                    auto beam_spec_res = tensorstore::Spec::FromJson(beam_spec_json);
                    if (!beam_spec_res.ok()) {
                        spdlog::warn("Failed to create TensorStore spec for BEAM: {}", beam_spec_res.status().ToString());
                    } else {
                        auto beam_spec = beam_spec_res.value();
                        auto open_res = tensorstore::Open(beam_spec, _context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read).result();
                        if (!open_res.ok()) {
                            spdlog::warn("Failed to open BEAM TensorStore: {}", open_res.status().ToString());
                        } else {
                            auto beam_store = open_res.value();

                            // Read beam data from specific chunk 0.0.0.0 (time=0, freq=0, pol=0, all beam params)
                            // BEAM array shape is typically [time, frequency, polarization, beam_param] where beam_param=[bmaj, bmin, bpa]
                            auto domain = beam_store.domain();
                            auto shape = domain.shape();
                            
                            // Convert shape to printable format
                            std::string shape_str = "[";
                            for (size_t i = 0; i < shape.size(); ++i) {
                                if (i > 0) shape_str += ", ";
                                shape_str += std::to_string(shape[i]);
                            }
                            shape_str += "]";
                            spdlog::debug("BEAM array shape: {}", shape_str);

                            if (shape.size() >= 4 && shape[3] >= 3) {
                                // Read the first beam entry [0, 0, 0, :] which contains [bmaj, bmin, bpa]
                                std::vector<tensorstore::Index> start(shape.size(), 0);
                                std::vector<tensorstore::Index> lengths(shape.size(), 1);
                                lengths[3] = 3; // Read first 3 beam parameters

                                auto read_res = tensorstore::Read(beam_store | tensorstore::AllDims().SizedInterval(start, lengths)).result();
                                if (!read_res.ok()) {
                                    spdlog::warn("Failed to read BEAM chunk: {}", read_res.status().ToString());
                                } else {
                                    auto beam_data = std::move(read_res.value());
                                    
                                    // Extract beam parameters from the read data
                                    if (beam_data.num_elements() >= 3) {
                                        const double* data_ptr = reinterpret_cast<const double*>(beam_data.data());
                                        if (data_ptr) {
                                            // Read bmaj, bmin, bpa from the first beam entry
                                            double major = data_ptr[0];
                                            double minor = data_ptr[1];
                                            double pa = data_ptr[2];
                                            
                                            spdlog::debug("BEAM raw values: bmaj={}, bmin={}, bpa={}", major, minor, pa);

                                            // Read units from BEAM/.zattrs if available (default to radians)
                                            casacore::String beam_unit = "rad";
                                            std::filesystem::path beam_zattrs_path = beam_array / ".zattrs";
                                            if (std::filesystem::exists(beam_zattrs_path)) {
                                                try {
                                                    std::ifstream bz(beam_zattrs_path);
                                                    nlohmann::json bzjson;
                                                    bz >> bzjson;
                                                    if (bzjson.contains("units") && bzjson["units"].is_string()) {
                                                        beam_unit = bzjson["units"].get<std::string>();
                                                    }
                                                } catch (const std::exception& e) {
                                                    spdlog::debug("Failed to parse BEAM/.zattrs units: {}", e.what());
                                                }
                                            }

                                            try {
                                                casacore::Quantity qmajor(major, beam_unit);
                                                casacore::Quantity qminor(minor, beam_unit);
                                                casacore::Quantity qpa(pa, beam_unit);

                                                // Set restoring beam on image info
                                                casacore::ImageInfo ii = imageInfo();
                                                ii.setRestoringBeam(qmajor, qminor, qpa);
                                                setImageInfo(ii);
                                                spdlog::info("ZARR BEAM: Set restoring beam from BEAM/0.0.0.0: major={} {}, minor={} {}, pa={} {}",
                                                             qmajor.getValue(), qmajor.getUnit(), qminor.getValue(), qminor.getUnit(), qpa.getValue(), qpa.getUnit());
                                            } catch (const std::exception& e) {
                                                spdlog::warn("Failed to set restoring beam from BEAM array: {}", e.what());
                                            }
                                        } else {
                                            spdlog::warn("BEAM data pointer is null or insufficient data");
                                        }
                                    } else {
                                        spdlog::warn("BEAM data has insufficient elements: {}", beam_data.num_elements());
                                    }
                                }
                            } else {
                                spdlog::warn("BEAM array has unexpected shape: {}", shape_str);
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                spdlog::warn("Exception while extracting BEAM from Zarr: {}", e.what());
            }
            
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
        // spdlog::debug("ZARR WCS: Starting parseWCSFromZattrs");
        
        // First, check if we have precise coordinate arrays
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path ra_path = zarr_path / "right_ascension";
        std::filesystem::path dec_path = zarr_path / "declination";
        std::filesystem::path freq_path = zarr_path / "frequency";
        
        bool has_precise_coords = std::filesystem::exists(ra_path) && std::filesystem::exists(dec_path);
        // spdlog::debug("ZARR WCS: Precise coordinate arrays available: {}", has_precise_coords);
        
        if (has_precise_coords) {
            return parseWCSFromCoordinateArrays(ra_path, dec_path, freq_path);
        }
        
        // Fallback to metadata-based parsing
        return parseWCSFromMetadata(zattrs);
    } catch (std::exception& e) {
        // spdlog::error("ZARR WCS: Exception in parseWCSFromZattrs: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::parseWCSFromCoordinateArrays(const std::filesystem::path& ra_path, const std::filesystem::path& dec_path, const std::filesystem::path& freq_path) {
    try {
        spdlog::debug("ZARR WCS: Reading precise coordinate arrays from TensorStore");
        
        // Read array metadata to understand structure
        std::ifstream ra_zarray(ra_path / ".zarray");
        nlohmann::json ra_meta;
        ra_zarray >> ra_meta;
        
        auto shape = ra_meta["shape"];
        size_t height = shape[0].get<size_t>();  // l dimension
        size_t width = shape[1].get<size_t>();   // m dimension
        
        // spdlog::debug("ZARR WCS: Coordinate arrays shape: {}x{}", height, width);
        
        // Read actual RA coordinate data using TensorStore
        spdlog::info("ZARR COORDS: Reading RA coordinate array: {}x{}", height, width);
        
        // Create TensorStore spec for RA array
        nlohmann::json ra_spec = {
            {"driver", "zarr2"},
            {"kvstore", {{"driver", "file"}, {"path", ra_path.string()}}}
        };
        
        auto ra_spec_result = tensorstore::Spec::FromJson(ra_spec);
        if (!ra_spec_result.ok()) {
            spdlog::error("Failed to create RA TensorStore spec: {}", ra_spec_result.status().ToString());
            return false;
        }
        
        auto ra_open = tensorstore::Open(ra_spec_result.value(), _context, 
                                        tensorstore::OpenMode::open, 
                                        tensorstore::ReadWriteMode::read).result();
        if (!ra_open.ok()) {
            spdlog::error("Failed to open RA TensorStore: {}", ra_open.status().ToString());
            return false;
        }
        
        auto ra_store = std::move(ra_open.value());
        
        // Sample RA coordinates to calculate pixel increment
        // Read center row and a few sample points
        size_t center_l = height / 2;
        size_t sample_cols[] = {width/4, width/2, 3*width/4};
        
        std::vector<double> ra_samples;
        std::vector<size_t> col_indices;
        
        for (size_t col : sample_cols) {
            if (col < width) {
                // Create slice for single pixel
                std::vector<tensorstore::Index> origin = {static_cast<tensorstore::Index>(center_l), static_cast<tensorstore::Index>(col)};
                std::vector<tensorstore::Index> shape = {1, 1};
                tensorstore::Box<> slice_box(origin, shape);
                
                auto sliced_store = ra_store | tensorstore::AllDims().BoxSlice(slice_box);
                if (sliced_store.ok()) {
                    auto read_result = tensorstore::Read<tensorstore::zero_origin>(sliced_store.value()).result();
                    if (read_result.ok()) {
                        auto data_array = std::move(read_result.value());
                        if (data_array.dtype().name() == "float64") {
                            const double* data = reinterpret_cast<const double*>(data_array.data());
                            ra_samples.push_back(data[0]);
                            col_indices.push_back(col);
                            spdlog::debug("RA sample at [{},{}]: {:.6f} rad ({:.6f}°)", 
                                        center_l, col, data[0], data[0] * 180.0 / M_PI);
                        }
                    }
                }
            }
        }
        
        // Calculate RA pixel increment from samples
        double ra_cdelt = 0.0;
        if (ra_samples.size() >= 2) {
            // Use linear fit or simple difference
            double delta_ra = ra_samples.back() - ra_samples.front();
            double delta_col = static_cast<double>(col_indices.back() - col_indices.front());
            ra_cdelt = delta_ra / delta_col;  // radians per pixel
            spdlog::info("ZARR COORDS: Calculated RA cdelt: {:.6e} rad/pix ({:.3f} arcsec/pix)", 
                        ra_cdelt, ra_cdelt * 180.0 * 3600.0 / M_PI);
        } else {
            spdlog::warn("ZARR COORDS: Could not read RA samples, using default");
            ra_cdelt = -2.5e-6;  // Default ~2.5 arcsec
        }
        
        // Get reference RA value from center
        double ra_rad = 0.0;
        if (!ra_samples.empty()) {
            ra_rad = ra_samples[ra_samples.size()/2];  // Use middle sample as reference
        }
        // Read actual DEC coordinate data using TensorStore
        spdlog::info("ZARR COORDS: Reading DEC coordinate array: {}x{}", height, width);
        
        // Create TensorStore spec for DEC array
        nlohmann::json dec_spec = {
            {"driver", "zarr2"},
            {"kvstore", {{"driver", "file"}, {"path", dec_path.string()}}}
        };
        
        auto dec_spec_result = tensorstore::Spec::FromJson(dec_spec);
        if (!dec_spec_result.ok()) {
            spdlog::error("Failed to create DEC TensorStore spec: {}", dec_spec_result.status().ToString());
            return false;
        }
        
        auto dec_open = tensorstore::Open(dec_spec_result.value(), _context, 
                                         tensorstore::OpenMode::open, 
                                         tensorstore::ReadWriteMode::read).result();
        if (!dec_open.ok()) {
            spdlog::error("Failed to open DEC TensorStore: {}", dec_open.status().ToString());
            return false;
        }
        
        auto dec_store = std::move(dec_open.value());
        
        // Sample DEC coordinates to calculate pixel increment
        // Read center column and a few sample points
        size_t center_m = width / 2;
        size_t sample_rows[] = {height/4, height/2, 3*height/4};
        
        std::vector<double> dec_samples;
        std::vector<size_t> row_indices;
        
        for (size_t row : sample_rows) {
            if (row < height) {
                // Create slice for single pixel
                std::vector<tensorstore::Index> origin = {static_cast<tensorstore::Index>(row), static_cast<tensorstore::Index>(center_m)};
                std::vector<tensorstore::Index> shape = {1, 1};
                tensorstore::Box<> slice_box(origin, shape);
                
                auto sliced_store = dec_store | tensorstore::AllDims().BoxSlice(slice_box);
                if (sliced_store.ok()) {
                    auto read_result = tensorstore::Read<tensorstore::zero_origin>(sliced_store.value()).result();
                    if (read_result.ok()) {
                        auto data_array = std::move(read_result.value());
                        if (data_array.dtype().name() == "float64") {
                            const double* data = reinterpret_cast<const double*>(data_array.data());
                            dec_samples.push_back(data[0]);
                            row_indices.push_back(row);
                            spdlog::debug("DEC sample at [{},{}]: {:.6f} rad ({:.6f}°)", 
                                        row, center_m, data[0], data[0] * 180.0 / M_PI);
                        }
                    }
                }
            }
        }
        
        // Calculate DEC pixel increment from samples
        double dec_cdelt = 0.0;
        if (dec_samples.size() >= 2) {
            // Use linear fit or simple difference
            double delta_dec = dec_samples.back() - dec_samples.front();
            double delta_row = static_cast<double>(row_indices.back() - row_indices.front());
            dec_cdelt = delta_dec / delta_row;  // radians per pixel
            spdlog::info("ZARR COORDS: Calculated DEC cdelt: {:.6e} rad/pix ({:.3f} arcsec/pix)", 
                        dec_cdelt, dec_cdelt * 180.0 * 3600.0 / M_PI);
        } else {
            spdlog::warn("ZARR COORDS: Could not read DEC samples, using default");
            dec_cdelt = 2.5e-6;  // Default ~2.5 arcsec
        }
        
        // Get reference DEC value from center
        double dec_rad = 0.0;
        if (!dec_samples.empty()) {
            dec_rad = dec_samples[dec_samples.size()/2];  // Use middle sample as reference
        }
        
        // Fallback to metadata if coordinate reading failed
        if (ra_rad == 0.0 && dec_rad == 0.0) {
            spdlog::warn("ZARR COORDS: Failed to read coordinate arrays, checking metadata");
            
            std::filesystem::path main_zattrs = ra_path.parent_path() / ".zattrs";
            std::ifstream main_file(main_zattrs);
            if (main_file.is_open()) {
                nlohmann::json main_json;
                main_file >> main_json;
                
                if (main_json.contains("pointing_center")) {
                    auto center_data = main_json["pointing_center"]["data"];
                    if (center_data.is_array() && center_data.size() >= 2) {
                        ra_rad = center_data[0].get<double>();
                        dec_rad = center_data[1].get<double>();
                        spdlog::info("ZARR COORDS: Using pointing_center from metadata: RA={:.6f} rad, DEC={:.6f} rad", 
                                    ra_rad, dec_rad);
                    }
                }
            }
            
            if (ra_rad == 0.0 && dec_rad == 0.0) {
                // Final fallback
                ra_rad = 5.5;  // ~315 degrees, typical for Hydra field
                dec_rad = -0.65; // ~-37 degrees, typical for southern sky
                spdlog::warn("ZARR COORDS: Using default center: RA={:.6f} rad, DEC={:.6f} rad", 
                            ra_rad, dec_rad);
            }
        }

        // Parse frequency coordinate array
        double freq_cdelt = 1e6;  // Default 1 MHz
        double freq_hz = 1.4e9;  // Default 1.4 GHz
        
        std::ifstream freq_zarray(freq_path / ".zarray");
        nlohmann::json freq_meta;
        freq_zarray >> freq_meta;

        auto freq_shape = freq_meta["shape"];
        size_t depth = freq_shape[0].get<size_t>();  // channel numbers

        spdlog::info("ZARR COORDS: Reading frequency coordinate array, {} channels", depth);
        
        // Read actual frequency coordinate data using TensorStore
        nlohmann::json freq_spec = {
            {"driver", "zarr2"},
            {"kvstore", {{"driver", "file"}, {"path", freq_path.string()}}}
        };
        
        auto freq_spec_result = tensorstore::Spec::FromJson(freq_spec);
        if (freq_spec_result.ok()) {
            auto freq_open = tensorstore::Open(freq_spec_result.value(), _context, 
                                             tensorstore::OpenMode::open, 
                                             tensorstore::ReadWriteMode::read).result();
            if (freq_open.ok()) {
                auto freq_store = std::move(freq_open.value());
                
                // Read frequency samples to calculate increment
                std::vector<double> freq_samples;
                std::vector<size_t> freq_indices;
                
                // Sample a few frequency channels
                size_t sample_channels[] = {0, depth/4, depth/2, 3*depth/4, depth-1};
                
                for (size_t ch : sample_channels) {
                    if (ch < depth) {
                        // Create slice for single frequency
                        std::vector<tensorstore::Index> origin = {static_cast<tensorstore::Index>(ch)};
                        std::vector<tensorstore::Index> shape = {1};
                        tensorstore::Box<> slice_box(origin, shape);
                        
                        auto sliced_store = freq_store | tensorstore::AllDims().BoxSlice(slice_box);
                        if (sliced_store.ok()) {
                            auto read_result = tensorstore::Read<tensorstore::zero_origin>(sliced_store.value()).result();
                            if (read_result.ok()) {
                                auto data_array = std::move(read_result.value());
                                if (data_array.dtype().name() == "float64") {
                                    const double* data = reinterpret_cast<const double*>(data_array.data());
                                    freq_samples.push_back(data[0]);
                                    freq_indices.push_back(ch);
                                    spdlog::debug("Frequency sample at channel {}: {:.3f} Hz ({:.3f} MHz)", 
                                                ch, data[0], data[0] / 1e6);
                                }
                            }
                        }
                    }
                }
                
                // Calculate frequency increment from samples
                if (freq_samples.size() >= 2) {
                    double delta_freq = freq_samples.back() - freq_samples.front();
                    double delta_ch = static_cast<double>(freq_indices.back() - freq_indices.front());
                    freq_cdelt = delta_freq / delta_ch;  // Hz per channel
                    freq_hz = freq_samples[0];  // First frequency as reference
                    spdlog::info("ZARR COORDS: Calculated frequency cdelt: {:.3f} Hz/ch ({:.3f} MHz/ch)", 
                                freq_cdelt, freq_cdelt / 1e6);
                    spdlog::info("ZARR COORDS: Using frequency reference: {:.3f} Hz ({:.3f} MHz)", 
                                freq_hz, freq_hz / 1e6);
                } else {
                    spdlog::warn("ZARR COORDS: Could not read frequency samples, trying metadata fallback");
                    
                    // Try to get frequency center from main .zattrs
                    std::filesystem::path main_freq_zattrs = freq_path.parent_path() / ".zattrs";
                    std::ifstream main_freq_file(main_freq_zattrs);
                    if (main_freq_file.is_open()) {
                        nlohmann::json main_freq_json;
                        main_freq_file >> main_freq_json;
                        
                        if (main_freq_json.contains("spectral") && main_freq_json["spectral"].contains("reference")) {
                            auto freq_ref_data = main_freq_json["spectral"]["reference"]["data"];
                            if (freq_ref_data.is_array() && freq_ref_data.size() > 0) {
                                freq_hz = freq_ref_data[0].get<double>();
                                spdlog::info("ZARR COORDS: Using frequency reference from metadata: {:.3f} Hz ({:.3f} MHz)", 
                                            freq_hz, freq_hz / 1e6);
                            }
                        }
                    }
                }
            }
        }

        
        // Store calculated coordinate values for use in coordinate system creation
        double ra_deg = ra_rad * 180.0 / M_PI;
        double dec_deg = dec_rad * 180.0 / M_PI;
        double ra_cdelt_deg = ra_cdelt * 180.0 / M_PI;  // Convert to degrees
        double dec_cdelt_deg = dec_cdelt * 180.0 / M_PI;  // Convert to degrees
        
        spdlog::info("ZARR COORDS: Final coordinate parameters:");
        spdlog::info("  RA center: {:.6f}° (cdelt: {:.6f}°/pix)", ra_deg, ra_cdelt_deg);
        spdlog::info("  DEC center: {:.6f}° (cdelt: {:.6f}°/pix)", dec_deg, dec_cdelt_deg);
        spdlog::info("  FREQ reference: {:.3f} MHz (cdelt: {:.3f} MHz/ch)", freq_hz / 1e6, freq_cdelt / 1e6);

        return buildDirectionCoordinateFromArrays(ra_rad, dec_rad, freq_hz, ra_cdelt_deg, dec_cdelt_deg, freq_cdelt, height, width, depth);
    } catch (std::exception& e) {
        spdlog::error("ZARR COORDS: Exception in parseWCSFromCoordinateArrays: {}", e.what());
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

bool CartaZarrImage::buildDirectionCoordinateFromArrays(double ra_rad, double dec_rad, double freq_hz, 
                                                      double ra_cdelt_deg, double dec_cdelt_deg, double freq_cdelt_hz,
                                                      size_t height, size_t width, size_t depth) {
    try {
        // spdlog::debug("ZARR WCS: Building DirectionCoordinate from coordinate arrays");
        // spdlog::debug("ZARR WCS: Reference RA={:.6f} rad ({:.6f}°), DEC={:.6f} rad ({:.6f}°)", 
        //              ra_rad, ra_rad * 180.0 / M_PI, dec_rad, dec_rad * 180.0 / M_PI);
        // spdlog::debug("ZARR WCS: Array dimensions: {}x{}", height, width);
        
        // Create DirectionCoordinate using available information
        casacore::Vector<double> ref_val(2);
        ref_val(0) = ra_rad;   // RA in radians
        ref_val(1) = dec_rad;  // DEC in radians
        
        // Use calculated pixel increments from coordinate arrays
        casacore::Vector<double> inc(2);
        inc(0) = -ra_cdelt_deg * M_PI / 180.0;    // RA increment in radians (negative for RA)
        inc(1) = dec_cdelt_deg * M_PI / 180.0;    // DEC increment in radians
        
        spdlog::info("ZARR WCS: Using calculated pixel increments:");
        spdlog::info("  RA increment: {:.6e} rad ({:.3f} arcsec)", inc(0), inc(0) * 180.0 * 3600.0 / M_PI);
        spdlog::info("  DEC increment: {:.6e} rad ({:.3f} arcsec)", inc(1), inc(1) * 180.0 * 3600.0 / M_PI);
        
        // Reference pixel (center of image)
        casacore::Vector<double> ref_pix(2);
        ref_pix(0) = (width - 1) / 2.0;   // Center of x axis (m)
        ref_pix(1) = (height - 1) / 2.0;  // Center of y axis (l)
        
        // Linear transformation matrix (identity for now)
        casacore::Matrix<double> xform(2, 2);
        xform = 0.0;
        xform(0, 0) = 1.0;
        xform(1, 1) = 1.0;
        
        // spdlog::debug("ZARR WCS: From Arrays Creating DirectionCoordinate with:");
        // spdlog::debug("  Reference value: RA={:.6f}° DEC={:.6f}°", 
        //              ra_rad * 180.0 / M_PI, dec_rad * 180.0 / M_PI);
        // spdlog::debug("  Reference pixel: ({:.1f}, {:.1f})", ref_pix(0), ref_pix(1));
        // spdlog::debug("  Pixel increment: ({:.3f} arcsec, {:.3f} arcsec)", 
        //              inc(0) * 180.0 * 3600.0 / M_PI, inc(1) * 180.0 * 3600.0 / M_PI);
        
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
            
            // spdlog::debug("ZARR WCS: DirectionCoordinate created successfully with CAR projection");
            
            // Test coordinate conversion
            casacore::Vector<double> world_coord(2);
            casacore::Vector<double> pixel_coord(2);
            pixel_coord(0) = ref_pix(0);
            pixel_coord(1) = ref_pix(1);
            
            // if (dir_coord.toWorld(world_coord, pixel_coord)) {
            //     spdlog::debug("ZARR WCS: Reference pixel ({:.1f}, {:.1f}) -> World ({:.6f}, {:.6f}) radians",
            //                 pixel_coord(0), pixel_coord(1),
            //                 world_coord(0), world_coord(1));
            //     spdlog::debug("ZARR WCS: World coordinates: RA={:.6f}° DEC={:.6f}°",
            //                 world_coord(0) * 180.0 / M_PI, 
            //                 world_coord(1) * 180.0 / M_PI);
            // } else {
            //     spdlog::warn("ZARR WCS: Failed to convert reference pixel to world coordinates");
            // }
            
        } catch (const std::exception& coord_e) {
            // spdlog::error("ZARR WCS: Failed to create DirectionCoordinate: {}", coord_e.what());
            dir_coord = DirectionCoordinate(); // Fallback to default
        }

        // Create SpectralCoordinate

        try {
            // Use calculated frequency values from coordinate arrays
            double rest_freq = 1420405751.786; // in Hz (HI line, could be updated from metadata)
            double spectral_crval = freq_hz; // Reference frequency in Hz from coordinate arrays
            double spectral_cdelt = freq_cdelt_hz; // Channel width from coordinate arrays
            double spectral_crpix = (depth - 1) / 2.0 + 1; // 1-based pixel
            // Use casacore MFrequency type for the SpectralCoordinate constructor
            casacore::MFrequency::Types frequency_type = casacore::MFrequency::TOPO; // default to TOPO

            spec_coord = SpectralCoordinate(frequency_type, spectral_crval, spectral_cdelt, spectral_crpix, rest_freq);

            spdlog::info("ZARR WCS: SpectralCoordinate created with calculated values:");
            spdlog::info("  Reference frequency: {:.3f} MHz", spectral_crval / 1e6);
            spdlog::info("  Channel width: {:.3f} MHz", spectral_cdelt / 1e6);
            spdlog::info("  Reference pixel: {:.1f}", spectral_crpix);
            
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
            // spdlog::debug("ZARR WCS: Successfully added DirectionCoordinate from arrays");
            
            _coord_sys.addCoordinate(spec_coord);      
            // spdlog::debug("ZARR WCS: Successfully added SpectralCoordinate");
            
            _coord_sys.addCoordinate(stokes_coord);    
            // spdlog::debug("ZARR WCS: Successfully added StokesCoordinate");
            
            // spdlog::debug("ZARR WCS: CoordinateSystem has {} coordinates", _coord_sys.nCoordinates());
            return true;
            
        } catch (const std::exception& e) {
            // spdlog::error("ZARR WCS: Failed to add coordinates to CoordinateSystem: {}", e.what());
            return false;
        }
        
    } catch (std::exception& e) {
        // spdlog::error("ZARR WCS: Exception in buildDirectionCoordinateFromArrays: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::parseWCSFromMetadata(const nlohmann::json& zattrs) {
    try {
        // spdlog::debug("ZARR WCS: Starting parseWCSFromZattrs");
        
        // Check for ZARR-style coordinate information
        bool has_array_dimensions = zattrs.contains("_ARRAY_DIMENSIONS");
        bool has_direction_info = zattrs.contains("direction") || zattrs.contains("pointing_center");
        
        // spdlog::debug("ZARR WCS: has_array_dimensions = {}, has_direction_info = {}", 
        //              has_array_dimensions, has_direction_info);
        
        if (!has_array_dimensions && !has_direction_info) {
            // spdlog::debug("ZARR WCS: No required info found, returning false");
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
            // spdlog::debug("ZARR WCS: Processing 5D ZARR file with dimensions: [{}]", 
            //              fmt::join(axis_names, ", "));
            
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
                            // spdlog::debug("ZARR WCS: pointing_center RA={} rad ({}°), DEC={} rad ({}°)", 
                            //             ra_rad, ra_rad * 180.0 / M_PI, 
                            //             dec_rad, dec_rad * 180.0 / M_PI);
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
                        
                        // spdlog::debug("ZARR WCS: From Meta Data Creating DirectionCoordinate with:");
                        // spdlog::debug("  Reference value: RA={}° DEC={}°", 
                        //             ra_rad * 180.0 / M_PI, dec_rad * 180.0 / M_PI);
                        // spdlog::debug("  Reference pixel: ({}, {})", ref_pix(0), ref_pix(1));
                        // spdlog::debug("  Pixel increment: ({} arcsec, {} arcsec)", 
                        //             inc(0) * 180.0 * 3600.0 / M_PI, inc(1) * 180.0 * 3600.0 / M_PI);
                        // spdlog::debug("  Image shape: ({}, {})", _shape(0), _shape(1));
                        
                        try {
                            // Try CAR projection first (common for radio astronomy)
                            dir_coord = DirectionCoordinate(MDirection::J2000, 
                                                          Projection::CAR,
                                                          ref_val(0), ref_val(1),
                                                          inc(0), inc(1),
                                                          xform,
                                                          ref_pix(0), ref_pix(1));
                            
                            // spdlog::debug("ZARR WCS: DirectionCoordinate created successfully with CAR projection");
                            
                            // Test coordinate conversion
                            casacore::Vector<double> world_coord(2);
                            casacore::Vector<double> pixel_coord(2);
                            pixel_coord(0) = ref_pix(0);
                            pixel_coord(1) = ref_pix(1);
                            
                            // if (dir_coord.toWorld(world_coord, pixel_coord)) {
                            //     spdlog::debug("ZARR WCS: Reference pixel ({}, {}) -> World ({}, {}) radians",
                            //                 pixel_coord(0), pixel_coord(1),
                            //                 world_coord(0), world_coord(1));
                            //     spdlog::debug("ZARR WCS: World coordinates: RA={}° DEC={}°",
                            //                 world_coord(0) * 180.0 / M_PI, 
                            //                 world_coord(1) * 180.0 / M_PI);
                            // } else {
                            //     spdlog::warn("ZARR WCS: Failed to convert reference pixel to world coordinates");
                            // }
                            
                        } catch (const std::exception& coord_e) {
                            // spdlog::error("ZARR WCS: Failed to create DirectionCoordinate: {}", coord_e.what());
                            dir_coord = DirectionCoordinate(); // Fallback to default
                        }
                    } else {
                        // spdlog::debug("ZARR WCS: No valid pointing center found, using default DirectionCoordinate");
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
                    // spdlog::debug("ZARR WCS: Found frequency/.zattrs at {}", freq_zattrs.string());
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
                                // spdlog::info("ZARR WCS: Built SpectralCoordinate from frequency/.zattrs (data) nchan={}", m);
                            } catch (const std::exception& e) {
                                // spdlog::warn("ZARR WCS: Failed to build spectral coord from frequency.data: {}", e.what());
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
                                // spdlog::info("ZARR WCS: Built SpectralCoordinate from frequency/.zattrs (reference+increment) nchan={}", nchan);
                            } catch (const std::exception& e) {
                                // spdlog::warn("ZARR WCS: Failed to build spectral coord from frequency.reference/increment: {}", e.what());
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                // spdlog::warn("ZARR WCS: Error reading frequency/.zattrs: {}", e.what());
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
                // spdlog::debug("ZARR WCS: Successfully added DirectionCoordinate");
                
                _coord_sys.addCoordinate(spec_coord);      // axis 2: frequency
                // spdlog::debug("ZARR WCS: Successfully added SpectralCoordinate");
                
                _coord_sys.addCoordinate(stokes_coord);    // axis 3: polarization
                // spdlog::debug("ZARR WCS: Successfully added StokesCoordinate");
                
                // spdlog::debug("ZARR WCS: Successfully added all coordinates to CoordinateSystem");
                // spdlog::debug("ZARR WCS: CoordinateSystem has {} coordinates", _coord_sys.nCoordinates());
                
                return true;
                
            } catch (const std::exception& e) {
                // spdlog::error("ZARR WCS: Failed to add coordinates to CoordinateSystem: {}", e.what());
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
            tensorstore::ReadWriteMode::read,
            tensorstore::dtype_v<float>  // Explicitly specify float data type
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
    return _actual_data_type;
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

        spdlog::info("doGetSlice called with start={}, length={}, stride={}", 
                    start.toString(), length.toString(), stride.toString());
        
        // Detect if this call is likely for histogram/statistics calculation
        // Histogram calls typically have pattern: length=[full_width, 1, 1, 1] or similar
        bool is_histogram_call = false;
        if (length.size() >= 4) {
            // Check for full-width or full-height reads with single channel/stokes
            int spatial_dims = 0;
            int singleton_dims = 0;
            for (size_t i = 0; i < length.size(); ++i) {
                if (length[i] > 100000) spatial_dims++;  // Large spatial dimension
                else if (length[i] == 1) singleton_dims++;  // Single slice
            }
            spdlog::info("doGetSlice: spatial_dims={}, singleton_dims={}", spatial_dims, singleton_dims);
            // Histogram typically reads full spatial data with single channel/stokes
            is_histogram_call = (spatial_dims >= 1 && singleton_dims >= 2);
        }
        
        // Track statistics calculation state for batch optimization
        static bool in_statistics_mode = false;
        static int stats_freq = -1;
        static int stats_stokes = -1;
        static std::chrono::time_point<std::chrono::steady_clock> stats_start_time;
        
        if (is_histogram_call) {
            spdlog::info("doGetSlice: HISTOGRAM/STATISTICS CALCULATION DETECTED - start={}, length={}", start.toString(), length.toString());
            
            // For statistics calculation, try to optimize by preloading the entire channel
            // This will reduce the number of individual doGetSlice calls from 4742 to much fewer
            int current_freq = (start.size() > 2) ? start[2] : 0;
            int current_stokes = (start.size() > 3) ? start[3] : 0;
            
            // Check if coordinates are reordered for statistics calls
            if (start.size() >= 4 && start[0] == 0 && start[1] == 0) {
                current_freq = start[0];
                current_stokes = start[1];
            }
            
            // Initialize statistics tracking
            if (!in_statistics_mode) {
                in_statistics_mode = true;
                stats_freq = current_freq;
                stats_stokes = current_stokes;
                stats_start_time = std::chrono::steady_clock::now();
                spdlog::info("doGetSlice: Starting STATISTICS BATCH MODE for freq={}, stokes={}", 
                            stats_freq, stats_stokes);
            }
            
            spdlog::info("doGetSlice: Statistics optimization - ensuring full channel cache for freq={}, stokes={}", 
                        current_freq, current_stokes);
        } else {
            // Reset statistics mode if we're not doing histogram/stats calls anymore
            if (in_statistics_mode) {
                auto duration = std::chrono::steady_clock::now() - stats_start_time;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
                spdlog::info("doGetSlice: STATISTICS BATCH MODE COMPLETED in {} ms", ms);
                in_statistics_mode = false;
            }
        }
        
        // CRITICAL FIX: The coordinates are being reordered somewhere before doGetSlice
        // Based on the error pattern, we need to detect and correct this reordering
        
        // Add diagnostic to understand coordinate format
        spdlog::warn("doGetSlice: COORDINATE REORDERING DEBUG - start={}, length={}", start.toString(), length.toString());
        spdlog::warn("doGetSlice: CARTA shape={}, ZARR shape={}", _shape.toString(), _original_zarr_shape.toString());
        
        int freq_index, stokes_index;
        
        // PATTERN DETECTION: From the error logs, we can see the reordering pattern:
        // GetChunk request: [512,4096,0,0] -> doGetSlice receives: [0,0,4096,512]
        // This suggests: [x,y,freq,stokes] -> [freq,stokes,y,x]
        
        bool coordinates_reordered = false;
        if (start.size() >= 4) {
            // Check if this looks like the reordered pattern
            bool zero_in_spatial_pos = (start[0] == 0 && start[1] == 0);  // x=0, y=0
            bool large_values_in_freq_stokes = (start[2] > 1000 || start[3] > 100);  // freq or stokes position has large values
            bool length_pattern_matches = (length.size() >= 4 && length[0] == 1 && length[1] == 1 && 
                                          (length[2] > 1 || length[3] > 1));  // spatial lengths in freq/stokes positions
            
            coordinates_reordered = zero_in_spatial_pos && large_values_in_freq_stokes && length_pattern_matches;
            
            spdlog::warn("doGetSlice: Reordering detection - zero_spatial={}, large_freq_stokes={}, length_pattern={}, REORDERED={}",
                        zero_in_spatial_pos, large_values_in_freq_stokes, length_pattern_matches, coordinates_reordered);
        }
        
        if (coordinates_reordered) {
            // REVERSE THE REORDERING: [freq,stokes,y,x] back to [x,y,freq,stokes]
            spdlog::warn("doGetSlice: Detected coordinate reordering - correcting from [freq,stokes,y,x] to [x,y,freq,stokes]");
            
            // Based on the pattern: GetChunk [7680,2560,0,0] -> doGetSlice [0,0,2560,7680]
            // This means: [x,y,freq,stokes] -> [freq,stokes,y,x]
            // So to reverse: [freq,stokes,y,x] -> [x,y,freq,stokes]
            // start=[0,0,2560,7680] represents [freq=0, stokes=0, y=2560, x=7680]
            // We want: [x=7680, y=2560, freq=0, stokes=0]
            
            freq_index = start[0];    // freq from reordered position 0
            stokes_index = start[1];  // stokes from reordered position 1
            // Note: spatial coordinates start[2]=y, start[3]=x are in wrong positions but we'll handle this later
            
            spdlog::warn("doGetSlice: Corrected coordinates - freq={}, stokes={} (spatial coords will be fixed in cache lookup)", 
                        freq_index, stokes_index);
        } else {
            // Normal case: assume correct CARTA [x, y, freq, stokes] format
            freq_index = (start.size() > 2) ? start[2] : 0;
            stokes_index = (start.size() > 3) ? start[3] : 0;
        }
        
        // DEBUG: Track all frequency index requests to identify source of freq=128
        spdlog::warn("doGetSlice FREQ TRACKING: freq_index={}, stokes_index={}, start={}, length={}", 
                     freq_index, stokes_index, start.toString(), length.toString());
        
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
        // NEVER use direct read - spatial profiles should ALWAYS use cache system
        // For histogram/statistics calculation, ALWAYS use full channel cache for efficiency
        bool use_region_cache = !is_1d_profile && !is_histogram_call && (req_width < full_width / 4 && req_height < full_height / 4);
        bool use_direct_read = false;  // DISABLED: Spatial profiles must use cache system
        
        // spdlog::debug("CACHE STRATEGY DECISION:");
        // spdlog::debug("  Request size: {}x{}, Full size: {}x{}", req_width, req_height, full_width, full_height);
        // spdlog::debug("  1D Profile detected: {} (horizontal={}, vertical={})", is_1d_profile ? "YES" : "NO", is_horizontal_profile, is_vertical_profile);
        // spdlog::debug("  Single point request: {}", is_single_point ? "YES" : "NO");
        // spdlog::debug("  Use region cache: {}", use_region_cache ? "YES" : "NO");
        // spdlog::debug("  Use direct read: {}", use_direct_read ? "YES" : "NO");
        // spdlog::debug("  Current cache status: loaded={}, channel={}, is_full={}", _channel_cache_loaded, _cached_channel, _is_full_channel_cache);
        // spdlog::debug("  Target channel: {}", current_channel);
        
        // CACHE-FIRST STRATEGY (inspired by CartaFitsImage's GetDataSubset approach)
        // CartaFitsImage always tries to avoid direct file access by using efficient data reading patterns
        // We implement similar strategy for ZARR: prioritize cache, force cache for statistics, avoid row-by-row
        
        bool cache_hit = false;
        
        // For single point requests, still try cache first (unlike previous direct read approach)
        if (use_direct_read && _channel_cache_loaded && _cached_channel == current_channel) {
            spdlog::debug("ZARR doGetSlice: Single point request - trying cache first before direct TensorStore");
            if (getSliceFromCache(buffer, section)) {
                return true;
            }
            // If cache miss, fall through to TensorStore read
            spdlog::debug("ZARR doGetSlice: Cache miss for single point, falling back to TensorStore");
            return readPixelFromTensorStore(buffer, section);
        }
        
        // Check if current cache is suitable (similar to CartaFitsImage's _equiv_bitpix check)
        bool cache_suitable = (_channel_cache_loaded && _cached_channel == current_channel);
        bool cache_type_suitable = true;
        
        // For statistics/histogram, ensure we have full channel cache (like CartaFitsImage loads full data subset)
        if (is_histogram_call && (!_is_full_channel_cache || !cache_suitable)) {
            cache_type_suitable = false;
            spdlog::debug("ZARR doGetSlice: Statistics/histogram requires full channel cache - loading now");
        }
        
        // For region requests, check if we want region cache instead of full cache
        if (!is_histogram_call && use_region_cache && (_is_full_channel_cache || !cache_suitable)) {
            cache_type_suitable = false;
        }
        
        // Load appropriate cache if needed (like CartaFitsImage's GetDataSubset template selection)
        if (!cache_suitable || !cache_type_suitable) {
            
            if (use_region_cache && !is_histogram_call) {
                // Load region cache for small requests
                int padding = std::min(100, std::min(req_width, req_height));
                int region_start_x = std::max(0, static_cast<int>(start[0]) - padding);
                int region_start_y = std::max(0, static_cast<int>(start[1]) - padding);
                int region_width = std::min(full_width - region_start_x, req_width + 2 * padding);
                int region_height = std::min(full_height - region_start_y, req_height + 2 * padding);
                
                spdlog::debug("ZARR doGetSlice: Loading region cache [freq={}, stokes={}]: {}x{} at ({},{}) with padding {}",
                            freq_index, stokes_index, region_width, region_height, region_start_x, region_start_y, padding);
                
                if (loadRegionCache(freq_index, stokes_index, region_start_x, region_start_y, region_width, region_height)) {
                    cache_hit = getSliceFromCache(buffer, section);
                }
                
                if (!cache_hit) {
                    spdlog::warn("ZARR doGetSlice: Region cache failed, falling back to full channel cache");
                    use_region_cache = false;  // Fall back to full channel cache
                }
            }
            
            if (!use_region_cache || !cache_hit) {
                // Load full channel cache (equivalent to CartaFitsImage's full data subset read)
                spdlog::debug("ZARR doGetSlice: Loading full channel [freq={}, stokes={}] - FITS-style efficient data access",
                            freq_index, stokes_index);
                
                if (loadChannelCache(freq_index, stokes_index)) {
                    cache_hit = getSliceFromCache(buffer, section);
                }
                
                if (!cache_hit) {
                    spdlog::warn("ZARR doGetSlice: Full channel cache failed, falling back to direct TensorStore read");
                }
            }
        } else {
            // Cache is already loaded and suitable, use it (like CartaFitsImage using already loaded data)
            cache_hit = getSliceFromCache(buffer, section);
        }
        
        // If cache served the request successfully, return (like CartaFitsImage successful GetDataSubset)
        if (cache_hit) {
            spdlog::debug("ZARR doGetSlice: Successfully served {} from cache (FITS-style efficient access)",
                         is_histogram_call ? "statistics/histogram" : "data");
            return true;
        }
        
        // Direct read from TensorStore (fallback or non-first-channel)
        // Map CARTA dimensions back to original ZARR dimensions for 5D case
        IPosition zarr_start = start;
        IPosition zarr_length = length;
        
        spdlog::debug("COORDINATE MAPPING DEBUG:");
        spdlog::debug("  Original CARTA request: start={}, length={}", start.toString(), length.toString());
        spdlog::debug("  CARTA shape: {}", _shape.toString());
        spdlog::debug("  ZARR original shape: {}", _original_zarr_shape.toString());
        
        if (_original_zarr_shape.size() == 5 && start.size() >= 4) {
            // CRITICAL: Use the corrected coordinates, not the original reordered ones
            int correct_x, correct_y, correct_freq, correct_stokes;
            
            if (coordinates_reordered) {
                // Use the corrected coordinates we calculated earlier
                correct_freq = freq_index;    // Already corrected
                correct_stokes = stokes_index; // Already corrected
                correct_x = start[3];         // x from reordered position 3
                correct_y = start[2];         // y from reordered position 2
                
                spdlog::debug("  Using corrected coordinates: x={}, y={}, freq={}, stokes={}", 
                             correct_x, correct_y, correct_freq, correct_stokes);
            } else {
                // Normal case: use coordinates as-is
                correct_x = start[0];
                correct_y = start[1];
                correct_freq = start[2];
                correct_stokes = start[3];
            }
            
            // Map corrected CARTA [x, y, freq, stokes] -> ZARR [time, freq, stokes, y, x]
            zarr_start.resize(5);
            zarr_length.resize(5);
            
            zarr_start[0] = 0;                    // ZARR[0]=time (always 0)
            zarr_start[1] = correct_freq;         // ZARR[1]=freq (corrected frequency)
            zarr_start[2] = correct_stokes;       // ZARR[2]=stokes (corrected stokes)
            zarr_start[3] = correct_y;            // ZARR[3]=y (corrected y)
            zarr_start[4] = correct_x;            // ZARR[4]=x (corrected x)
            
            zarr_length[0] = 1;                   // ZARR[0]=time (always 1)
            zarr_length[1] = 1;                   // ZARR[1]=freq (single frequency)
            zarr_length[2] = 1;                   // ZARR[2]=stokes (single stokes)
            
            // CRITICAL: Clip lengths to stay within ZARR bounds
            int requested_height = coordinates_reordered ? length[2] : length[1];
            int requested_width = coordinates_reordered ? length[3] : length[0];
            
            // Clip y dimension (ZARR dimension 3)
            int max_y_length = _original_zarr_shape[3] - correct_y;
            zarr_length[3] = std::max(0, std::min(requested_height, max_y_length));
            
            // Clip x dimension (ZARR dimension 4) 
            int max_x_length = _original_zarr_shape[4] - correct_x;
            zarr_length[4] = std::max(0, std::min(requested_width, max_x_length));
            
            // If the request is completely outside bounds, return empty buffer
            if (zarr_length[3] <= 0 || zarr_length[4] <= 0) {
                spdlog::warn("Request completely outside ZARR bounds: x=[{},{}), y=[{},{}), ZARR bounds: x_max={}, y_max={}",
                           correct_x, correct_x + requested_width,
                           correct_y, correct_y + requested_height,
                           _original_zarr_shape[4], _original_zarr_shape[3]);
                
                // Return buffer filled with NaN values to indicate invalid region
                IPosition requested_shape = coordinates_reordered ? 
                    IPosition(4, length[3], length[2], length[1], length[0]) :
                    length;
                buffer.resize(requested_shape);
                buffer = std::numeric_limits<float>::quiet_NaN();
                return true;
            }
            
            if (zarr_length[3] != requested_height || zarr_length[4] != requested_width) {
                spdlog::warn("  CLIPPED REQUEST: original height={} width={}, clipped height={} width={}", 
                            requested_height, requested_width, zarr_length[3], zarr_length[4]);
                spdlog::warn("  ZARR bounds: y_max={}, x_max={}, requested y=[{},{}), x=[{},{})", 
                            _original_zarr_shape[3], _original_zarr_shape[4],
                            correct_y, correct_y + requested_height,
                            correct_x, correct_x + requested_width);
            }
            
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
        // IMPORTANT: Use the actual read dimensions (which may be clipped), not the original request
        IPosition actual_read_shape;
        if (_original_zarr_shape.size() == 5 && start.size() >= 4) {
            // For 5D case, use the clipped dimensions
            actual_read_shape.resize(4);
            actual_read_shape[0] = zarr_length[4];  // width (x)
            actual_read_shape[1] = zarr_length[3];  // height (y)
            actual_read_shape[2] = zarr_length[1];  // freq (should be 1)
            actual_read_shape[3] = zarr_length[2];  // stokes (should be 1)
        } else {
            actual_read_shape = length;
        }
        
        buffer.resize(actual_read_shape);
        
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
        size_t zarr_elements = zarr_array.num_elements();
        
        if (num_elements != zarr_elements) {
            spdlog::warn("Buffer size mismatch: buffer={}, zarr={}, using smaller size", 
                        num_elements, zarr_elements);
            num_elements = std::min(num_elements, zarr_elements);
        }
        
        // Copy the actual data read
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
        
        // CRITICAL: Handle coordinate reordering in cache access too
        // The same reordering that affects doGetSlice also affects cache access
        int start_x, start_y, req_width, req_height;
        
        // Detect if coordinates are reordered using the same logic as doGetSlice
        bool coordinates_reordered = false;
        if (start.size() >= 4) {
            bool zero_in_spatial_pos = (start[0] == 0 && start[1] == 0);
            bool large_values_in_freq_stokes = (start[2] > 1000 || start[3] > 100);
            bool length_pattern_matches = (length.size() >= 4 && length[0] == 1 && length[1] == 1 && 
                                          (length[2] > 1 || length[3] > 1));
            coordinates_reordered = zero_in_spatial_pos && large_values_in_freq_stokes && length_pattern_matches;
        }
        
        if (coordinates_reordered) {
            // Reverse the reordering: [freq,stokes,y,x] -> [x,y,freq,stokes]
            // start=[0,0,2560,7680] represents [freq=0, stokes=0, y=2560, x=7680]
            // We want: [x=7680, y=2560, ...]
            start_x = start[3];      // x from reordered position 3
            start_y = start[2];      // y from reordered position 2
            req_width = length[3];   // width from reordered position 3
            req_height = length[2];  // height from reordered position 2
            
            spdlog::debug("getSliceFromCache: Corrected spatial coordinates - x={}, y={}, width={}, height={}", 
                         start_x, start_y, req_width, req_height);
        } else {
            // Normal case: assume correct CARTA [x, y, freq, stokes] format
            start_x = start[0];      // CARTA x = ZARR l dimension
            start_y = start[1];      // CARTA y = ZARR m dimension  
            req_width = length[0];   // Requested slice width
            req_height = length[1];  // Requested slice height
        }
        
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
        
        // spdlog::debug("CACHE SLICE REQUEST:");
        // spdlog::debug("  Request: start_x={}, start_y={}, width={}, height={}", start_x, start_y, req_width, req_height);
        // spdlog::debug("  Cache size: {} elements, expected: {}", _channel_cache.size(), _cache_width * _cache_height);
        
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
        
        // spdlog::debug("DOWNSAMPLING PARAMETERS:");
        // spdlog::debug("  Full width: {}, Requested width: {}", full_width, req_width);
        // spdlog::debug("  Downsample factor: {:.2f}", downsample_factor);
        // spdlog::debug("  Pixels per bin: {:.2f}", downsample_factor);
        
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
            // spdlog::debug("Downsampled data: {} valid, {} NaN pixels (factor: {:.2f})", 
            //              valid_count, nan_count, downsample_factor);
            
            // Special debug check for downsampled matrices
            if (req_width > 100) {  // Only for significant width requests
                bool first_is_nan = std::isnan(dest_data[0]);
                bool last_is_nan = std::isnan(dest_data[req_width - 1]);
                
                // spdlog::debug("Downsampled matrix debug: first_pixel={}, last_pixel={}", 
                //              first_is_nan ? "NaN" : std::to_string(dest_data[0]),
                //              last_is_nan ? "NaN" : std::to_string(dest_data[req_width - 1]));
                
                // // Additional check for row position and data transition
                // spdlog::debug("Row position: y={} out of {} total rows (0-indexed)", start_y, _cache_height);
                // if (start_y >= _cache_height - 10) {
                //     spdlog::debug("→ This is near the bottom edge, NaN values are expected");
                // } else if (start_y < 183) {  // Actual data transition boundary discovered
                //     if (valid_count > 0) {
                //         spdlog::info("✓ DATA TRANSITION: Row {} has {} valid pixels - data region starting!", 
                //                     start_y, valid_count);
                //     } else {
                //         spdlog::debug("→ Row {} in header region, all NaN expected", start_y);
                //     }
                // } else {
                //     spdlog::debug("→ Row {} in main data region, expecting valid pixels", start_y);
                //     if (valid_count == 0) {
                //         spdlog::warn("Unexpected: Row {} in data region has no valid pixels", start_y);
                //     }
                // }
                
                // // Show sample values for rows with mixed data
                // if (valid_count > 0 && nan_count > 0) {
                //     spdlog::debug("Mixed data row - showing first few valid values:");
                //     int shown = 0;
                //     for (int i = 0; i < req_width && shown < 5; ++i) {
                //         if (!std::isnan(dest_data[i])) {
                //             spdlog::debug("  pixel[{}] = {}", i, dest_data[i]);
                //             shown++;
                //         }
                //     }
                // }
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
            // 5D case: Assume already in ZARR order [time, freq, stokes, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            for (int i = 0; i < 5; ++i) {
                box_origin[i] = start[i];
                box_shape[i] = length[i];
            }
        } else if (start.size() == 4) {
            // 4D case: CARTA [x, y, freq, stokes] -> ZARR [time, freq, stokes, y, x]
            // Based on error log: CARTA shape [7763, 4742, 128, 1] = [width, height, freq, stokes]
            // ZARR shape [1, 128, 1, 7763, 4742] = [time, freq, stokes, height, width]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;           // time = 0
            box_origin[1] = start[2];    // freq (from CARTA position 2)
            box_origin[2] = start[3];    // stokes (from CARTA position 3) 
            box_origin[3] = start[1];    // y (from CARTA position 1)
            box_origin[4] = start[0];    // x (from CARTA position 0)
            box_shape[0] = 1;            // time = 1
            box_shape[1] = length[2];    // freq length
            box_shape[2] = length[3];    // stokes length
            box_shape[3] = length[1];    // y length
            box_shape[4] = length[0];    // x length
        } else if (start.size() == 3) {
            // 3D case: ZarrLoader passes [freq, y, x] -> map to ZARR [time, freq, stokes, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;           // time = 0
            box_origin[1] = start[0];    // freq (from ZarrLoader position 0)
            box_origin[2] = 0;           // stokes = 0
            box_origin[3] = start[1];    // y (from ZarrLoader position 1)
            box_origin[4] = start[2];    // x (from ZarrLoader position 2)
            box_shape[0] = 1;            // time = 1
            box_shape[1] = length[0];    // freq length
            box_shape[2] = 1;            // stokes = 1
            box_shape[3] = length[1];    // y length
            box_shape[4] = length[2];    // x length
        } else if (start.size() == 2) {
            // 2D case: ZarrLoader passes [y, x] -> map to ZARR [time, freq, stokes, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;           // time = 0
            box_origin[1] = 0;           // freq = 0 (first channel)
            box_origin[2] = 0;           // stokes = 0
            box_origin[3] = start[0];    // y (from ZarrLoader position 0)
            box_origin[4] = start[1];    // x (from ZarrLoader position 1)
            box_shape[0] = 1;            // time = 1
            box_shape[1] = 1;            // freq = 1 (single channel)
            box_shape[2] = 1;            // stokes = 1
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
        
        spdlog::debug("DIRECT READ: TensorStore slice [time={}, freq={}, stokes={}, y={}:{}, x={}:{}]",
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
        
        // spdlog::debug("DIRECT READ: Successfully read {} elements (shape={})", total_elements, length.toString());
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in readDirectFromTensorStore: {}", e.what());
        return false;
    }
}

Bool CartaZarrImage::readPixelFromTensorStore(Array<float>& buffer, const Slicer& section) {
    try {
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        
        // CRITICAL FIX: Handle coordinate mapping based on actual section dimensions
        // The issue is that ZarrLoader is passing coordinates in ZARR order already!
        // We need to map the incoming slicer coordinates correctly
        
        spdlog::debug("readPixelFromTensorStore: section start={}, length={}, shape={}", 
                     start.toString(), length.toString(), _original_zarr_shape.toString());
        
        std::vector<tensorstore::Index> box_origin;
        std::vector<tensorstore::Index> box_shape;
        
        if (start.size() == 5) {
            // 5D case: ZarrLoader already provides ZARR order [time, freq, stokes, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            for (int i = 0; i < 5; ++i) {
                box_origin[i] = start[i];
                box_shape[i] = length[i];
            }
        } else if (start.size() == 4) {
            spdlog::debug("start={}, length={}", start.toString(), length.toString());
            // 4D case: CARTA internal format [x, y, freq, stokes]
            // Map CARTA [x,y,freq,stokes] -> ZARR [time=0, freq, stokes, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;                 // time = 0
            box_origin[1] = start[2];          // freq (from CARTA position 2)
            box_origin[2] = start[3];          // stokes (from CARTA position 3)
            box_origin[3] = start[1];          // y (from CARTA position 1)
            box_origin[4] = start[0];          // x (from CARTA position 0)
            box_shape[0] = 1;                  // time = 1
            box_shape[1] = length[2];          // freq length
            box_shape[2] = length[3];          // stokes length
            box_shape[3] = length[1];          // y length
            box_shape[4] = length[0];          // x length
            
            spdlog::debug("4D coordinate mapping: CARTA[{},{},{},{}] -> ZARR[{},{},{},{},{}]", 
                         start[0], start[1], start[2], start[3],
                         box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4]);
        } else if (start.size() == 3) {
            // 3D case: incoming CARTA [freq,y,x] to match 4D pattern
            // Map CARTA [freq,y,x] -> ZARR [time=0, freq, stokes=0, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;                 // time = 0
            box_origin[1] = start[0];          // freq (from CARTA position 0)
            box_origin[2] = 0;                 // stokes = 0
            box_origin[3] = start[1];          // y (from CARTA position 1)
            box_origin[4] = start[2];          // x (from CARTA position 2)
            box_shape[0] = 1;                  // time = 1
            box_shape[1] = length[0];          // freq length
            box_shape[2] = 1;                  // stokes = 1
            box_shape[3] = length[1];          // y length
            box_shape[4] = length[2];          // x length
        } else if (start.size() == 2) {
            // 2D case: incoming CARTA [y,x]
            // Map CARTA [y,x] -> ZARR [time=0, freq=0, stokes=0, y, x]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;                 // time = 0
            box_origin[1] = 0;                 // freq = 0
            box_origin[2] = 0;                 // stokes = 0
            box_origin[3] = start[0];          // y (from CARTA position 0)
            box_origin[4] = start[1];          // x (from CARTA position 1)
            box_shape[0] = 1;                  // time = 1
            box_shape[1] = 1;                  // freq = 1
            box_shape[2] = 1;                  // stokes = 1
            box_shape[3] = length[0];          // y length
            box_shape[4] = length[1];          // x length
        } else {
            spdlog::error("readPixelFromTensorStore: Unsupported section dimensions: {}", start.size());
            return false;
        }
        spdlog::debug("dim = {}", start.size());
        
        // Validate coordinates against ZARR bounds
        for (size_t i = 0; i < box_origin.size(); ++i) {
            if (box_origin[i] < 0 || 
                (i < _original_zarr_shape.size() && box_origin[i] + box_shape[i] > _original_zarr_shape[i])) {
                spdlog::error("readPixelFromTensorStore: Coordinate {} out of bounds: origin={}, shape={}, max={} (ZARR dim names: [time,freq,stokes,y,x])", 
                             i, box_origin[i], box_shape[i], 
                             i < _original_zarr_shape.size() ? _original_zarr_shape[i] : -1);
                spdlog::error("Failed mapping: box_origin=[{},{},{},{},{}], ZARR_shape=[{},{},{},{},{}]",
                             box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4],
                             _original_zarr_shape.size() > 0 ? _original_zarr_shape[0] : -1,
                             _original_zarr_shape.size() > 1 ? _original_zarr_shape[1] : -1,
                             _original_zarr_shape.size() > 2 ? _original_zarr_shape[2] : -1,
                             _original_zarr_shape.size() > 3 ? _original_zarr_shape[3] : -1,
                             _original_zarr_shape.size() > 4 ? _original_zarr_shape[4] : -1);
                return false;
            }
        }
        
        spdlog::debug("DIRECT PIXEL READ: TensorStore slice [time={}, freq={}, stokes={}, y={}:{}, x={}:{}]",
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
        
        // spdlog::debug("DIRECT READ: Successfully read {} elements (shape={})", total_elements, length.toString());
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in readDirectFromTensorStore: {}", e.what());
        return false;
    }
}

} // namespace carta