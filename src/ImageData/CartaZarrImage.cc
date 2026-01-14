/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"
#include "Logger/Logger.h"
#include "ThreadManager/ThreadManager.h"

#include "Util/Casacore.h"

#include <limits>
#include <cmath>
#include <thread>
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

CartaZarrImage::CartaZarrImage(const std::string& filename) : ImageInterface<float>(), _name(filename), _is_copy(false) {
    // Initialize shared_ptr members for cache sharing across copies
    _channel_cache = std::make_shared<std::vector<float>>();
    _all_channel_stats = std::make_shared<std::map<int, ChannelStats>>();
    
    // Initialize TensorStore context with memory limit
    // Note: Full context initialization happens in initializeTensorStore()
    // This is just a placeholder that will be replaced
    _context = tensorstore::Context::Default();
    
    // Check if this is a repeated initialization for the same file
    static std::unordered_map<std::string, bool> initialized_files;
    static std::unordered_map<std::string, IPosition> cached_shapes;
    static std::unordered_map<std::string, IPosition> cached_original_shapes;
    static std::unordered_map<std::string, casacore::DataType> cached_data_types;
    static std::unordered_map<std::string, CoordinateSystem> cached_coord_sys;
    // Cache TensorStore and Context objects for performance - reuse across multiple image instances
    static std::unordered_map<std::string, tensorstore::TensorStore<>> cached_tensorstores;
    static std::unordered_map<std::string, tensorstore::Context> cached_contexts;
    static std::unordered_map<std::string, std::string> cached_zarr_paths;
    static std::unordered_map<std::string, bool> brightness_unit_loaded;
    static std::unordered_map<std::string, bool> image_info_loaded;
    
    bool is_repeat_init = initialized_files[filename];
    if (is_repeat_init) {
        auto t0 = std::chrono::high_resolution_clock::now();
        spdlog::info("CartaZarrImage: Using cached metadata for already initialized file: {}", filename);
        
        // Use cached values to avoid expensive re-initialization
        auto t1 = std::chrono::high_resolution_clock::now();
        _shape = cached_shapes[filename];
        _original_zarr_shape = cached_original_shapes[filename];
        _actual_data_type = cached_data_types[filename];
        _coord_sys = cached_coord_sys[filename];
        _ndim = _shape.size();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto copy_cached_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        spdlog::info("  Copy cached metadata: {} μs", copy_cached_us);
        
        // Reuse cached TensorStore and Context - much faster than re-initializing
        auto t3 = std::chrono::high_resolution_clock::now();
        if (cached_tensorstores.find(filename) != cached_tensorstores.end()) {
            _tensorstore = cached_tensorstores[filename];
            _context = cached_contexts[filename];
            _tensorstore_initialized = true;
            spdlog::info("  Reusing cached TensorStore (no re-initialization needed)");
        } else {
            // Fallback: initialize if not in cache (shouldn't happen)
            spdlog::warn("  TensorStore not in cache, re-initializing...");
            initializeTensorStore();
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        auto init_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        spdlog::info("  TensorStore setup: {} μs ({:.2f} ms)", init_ts_us, init_ts_us / 1000.0);
        
        // Set coordinate system in ImageInterface base class
        auto t5 = std::chrono::high_resolution_clock::now();
        setCoordinateInfo(_coord_sys);
        auto t6 = std::chrono::high_resolution_clock::now();
        auto set_coord_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        spdlog::info("  setCoordinateInfo: {} μs", set_coord_us);
        
        // Only read brightness unit and setup image info once per file
        auto t7 = std::chrono::high_resolution_clock::now();
        readBrightnessUnitIfNeeded();
        setupImageInfoIfNeeded();
        auto t8 = std::chrono::high_resolution_clock::now();
        auto setup_info_us = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
        spdlog::info("  readBrightnessUnit + setupImageInfo: {} μs", setup_info_us);
        
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t0).count();
        spdlog::info("Total cached metadata init: {} μs ({:.2f} ms)", total_us, total_us / 1000.0);
        
        return;
    } else {
        // spdlog::info("CartaZarrImage: First-time initialization for file: {}", filename);
        initialized_files[filename] = true;
    }
    
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
                spdlog::debug("ZARR dtype: {}", dtype_str);
                
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
                spdlog::debug("Mapped ZARR dtype {} to casacore DataType", dtype_str);
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
                    spdlog::debug("Zarr image shape (CARTA order): {}", _shape.toString());
                    spdlog::debug("Original ZARR shape: {}", _original_zarr_shape.toString());
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
    
    // Cache the initialized values for future instances
    cached_shapes[filename] = _shape;
    cached_original_shapes[filename] = _original_zarr_shape;
    cached_data_types[filename] = _actual_data_type;
    cached_coord_sys[filename] = _coord_sys;
    // Cache TensorStore and Context for fast reuse (shares data via cache_pool)
    if (_tensorstore_initialized) {
        cached_tensorstores[filename] = _tensorstore;
        cached_contexts[filename] = _context;
        spdlog::info("Cached TensorStore and Context for future reuse");
    }
    
    // Set coordinate system in ImageInterface base class
    setCoordinateInfo(_coord_sys);
    
    // Load brightness unit and image info (beam) once per file at the end of constructor
    readBrightnessUnitIfNeeded();
    setupImageInfoIfNeeded();
}

// Copy constructor for efficient SubImage creation (shares TensorStore and cache)
CartaZarrImage::CartaZarrImage(const CartaZarrImage& other)
    : ImageInterface<float>(other),
      _coord_sys(other._coord_sys),
      _shape(other._shape),
      _original_zarr_shape(other._original_zarr_shape),
      _name(other._name),
      _ndim(other._ndim),
      _context(other._context),
      _tensorstore(other._tensorstore),  // Share TensorStore instance!
      _tensorstore_initialized(other._tensorstore_initialized),
      _actual_data_type(other._actual_data_type),
      _channel_cache(other._channel_cache),  // Share cache!
      _channel_cache_loaded(other._channel_cache_loaded),
      _cached_channel(other._cached_channel),
      _num_cached_channels(other._num_cached_channels),
      _cache_width(other._cache_width),
      _cache_height(other._cache_height),
      _cache_start_x(other._cache_start_x),
      _cache_start_y(other._cache_start_y),
      _cache_num_freq(other._cache_num_freq),
      _cache_num_stokes(other._cache_num_stokes),
      _cache_freq_start(other._cache_freq_start),
      _cache_stokes_start(other._cache_stokes_start),
      _is_full_channel_cache(other._is_full_channel_cache),
      _is_copy(true),  // Mark as copy!
      _file_last_modified(other._file_last_modified),
      _coordinate_system_initialized(other._coordinate_system_initialized),
      _frequency_type_cached(other._frequency_type_cached),
      _cached_frequency_type(other._cached_frequency_type) {
    spdlog::debug("CartaZarrImage copy constructor: sharing TensorStore and cache for file: {}", _name);
    setCoordinateInfo(_coord_sys);
}

// Destructor - only original instances clean up shared resources
CartaZarrImage::~CartaZarrImage() {
    if (_is_copy) {
        spdlog::debug("CartaZarrImage destructor: skipping cleanup for copy (file: {})", _name);
    } else {
        spdlog::debug("CartaZarrImage destructor: cleaning up original instance (file: {})", _name);
        // Default cleanup for non-copy instances
    }
}

void CartaZarrImage::setupCoordinateSystem() {
    // Check if coordinate system is already set up
    if (_coord_sys.nCoordinates() > 0) {
        spdlog::debug("CartaZarrImage: Coordinate system already initialized, skipping setup");
        return;
    }
    
    // Try to read coordinate system from .zattrs file
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        if (std::filesystem::exists(zattrs_path)) {
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
            spdlog::debug("CartaZarrImage: Reading coordinate system from .zattrs: {}", _name);

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
        // For ZARR files with l,m coordinate arrays (SIN projection style):
        // - l, m represent angular offsets from the phase center
        // - We need to find where l=0, m=0 (the reference pixel/phase center)
        // - CDELT can be calculated from the l,m array spacing
        // - CRVAL comes from direction.reference metadata
        
        spdlog::debug("ZARR WCS: Reading l,m coordinate arrays for SIN projection");
        
        // Check if l,m arrays exist instead of ra,dec
        std::filesystem::path l_path = ra_path.parent_path() / "l";
        std::filesystem::path m_path = ra_path.parent_path() / "m";
        
        bool use_lm_coords = std::filesystem::exists(l_path) && std::filesystem::exists(m_path);
        
        if (use_lm_coords) {
            spdlog::debug("ZARR WCS: Using l,m coordinate arrays for SIN projection");
            return parseWCSFromLMArrays(l_path, m_path, freq_path);
        }
        
        // Original ra,dec array parsing code
        // Read array metadata to understand structure
        std::ifstream ra_zarray(ra_path / ".zarray");
        nlohmann::json ra_meta;
        ra_zarray >> ra_meta;
        
        auto shape = ra_meta["shape"];
        size_t height = shape[0].get<size_t>();  // l dimension
        size_t width = shape[1].get<size_t>();   // m dimension
        
        // spdlog::debug("ZARR WCS: Coordinate arrays shape: {}x{}", height, width);
        
        // Read actual RA coordinate data using TensorStore
        spdlog::debug("ZARR COORDS: Reading RA coordinate array: {}x{}", height, width);
        
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
            spdlog::error("ZARR COORDS: Could not read RA samples from coordinate array");
            return false;
        }
        
        // Get reference RA value from center
        double ra_rad = 0.0;
        if (!ra_samples.empty()) {
            ra_rad = ra_samples[ra_samples.size()/2];  // Use middle sample as reference
        }
        // Read actual DEC coordinate data using TensorStore
        spdlog::debug("ZARR COORDS: Reading DEC coordinate array: {}x{}", height, width);
        
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
            spdlog::error("ZARR COORDS: Could not read DEC samples from coordinate array");
            return false;
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

        spdlog::debug("ZARR COORDS: Reading frequency coordinate array, {} channels", depth);
        
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
        
        spdlog::debug("ZARR COORDS: Final coordinate parameters:");
        spdlog::info("  RA center: {:.6f}° (cdelt: {:.6f}°/pix)", ra_deg, ra_cdelt_deg);
        spdlog::info("  DEC center: {:.6f}° (cdelt: {:.6f}°/pix)", dec_deg, dec_cdelt_deg);
        spdlog::info("  FREQ reference: {:.3f} MHz (cdelt: {:.3f} MHz/ch)", freq_hz / 1e6, freq_cdelt / 1e6);

        return buildDirectionCoordinateFromArrays(ra_rad, dec_rad, freq_hz, ra_cdelt_deg, dec_cdelt_deg, freq_cdelt, height, width, depth);
    } catch (std::exception& e) {
        spdlog::error("ZARR COORDS: Exception in parseWCSFromCoordinateArrays: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::parseWCSFromLMArrays(const std::filesystem::path& l_path, const std::filesystem::path& m_path, const std::filesystem::path& freq_path) {
    try {
        spdlog::debug("ZARR WCS: Parsing WCS from l,m coordinate arrays (SIN projection)");
        
        // Read l array metadata
        std::ifstream l_zarray(l_path / ".zarray");
        nlohmann::json l_meta;
        l_zarray >> l_meta;
        size_t nl = l_meta["shape"][0].get<size_t>();
        
        // Read m array metadata
        std::ifstream m_zarray(m_path / ".zarray");
        nlohmann::json m_meta;
        m_zarray >> m_meta;
        size_t nm = m_meta["shape"][0].get<size_t>();
        
        spdlog::debug("ZARR WCS: l,m array dimensions: {} x {}", nl, nm);
        
        // Read reference coordinates (phase center) from main .zattrs
        std::filesystem::path main_zattrs = l_path.parent_path() / ".zattrs";
        std::ifstream main_file(main_zattrs);
        nlohmann::json main_json;
        main_file >> main_json;
        
        double ref_ra_rad = 0.0, ref_dec_rad = 0.0;
        if (main_json.contains("direction") && main_json["direction"].contains("reference")) {
            auto ref_data = main_json["direction"]["reference"]["data"];
            if (ref_data.is_array() && ref_data.size() >= 2) {
                ref_ra_rad = ref_data[0].get<double>();
                ref_dec_rad = ref_data[1].get<double>();
                spdlog::info("ZARR WCS: Phase center from metadata: RA={:.6f}° DEC={:.6f}°",
                            ref_ra_rad * 180.0 / M_PI, ref_dec_rad * 180.0 / M_PI);
            }
        }
        
        // Read a few samples from l array to calculate CDELT and find CRPIX
        // For l,m coordinates: l=0, m=0 is the phase center (reference pixel)
        // CDELT is the spacing between adjacent pixels in the l,m arrays
        
        nlohmann::json l_spec = {
            {"driver", "zarr2"},
            {"kvstore", {{"driver", "file"}, {"path", l_path.string()}}}
        };
        
        auto l_spec_result = tensorstore::Spec::FromJson(l_spec);
        if (!l_spec_result.ok()) {
            spdlog::error("Failed to create l array TensorStore spec");
            return false;
        }
        
        auto l_open = tensorstore::Open(l_spec_result.value(), _context,
                                       tensorstore::OpenMode::open,
                                       tensorstore::ReadWriteMode::read).result();
        if (!l_open.ok()) {
            spdlog::error("Failed to open l array TensorStore");
            return false;
        }
        
        auto l_store = std::move(l_open.value());
        
        // Read several l values to calculate spacing and find l=0
        std::vector<double> l_values;
        std::vector<size_t> l_indices = {0, nl/4, nl/2, 3*nl/4, nl-1};
        
        for (size_t idx : l_indices) {
            if (idx < nl) {
                std::vector<tensorstore::Index> origin = {static_cast<tensorstore::Index>(idx)};
                std::vector<tensorstore::Index> shape = {1};
                tensorstore::Box<> slice_box(origin, shape);
                
                auto sliced = l_store | tensorstore::AllDims().BoxSlice(slice_box);
                if (sliced.ok()) {
                    auto read_result = tensorstore::Read<tensorstore::zero_origin>(sliced.value()).result();
                    if (read_result.ok()) {
                        auto data_array = std::move(read_result.value());
                        if (data_array.dtype().name() == "float64") {
                            const double* data = reinterpret_cast<const double*>(data_array.data());
                            l_values.push_back(data[0]);
                        }
                    }
                }
            }
        }
        
        // Calculate l spacing (CDELT1 in radians)
        double l_cdelt_rad = 0.0;
        double crpix_l = 0.0;  // 0-based pixel where l=0
        
        if (l_values.size() >= 2) {
            // Calculate spacing from samples
            double delta_l = l_values.back() - l_values.front();
            double delta_idx = static_cast<double>(l_indices.back() - l_indices.front());
            l_cdelt_rad = delta_l / delta_idx;
            
            // Find where l=0 (reference pixel) using linear interpolation
            // l(pixel) = l[0] + pixel * cdelt
            // 0 = l[0] + crpix * cdelt
            // crpix = -l[0] / cdelt
            crpix_l = -l_values[0] / l_cdelt_rad;
            
            spdlog::info("ZARR WCS: l axis: cdelt={:.6e} rad ({:.3f} arcsec), CRPIX={:.1f}",
                        l_cdelt_rad, l_cdelt_rad * 180.0 * 3600.0 / M_PI, crpix_l + 1.0);  // +1 for FITS convention
        }
        
        // Similarly for m array
        nlohmann::json m_spec = {
            {"driver", "zarr2"},
            {"kvstore", {{"driver", "file"}, {"path", m_path.string()}}}
        };
        
        auto m_spec_result = tensorstore::Spec::FromJson(m_spec);
        if (!m_spec_result.ok()) {
            spdlog::error("Failed to create m array TensorStore spec");
            return false;
        }
        
        auto m_open = tensorstore::Open(m_spec_result.value(), _context,
                                       tensorstore::OpenMode::open,
                                       tensorstore::ReadWriteMode::read).result();
        if (!m_open.ok()) {
            spdlog::error("Failed to open m array TensorStore");
            return false;
        }
        
        auto m_store = std::move(m_open.value());
        
        std::vector<double> m_values;
        std::vector<size_t> m_indices = {0, nm/4, nm/2, 3*nm/4, nm-1};
        
        for (size_t idx : m_indices) {
            if (idx < nm) {
                std::vector<tensorstore::Index> origin = {static_cast<tensorstore::Index>(idx)};
                std::vector<tensorstore::Index> shape = {1};
                tensorstore::Box<> slice_box(origin, shape);
                
                auto sliced = m_store | tensorstore::AllDims().BoxSlice(slice_box);
                if (sliced.ok()) {
                    auto read_result = tensorstore::Read<tensorstore::zero_origin>(sliced.value()).result();
                    if (read_result.ok()) {
                        auto data_array = std::move(read_result.value());
                        if (data_array.dtype().name() == "float64") {
                            const double* data = reinterpret_cast<const double*>(data_array.data());
                            m_values.push_back(data[0]);
                        }
                    }
                }
            }
        }
        
        double m_cdelt_rad = 0.0;
        double crpix_m = 0.0;  // 0-based pixel where m=0
        
        if (m_values.size() >= 2) {
            double delta_m = m_values.back() - m_values.front();
            double delta_idx = static_cast<double>(m_indices.back() - m_indices.front());
            m_cdelt_rad = delta_m / delta_idx;
            
            crpix_m = -m_values[0] / m_cdelt_rad;
            
            spdlog::info("ZARR WCS: m axis: cdelt={:.6e} rad ({:.3f} arcsec), CRPIX={:.1f}",
                        m_cdelt_rad, m_cdelt_rad * 180.0 * 3600.0 / M_PI, crpix_m + 1.0);
        }
        
        // Parse frequency array  
        double freq_hz = 1.4e9;
        double freq_cdelt_hz = 1e6;
        size_t depth = 1;
        
        if (std::filesystem::exists(freq_path)) {
            std::ifstream freq_zarray(freq_path / ".zarray");
            nlohmann::json freq_meta;
            freq_zarray >> freq_meta;
            depth = freq_meta["shape"][0].get<size_t>();
            
            // Sample frequency array similar to above
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
                    
                    std::vector<double> freq_samples;
                    std::vector<size_t> freq_indices = {0, depth-1};
                    
                    for (size_t idx : freq_indices) {
                        if (idx < depth) {
                            std::vector<tensorstore::Index> origin = {static_cast<tensorstore::Index>(idx)};
                            std::vector<tensorstore::Index> shape = {1};
                            tensorstore::Box<> slice_box(origin, shape);
                            
                            auto sliced = freq_store | tensorstore::AllDims().BoxSlice(slice_box);
                            if (sliced.ok()) {
                                auto read_result = tensorstore::Read<tensorstore::zero_origin>(sliced.value()).result();
                                if (read_result.ok()) {
                                    auto data_array = std::move(read_result.value());
                                    if (data_array.dtype().name() == "float64") {
                                        const double* data = reinterpret_cast<const double*>(data_array.data());
                                        freq_samples.push_back(data[0]);
                                    }
                                }
                            }
                        }
                    }
                    
                    if (freq_samples.size() >= 2) {
                        freq_cdelt_hz = (freq_samples.back() - freq_samples.front()) / (depth - 1);
                        freq_hz = freq_samples[0];
                        spdlog::info("ZARR WCS: Frequency: reference={:.3f} MHz, cdelt={:.3f} MHz/ch",
                                    freq_hz / 1e6, freq_cdelt_hz / 1e6);
                    }
                }
            }
        }
        
        // Convert to degrees for buildDirectionCoordinateFromLM
        double l_cdelt_deg = l_cdelt_rad * 180.0 / M_PI;
        double m_cdelt_deg = m_cdelt_rad * 180.0 / M_PI;
        
        return buildDirectionCoordinateFromLM(ref_ra_rad, ref_dec_rad, freq_hz,
                                             l_cdelt_deg, m_cdelt_deg, freq_cdelt_hz,
                                             crpix_l, crpix_m, nl, nm, depth);
                                             
    } catch (std::exception& e) {
        spdlog::error("ZARR WCS: Exception in parseWCSFromLMArrays: {}", e.what());
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
        
        spdlog::debug("ZARR WCS: Using calculated pixel increments:");
        spdlog::info("  RA increment: {:.6e} rad ({:.3f} arcsec)", inc(0), inc(0) * 180.0 * 3600.0 / M_PI);
        spdlog::info("  DEC increment: {:.6e} rad ({:.3f} arcsec)", inc(1), inc(1) * 180.0 * 3600.0 / M_PI);
        
        // Reference pixel: For SIN projection with l,m coordinates, the reference pixel is where l=0, m=0
        // which corresponds to the phase center. This is NOT the image center.
        // For a converted FITS file, the reference pixel should match the original FITS CRPIX values.
        // We need to read from l,m coordinate arrays to find where they equal zero.
        // As a practical approach: phase center offset from corner = CRPIX - 1 (FITS is 1-based, casacore is 0-based)
        casacore::Vector<double> ref_pix(2);
        
        // Try to determine reference pixel from l,m arrays if available
        // For now, use a reasonable estimate based on typical radio interferometry data
        // where the phase center is often near but not at the image center
        // In the future, this should read actual l,m array values to find l=0, m=0
        ref_pix(0) = (width - 1) / 2.0;   // Default to center of m axis
        ref_pix(1) = (height - 1) / 2.0;  // Default to center of l axis
        
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
            // Get direction type from metadata (reads "frame" from direction.reference.attrs)
            casacore::MDirection::Types direction_type = GetDirectionType();
            
            // Get projection type from metadata (reads "projection" from direction)
            casacore::Projection projection = GetProjectionType();
            
            // Create DirectionCoordinate with dynamic projection
            dir_coord = DirectionCoordinate(direction_type, 
                                          projection,
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
            // Read rest frequency and reference frequency from metadata
            double rest_freq = 0.0; // Default 0 means no rest frequency
            double reference_freq = freq_hz; // Default to first frequency from array
            
            // Try to read rest_frequency and reference_frequency from frequency/.zattrs
            try {
                std::filesystem::path zarr_base(_name.c_str());
                std::filesystem::path freq_zattrs_path = zarr_base / "frequency" / ".zattrs";
                if (std::filesystem::exists(freq_zattrs_path)) {
                    std::ifstream freq_zattrs_file(freq_zattrs_path);
                    nlohmann::json freq_zattrs_json;
                    freq_zattrs_file >> freq_zattrs_json;
                    
                    // Check for rest_frequency field
                    if (freq_zattrs_json.contains("rest_frequency")) {
                        if (freq_zattrs_json["rest_frequency"].contains("data")) {
                            rest_freq = freq_zattrs_json["rest_frequency"]["data"].get<double>();
                            spdlog::info("ZARR COORDS: Found rest_frequency in metadata: {:.3f} Hz ({:.3f} MHz)", 
                                        rest_freq, rest_freq / 1e6);
                        } else if (freq_zattrs_json["rest_frequency"].is_number()) {
                            rest_freq = freq_zattrs_json["rest_frequency"].get<double>();
                            spdlog::info("ZARR COORDS: Found rest_frequency in metadata: {:.3f} Hz ({:.3f} MHz)", 
                                        rest_freq, rest_freq / 1e6);
                        }
                    }
                    
                    // Check for reference_frequency field (CRVAL3)
                    if (freq_zattrs_json.contains("reference_frequency")) {
                        if (freq_zattrs_json["reference_frequency"].contains("data")) {
                            reference_freq = freq_zattrs_json["reference_frequency"]["data"].get<double>();
                            spdlog::info("ZARR COORDS: Found reference_frequency (CRVAL3) in metadata: {:.3f} Hz ({:.3f} MHz)", 
                                        reference_freq, reference_freq / 1e6);
                        } else if (freq_zattrs_json["reference_frequency"].is_number()) {
                            reference_freq = freq_zattrs_json["reference_frequency"].get<double>();
                            spdlog::info("ZARR COORDS: Found reference_frequency (CRVAL3) in metadata: {:.3f} Hz ({:.3f} MHz)", 
                                        reference_freq, reference_freq / 1e6);
                        }
                    }
                }
            } catch (const std::exception& e) {
                spdlog::debug("ZARR COORDS: Could not read frequency metadata: {}", e.what());
            }
            
            // If rest frequency not found or is 0, use default HI line frequency
            if (rest_freq == 0.0) {
                rest_freq = 1420405751.786; // HI line frequency in Hz
                spdlog::debug("ZARR COORDS: Using default HI line rest frequency: {:.3f} Hz ({:.3f} MHz)", 
                            rest_freq, rest_freq / 1e6);
            }
            
            // CRVAL3: Use reference_frequency from metadata
            // This is the frequency at the reference pixel (CRPIX3)
            double spectral_crval = reference_freq;
            
            // CDELT3: Channel width from coordinate arrays
            double spectral_cdelt = freq_cdelt_hz;
            
            // CRPIX3: Reference pixel for casacore SpectralCoordinate is 0-based
            // The reference_frequency in metadata corresponds to the first frequency channel (pixel 0 in 0-based indexing)
            // This will be converted to FITS CRPIX3 = 1 when exported to FITS header
            double spectral_crpix = 0.0;
            
            // Get frequency reference frame from metadata instead of hardcoding TOPO
            casacore::MFrequency::Types frequency_type = GetFrequencyType();

            spec_coord = SpectralCoordinate(frequency_type, spectral_crval, spectral_cdelt, spectral_crpix, rest_freq);

            spdlog::info("ZARR WCS: SpectralCoordinate created with FITS-like parameters:");
            spdlog::info("  CRVAL3 (Reference frequency): {:.6e} Hz ({:.3f} MHz)", spectral_crval, spectral_crval / 1e6);
            spdlog::info("  CDELT3 (Channel width): {:.6e} Hz ({:.3f} MHz)", spectral_cdelt, spectral_cdelt / 1e6);
            spdlog::info("  CRPIX3 (Reference pixel, 0-based internal): {:.1f} (FITS will show as 1)", spectral_crpix);
            spdlog::info("  Rest frequency: {:.3f} MHz", rest_freq / 1e6);
            spdlog::info("  Frequency reference frame: {}", casacore::MFrequency::showType(frequency_type));
            
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

bool CartaZarrImage::buildDirectionCoordinateFromLM(double ref_ra_rad, double ref_dec_rad, double freq_hz,
                                                    double l_cdelt_deg, double m_cdelt_deg, double freq_cdelt_hz,
                                                    double crpix_l, double crpix_m,
                                                    size_t nl, size_t nm, size_t depth) {
    try {
        spdlog::info("ZARR WCS: Building DirectionCoordinate from l,m arrays (SIN projection)");
        spdlog::info("  Phase center: RA={:.6f}° DEC={:.6f}°", 
                    ref_ra_rad * 180.0 / M_PI, ref_dec_rad * 180.0 / M_PI);
        spdlog::info("  CRPIX (0-based): l={:.1f}, m={:.1f}", crpix_l, crpix_m);
        spdlog::info("  CDELT: l={:.6f}° ({:.3f}\"), m={:.6f}° ({:.3f}\")",
                    l_cdelt_deg, l_cdelt_deg * 3600.0, m_cdelt_deg, m_cdelt_deg * 3600.0);
        
        // Create DirectionCoordinate
        casacore::Vector<double> ref_val(2);
        ref_val(0) = ref_ra_rad;   // CRVAL1: RA at reference pixel
        ref_val(1) = ref_dec_rad;  // CRVAL2: DEC at reference pixel
        
        // Pixel increments in radians
        casacore::Vector<double> inc(2);
        inc(0) = l_cdelt_deg * M_PI / 180.0;    // CDELT1: l increment (already has correct sign)
        inc(1) = m_cdelt_deg * M_PI / 180.0;    // CDELT2: m increment
        
        // Reference pixel (0-based for casacore, will be +1 in FITS)
        casacore::Vector<double> ref_pix(2);
        ref_pix(0) = crpix_l;  // CRPIX1 - 1 (casacore is 0-based)
        ref_pix(1) = crpix_m;  // CRPIX2 - 1
        
        // Linear transformation matrix (identity)
        casacore::Matrix<double> xform(2, 2);
        xform = 0.0;
        xform(0, 0) = 1.0;
        xform(1, 1) = 1.0;
        
        DirectionCoordinate dir_coord;
        SpectralCoordinate spec_coord;
        
        try {
            // Get direction type and projection from metadata
            casacore::MDirection::Types direction_type = GetDirectionType();
            casacore::Projection projection = GetProjectionType();
            
            // Create DirectionCoordinate
            dir_coord = DirectionCoordinate(direction_type,
                                          projection,
                                          ref_val(0), ref_val(1),
                                          inc(0), inc(1),
                                          xform,
                                          ref_pix(0), ref_pix(1));
            
            spdlog::info("ZARR WCS: DirectionCoordinate created successfully with {} projection",
                        projection.name());
            
            // Verify coordinate conversion
            casacore::Vector<double> world_coord(2);
            casacore::Vector<double> pixel_coord(2);
            pixel_coord(0) = ref_pix(0);
            pixel_coord(1) = ref_pix(1);
            
            if (dir_coord.toWorld(world_coord, pixel_coord)) {
                spdlog::info("ZARR WCS: Verification - CRPIX ({:.1f}, {:.1f}) -> RA={:.6f}° DEC={:.6f}°",
                            pixel_coord(0) + 1.0, pixel_coord(1) + 1.0,  // +1 for FITS display
                            world_coord(0) * 180.0 / M_PI,
                            world_coord(1) * 180.0 / M_PI);
            }
            
        } catch (const std::exception& coord_e) {
            spdlog::error("ZARR WCS: Failed to create DirectionCoordinate: {}", coord_e.what());
            return false;
        }
        
        // Create SpectralCoordinate
        try {
            double rest_freq = 1420405751.786; // Default HI line
            double reference_freq = freq_hz;
            double spectral_cdelt = freq_cdelt_hz;
            double spectral_crpix = 0.0;  // 0-based
            
            // Try to read rest frequency from metadata
            std::filesystem::path zarr_base(_name.c_str());
            std::filesystem::path freq_zattrs_path = zarr_base / "frequency" / ".zattrs";
            if (std::filesystem::exists(freq_zattrs_path)) {
                std::ifstream freq_zattrs_file(freq_zattrs_path);
                nlohmann::json freq_zattrs_json;
                freq_zattrs_file >> freq_zattrs_json;
                
                if (freq_zattrs_json.contains("rest_frequency")) {
                    if (freq_zattrs_json["rest_frequency"].contains("data")) {
                        rest_freq = freq_zattrs_json["rest_frequency"]["data"].get<double>();
                    } else if (freq_zattrs_json["rest_frequency"].is_number()) {
                        rest_freq = freq_zattrs_json["rest_frequency"].get<double>();
                    }
                }
            }
            
            casacore::MFrequency::Types frequency_type = GetFrequencyType();
            spec_coord = SpectralCoordinate(frequency_type, reference_freq, spectral_cdelt, spectral_crpix, rest_freq);
            
            spdlog::info("ZARR WCS: SpectralCoordinate created: CRVAL={:.3f} MHz, CDELT={:.3f} MHz/ch",
                        reference_freq / 1e6, spectral_cdelt / 1e6);
                        
        } catch (const std::exception& spec_e) {
            spdlog::error("ZARR WCS: Failed to create SpectralCoordinate: {}", spec_e.what());
            spec_coord = SpectralCoordinate(); // Fallback
        }
        
        // Create StokesCoordinate
        casacore::Vector<int> stokes_types(_shape(3));
        for (int i = 0; i < _shape(3); ++i) {
            stokes_types(i) = casacore::Stokes::I;
        }
        StokesCoordinate stokes_coord(stokes_types);
        
        // Add coordinates to system
        try {
            _coord_sys.addCoordinate(dir_coord);
            _coord_sys.addCoordinate(spec_coord);
            _coord_sys.addCoordinate(stokes_coord);
            
            spdlog::info("ZARR WCS: CoordinateSystem successfully created with {} coordinates", _coord_sys.nCoordinates());
            return true;
            
        } catch (const std::exception& e) {
            spdlog::error("ZARR WCS: Failed to add coordinates: {}", e.what());
            return false;
        }
        
    } catch (std::exception& e) {
        spdlog::error("ZARR WCS: Exception in buildDirectionCoordinateFromLM: {}", e.what());
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
                            // Get direction type from metadata (reads "frame" from direction.reference.attrs)
                            casacore::MDirection::Types direction_type = GetDirectionType();
                            
                            // Get projection type from metadata (reads "projection" from direction)
                            casacore::Projection projection = GetProjectionType();
                            
                            // Create DirectionCoordinate with dynamic projection
                            dir_coord = DirectionCoordinate(direction_type, 
                                                          projection,
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
                                // Use GetFrequencyType() instead of hardcoding TOPO
                                casacore::MFrequency::Types freq_type = GetFrequencyType();
                                spec_coord = SpectralCoordinate(freq_type, vals, spec_unit, spec_rest);
                                spec_built = true;
                                spdlog::info("ZARR WCS: Built SpectralCoordinate from frequency/.zattrs (data) nchan={}, frame={}", 
                                           m, casacore::MFrequency::showType(freq_type));
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
                                // Use GetFrequencyType() instead of hardcoding TOPO
                                casacore::MFrequency::Types freq_type = GetFrequencyType();
                                spec_coord = SpectralCoordinate(freq_type, vals, spec_unit, spec_rest);
                                spec_built = true;
                                spdlog::info("ZARR WCS: Built SpectralCoordinate from frequency/.zattrs (reference+increment) nchan={}, frame={}", 
                                           nchan, casacore::MFrequency::showType(freq_type));
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
    auto ts_start = std::chrono::high_resolution_clock::now();
    try {
        // Detect number of CPU cores for optimal parallelization
        unsigned int num_cpus = std::thread::hardware_concurrency();
        if (num_cpus == 0) num_cpus = 8;  // fallback to 8 if detection fails
        
        // Create TensorStore context with aggressive parallelization
        // Using all available CPU cores for maximum I/O and decode throughput
        // Cache size calculation: 4 channels × 7763×4742 pixels × 4 bytes/pixel = ~560MB
        // Set to 128MB to test smaller cache for PV diagram performance
        nlohmann::json context_spec = {
            {"cache_pool", {
                {"total_bytes_limit", 128ULL << 20}  // 128MB cache limit - testing smaller cache
            }},
            {"data_copy_concurrency", {
                {"limit", num_cpus}  // Use all CPU cores for chunk decode operations
            }},
            {"file_io_concurrency", {
                {"limit", num_cpus}  // Use all CPU cores for parallel file I/O
            }}
        };
        
        auto t_ctx_start = std::chrono::high_resolution_clock::now();
        auto context_result = tensorstore::Context::FromJson(context_spec);
        if (context_result.ok()) {
            _context = context_result.value();
            auto t_ctx_end = std::chrono::high_resolution_clock::now();
            auto ctx_us = std::chrono::duration_cast<std::chrono::microseconds>(t_ctx_end - t_ctx_start).count();
            spdlog::info("    Context creation: {} μs - {} CPU cores, 128MB cache, {}-thread data_copy_concurrency",
                        ctx_us, num_cpus, num_cpus);
        } else {
            spdlog::warn("Failed to create TensorStore context with cache and concurrency: {}, using default", 
                        context_result.status().ToString());
            _context = tensorstore::Context::Default();
        }
        
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
                spdlog::debug("Found Zarr array in subdirectory: {}", zarr_path);
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
            spdlog::debug("Using direct Zarr path: {}", zarr_path);
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
        auto t_open_start = std::chrono::high_resolution_clock::now();
        auto open_future = tensorstore::Open(
            input_spec, 
            _context, 
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read,
            tensorstore::dtype_v<float>  // Explicitly specify float data type
        );
        
        auto open_result = open_future.result();
        auto t_open_end = std::chrono::high_resolution_clock::now();
        auto open_us = std::chrono::duration_cast<std::chrono::microseconds>(t_open_end - t_open_start).count();
        spdlog::info("    TensorStore::Open: {} μs ({:.2f} ms)", open_us, open_us / 1000.0);
        
        if (!open_result.ok()) {
            spdlog::error("Failed to open TensorStore for {}: {}", _name, open_result.status().ToString());
            return;
        }
        
        _tensorstore = std::move(open_result).value();
        _tensorstore_initialized = true;
        
        // Verify that the data type is float32 as expected
        auto ts_dtype = _tensorstore.dtype();
        spdlog::debug("TensorStore data type: {}", ts_dtype.name());
        
        // Check if data type is float32 (TensorStore uses "float32" as the name)
        if (ts_dtype.name() != "float32") {
            spdlog::warn("TensorStore data type is not float32, got: {}", ts_dtype.name());
        }
        
        spdlog::debug("Successfully initialized TensorStore for {}", _name);
        
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
        
        spdlog::debug("Updated Zarr image shape from TensorStore: {}", _shape.toString());
        
        spdlog::debug("TensorStore rank: {}", ts_domain.rank());
        spdlog::debug("TensorStore shape: [{}]", 
                    [&ts_shape]() {
                        std::string result;
                        for (size_t i = 0; i < ts_shape.size(); ++i) {
                            if (i > 0) result += ", ";
                            result += std::to_string(ts_shape[i]);
                        }
                        return result;
                    }());
        
        auto ts_end = std::chrono::high_resolution_clock::now();
        auto ts_total_us = std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_start).count();
        spdlog::info("  initializeTensorStore total: {} μs ({:.2f} ms)", ts_total_us, ts_total_us / 1000.0);
        
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
    spdlog::debug("STEP 1: doGetSlice - Request received");
    
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for {}", _name);
        return false;
    }
    
    try {
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        const IPosition& stride = section.stride();

        static int total_calls = 0;
        total_calls++;
        
        // Only log every 10th call or first 5 calls to reduce spam
        if (total_calls <= 5 || total_calls % 10 == 0) {
            spdlog::debug("doGetSlice #{} called with start={}, length={}, stride={}", 
                        total_calls, start.toString(), length.toString(), stride.toString());
        }
        
        // Detect coordinate reordering for cache logic (but don't transform coordinates)
        bool coordinates_reordered = false;
        
        // Detect coordinate reordering based on the pattern observed in error logs
        // Pattern: GetChunk request [512,4096,0,0] -> doGetSlice receives [0,0,4096,512]
        // This suggests coordinates are reordered from [x,y,freq,stokes] to [freq,stokes,y,x]
        if (start.size() >= 4) {
            bool zero_in_spatial_pos = (start[0] == 0 && start[1] == 0);
            bool large_values_in_freq_stokes = (start[2] > 1000 || start[3] > 100);
            bool length_pattern_matches = (length.size() >= 4 && length[0] == 1 && length[1] == 1 && 
                                          (length[2] > 1 || length[3] > 1));
            coordinates_reordered = zero_in_spatial_pos && large_values_in_freq_stokes && length_pattern_matches;
            
            if (coordinates_reordered) {
                spdlog::debug("doGetSlice: Coordinate reordering detected - start={}, length={}", 
                             start.toString(), length.toString());
            }
            
            // Store the current coordinate reordering state for use in getSliceFromCache
            _current_coordinates_reordered = coordinates_reordered;
        }
        
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
            if (coordinates_reordered) {
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
        
        // // Add diagnostic to understand coordinate format - reduce logging frequency
        // static int debug_call_count = 0;
        // static std::unordered_set<std::string> logged_patterns;
        
        // std::string pattern = fmt::format("{}_{}", start.toString(), length.toString());
        // bool should_log = (debug_call_count < 10) || (logged_patterns.find(pattern) == logged_patterns.end());
        
        // if (should_log) {
        //     spdlog::warn("doGetSlice: COORDINATE DEBUG #{} - start={}, length={}", debug_call_count, start.toString(), length.toString());
        //     spdlog::warn("doGetSlice: CARTA shape={}, ZARR shape={}", _shape.toString(), _original_zarr_shape.toString());
        //     logged_patterns.insert(pattern);
        // }
        // debug_call_count++;
        
        int freq_index, stokes_index;
        
        // PATTERN DETECTION: From the error logs, we can see the reordering pattern:
        // GetChunk request: [512,4096,0,0] -> doGetSlice receives: [0,0,4096,512]
        // This suggests: [x,y,freq,stokes] -> [freq,stokes,y,x]
        
        if (coordinates_reordered) {
            // After coordinate reordering, freq and stokes are now in their correct positions
            freq_index = start[0];    // freq 
            stokes_index = start[1];  // stokes
            
            spdlog::warn("doGetSlice: Using reordered coordinates - freq={}, stokes={}", 
                        freq_index, stokes_index);
        } else {
            // Normal case: assume correct CARTA [x, y, freq, stokes] format
            freq_index = (start.size() > 2) ? start[2] : 0;
            stokes_index = (start.size() > 3) ? start[3] : 0;
        }
        
        // Create a unique channel identifier combining freq and stokes
        int current_channel = freq_index * 1000 + stokes_index;  // Assuming max 1000 stokes per freq
        
        // CRITICAL FIX: Calculate request size correctly based on coordinate reordering
        int req_width, req_height;
        if (coordinates_reordered) {
            // Reordered format: [freq, stokes, y, x] -> spatial dimensions are at positions 2,3
            req_width = (length.size() > 3) ? length[3] : 1;   // x dimension
            req_height = (length.size() > 2) ? length[2] : 1;  // y dimension
        } else {
            // Normal format: [x, y, freq, stokes] -> spatial dimensions are at positions 0,1
            req_width = length[0];   // x dimension
            req_height = length[1];  // y dimension
        }
        
        int full_width = _shape[0];  // CARTA width
        int full_height = _shape[1]; // CARTA height
        spdlog::debug("doGetSlice: Request size = {}x{}, Full image size = {}x{} (coordinates_reordered={})", 
                    req_width, req_height, full_width, full_height, coordinates_reordered);

        // Detect region spectral operations: small spatial region reads for spectral analysis
        // These typically read a small region across multiple channels/frequencies
        bool is_region_spectral = false;
        if (start.size() >= 4) {
            int spatial_area = req_width * req_height;
            int full_area = full_width * full_height;
            double area_ratio = static_cast<double>(spatial_area) / static_cast<double>(full_area);
            
            // Region spectral characteristics:
            // 1. Small to moderate spatial area (< 50% of full image)
            // 2. Multi-channel request (length[2] > 1) OR single channel in spectral context
            // 3. NOT a single pixel request (width > 1 OR height > 1)
            // 4. NOT a 1D profile (both width > 1 AND height > 1)
            
            bool is_small_spatial_region = (area_ratio < 0.5 && req_width < full_width && req_height < full_height);
            
            // Check if this is multi-channel based on coordinate format
            bool is_multi_channel = false;
            if (coordinates_reordered) {
                // Reordered format: [freq, stokes, y, x] -> check freq/stokes dimensions
                is_multi_channel = (length[0] > 1 || length[1] > 1);
            } else {
                // Normal format: [x, y, freq, stokes] -> check freq/stokes dimensions
                is_multi_channel = (length[2] > 1 || length[3] > 1);
            }
            
            bool is_not_single_pixel = (req_width > 1 || req_height > 1);
            bool is_not_1d_profile = (req_width > 1 && req_height > 1);
            bool is_not_pixel_spectral = !(req_width == 1 && req_height == 1 && is_multi_channel);  // Exclude pixel spectral
            
            if (is_small_spatial_region && is_not_single_pixel && is_not_1d_profile && is_not_pixel_spectral) {
                is_region_spectral = true;
                spdlog::debug("REGION SPECTRAL DETECTED: area={:.2f}%, multi_ch={}, {}x{} spatial", 
                             area_ratio * 100, is_multi_channel, req_width, req_height);
            }
        }

        // Handle special cases for 1D slices (profiles) and single point requests
        bool is_horizontal_profile = (req_height == 1);  // Horizontal line (y-profile)
        bool is_vertical_profile = (req_width == 1);     // Vertical line (x-profile)
        bool is_single_point = (req_width == 1 && req_height == 1); // Single pixel request
        bool is_1d_profile = (is_horizontal_profile || is_vertical_profile) && !is_single_point;
        
        // CRITICAL FIX: Detect pixel spectral (1x1 spatial + multi-channel) - handle coordinate reordering
        bool is_pixel_spectral = false;
        if (req_width == 1 && req_height == 1) {
            // Check if this is multi-channel based on coordinate format
            if (coordinates_reordered) {
                // Reordered format: [freq, stokes, y, x] -> check freq/stokes dimensions
                is_pixel_spectral = (length[0] > 1 || length[1] > 1);
            } else {
                // Normal format: [x, y, freq, stokes] -> check freq/stokes dimensions
                is_pixel_spectral = (length[2] > 1 || length[3] > 1);
            }
        }
        
        // Use region cache for small requests (e.g., z-profile regions, small tiles)
        // For 1D profiles, always use full channel cache to avoid complexity
        // NEVER use direct read - spatial profiles should ALWAYS use cache system
        // For histogram/statistics calculation, ALWAYS use full channel cache for efficiency
        // CRITICAL: For ALL spectral operations (pixel + region), use direct TensorStore read to avoid cache overhead
        // ADDITIONAL: If coordinates are reordered, prefer full channel cache to avoid boundary issues
        bool use_region_cache = !is_1d_profile && !is_histogram_call && !is_pixel_spectral && !is_region_spectral &&
                               (req_width < full_width / 4 && req_height < full_height / 4) && !coordinates_reordered;
        bool use_direct_read = is_pixel_spectral || is_region_spectral;  // ENABLED for ALL spectral operations
        
        spdlog::debug("CACHE STRATEGY DECISION:");
        spdlog::debug("  Request size: {}x{}, Full size: {}x{}", req_width, req_height, full_width, full_height);
        spdlog::debug("  1D Profile detected: {} (horizontal={}, vertical={})", is_1d_profile ? "YES" : "NO", is_horizontal_profile, is_vertical_profile);
        spdlog::debug("  Single point request: {}", is_single_point ? "YES" : "NO");
        spdlog::debug("  Pixel spectral detected: {}", is_pixel_spectral ? "YES" : "NO");
        spdlog::debug("  Region spectral: {}", is_region_spectral ? "YES" : "NO");
        spdlog::debug("  Use region cache: {}", use_region_cache ? "YES" : "NO");
        spdlog::debug("  Use direct read: {}", use_direct_read ? "YES" : "NO");
        int display_channel = (_cached_channel >= 0) ? _cached_channel / 1000 : _cached_channel;  // Decode frequency channel
        spdlog::debug("  Current cache status: loaded={}, channel={}, is_full={}", _channel_cache_loaded, display_channel, _is_full_channel_cache);
        // spdlog::debug("  Target channel: {}", current_channel);
        
        // CACHE-FIRST STRATEGY (inspired by CartaFitsImage's GetDataSubset approach)
        // CartaFitsImage always tries to avoid direct file access by using efficient data reading patterns
        // We implement similar strategy for ZARR: prioritize cache, force cache for statistics, avoid row-by-row
        
        bool cache_hit = false;
        
        // For single point requests, still try cache first (unlike previous direct read approach)
        if (use_direct_read && (is_pixel_spectral || is_region_spectral)) {
            spdlog::debug("ZARR doGetSlice: Spectral request (pixel or region) - using direct TensorStore read, bypassing cache");
            return readPixelFromTensorStore(buffer, section);
        }
        
        // Check if current cache is suitable (similar to CartaFitsImage's _equiv_bitpix check)
        bool cache_suitable = (_channel_cache_loaded && _cached_channel == current_channel);
        bool cache_type_suitable = true;
        
        // For coordinate reordered requests (channel switches), skip cache system entirely
        bool skip_cache_system = coordinates_reordered;
        
        // OPTIMIZED: For histogram requests, use cached statistics instead of loading full channel
        if (is_histogram_call) {
            // Check if we have cached histogram for this channel
            int channel_id = freq_index * 1000 + stokes_index;
            auto it = _all_channel_stats->find(channel_id);
            bool have_cached_histogram = (it != _all_channel_stats->end() && it->second.valid && !it->second.histogram_bins.empty());
            
            if (have_cached_histogram) {
                // Return cached histogram data without loading _channel_cache
                spdlog::debug("Using cached histogram for channel z={}, stokes={} - no need to load pixel data", 
                             freq_index, stokes_index);
                // The histogram data will be retrieved by ZarrLoader from _all_channel_stats
                return true;  // Skip loading _channel_cache
            } else {
                // Need to compute histogram - but DON'T load into _channel_cache
                spdlog::debug("ZARR doGetSlice: Computing histogram for channel z={}, stokes={} without loading full cache", 
                             freq_index, stokes_index);
                return computeAndCacheHistogram(freq_index, stokes_index, buffer, section);
            }
        }
        
        // For region requests, check if we want region cache instead of full cache
        if (!is_histogram_call && use_region_cache && (_is_full_channel_cache || !cache_suitable)) {
            cache_type_suitable = false;
        }
        
        // Load appropriate cache if needed (like CartaFitsImage's GetDataSubset template selection)
        if ((!cache_suitable || !cache_type_suitable) && !skip_cache_system) {
            
            // BACKGROUND PRELOAD STRATEGY (FITS approach):
            // 1. Load current channel immediately (foreground)
            // 2. Queue background loading of next few channels (4-9 channels ahead)
            // 3. Maintain LRU cache to avoid memory overflow
            
            if (use_region_cache && !is_histogram_call) {
                // IMMEDIATE FOREGROUND LOADING (like FITS FillImageCache)
                // Load current channel/region immediately for user responsiveness
                // Enhanced 4D request detection for region spectral operations
                bool is_4d_request = false;
                int num_freq = 1, num_stokes = 1;
                
                if (length.size() >= 4) {
                    // Check for multi-channel/multi-stokes requests
                    if (length[2] > 1 || length[3] > 1) {
                        is_4d_request = true;
                        // Limit to max 4 channels per batch to prevent memory overflow
                        // For 10000x10000 float32: 4 channels = 1.6GB
                        num_freq = std::min(static_cast<int>(length[2]), 4);
                        num_stokes = length[3];
                        
                        if (length[2] > 4) {
                            spdlog::warn("Request for {} channels exceeds limit, capping at 4 channels per batch", length[2]);
                        }
                    }
                    
                    // DISABLED: For region spectral, use single-channel strategy to avoid cache conflicts
                    // The previous batch loading caused identical values across channels 0-9
                    // Now using single-channel processing in ZarrLoader::GetRegionSpectralData
                    if (false && is_region_spectral && !is_4d_request) {
                        // DISABLED: Load a small group of channels for region spectral efficiency
                        int max_freq = _original_zarr_shape.size() > 1 ? _original_zarr_shape[1] : 1;
                        int current_freq = freq_index;
                        
                        // DISABLED: Load a small batch around current frequency for spectral analysis
                        int batch_size = std::min(10, max_freq - current_freq);
                        if (batch_size > 1) {
                            is_4d_request = true;
                            num_freq = batch_size;
                            num_stokes = std::max(1, static_cast<int>(length[3]));
                            
                            spdlog::debug("REGION SPECTRAL OPTIMIZATION: Loading {} freq channels starting from {}", 
                                         num_freq, current_freq);
                        }
                    }
                }
                
                if (is_4d_request) {
                    // 4D request: load multiple frequencies at once
                    
                    // Extract spatial coordinates
                    int spatial_start_x, spatial_start_y, spatial_width, spatial_height;
                    if (coordinates_reordered) {
                        spatial_start_x = start[3];
                        spatial_start_y = start[2];
                        spatial_width = length[3];
                        spatial_height = length[2];
                    } else {
                        spatial_start_x = start[0];
                        spatial_start_y = start[1];
                        spatial_width = length[0];
                        spatial_height = length[1];
                    }
                    
                    spdlog::debug("ZARR doGetSlice: 4D Region spectral request - {}x{} spatial, {} freq, {} stokes starting from freq={}", 
                                 spatial_width, spatial_height, num_freq, num_stokes, freq_index);
                    
                    // Load 4D region cache with frequency offset
                    if (load4DRegionCache(spatial_start_x, spatial_start_y, spatial_width, spatial_height, 
                                         num_freq, num_stokes, freq_index, stokes_index)) {
                        cache_hit = getSliceFromCache(buffer, section);
                    }
                } else {
                    // IMMEDIATE FOREGROUND LOADING for current channel (like FITS FillImageCache)
                    // Then BACKGROUND PRELOAD STRATEGY: Queue loading of next 4-9 channels
                    
                    // For small requests, use adaptive padding based on region size
                    int padding = std::min(100, std::min(req_width, req_height));        // Standard padding for small requests
                    
                    // Extract correct spatial coordinates based on reordering detection
                    int spatial_start_x, spatial_start_y;
                    if (coordinates_reordered) {
                        // Reordered format: [freq, stokes, y, x] -> extract y,x as spatial coords
                        spatial_start_x = (start.size() > 3) ? start[3] : 0;
                        spatial_start_y = (start.size() > 2) ? start[2] : 0;
                    } else {
                        // Normal format: [x, y, freq, stokes]
                        spatial_start_x = start[0];
                        spatial_start_y = start[1];
                    }
                    
                    int region_start_x = std::max(0, spatial_start_x - padding);
                    int region_start_y = std::max(0, spatial_start_y - padding);
                    int region_width = std::min(full_width - region_start_x, req_width + 2 * padding);
                    int region_height = std::min(full_height - region_start_y, req_height + 2 * padding);
                    
                    spdlog::debug("ZARR doGetSlice: Loading region cache [freq={}, stokes={}]: {}x{} at ({},{}) with padding {} (small_request)",
                                freq_index, stokes_index, region_width, region_height, region_start_x, region_start_y, padding);
                    
                    if (loadRegionCache(freq_index, stokes_index, region_start_x, region_start_y, region_width, region_height)) {
                        cache_hit = getSliceFromCache(buffer, section);
                    }
                }
                
                if (!cache_hit) {
                    spdlog::warn("ZARR doGetSlice: Region cache failed, falling back to full channel cache");
                    use_region_cache = false;  // Fall back to full channel cache
                    // For region spectral, prefer direct read over full channel cache
                    return readPixelFromTensorStore(buffer, section);
                }
            }
            
            if ((!use_region_cache && !is_region_spectral) || !cache_hit) {
                // Load full channel cache (equivalent to CartaFitsImage's full data subset read)
                spdlog::debug("ZARR doGetSlice: Loading full channel [freq={}, stokes={}] - FITS-style efficient data access",
                            freq_index, stokes_index);
                
                if (loadChannelCache(freq_index, stokes_index)) {
                    // OPTIMIZATION: For full channel requests without offset, copy cache directly
                    bool can_copy_directly = false;
                    if (_is_full_channel_cache && _channel_cache_loaded) {
                        // Check if this is a full channel request (no offset, matches cache dimensions)
                        int req_start_x, req_start_y, req_width, req_height;
                        if (coordinates_reordered) {
                            req_start_x = start[2];
                            req_start_y = start[3];
                            req_width = length[2];
                            req_height = length[3];
                        } else {
                            req_start_x = start[1];
                            req_start_y = start[0];
                            req_width = length[1];
                            req_height = length[0];
                        }
                        spdlog::info("length[0]={}, length[1]={}, length[2]={}, length[3]={}", 
                                    length[0], length[1], length[2], length[3]);
                        
                        // For simple channel switches: request entire image without offset
                        if (req_start_x == 0 && req_start_y == 0 && 
                            req_width == _cache_width && req_height == _cache_height) {
                            can_copy_directly = true;
                            spdlog::debug("STEP 6 OPTIMIZATION: Direct cache copy after loading full channel - skipping getSliceFromCache");
                            
                            // Direct copy from cache to buffer
                            buffer.resize(length);
                            float* dest_data = buffer.data();
                            std::copy(_channel_cache->begin(), _channel_cache->end(), dest_data);
                            
                            cache_hit = true;
                        }
                    }
                    
                    if (!can_copy_directly) {
                        cache_hit = getSliceFromCache(buffer, section);
                    }
                }
                
                if (!cache_hit) {
                    spdlog::warn("ZARR doGetSlice: Full channel cache failed, falling back to direct TensorStore read");
                }
            }
        } else {
            // Cache is already loaded and suitable, use it (like CartaFitsImage using already loaded data)
            // But skip cache system for coordinate reordered requests (channel switches)
            if (!skip_cache_system) {
                // OPTIMIZATION: For full channel requests without offset, copy cache directly
                bool can_copy_directly = false;
                if (_is_full_channel_cache && _channel_cache_loaded) {
                    // Check if this is a full channel request (no offset, matches cache dimensions)
                    int req_start_x, req_start_y, req_width, req_height;
                    if (coordinates_reordered) {
                        req_start_x = start[2];
                        req_start_y = start[3];
                        req_width = length[2];
                        req_height = length[3];
                    } else {
                        req_start_x = start[1];
                        req_start_y = start[0];
                        req_width = length[1];
                        req_height = length[0];
                    }
                    spdlog::info("length[0]={}, length[1]={}, length[2]={}, length[3]={}", 
                                    length[0], length[1], length[2], length[3]);
                    
                    // For simple channel switches: request entire image without offset
                    if (req_start_x == 0 && req_start_y == 0 && 
                        req_width == _cache_width && req_height == _cache_height) {
                        can_copy_directly = true;
                        spdlog::debug("STEP 6 OPTIMIZATION: Direct cache copy for full channel request - skipping getSliceFromCache");
                        
                        // Direct copy from cache to buffer
                        buffer.resize(length);
                        float* dest_data = buffer.data();
                        std::copy(_channel_cache->begin(), _channel_cache->end(), dest_data);
                        
                        cache_hit = true;
                    }
                }
                
                if (!can_copy_directly) {
                    cache_hit = getSliceFromCache(buffer, section);
                }
            }
        }
        
        // If cache served the request successfully, return (like CartaFitsImage successful GetDataSubset)
        if (cache_hit) {
            spdlog::debug("STEP 7: Cache Hit - Successfully served {} from cache", 
                        is_histogram_call ? "statistics/histogram" : "data");
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
                // When coordinates are reordered: [freq,stokes,y,x] instead of [x,y,freq,stokes]
                correct_freq = freq_index;    // Already corrected
                correct_stokes = stokes_index; // Already corrected
                correct_x = start[3];         // x is in position 3 after reordering
                correct_y = start[2];         // y is in position 2 after reordering
                
                spdlog::debug("  Using corrected coordinates from reordered: x={}, y={}, freq={}, stokes={}", 
                             correct_x, correct_y, correct_freq, correct_stokes);
            } else {
                // Normal case: use standard CARTA [x, y, freq, stokes] format
                correct_x = start[0];
                correct_y = start[1];
                correct_freq = start[2];
                correct_stokes = start[3];
                
                spdlog::debug("  Using standard CARTA coordinates: x={}, y={}, freq={}, stokes={}", 
                             correct_x, correct_y, correct_freq, correct_stokes);
            }
            
            // Map corrected CARTA [x, y, freq, stokes] -> ZARR [time, freq, stokes, l, m]
            // ZARR format: [time, freq, pol, l, m] where l=width(x), m=height(y)
            zarr_start.resize(5);
            zarr_length.resize(5);
            
            zarr_start[0] = 0;                    // ZARR[0]=time (always 0)
            zarr_start[1] = correct_freq;         // ZARR[1]=freq
            zarr_start[2] = correct_stokes;       // ZARR[2]=stokes  
            zarr_start[3] = correct_x;            // ZARR[3]=l (width/x)
            zarr_start[4] = correct_y;            // ZARR[4]=m (height/y)
            
            zarr_length[0] = 1;                   // ZARR[0]=time (always 1)
            zarr_length[1] = 1;                   // ZARR[1]=freq (single frequency)
            zarr_length[2] = 1;                   // ZARR[2]=stokes (single stokes)
            
            // CRITICAL: Clip lengths to stay within ZARR bounds
            int requested_width = coordinates_reordered ? length[3] : length[0];
            int requested_height = coordinates_reordered ? length[2] : length[1];
            
            // Clip l dimension (ZARR dimension 3 = width/x)  
            int max_x_length = _original_zarr_shape[3] - correct_x;
            zarr_length[3] = std::max(0, std::min(requested_width, max_x_length));
            
            // Clip m dimension (ZARR dimension 4 = height/y)
            int max_y_length = _original_zarr_shape[4] - correct_y;
            zarr_length[4] = std::max(0, std::min(requested_height, max_y_length));
            
            // If the request is completely outside bounds, return empty buffer
            if (zarr_length[3] <= 0 || zarr_length[4] <= 0) {
                spdlog::warn("Request completely outside ZARR bounds: x=[{},{}), y=[{},{}), ZARR bounds: x_max={}, y_max={}",
                           correct_x, correct_x + requested_width,
                           correct_y, correct_y + requested_height,
                           _original_zarr_shape[3], _original_zarr_shape[4]);
                
                // Return buffer filled with NaN values to indicate invalid region
                IPosition requested_shape = coordinates_reordered ? 
                    IPosition(4, length[3], length[2], length[1], length[0]) :
                    length;
                buffer.resize(requested_shape);
                buffer = std::numeric_limits<float>::quiet_NaN();
                return true;
            }
            
            if (zarr_length[3] != requested_width || zarr_length[4] != requested_height) {
                spdlog::warn("  CLIPPED REQUEST: original width={} height={}, clipped width={} height={}", 
                            requested_width, requested_height, zarr_length[3], zarr_length[4]);
                spdlog::warn("  ZARR bounds: x_max={}, y_max={}, requested x=[{},{}), y=[{},{})", 
                            _original_zarr_shape[3], _original_zarr_shape[4],
                            correct_x, correct_x + requested_width,
                            correct_y, correct_y + requested_height);
            }
            
            spdlog::debug("  Mapped to ZARR 5D: start={}, length={}", zarr_start.toString(), zarr_length.toString());
            spdlog::debug("  ZARR coordinate check: time={}, freq={}, pol={}, l=[{},{}), m=[{},{})", 
                         zarr_start[0], zarr_start[1], zarr_start[2], 
                         zarr_start[3], zarr_start[3] + zarr_length[3],
                         zarr_start[4], zarr_start[4] + zarr_length[4]);
        }
        
        // Skip TensorStore read for reordered coordinates as requested
        if (coordinates_reordered) {
            spdlog::info("req_width={}, req_height={}, coordinates_reordered=true - skipping TensorStore read", 
                        req_width, req_height);
            return false;
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
            actual_read_shape[0] = zarr_length[3];  // width (x) - ZARR l dimension
            actual_read_shape[1] = zarr_length[4];  // height (y) - ZARR m dimension
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
    // Get data for the section
    Array<float> data_buffer;
    if (!doGetSlice(data_buffer, section)) {
        return false;
    }
    
    // Create mask where finite values are true (valid), NaN/infinite values are false (invalid)
    buffer = isFinite(data_buffer);
    return true;
}

Bool CartaZarrImage::isMasked() const {
    return true; // Enable masking to handle NaN values
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
    return true; // Enable pixel mask to handle NaN values
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
    // Use copy constructor to share TensorStore and cache (like CartaFitsImage)
    spdlog::debug("CartaZarrImage::cloneII called - using copy constructor for efficiency");
    return new CartaZarrImage(*this);
}

const CoordinateSystem& CartaZarrImage::coordinates() const {
    return _coord_sys;
}

bool CartaZarrImage::loadChannelCache(int freq_channel, int stokes_channel) {
    spdlog::debug("STEP 2: loadChannelCache - Starting to load channel cache [freq={}, stokes={}]", freq_channel, stokes_channel);
    
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for channel cache loading");
        return false;
    }
    
    // Check if this channel is already cached - if so, just return success
    int requested_channel_id = freq_channel * 1000 + stokes_channel;
    if (_channel_cache_loaded && _cached_channel == requested_channel_id) {
        spdlog::debug("Channel z={}, stokes={} already cached, skipping reload", freq_channel, stokes_channel);
        return true;
    }
    
    // IMPORTANT: Do NOT clear or load _channel_cache for histogram calculations
    // Histogram data is now cached in _all_channel_stats[channel_id].histogram_bins
    // Only clear _channel_cache when switching viewing channels (not for stats)
    if (_channel_cache_loaded && _cached_channel != requested_channel_id) {
        _channel_cache->clear();  // Clear but keep allocated capacity for reuse
        _channel_cache->shrink_to_fit();  // Force deallocation
        _channel_cache_loaded = false;
        spdlog::debug("Cleared old channel cache before loading new channel");
    }
    
    try {
        // For 5D ZARR, load the entire specified channel [time=0, freq=freq_channel, pol=stokes_channel, l=all, m=all]
        if (_original_zarr_shape.size() == 5) {
            _cache_width = _original_zarr_shape[3];   // l dimension (width)
            _cache_height = _original_zarr_shape[4];  // m dimension (height)
            
            spdlog::debug("Cache dimensions from ZARR shape: width={}, height={}", _cache_width, _cache_height);
            spdlog::info("CARTA shape: [{}, {}]", _shape[0], _shape[1]);
            spdlog::info("ZARR original shape: [{}]", fmt::join(_original_zarr_shape, ", "));
            
            // CRITICAL FIX: The freq_channel and stokes_channel parameters are CARTA coordinates
            // We need to validate they're within ZARR bounds and map them correctly
            spdlog::debug("CACHE LOAD: Input CARTA coordinates - freq={}, stokes={}", freq_channel, stokes_channel);
            spdlog::debug("CACHE LOAD: ZARR bounds - freq=[0,{}), stokes=[0,{})", _original_zarr_shape[1], _original_zarr_shape[2]);
            
            // Validate coordinates are within ZARR bounds
            if (freq_channel < 0 || freq_channel >= _original_zarr_shape[1]) {
                spdlog::error("CACHE LOAD: Frequency index {} out of range [0,{})", freq_channel, _original_zarr_shape[1]);
                return false;
            }
            
            if (stokes_channel < 0 || stokes_channel >= _original_zarr_shape[2]) {
                spdlog::error("CACHE LOAD: Stokes index {} out of range [0,{})", stokes_channel, _original_zarr_shape[2]);
                return false;
            }
            
            // Load only ONE channel at a time to minimize memory usage
            // Memory per channel: 7763×4742 pixels × 4 bytes = ~147MB
            // Loading only current channel reduces cache from 588MB (4 channels) to 147MB
            int num_cache_channels = 1;  // Changed from 4 to 1
            std::vector<tensorstore::Index> box_origin = {0, freq_channel, stokes_channel, 0, 0};
            std::vector<tensorstore::Index> box_shape = {1, num_cache_channels, 1, _cache_width, _cache_height};
            
            // spdlog::info("CACHE LOADING {} CHANNEL (freq={})", num_cache_channels, freq_channel);
            // spdlog::info("  ZARR 5D cache box: origin=[{},{},{},{},{}], shape=[{},{},{},{},{}]", 
            //              box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4],
            //              box_shape[0], box_shape[1], box_shape[2], box_shape[3], box_shape[4]);
            // spdlog::info("  This reads ZARR coordinates:");
            // spdlog::info("    time: [0, 1)");
            // spdlog::info("    freq: [{}, {})", freq_channel, freq_channel + num_cache_channels);
            // spdlog::info("    stokes: [{}, {})", stokes_channel, stokes_channel + 1);
            // spdlog::info("    l (width/x): [0, {})", _cache_width);
            // spdlog::info("    m (height/y): [0, {})", _cache_height);
            // spdlog::info("  So y=1000 in ZARR corresponds to m=1000 coordinate");
            
            tensorstore::Box<> cache_box(box_origin, box_shape);
            
            spdlog::debug("STEP 3: TensorStore I/O - Reading {}x{} pixels from disk", _cache_width, _cache_height);
            
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
            
            // DEBUG: Comprehensive TensorStore data validation
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            
            // // Check array dimensions from TensorStore
            // spdlog::info("DEBUG: TensorStore array info:");
            // spdlog::info("  Shape: {}", zarr_array.shape().size());
            // for (size_t i = 0; i < zarr_array.shape().size(); ++i) {
            //     spdlog::info("    Dimension[{}]: {}", i, zarr_array.shape()[i]);
            // }
            // spdlog::info("  Total elements: {}", zarr_array.num_elements());
            // spdlog::info("  Expected elements: {} * {} * {} = {}", 
            //             num_cache_channels, _cache_width, _cache_height,
            //             num_cache_channels * _cache_width * _cache_height);
            
            // CRITICAL: TensorStore 5D array layout analysis
            // Shape: [time=1, freq=4, stokes=1, l=7763, m=4742]
            // Need to determine memory layout order
            spdlog::info("DEBUG: Analyzing TensorStore memory layout for 5D array:");
            spdlog::info("  Requested box: time=[0,1), freq=[0,{}), stokes=[0,1), l=[0,{}), m=[0,{})", 
                        num_cache_channels, _cache_width, _cache_height);
            
            // Test different possible layouts by checking known boundary positions
            // For this image, boundaries (y=0, y=last, x=0, x=last) should have NaN
            
            // if (_cache_height > 1000 && _cache_width > 1000) {
            //     // Try different stride calculations to find correct layout
            //     spdlog::info("DEBUG: Testing different memory layout interpretations:");
                
            //     // Layout 1: Row-major C-style [time][freq][stokes][l][m] 
            //     // Element at (t=0, f=0, s=0, l=x, m=y) is at: t*F*S*L*M + f*S*L*M + s*L*M + l*M + m
            //     spdlog::info("  Layout 1 (C-style row-major [time][freq][stokes][l][m]):");
            //     size_t layout1_stride_freq = _cache_width * _cache_height;  // S*L*M (stokes=1)
            //     size_t layout1_y0_offset = 0;  // y=0, x=0, first channel
            //     size_t layout1_y1000_offset = 1000;  // y=1000, x=0
            //     size_t layout1_ylast_offset = _cache_height - 1;  // y=last, x=0
                
            //     spdlog::info("    y=0, x=0: {:.6e} (isnan={})", 
            //                 src_data[layout1_y0_offset], std::isnan(src_data[layout1_y0_offset]));
            //     spdlog::info("    y=1000, x=0: {:.6e} (isnan={})", 
            //                 src_data[layout1_y1000_offset], std::isnan(src_data[layout1_y1000_offset]));
            //     spdlog::info("    y=last({}), x=0: {:.6e} (isnan={})", 
            //                 _cache_height - 1, src_data[layout1_ylast_offset], std::isnan(src_data[layout1_ylast_offset]));
                
            //     // Layout 2: Row-major with different axis order [time][freq][stokes][m][l]
            //     // Element at (t=0, f=0, s=0, l=x, m=y) is at: t*F*S*M*L + f*S*M*L + s*M*L + m*L + l
            //     spdlog::info("  Layout 2 (row-major [time][freq][stokes][m][l]):");
            //     size_t layout2_stride_m = _cache_width;  // L
            //     size_t layout2_y0_offset = 0 * layout2_stride_m + 0;  // y=0, x=0
            //     size_t layout2_y1000_offset = 1000 * layout2_stride_m + 0;  // y=1000, x=0
            //     size_t layout2_ylast_offset = (_cache_height - 1) * layout2_stride_m + 0;  // y=last, x=0
                
            //     spdlog::info("    y=0, x=0: {:.6e} (isnan={})", 
            //                 src_data[layout2_y0_offset], std::isnan(src_data[layout2_y0_offset]));
            //     spdlog::info("    y=1000, x=0: {:.6e} (isnan={})", 
            //                 src_data[layout2_y1000_offset], std::isnan(src_data[layout2_y1000_offset]));
            //     spdlog::info("    y=last({}), x=0: {:.6e} (isnan={})", 
            //                 _cache_height - 1, src_data[layout2_ylast_offset], std::isnan(src_data[layout2_ylast_offset]));
                
            //     // Check which layout gives NaN at boundaries
            //     bool layout1_has_boundary_nan = std::isnan(src_data[layout1_y0_offset]) && std::isnan(src_data[layout1_ylast_offset]);
            //     bool layout2_has_boundary_nan = std::isnan(src_data[layout2_y0_offset]) && std::isnan(src_data[layout2_ylast_offset]);
                
            //     spdlog::info("  Layout 1 boundary check: {}", layout1_has_boundary_nan ? "CORRECT (NaN at y=0 and y=last)" : "WRONG");
            //     spdlog::info("  Layout 2 boundary check: {}", layout2_has_boundary_nan ? "CORRECT (NaN at y=0 and y=last)" : "WRONG");
            // }
            
            // Verify data type
            if (zarr_array.dtype().name() != "float32") {
                spdlog::error("Unexpected data type in cache: {}", zarr_array.dtype().name());
                return false;
            }
            
            // Copy data to cache (now contains num_cache_channels channels)
            size_t total_elements = num_cache_channels * _cache_width * _cache_height;
            _channel_cache->resize(total_elements);
            
            std::copy(src_data, src_data + total_elements, _channel_cache->data());
            
            spdlog::debug("STEP 4: Data Copy - Copied {} elements to cache", total_elements);
            
            // Store the cached channel identifier and number of channels
            _cached_channel = freq_channel * 1000 + stokes_channel;
            _num_cached_channels = num_cache_channels;
            _cache_start_x = 0;  // Full channel cache starts at origin
            _cache_start_y = 0;
            _channel_cache_loaded = true;
            _is_full_channel_cache = true;  // This is a full channel cache
            
            spdlog::debug("Loaded SINGLE channel [freq={}, stokes={}] cache: {}x{} pixels ({} MB)", 
                        freq_channel, stokes_channel,
                        _cache_width, _cache_height, 
                        (total_elements * sizeof(float)) / (1024 * 1024));
            
            // Calculate and cache statistics for this channel (only if not already computed)
            int channel_id = freq_channel * 1000 + stokes_channel;
            auto it = _all_channel_stats->find(channel_id);
            bool need_compute_stats = (it == _all_channel_stats->end() || !it->second.valid);
            
            if (need_compute_stats) {
                ChannelStats& stats = (*_all_channel_stats)[channel_id];
                stats.freq_channel = freq_channel;
                stats.stokes_channel = stokes_channel;
                stats.valid_pixels = 0;
                stats.sum = 0.0;
                stats.sum_sq = 0.0;
                stats.min_val = std::numeric_limits<float>::max();
                stats.max_val = std::numeric_limits<float>::lowest();
                
                for (size_t i = 0; i < total_elements; ++i) {
                    float val = (*_channel_cache)[i];
                    if (!std::isnan(val) && std::isfinite(val)) {
                        stats.valid_pixels++;
                        stats.sum += val;
                        stats.sum_sq += val * val;
                        stats.min_val = std::min(stats.min_val, (double)val);
                        stats.max_val = std::max(stats.max_val, (double)val);
                    }
                }
                stats.valid = (stats.valid_pixels > 0);
                
                if (stats.valid) {
                    spdlog::info("Computed stats for channel z={}, stokes={}: min={:.6e}, max={:.6e}, valid_pixels={}", 
                                freq_channel, stokes_channel, 
                                stats.min_val, stats.max_val, 
                                stats.valid_pixels);
                }
                spdlog::debug("STEP 5: Statistics - Computed channel statistics");
            } else {
                spdlog::debug("Using cached stats for channel z={}, stokes={} (min={:.6e}, max={:.6e})", 
                             freq_channel, stokes_channel,
                             it->second.min_val, it->second.max_val);
            }
            
            // DEBUG: Output y=1000 x-profile for first channel to verify cache storage
            // if (_cache_height > 1000) {
            //     spdlog::info("DEBUG: Channel 0 (freq={}) y=1000 x-profile (first 10 pixels):", freq_channel);
            //     size_t row_offset = 1000 * _cache_width;
            //     for (int x = 0; x < std::min(10, static_cast<int>(_cache_width)); ++x) {
            //         float val = _channel_cache[row_offset + x];
            //         spdlog::info("  x={}: {:.6e} (isnan={})", x, val, std::isnan(val));
            //     }
                
            //     // Verify by reading directly from cache
            //     spdlog::info("DEBUG: Verifying by re-reading y=1000 x-profile from cache:");
            //     for (int x = 0; x < std::min(10, static_cast<int>(_cache_width)); ++x) {
            //         size_t cache_idx = row_offset + x;
            //         if (cache_idx < _channel_cache.size()) {
            //             float val = _channel_cache[cache_idx];
            //             spdlog::info("  x={}: {:.6e} (isnan={})", x, val, std::isnan(val));
            //         }
            //     }
            // }
            
            // // DEBUG: Check TensorStore data layout and verify expected NaN boundaries
            // spdlog::debug("TENSORSTORE DATA LAYOUT DEBUG:");
            // spdlog::debug("  TensorStore array shape: {}", zarr_array.shape().size());
            // for (size_t i = 0; i < zarr_array.shape().size(); ++i) {
            //     spdlog::debug("    Dimension {}: {}", i, zarr_array.shape()[i]);
            // }
            
            // // Check first few and expected boundary rows specifically 
            // std::vector<int> test_rows = {0, 1, 2, 183, 184, 249, 250, 251, 252, 253};
            // for (int row : test_rows) {
            //     if (row >= _cache_height) continue;
                
            //     // Sample a few columns in this row
            //     std::vector<int> sample_cols = {100, 1000, 3000, 5000, 7000};
            //     int nan_count = 0, finite_count = 0;
                
            //     for (int col : sample_cols) {
            //         if (col >= _cache_width) continue;
                    
            //         size_t cache_idx = row * _cache_width + col;
            //         if (cache_idx < _channel_cache.size()) {
            //             float val = _channel_cache[cache_idx];
            //             if (std::isnan(val)) {
            //                 nan_count++;
            //             } else if (std::isfinite(val)) {
            //                 finite_count++;
            //             }
            //         }
            //     }
                
            //     spdlog::debug("  Row {}: {}/{} NaN, {}/{} finite (expected: row<250 should be NaN)", 
            //                  row, nan_count, sample_cols.size(), finite_count, sample_cols.size());
            // }
            
            // // Sample a few key positions to understand data layout
            // spdlog::debug("CACHE DATA SAMPLING:");
            // for (int y = 0; y < std::min(5, _cache_height); y++) {
            //     int x = 0;
            //     int idx = y * _cache_width + x;
            //     float val = _channel_cache[idx];
            //     spdlog::debug("  row={}, col={}, idx={}, value={}, isNaN={}", y, x, idx, val, std::isnan(val));
            // }
            
            // // Precise boundary analysis based on image data
            // if (_cache_height > 250) {
            //     spdlog::debug("PRECISE BOUNDARY ANALYSIS:");
                
            //     // Sample multiple columns to get accurate boundary
            //     std::vector<int> sample_columns = {100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000};
            //     int consensus_first_row = -1;
            //     int consensus_last_row = -1;
                
            //     // Check each sample column
            //     for (int col : sample_columns) {
            //         if (col >= _cache_width) continue;
                    
            //         int col_first_row = -1;
            //         int col_last_row = -1;
                    
            //         for (int y = 0; y < _cache_height; y++) {
            //             int idx = y * _cache_width + col;
            //             float val = _channel_cache[idx];
            //             if (!std::isnan(val) && std::isfinite(val)) {
            //                 if (col_first_row == -1) col_first_row = y;
            //                 col_last_row = y;
            //             }
            //         }
                    
            //         spdlog::debug("  Column {}: first_row={}, last_row={}", col, col_first_row, col_last_row);
                    
            //         // Update consensus (use most restrictive bounds)
            //         if (col_first_row >= 0) {
            //             if (consensus_first_row == -1 || col_first_row < consensus_first_row) {
            //                 consensus_first_row = col_first_row;
            //             }
            //         }
            //         if (col_last_row >= 0) {
            //             if (consensus_last_row == -1 || col_last_row > consensus_last_row) {
            //                 consensus_last_row = col_last_row;
            //             }
            //         }
            //     }
                
            //     // Additional detailed analysis around discovered boundaries
            //     if (consensus_first_row >= 0) {
            //         spdlog::debug("DETAILED BOUNDARY EXAMINATION:");
            //         int check_start = std::max(0, consensus_first_row - 5);
            //         int check_end = std::min(_cache_height - 1, consensus_first_row + 5);
                    
            //         for (int y = check_start; y <= check_end; y++) {
            //             // Count valid pixels in this row
            //             int valid_pixels = 0;
            //             int total_pixels = 0;
            //             for (int x = 0; x < _cache_width; x += 100) { // Sample every 100 pixels
            //                 int idx = y * _cache_width + x;
            //                 float val = _channel_cache[idx];
            //                 if (!std::isnan(val) && std::isfinite(val)) {
            //                     valid_pixels++;
            //                 }
            //                 total_pixels++;
            //             }
            //             float valid_ratio = (float)valid_pixels / total_pixels;
            //             spdlog::debug("    Row {}: {}/{} valid pixels ({:.1f}%)", 
            //                          y, valid_pixels, total_pixels, valid_ratio * 100);
            //         }
            //     }
                
            //     spdlog::info("FINAL BOUNDARY ANALYSIS:");
            //     spdlog::info("  Consensus first data row: {}", consensus_first_row);
            //     spdlog::info("  Consensus last data row: {}", consensus_last_row);
            //     spdlog::info("  Total data rows: {}", (consensus_first_row >= 0 && consensus_last_row >= 0) ? (consensus_last_row - consensus_first_row + 1) : 0);
            //     spdlog::info("  NaN header rows: {}", consensus_first_row >= 0 ? consensus_first_row : _cache_height);
            //     spdlog::info("  NaN footer rows: {}", consensus_last_row >= 0 ? (_cache_height - consensus_last_row - 1) : 0);
            //     spdlog::info("  Data coverage: {:.1f}% of image height", 
            //                 consensus_first_row >= 0 && consensus_last_row >= 0 ? 
            //                 ((float)(consensus_last_row - consensus_first_row + 1) / _cache_height * 100) : 0.0);
            // }
            
            return true;
        } else {
            // For non-5D arrays, load the entire 2D image (ignore freq/stokes parameters)
            spdlog::debug("Check _shape", _shape.toString());
            spdlog::debug("Check _cache_width _cache_height", _cache_width, _cache_height);
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
            _channel_cache->resize(total_elements);
            
            // Check if data type is float32
            if (zarr_array.dtype() != tensorstore::dtype_v<float>) {
                spdlog::error("ZARR array dtype is not float32: {}", zarr_array.dtype().name());
                return false;
            }
            
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            
            // Debug: Check first few values
            spdlog::debug("First 5 ZARR values: [{}, {}, {}, {}, {}]", 
                         src_data[0], src_data[1], src_data[2], src_data[3], src_data[4]);
            
            std::copy(src_data, src_data + total_elements, _channel_cache->data());
            
            // Debug: Check first few cached values
            spdlog::debug("First 5 cached values: [{}, {}, {}, {}, {}]", 
                         (*_channel_cache)[0], (*_channel_cache)[1], (*_channel_cache)[2], (*_channel_cache)[3], (*_channel_cache)[4]);
            
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

bool CartaZarrImage::computeAndCacheHistogram(int freq_channel, int stokes_channel, 
                                              casacore::Array<float>& buffer, const casacore::Slicer& section) {
    try {
        if (_original_zarr_shape.size() != 5) {
            spdlog::error("computeAndCacheHistogram: Only 5D ZARR supported");
            return false;
        }
        
        size_t width = _original_zarr_shape[3];
        size_t height = _original_zarr_shape[4];
        
        // Read channel data directly from TensorStore for statistics calculation
        std::vector<tensorstore::Index> box_origin = {0, freq_channel, stokes_channel, 0, 0};
        std::vector<tensorstore::Index> box_shape = {1, 1, 1, static_cast<tensorstore::Index>(width), static_cast<tensorstore::Index>(height)};
        tensorstore::Box<> cache_box(box_origin, box_shape);
        
        auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(cache_box);
        if (!constrained_store.ok()) {
            spdlog::error("computeAndCacheHistogram: Failed to create slice: {}", constrained_store.status().ToString());
            return false;
        }
        
        auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
        if (!read_result.ok()) {
            spdlog::error("computeAndCacheHistogram: Failed to read data: {}", read_result.status().ToString());
            return false;
        }
        
        auto zarr_array = std::move(read_result.value());
        const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
        size_t total_elements = width * height;
        
        // Compute basic statistics
        int channel_id = freq_channel * 1000 + stokes_channel;
        ChannelStats& stats = (*_all_channel_stats)[channel_id];
        stats.freq_channel = freq_channel;
        stats.stokes_channel = stokes_channel;
        stats.valid_pixels = 0;
        stats.sum = 0.0;
        stats.sum_sq = 0.0;
        stats.min_val = std::numeric_limits<float>::max();
        stats.max_val = std::numeric_limits<float>::lowest();
        
        for (size_t i = 0; i < total_elements; ++i) {
            float val = src_data[i];
            if (!std::isnan(val) && std::isfinite(val)) {
                stats.valid_pixels++;
                stats.sum += val;
                stats.sum_sq += val * val;
                stats.min_val = std::min(stats.min_val, (double)val);
                stats.max_val = std::max(stats.max_val, (double)val);
            }
        }
        
        if (stats.valid_pixels == 0) {
            spdlog::warn("computeAndCacheHistogram: No valid pixels in channel z={}, stokes={}", freq_channel, stokes_channel);
            stats.valid = false;
            return false;
        }
        
        // Compute histogram (default 10000 bins like FITS)
        int num_bins = 10000;
        double range = stats.max_val - stats.min_val;
        double bin_width = range / num_bins;
        
        stats.histogram_bins.resize(num_bins, 0);
        stats.num_bins = num_bins;
        stats.bin_width = bin_width;
        stats.bin_center = stats.min_val;
        
        for (size_t i = 0; i < total_elements; ++i) {
            float val = src_data[i];
            if (!std::isnan(val) && std::isfinite(val)) {
                int bin = static_cast<int>((val - stats.min_val) / bin_width);
                if (bin >= num_bins) bin = num_bins - 1;
                if (bin < 0) bin = 0;
                stats.histogram_bins[bin]++;
            }
        }
        
        stats.valid = true;
        
        spdlog::info("Computed and cached histogram for channel z={}, stokes={}: {} bins, min={:.6e}, max={:.6e}, valid_pixels={}", 
                    freq_channel, stokes_channel, num_bins,
                    stats.min_val, stats.max_val, stats.valid_pixels);
        
        // Fill buffer with the histogram data (if requested)
        if (buffer.size() == num_bins) {
            float* dest = buffer.data();
            for (int i = 0; i < num_bins; ++i) {
                dest[i] = static_cast<float>(stats.histogram_bins[i]);
            }
        }
        
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("computeAndCacheHistogram: Exception: {}", e.what());
        return false;
    }
}

bool CartaZarrImage::getSliceFromCache(casacore::Array<float>& buffer, const casacore::Slicer& section) {
    if (!_channel_cache_loaded) {
        return false;
    }
    
    spdlog::info("STEP 6: getSliceFromCache - Extracting data from cache");
    
    try {
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        
        // CRITICAL: Handle coordinate reordering in cache access too
        // The same reordering that affects doGetSlice also affects cache access
        int start_x, start_y, req_width, req_height;
        
        // Use the coordinate reordering state detected in doGetSlice
        bool coordinates_reordered = _current_coordinates_reordered;
        
        if (coordinates_reordered) {
            // Since we fixed the buffer shape in doGetSlice, coordinates are now correct
            // No need to reorder - use normal CARTA [x, y, freq, stokes] format
            start_x = start[0];      // x from position 0
            start_y = start[1];      // y from position 1
            req_width = length[0];   // width from position 0
            req_height = length[1];  // height from position 1
            
            spdlog::debug("getSliceFromCache: Using corrected coordinates - x={}, y={}, width={}, height={}", 
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
        
        if (_channel_cache->empty()) {
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
        
        // CRITICAL FIX: Handle 4D cache data for region spectral operations
        // Determine if we're dealing with 4D data (width, height, freq, stokes)
        bool is_4d_data = false;
        int num_freq = 1, num_stokes = 1;
        
        if (length.size() >= 4 && length[2] > 1) {
            // This is a 4D request [x, y, freq, stokes] with multiple frequencies
            is_4d_data = true;
            num_freq = length[2];
            num_stokes = length[3];
            
            spdlog::debug("MULTI-CHANNEL CACHE ACCESS: 4D data detected - width={}, height={}, freq={}, stokes={}", 
                         req_width, req_height, num_freq, num_stokes);
            spdlog::debug("Cache size: {} elements, expected 4D size: {}", 
                         _channel_cache->size(), _cache_width * _cache_height * num_freq * num_stokes);
        }
        
        if (is_4d_data) {
            // ENHANCED 4D ACCESS for region spectral data with frequency/stokes offset validation
            // Cache layout: [width, height, freq, stokes] in row-major order
            // dest layout: [x, y, freq, stokes] in row-major order
            
            // Get the frequency and stokes indices from the current request
            const IPosition& request_start = section.start();
            int request_freq_start = (request_start.size() > 2) ? request_start[2] : 0;
            int request_stokes_start = (request_start.size() > 3) ? request_start[3] : 0;
            
            spdlog::debug("4D CACHE ACCESS: Request freq={}:{} stokes={}:{}, Cache freq={}:{} stokes={}:{}", 
                         request_freq_start, request_freq_start + num_freq - 1,
                         request_stokes_start, request_stokes_start + num_stokes - 1,
                         _cache_freq_start, _cache_freq_start + _cache_num_freq - 1,
                         _cache_stokes_start, _cache_stokes_start + _cache_num_stokes - 1);
            
            // Check if requested range is within cached range
            bool freq_in_range = (request_freq_start >= _cache_freq_start && 
                                 request_freq_start + num_freq <= _cache_freq_start + _cache_num_freq);
            bool stokes_in_range = (request_stokes_start >= _cache_stokes_start && 
                                   request_stokes_start + num_stokes <= _cache_stokes_start + _cache_num_stokes);
            
            if (!freq_in_range || !stokes_in_range) {
                spdlog::warn("4D CACHE MISS: Requested range outside cached range");
                return false;
            }
            
            for (int freq = 0; freq < num_freq; ++freq) {
                for (int stokes = 0; stokes < num_stokes; ++stokes) {
                    for (int y = 0; y < req_height; ++y) {
                        for (int x = 0; x < req_width; ++x) {
                            // Destination index in output buffer
                            size_t dest_idx = ((freq * num_stokes + stokes) * req_height + y) * req_width + x;
                            
                            // Source coordinates in cache
                            int src_x = start_x + x;
                            int src_y = start_y + y;
                            
                            // Skip if out of bounds
                            if (src_x < 0 || src_x >= _cache_width || src_y < 0 || src_y >= _cache_height) {
                                dest_data[dest_idx] = std::numeric_limits<float>::quiet_NaN();
                                continue;
                            }
                            
                            // Calculate cache-relative frequency and stokes indices
                            int cache_freq_idx = (request_freq_start + freq) - _cache_freq_start;
                            int cache_stokes_idx = (request_stokes_start + stokes) - _cache_stokes_start;
                            
                            // Cache index: cache[width][height][freq][stokes] with offsets
                            // Layout: width * height * freq * stokes in row-major order
                            size_t cache_idx = (((cache_freq_idx * _cache_num_stokes + cache_stokes_idx) * _cache_height + src_y) * _cache_width + src_x);
                            
                            // Boundary check
                            if (cache_idx >= _channel_cache->size()) {
                                spdlog::error("4D BOUNDARY VIOLATION: cache_idx {} >= cache_size {}, coords=({},{}) cache_freq={} cache_stokes={}", 
                                             cache_idx, _channel_cache->size(), src_x, src_y, cache_freq_idx, cache_stokes_idx);
                                dest_data[dest_idx] = std::numeric_limits<float>::quiet_NaN();
                                continue;
                            }
                            
                            // Copy data directly
                            dest_data[dest_idx] = (*_channel_cache)[cache_idx];
                        }
                    }
                }
            }
            
            spdlog::debug("4D CACHE ACCESS: Successfully copied {}x{}x{}x{} data", 
                         req_width, req_height, num_freq, num_stokes);
            
        } else {
            // Original 2D processing for single channel data
            spdlog::debug("2D CACHE ACCESS: Single channel processing");
            
            // Do NOT pre-initialize with NaN - let each pixel be calculated properly
            // Only set NaN for pixels that truly have no valid data
            
            // Process each row with downsampling
            for (int y = 0; y < req_height; ++y) {
                int src_y = start_y + y;
                
                // For each output column, calculate average from corresponding input pixels
                for (int out_x = 0; out_x < req_width; ++out_x) {
                    size_t dest_idx = y * req_width + out_x;
                    
                    // Initialize this pixel as NaN, will be overwritten if valid data is found
                    dest_data[dest_idx] = std::numeric_limits<float>::quiet_NaN();
                    
                    // Skip if source row is out of bounds
                    if (src_y < 0 || src_y >= _cache_height) {
                        continue;
                    }
                    
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
                        if (cache_idx >= _channel_cache->size()) {
                            spdlog::error("BOUNDARY VIOLATION: cache_idx {} >= cache_size {}, src_x={}, src_y={}, cache_width={}, cache_height={}", 
                                         cache_idx, _channel_cache->size(), src_x, src_y, _cache_width, _cache_height);
                            continue;
                        }
                        
                        float val = (*_channel_cache)[cache_idx];
                        if (!std::isnan(val) && std::isfinite(val)) {
                            sum += val * weight;
                            weight_sum += weight;
                            valid_pixels++;
                        }
                    }
                    
                    // Set output value - dest_idx already calculated and validated above
                    if (weight_sum > 0.0f && valid_pixels > 0) {
                        dest_data[dest_idx] = sum / weight_sum;
                    }
                    // Note: dest_data[dest_idx] already initialized to NaN if no valid data found
                }
            }
        } // End of 2D vs 4D processing
        
        // Debug: Check extracted data with detailed sampling
        if (req_width > 0 && req_height > 0) {
            size_t valid_count = 0;
            size_t nan_count = 0;
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            
            // Calculate statistics in a single pass (optimized - removed unused rms/stddev)
            for (int i = 0; i < req_width * req_height; ++i) {
                float val = dest_data[i];
                if (std::isnan(val)) {
                    nan_count++;
                } else if (std::isfinite(val)) {
                    valid_count++;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
            
            // spdlog::debug("EXTRACTED DATA SUMMARY:");
            // spdlog::debug("  Total pixels: {}", req_width * req_height);
            // spdlog::debug("  Valid pixels: {}", valid_count);
            // spdlog::debug("  NaN pixels: {}", nan_count);
            // if (valid_count > 0) {
            //     spdlog::debug("  Value range: [{}, {}]", min_val, max_val);
                
            //     // Enhanced statistics output for CARTA Statistics Widget
            //     spdlog::info("ZARR STATISTICS CALCULATION RESULTS:");
            //     spdlog::info("  Sum (Flux Density): {:.6e}", sum);
            //     spdlog::info("  Mean: {:.6e}", mean);
            //     spdlog::info("  Min: {:.6e}", min_val);
            //     spdlog::info("  Max: {:.6e}", max_val);
            //     spdlog::info("  RMS: {:.6e}", rms);
            //     spdlog::info("  Std Dev: {:.6e}", stddev);
            //     spdlog::info("  Valid Pixels: {}", valid_count);
            //     spdlog::info("  Total Pixels: {}", req_width * req_height);
            //     spdlog::info("  Data Coverage: {:.1f}%", (100.0 * valid_count) / (req_width * req_height));
            // } else {
            //     spdlog::error("NO VALID DATA EXTRACTED - ALL PIXELS ARE NaN!");
            // }
            
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
            _channel_cache->resize(total_elements);
            
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            std::copy(src_data, src_data + total_elements, _channel_cache->data());
            
            // Store the cached region information
            _cached_channel = freq_channel * 1000 + stokes_channel;
            _cache_width = width;
            _cache_height = height;
            _cache_start_x = start_x;
            _cache_start_y = start_y;
            spdlog::debug("_cache_start_x={}, _cache_start_y={}, start_x={}, start_y={}", _cache_start_x, _cache_start_y, start_x, start_y);
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

bool CartaZarrImage::load4DRegionCache(int start_x, int start_y, int width, int height, int num_freq, int num_stokes, int freq_start, int stokes_start) {
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for 4D region cache loading");
        return false;
    }
    
    // Safety checks for region dimensions
    if (width <= 0 || height <= 0 || num_freq <= 0 || num_stokes <= 0) {
        spdlog::error("Invalid 4D region dimensions: {}x{}x{}x{} at ({},{}) freq_start={} stokes_start={}", 
                     width, height, num_freq, num_stokes, start_x, start_y, freq_start, stokes_start);
        return false;
    }
    
    try {
        // For 5D ZARR, load 4D region [time=0, freq=freq_start:freq_start+num_freq, pol=stokes_start:stokes_start+num_stokes, l=start_x:end_x, m=start_y:end_y]
        if (_original_zarr_shape.size() == 5) {
            // Clamp region to valid bounds
            int max_width = _original_zarr_shape[3];   // l dimension (width)
            int max_height = _original_zarr_shape[4];  // m dimension (height)
            int max_freq = _original_zarr_shape[1];    // freq dimension
            int max_stokes = _original_zarr_shape[2];  // stokes dimension
            
            start_x = std::max(0, std::min(start_x, max_width - 1));
            start_y = std::max(0, std::min(start_y, max_height - 1));
            width = std::min(width, max_width - start_x);
            height = std::min(height, max_height - start_y);
            
            // Clamp frequency and stokes ranges
            freq_start = std::max(0, std::min(freq_start, max_freq - 1));
            stokes_start = std::max(0, std::min(stokes_start, max_stokes - 1));
            num_freq = std::min(num_freq, max_freq - freq_start);
            num_stokes = std::min(num_stokes, max_stokes - stokes_start);
            
            if (width <= 0 || height <= 0 || num_freq <= 0 || num_stokes <= 0) {
                spdlog::warn("Invalid clamped 4D dimensions: {}x{}x{}x{} at ({},{}) freq={}:{} stokes={}:{}", 
                            width, height, num_freq, num_stokes, start_x, start_y, 
                            freq_start, freq_start + num_freq - 1, stokes_start, stokes_start + num_stokes - 1);
                return false;
            }
            
            spdlog::info("Loading 4D region cache: {}x{}x{}x{} region at ({},{}) from freq={}:{} stokes={}:{} of {}x{}x{}x{}", 
                        width, height, num_freq, num_stokes, start_x, start_y,
                        freq_start, freq_start + num_freq - 1, stokes_start, stokes_start + num_stokes - 1,
                        max_width, max_height, max_freq, max_stokes);
            
            // Create box for 4D region with frequency/stokes offsets
            std::vector<tensorstore::Index> box_origin = {0, freq_start, stokes_start, start_x, start_y};
            std::vector<tensorstore::Index> box_shape = {1, num_freq, num_stokes, width, height};
            
            spdlog::debug("4D REGION CACHE LOADING COORDINATES:");
            spdlog::debug("  ZARR 5D region box: origin=[{},{},{},{},{}], shape=[{},{},{},{},{}]", 
                         box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4],
                         box_shape[0], box_shape[1], box_shape[2], box_shape[3], box_shape[4]);
            
            tensorstore::Box<> cache_box(box_origin, box_shape);
            
            // Apply the box slice to the TensorStore
            auto constrained_store = _tensorstore | tensorstore::AllDims().BoxSlice(cache_box);
            if (!constrained_store.ok()) {
                spdlog::error("Failed to create 4D region cache slice: {}", constrained_store.status().ToString());
                return false;
            }
            
            // Read 4D region data
            auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
            if (!read_result.ok()) {
                spdlog::error("Failed to read 4D region cache: {}", read_result.status().ToString());
                return false;
            }
            
            auto zarr_array = std::move(read_result.value());
            
            // Verify data type
            if (zarr_array.dtype().name() != "float32") {
                spdlog::error("Unexpected data type in 4D region cache: {}", zarr_array.dtype().name());
                return false;
            }
            
            // Copy data to cache - layout: [width, height, freq, stokes] (row-major)
            size_t total_elements = width * height * num_freq * num_stokes;
            _channel_cache->resize(total_elements);
            
            const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
            std::copy(src_data, src_data + total_elements, _channel_cache->data());
            
            // Store the cached region information for 4D data
            _cached_channel = -1;  // Special marker for 4D cache
            _cache_width = width;
            _cache_height = height;
            _cache_start_x = start_x;
            _cache_start_y = start_y;
            _cache_num_freq = num_freq;
            _cache_num_stokes = num_stokes;
            _cache_freq_start = freq_start;     // Store frequency offset
            _cache_stokes_start = stokes_start; // Store stokes offset
            _channel_cache_loaded = true;
            _is_full_channel_cache = false;  // This is a 4D region cache
            
            spdlog::info("Loaded 4D region cache: {}x{}x{}x{} elements at ({},{}) freq={}:{} stokes={}:{} ({} MB)", 
                        width, height, num_freq, num_stokes, start_x, start_y,
                        freq_start, freq_start + num_freq - 1, stokes_start, stokes_start + num_stokes - 1,
                        (total_elements * sizeof(float)) / (1024 * 1024));
            
            return true;
        }
        
        spdlog::warn("4D region cache only supported for 5D ZARR files");
        return false;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in load4DRegionCache: {}", e.what());
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
        
        spdlog::info("Data copy: Copying {} elements to output buffer", total_elements);
        
        const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
        std::copy(src_data, src_data + total_elements, buffer.data());
        
        spdlog::info("DIRECT READ COMPLETE: Successfully read {} elements", total_elements);
        // spdlog::debug("DIRECT READ: Successfully read {} elements (shape={})", total_elements, length.toString());
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in readDirectFromTensorStore: {}", e.what());
        return false;
    }
}

Bool CartaZarrImage::readPixelFromTensorStore(Array<float>& buffer, const Slicer& section) {
    spdlog::info("DIRECT READ PATH: readPixelFromTensorStore - Bypassing cache");
    try {
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        
        spdlog::info("   Input slicer: start={}, length={}", start.toString(), length.toString());
        
        // CRITICAL FIX: Handle coordinate mapping based on actual section dimensions
        // The issue is that ZarrLoader is passing coordinates in ZARR order already!
        // We need to map the incoming slicer coordinates correctly
        
        // spdlog::debug("readPixelFromTensorStore: section start={}, length={}, shape={}", 
        //              start.toString(), length.toString(), _original_zarr_shape.toString());
        
        std::vector<tensorstore::Index> box_origin;
        std::vector<tensorstore::Index> box_shape;
        
        if (start.size() == 5) {
            // 5D case: ZarrLoader provides ZARR order [time, freq, stokes, x, y]
            // No coordinate transformation needed, use as-is
            box_origin.resize(5);
            box_shape.resize(5);
            for (int i = 0; i < 5; ++i) {
                box_origin[i] = start[i];
                box_shape[i] = length[i];
            }
            // spdlog::debug("5D coordinate mapping: Direct ZARR order [time={}, freq={}, stokes={}, x={}, y={}]", 
            //              box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4]);
        } else if (start.size() == 4) {
            // spdlog::debug("start={}, length={}", start.toString(), length.toString());
            // 4D case: CARTA internal format [x, y, freq, stokes]
            // Map CARTA [x,y,freq,stokes] -> ZARR [time=0, freq, stokes, x, y]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;                 // time = 0
            box_origin[1] = start[2];          // freq (from CARTA position 2)
            box_origin[2] = start[3];          // stokes (from CARTA position 3)
            box_origin[3] = start[0];          // x (from CARTA position 0)
            box_origin[4] = start[1];          // y (from CARTA position 1)
            box_shape[0] = 1;                  // time = 1
            box_shape[1] = length[2];          // freq length
            box_shape[2] = length[3];          // stokes length
            box_shape[3] = length[0];          // x length
            box_shape[4] = length[1];          // y length
            
            // spdlog::debug("4D coordinate mapping: CARTA[x={},y={},freq={},stokes={}] -> ZARR[time={},freq={},stokes={},x={},y={}]", 
            //              start[0], start[1], start[2], start[3],
            //              box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4]);
        } else if (start.size() == 3) {
            // 3D case: incoming CARTA [x,y,freq] to match ZarrLoader pattern
            // Map CARTA [x,y,freq] -> ZARR [time=0, freq, stokes=0, x, y]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;                 // time = 0
            box_origin[1] = start[2];          // freq (from CARTA position 2)
            box_origin[2] = 0;                 // stokes = 0
            box_origin[3] = start[0];          // x (from CARTA position 0)
            box_origin[4] = start[1];          // y (from CARTA position 1)
            box_shape[0] = 1;                  // time = 1
            box_shape[1] = length[2];          // freq length
            box_shape[2] = 1;                  // stokes = 1
            box_shape[3] = length[0];          // x length
            box_shape[4] = length[1];          // y length
        } else if (start.size() == 2) {
            // 2D case: incoming CARTA [x,y] 
            // Map CARTA [x,y] -> ZARR [time=0, freq=0, stokes=0, x, y]
            box_origin.resize(5);
            box_shape.resize(5);
            box_origin[0] = 0;                 // time = 0
            box_origin[1] = 0;                 // freq = 0
            box_origin[2] = 0;                 // stokes = 0
            box_origin[3] = start[0];          // x (from CARTA position 0)
            box_origin[4] = start[1];          // y (from CARTA position 1)
            box_shape[0] = 1;                  // time = 1
            box_shape[1] = 1;                  // freq = 1
            box_shape[2] = 1;                  // stokes = 1
            box_shape[3] = length[0];          // y length
            box_shape[4] = length[1];          // x length
        } else {
            spdlog::error("readPixelFromTensorStore: Unsupported section dimensions: {}", start.size());
            return false;
        }
        // spdlog::debug("dim = {}", start.size());
        
        spdlog::info("Coordinate mapping: ZARR box origin=[{},{},{},{},{}], shape=[{},{},{},{},{}]",
                    box_origin[0], box_origin[1], box_origin[2], box_origin[3], box_origin[4],
                    box_shape[0], box_shape[1], box_shape[2], box_shape[3], box_shape[4]);
        
        // Validate coordinates against ZARR bounds
        for (size_t i = 0; i < box_origin.size(); ++i) {
            if (box_origin[i] < 0 || 
                (i < _original_zarr_shape.size() && box_origin[i] + box_shape[i] > _original_zarr_shape[i])) {
                spdlog::error("readPixelFromTensorStore: Coordinate {} out of bounds: origin={}, shape={}, max={} (ZARR dim names: [time,freq,stokes,x,y])", 
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
        
        // spdlog::debug("DIRECT PIXEL READ: TensorStore slice [time={}, freq={}, stokes={}, x={}:{}, y={}:{}]",
        //              box_origin[0], box_origin[1], box_origin[2], 
        //              box_origin[3], box_origin[3] + box_shape[3] - 1,
        //              box_origin[4], box_origin[4] + box_shape[4] - 1);
        
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

casacore::uInt CartaZarrImage::advisedMaxPixels() const {
    // Return the total number of pixels in the image for memory management
    size_t total_pixels = 1;
    for (int i = 0; i < _shape.size(); ++i) {
        total_pixels *= _shape[i];
    }
    return static_cast<casacore::uInt>(std::min(total_pixels, static_cast<size_t>(INT_MAX)));
}

casacore::IPosition CartaZarrImage::doNiceCursorShape(casacore::uInt maxPixels) const {
    // Return a reasonable cursor shape for processing
    casacore::IPosition cursor_shape = _shape;
    
    // For large images, try to limit to maxPixels
    size_t total_pixels = cursor_shape.product();
    if (total_pixels > maxPixels && maxPixels > 0) {
        // For now, just return the full shape
        // In the future, we could implement more sophisticated cursor shaping
        cursor_shape = casacore::IPosition(_shape.size(), 1);
        cursor_shape[0] = std::min(static_cast<casacore::Int>(maxPixels), static_cast<casacore::Int>(_shape[0]));
        cursor_shape[1] = std::min(static_cast<casacore::Int>(maxPixels / cursor_shape[0]), static_cast<casacore::Int>(_shape[1]));
    }
    
    return cursor_shape;
}

casacore::MFrequency::Types CartaZarrImage::GetFrequencyType() {
    // Determine frequency reference frame from ZARR frequency axis .zattrs
    casacore::MFrequency::Types freq_type(casacore::MFrequency::TOPO); // Default to TOPO
    
    try {
        std::filesystem::path zarr_path(_name.c_str());
        
        // First, try to read from frequency axis .zattrs (correct location)
        std::filesystem::path frequency_zattrs = zarr_path / "frequency" / ".zattrs";
        if (std::filesystem::exists(frequency_zattrs)) {
            std::ifstream freq_zattrs_file(frequency_zattrs);
            nlohmann::json freq_zattrs_json;
            freq_zattrs_file >> freq_zattrs_json;
            
            spdlog::debug("ZARR FREQ: Found frequency/.zattrs, checking for observer field");
            
            // Look for reference_value.attrs.observer (the correct location)
            const std::vector<std::string> possible_keys = {"reference_value", "reference_frequency"};
            for(const auto& key : possible_keys) {
                if (freq_zattrs_json.contains(key) && 
                    freq_zattrs_json[key].contains("attrs") &&
                    freq_zattrs_json[key]["attrs"].contains("observer")) {
                    
                    std::string observer = freq_zattrs_json[key]["attrs"]["observer"].get<std::string>();
                    spdlog::info("ZARR FREQ: Found observer in frequency/.zattrs under '{}': '{}'", key, observer);
                    freq_type = ParseFrequencyFrame(observer);
                    return freq_type;
                }
            }
            
            // Also check for direct observer field (alternative location)
            if (freq_zattrs_json.contains("observer")) {
                std::string observer = freq_zattrs_json["observer"].get<std::string>();
                spdlog::info("ZARR FREQ: Found direct observer in frequency/.zattrs: '{}'", observer);
                freq_type = ParseFrequencyFrame(observer);
                return freq_type;
            }
            
            spdlog::debug("ZARR FREQ: No observer field found in frequency/.zattrs");
        }
        
        spdlog::warn("ZARR FREQ: No spectral reference frame found in frequency/.zattrs, using default TOPO");
        
    } catch (const std::exception& e) {
        spdlog::warn("ZARR FREQ: Exception while reading frequency reference frame: {}, using default TOPO", e.what());
    }
    
    return freq_type;
}

casacore::MFrequency::Types CartaZarrImage::ParseFrequencyFrame(const std::string& frame_str) {
    // Parse frequency reference frame string (similar to CartaFitsImage::GetFrequencyType)
    std::string frame_upper = frame_str;
    std::transform(frame_upper.begin(), frame_upper.end(), frame_upper.begin(), ::toupper);
    
    // Map common frequency reference frame names to casacore types
    std::unordered_map<std::string, casacore::MFrequency::Types> frame_types = {
        {"TOPOCENT", casacore::MFrequency::TOPO},
        {"TOPO", casacore::MFrequency::TOPO},
        {"TOPOCENTRIC", casacore::MFrequency::TOPO},
        {"GEOCENTR", casacore::MFrequency::GEO},
        {"GEO", casacore::MFrequency::GEO},
        {"GEOCENTRIC", casacore::MFrequency::GEO},
        {"BARYCENT", casacore::MFrequency::BARY},
        {"BARY", casacore::MFrequency::BARY},
        {"BARYCENTRIC", casacore::MFrequency::BARY},
        {"HELIOCEN", casacore::MFrequency::BARY},  // Map to BARY as in FITS
        {"HELIOCENTRIC", casacore::MFrequency::BARY},
        {"LSRK", casacore::MFrequency::LSRK},
        {"LSRD", casacore::MFrequency::LSRD},
        {"GALACTOC", casacore::MFrequency::GALACTO},
        {"GALACTIC", casacore::MFrequency::GALACTO},
        {"LOCALGRP", casacore::MFrequency::LGROUP},
        {"LOCAL_GROUP", casacore::MFrequency::LGROUP},
        {"CMBDIPOL", casacore::MFrequency::CMB},
        {"CMB", casacore::MFrequency::CMB},
        {"SOURCE", casacore::MFrequency::REST},
        {"REST", casacore::MFrequency::REST}
    };
    
    if (frame_types.count(frame_upper)) {
        casacore::MFrequency::Types result = frame_types[frame_upper];
        if (frame_upper == "HELIOCEN" || frame_upper == "HELIOCENTRIC") {
            spdlog::debug("ZARR FREQ: HELIOCENTRIC reference frame unsupported, using BARYCENTRIC instead.");
        }
        
        // Map casacore internal names to FITS standard names for logging
        std::string fits_name;
        switch (result) {
            case casacore::MFrequency::TOPO: fits_name = "TOPOCENT"; break;
            case casacore::MFrequency::GEO: fits_name = "GEOCENTR"; break;
            case casacore::MFrequency::BARY: fits_name = "BARYCENT"; break;
            case casacore::MFrequency::LSRK: fits_name = "LSRK"; break;
            case casacore::MFrequency::LSRD: fits_name = "LSRD"; break;
            case casacore::MFrequency::GALACTO: fits_name = "GALACTOC"; break;
            case casacore::MFrequency::LGROUP: fits_name = "LOCALGRP"; break;
            case casacore::MFrequency::CMB: fits_name = "CMBDIPOL"; break;
            case casacore::MFrequency::REST: fits_name = "SOURCE"; break;
            default: fits_name = casacore::MFrequency::showType(result); break;
        }
        
        spdlog::info("ZARR FREQ: Mapped '{}' to {}", frame_str, fits_name);
        return result;
    }
    
    spdlog::warn("ZARR FREQ: Unknown frequency reference frame '{}', using default TOPO", frame_str);
    return casacore::MFrequency::TOPO;
}

casacore::MFrequency::Types CartaZarrImage::ParseFrequencyFrameCode(int frame_code) {
    // Parse frequency reference frame from integer code (common in some ZARR formats)
    // These codes may be format-specific, but here are some common mappings
    
    std::unordered_map<int, casacore::MFrequency::Types> frame_codes = {
        {0, casacore::MFrequency::TOPO},      // Topocentric
        {1, casacore::MFrequency::GEO},       // Geocentric  
        {2, casacore::MFrequency::BARY},      // Barycentric
        {3, casacore::MFrequency::LSRK},      // LSR Kinematic
        {4, casacore::MFrequency::LSRD},      // LSR Dynamic
        {5, casacore::MFrequency::GALACTO},   // Galactocentric
        {6, casacore::MFrequency::LGROUP},    // Local Group
        {7, casacore::MFrequency::CMB},       // CMB
        {8, casacore::MFrequency::REST},      // Rest frame
        // Add more mappings as needed for specific ZARR formats
    };
    
    if (frame_codes.count(frame_code)) {
        casacore::MFrequency::Types result = frame_codes[frame_code];
        spdlog::info("ZARR FREQ: Mapped frame code {} to casacore::MFrequency::{}", 
                     frame_code, casacore::MFrequency::showType(result));
        return result;
    }
    
    spdlog::warn("ZARR FREQ: Unknown frequency reference frame code {}, using default TOPO", frame_code);
    return casacore::MFrequency::TOPO;
}

casacore::MDirection::Types CartaZarrImage::GetDirectionType() {
    // Get direction reference system from ZARR metadata
    // Reads "frame" field from direction.reference.attrs or SKY/.zattrs pointing_center.attrs (e.g., "icrs", "fk5", "fk4")
    casacore::MDirection::Types dir_type(casacore::MDirection::J2000); // Default to J2000
    std::string frame;
    bool frame_found = false;
    
    try {
        std::filesystem::path zarr_path(_name.c_str());
        
        // First, try to read from SKY/.zattrs (ASKAP ZARR format)
        std::filesystem::path sky_zattrs_path = zarr_path / "SKY" / ".zattrs";
        if (std::filesystem::exists(sky_zattrs_path)) {
            std::ifstream sky_zattrs_file(sky_zattrs_path);
            nlohmann::json sky_zattrs_json;
            sky_zattrs_file >> sky_zattrs_json;
            
            // Look for frame in pointing_center.attrs.frame
            if (sky_zattrs_json.contains("pointing_center") &&
                sky_zattrs_json["pointing_center"].contains("attrs") &&
                sky_zattrs_json["pointing_center"]["attrs"].contains("frame")) {
                
                frame = sky_zattrs_json["pointing_center"]["attrs"]["frame"].get<std::string>();
                frame_found = true;
                spdlog::debug("ZARR WCS: Found frame '{}' in SKY/.zattrs", frame);
            }
        }
        
        // If not found in SKY/.zattrs, try main .zattrs
        if (!frame_found) {
            std::filesystem::path zattrs_path = zarr_path / ".zattrs";
            if (std::filesystem::exists(zattrs_path)) {
                std::ifstream zattrs_file(zattrs_path);
                nlohmann::json zattrs_json;
                zattrs_file >> zattrs_json;
                
                // Look for frame in direction.reference.attrs
                if (zattrs_json.contains("direction") && 
                    zattrs_json["direction"].contains("reference") &&
                    zattrs_json["direction"]["reference"].contains("attrs") &&
                    zattrs_json["direction"]["reference"]["attrs"].contains("frame")) {
                    
                    frame = zattrs_json["direction"]["reference"]["attrs"]["frame"].get<std::string>();
                    frame_found = true;
                    spdlog::debug("ZARR WCS: Found frame '{}' in .zattrs", frame);
                }
            }
        }
        
        // Parse the frame string if found
        if (frame_found) {
            // Convert to uppercase for case-insensitive comparison
            std::transform(frame.begin(), frame.end(), frame.begin(), ::toupper);
            
            if (frame == "ICRS") {
                dir_type = casacore::MDirection::ICRS;
                spdlog::info("ZARR WCS: Using ICRS direction reference system");
            } else if (frame == "FK5" || frame == "J2000") {
                dir_type = casacore::MDirection::J2000;
                spdlog::info("ZARR WCS: Using FK5/J2000 direction reference system");
            } else if (frame == "FK4" || frame == "B1950") {
                dir_type = casacore::MDirection::B1950;
                spdlog::info("ZARR WCS: Using FK4/B1950 direction reference system");
            } else if (frame == "GALACTIC") {
                dir_type = casacore::MDirection::GALACTIC;
                spdlog::info("ZARR WCS: Using GALACTIC direction reference system");
            } else {
                spdlog::warn("ZARR WCS: Unknown frame '{}', using default J2000", frame);
            }
        } else {
            spdlog::debug("ZARR WCS: No direction frame found in metadata, using default J2000");
        }
    } catch (const std::exception& e) {
        spdlog::warn("ZARR WCS: Exception reading direction reference system: {}, using default J2000", e.what());
    }
    
    return dir_type;
}

casacore::Projection CartaZarrImage::GetProjectionType() {
    // Get projection type from ZARR metadata
    // Reads "projection" field from direction (e.g., "SIN", "CAR", "TAN")
    casacore::Projection projection(casacore::Projection::CAR); // Default to CAR
    
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        if (std::filesystem::exists(zattrs_path)) {
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
            // Look for projection in direction
            if (zattrs_json.contains("direction") && 
                zattrs_json["direction"].contains("projection")) {
                
                std::string proj_str = zattrs_json["direction"]["projection"].get<std::string>();
                
                // Convert to uppercase for case-insensitive comparison
                std::transform(proj_str.begin(), proj_str.end(), proj_str.begin(), ::toupper);
                
                try {
                    // Use casacore's Projection::type() to parse projection string
                    casacore::Projection::Type proj_type = casacore::Projection::type(proj_str);
                    projection = casacore::Projection(proj_type);
                    spdlog::info("ZARR WCS: Using {} projection from metadata", proj_str);
                    
                    // Read projection parameters if available
                    if (zattrs_json["direction"].contains("projection_parameters")) {
                        auto proj_params = zattrs_json["direction"]["projection_parameters"];
                        if (proj_params.is_array() && proj_params.size() >= 2) {
                            casacore::Vector<double> params(2);
                            params(0) = proj_params[0].get<double>();
                            params(1) = proj_params[1].get<double>();
                            projection = casacore::Projection(proj_type, params);
                            spdlog::debug("ZARR WCS: Projection parameters: [{}, {}]", params(0), params(1));
                        }
                    }
                } catch (const casacore::AipsError& e) {
                    spdlog::warn("ZARR WCS: Unknown projection '{}', using default CAR: {}", proj_str, e.getMesg());
                }
            } else {
                spdlog::debug("ZARR WCS: No projection found in metadata, using default CAR");
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("ZARR WCS: Exception reading projection: {}, using default CAR", e.what());
    }
    
    return projection;
}

void CartaZarrImage::setupImageInfo() {
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        if (std::filesystem::exists(zattrs_path)) {
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
            // Try to read beam information from BEAM zarr array if present
            try {
                std::filesystem::path beam_array = zarr_path / "BEAM";
                std::filesystem::path beam_zarray = beam_array / ".zarray";
                if (std::filesystem::exists(beam_zarray)) {
                    spdlog::info("setupImageInfo: Found BEAM array for Zarr image: {}", beam_array.string());

                    // Build TensorStore spec for BEAM array
                    nlohmann::json beam_spec_json = {
                        {"driver", "zarr2"},
                        {"kvstore", { {"driver", "file"}, {"path", beam_array.string()} }}
                    };

                    auto beam_spec_res = tensorstore::Spec::FromJson(beam_spec_json);
                    if (!beam_spec_res.ok()) {
                        spdlog::warn("setupImageInfo: Failed to create TensorStore spec for BEAM: {}", beam_spec_res.status().ToString());
                        return;
                    }

                    auto beam_spec = beam_spec_res.value();
                    auto open_res = tensorstore::Open(beam_spec, _context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read).result();
                    if (!open_res.ok()) {
                        spdlog::warn("setupImageInfo: Failed to open BEAM TensorStore: {}", open_res.status().ToString());
                        return;
                    }

                    auto beam_store = open_res.value();

                    // Read beam data from specific chunk 0.0.0.0 (time=0, freq=0, pol=0, all beam params)
                    auto domain = beam_store.domain();
                    auto shape = domain.shape();
                    
                    // Convert shape to printable format
                    std::string shape_str = "[";
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (i > 0) shape_str += ", ";
                        shape_str += std::to_string(shape[i]);
                    }
                    shape_str += "]";
                    spdlog::debug("setupImageInfo: BEAM array shape: {}", shape_str);

                    if (shape.size() >= 4 && shape[3] >= 3) {
                        // Read the first beam entry [0, 0, 0, :] which contains [bmaj, bmin, bpa]
                        std::vector<tensorstore::Index> start(shape.size(), 0);
                        std::vector<tensorstore::Index> lengths(shape.size(), 1);
                        lengths[3] = 3; // Read first 3 beam parameters

                        auto read_res = tensorstore::Read(beam_store | tensorstore::AllDims().SizedInterval(start, lengths)).result();
                        if (!read_res.ok()) {
                            spdlog::warn("setupImageInfo: Failed to read BEAM chunk: {}", read_res.status().ToString());
                            return;
                        }

                        auto beam_data = std::move(read_res.value());
                        
                        // Extract beam parameters from the read data
                        if (beam_data.num_elements() >= 3) {
                            const double* data_ptr = reinterpret_cast<const double*>(beam_data.data());
                            if (data_ptr) {
                                // Read bmaj, bmin, bpa from the first beam entry
                                double major = data_ptr[0];
                                double minor = data_ptr[1];
                                double pa = data_ptr[2];
                                
                                spdlog::debug("setupImageInfo: BEAM raw values: bmaj={}, bmin={}, bpa={}", major, minor, pa);

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
                                        spdlog::debug("setupImageInfo: Failed to parse BEAM/.zattrs units: {}", e.what());
                                    }
                                }

                                try {
                                    casacore::Quantity qmajor(major, beam_unit);
                                    casacore::Quantity qminor(minor, beam_unit);
                                    casacore::Quantity qpa(pa, beam_unit);

                                    // Get current image info and set restoring beam
                                    casacore::ImageInfo ii = imageInfo();
                                    ii.setRestoringBeam(qmajor, qminor, qpa);
                                    setImageInfo(ii);
                                    
                                    spdlog::info("setupImageInfo: Successfully set restoring beam from BEAM array: major={} {}, minor={} {}, pa={} {}",
                                                qmajor.getValue(), qmajor.getUnit(), qminor.getValue(), qminor.getUnit(), qpa.getValue(), qpa.getUnit());
                                } catch (const std::exception& e) {
                                    spdlog::warn("setupImageInfo: Failed to set restoring beam from BEAM array: {}", e.what());
                                }
                            } else {
                                spdlog::warn("setupImageInfo: BEAM data pointer is null or insufficient data");
                            }
                        } else {
                            spdlog::warn("setupImageInfo: BEAM data has insufficient elements: {}", beam_data.num_elements());
                        }
                    } else {
                        spdlog::warn("setupImageInfo: BEAM array has unexpected shape: {}", shape_str);
                    }
                }
            } catch (const std::exception& e) {
                spdlog::warn("setupImageInfo: Exception while extracting BEAM from Zarr: {}", e.what());
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("setupImageInfo: Exception while reading .zattrs for beam info: {}", e.what());
    }
}

std::string CartaZarrImage::readBrightnessUnit() {
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        if (std::filesystem::exists(zattrs_path)) {
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
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
                        spdlog::debug("readBrightnessUnit: Failed to read SKY/.zattrs: {}", e.what());
                    }
                }
            }

            if (!found_bunit.empty()) {
                casacore::String bunit_str(found_bunit);
                NormalizeUnit(bunit_str);
                if (casacore::UnitVal::check(bunit_str)) {
                    // Set image brightness unit
                    setUnits(casacore::Unit(bunit_str));
                    spdlog::info("readBrightnessUnit: Set image brightness unit from .zattrs: {}", bunit_str);
                    return found_bunit;
                } else {
                    spdlog::warn("readBrightnessUnit: Found brightness unit '{}' in .zattrs but failed to normalize/check", found_bunit);
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("readBrightnessUnit: Exception while parsing .zattrs for brightness unit: {}", e.what());
    }
    
    return "";
}

// Cached versions to avoid repeated loading per file
void CartaZarrImage::readBrightnessUnitIfNeeded() {
    static std::unordered_map<std::string, bool> brightness_unit_loaded;
    static std::unordered_map<std::string, casacore::Unit> cached_brightness_units;
    
    if (!brightness_unit_loaded[_name]) {
        std::string unit_str = readBrightnessUnit();
        if (!unit_str.empty()) {
            cached_brightness_units[_name] = units();
        }
        brightness_unit_loaded[_name] = true;
        spdlog::debug("CartaZarrImage: Loaded brightness unit for file: {}", _name);
    } else {
        // Apply cached brightness unit to this instance
        if (cached_brightness_units.find(_name) != cached_brightness_units.end()) {
            setUnits(cached_brightness_units[_name]);
            spdlog::debug("CartaZarrImage: Applied cached brightness unit for file: {}", _name);
        }
        spdlog::debug("CartaZarrImage: Skipping repeated brightness unit read for file: {}", _name);
    }
}

void CartaZarrImage::setupImageInfoIfNeeded() {
    static std::unordered_map<std::string, bool> image_info_loaded;
    static std::unordered_map<std::string, casacore::ImageInfo> cached_image_info;
    
    if (!image_info_loaded[_name]) {
        setupImageInfo();
        cached_image_info[_name] = imageInfo();
        image_info_loaded[_name] = true;
        spdlog::debug("CartaZarrImage: Loaded image info for file: {}", _name);
    } else {
        // Apply cached image info (including beam) to this instance
        if (cached_image_info.find(_name) != cached_image_info.end()) {
            setImageInfo(cached_image_info[_name]);
            spdlog::debug("CartaZarrImage: Applied cached image info (beam) for file: {}", _name);
        }
        spdlog::debug("CartaZarrImage: Skipping repeated image info setup for file: {}", _name);
    }
}

} // namespace carta