/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"
#include "CartaZarrImage.h"
#include "Logger/Logger.h"
#include "Util/Image.h"
#include "Main/ProgramSettings.h"

#include <filesystem>
#include <memory>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <future>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// xtensor 0.26.0 for optimized array operations (C++17 compatible)
#include <xtensor/containers/xadapt.hpp>   // Wrap raw pointer as xtensor array
#include <xtensor/views/xview.hpp>         // Array slicing and views
#include <xtensor/core/xmath.hpp>          // nansum, nanmean, nanmin, nanmax, nanvar, nanstd, count_nonnan
#include <xtensor/generators/xbuilder.hpp> // Array construction utilities

#define XUSE_XSIMD  // Enable SIMD optimizations in xtensor

using namespace carta;

ZarrLoader::ZarrLoader(const std::string& filename) : FileLoader(filename) {
    _num_dims = 0;
    _has_pixel_mask = false;
    
    spdlog::info("ZarrLoader created for: {}", _filename);
}

void ZarrLoader::AllocateImage(const std::string& hdu) {
    try {
        _image = std::shared_ptr<casacore::ImageInterface<float>>(new CartaZarrImage(_filename));
        
        if (_image) {
            casacore::IPosition shape = _image->shape();
            _num_dims = shape.size();
            _has_pixel_mask = _image->hasPixelMask();
            
            // CRITICAL: Initialize coordinate system from the image
            _coord_sys = std::shared_ptr<casacore::CoordinateSystem>(
                static_cast<casacore::CoordinateSystem*>(_image->coordinates().clone()));
            
            // Set image shape for coordinate axis finding
            _image_shape = shape;
            
            spdlog::info("Created CartaZarrImage: {} dims={}, has_mask={}", _filename, _num_dims, _has_pixel_mask);
        }
    } catch (std::exception& e) {
        spdlog::error("Failed to create CartaZarrImage for {}: {}", _filename, e.what());
        _image = nullptr;
        _num_dims = 0;
        _has_pixel_mask = false;
        _coord_sys = nullptr;
    }
}

bool ZarrLoader::HasZarrArrayMetadata(const std::string& path) const {
    std::filesystem::path zarr_path(path);
    std::filesystem::path zarray_path = zarr_path / ".zarray";
    return std::filesystem::exists(zarray_path) && std::filesystem::is_regular_file(zarray_path);
}

bool ZarrLoader::HasZmetadataFile(const std::string& path) const {
    std::filesystem::path zarr_path(path);
    std::filesystem::path zmeta_path = zarr_path / ".zmetadata";
    return std::filesystem::exists(zmeta_path) && std::filesystem::is_regular_file(zmeta_path);
}

// FileLoader virtual function implementations
bool ZarrLoader::HasData(FileInfo::Data ds) const {
    switch (ds) {
        case FileInfo::Data::Image:
            return _image != nullptr;
        case FileInfo::Data::XY:
            return _image != nullptr && _num_dims >= 2;
        case FileInfo::Data::XYZ:
            return _image != nullptr && _num_dims >= 3;
        case FileInfo::Data::XYZW:
            return _image != nullptr && _num_dims >= 4;
        case FileInfo::Data::MASK:
            return _has_pixel_mask;
        default:
            return FileLoader::HasData(ds);
    }
}

bool ZarrLoader::HasMip(int mip) const {
    return false;
}

bool ZarrLoader::UseTileCache() const {
    return true;
}

bool ZarrLoader::GetCursorSpectralData(std::vector<float>& data, int stokes, int cursor_x, int count_x,
    int cursor_y, int count_y, std::mutex& image_mutex) {
    
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        if (!_image) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: No image available");
            return false;
        }
        
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Image is not a CartaZarrImage");
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        // FIXED: For CARTA internal shape [x, y, freq, stokes] format:
        int img_width = shape[0];     // x dimension  
        int img_height = shape[1];    // y dimension
        int num_channels = shape[2];  // freq dimension (was incorrectly shape[1])
        
        if (cursor_x < 0 || cursor_y < 0 || cursor_x >= img_width || cursor_y >= img_height) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Cursor out of bounds: ({},{}) (image: {}x{})", 
                         cursor_x, cursor_y, img_width, img_height);
            return false;
        }
        
        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            return false;
        }
        
        spdlog::debug("GetCursorSpectralData: Using direct TensorStore read for {} channels at point ({},{}) stokes={}", 
                     num_channels, cursor_x, cursor_y, stokes);
        
        data.resize(num_channels);
        
        // Use direct TensorStore read to avoid cache loading and ensure success
        // Read all channels at once as a single spectral profile
        casacore::IPosition start, length;
        
        if (shape.size() == 5) {
            // 5D ZARR: [time, freq, stokes, x, y] - read all freq channels for single pixel
            start = casacore::IPosition(5, 0, 0, stokes, cursor_x, cursor_y);
            length = casacore::IPosition(5, 1, num_channels, 1, 1, 1);
        } else if (shape.size() == 4) {
            // 4D: CARTA internal format [x, y, freq, stokes] - read all freq channels for single pixel
            start = casacore::IPosition(4, cursor_x, cursor_y, 0, stokes);
            length = casacore::IPosition(4, 1, 1, num_channels, 1);
        } else if (shape.size() == 3) {
            // 3D: [x, y, freq] - read all freq channels for single pixel
            start = casacore::IPosition(3, cursor_x, cursor_y, 0);
            length = casacore::IPosition(3, 1, 1, num_channels);
        } else if (shape.size() == 2) {
            // 2D: [x, y] - single channel only
            start = casacore::IPosition(2, cursor_x, cursor_y);
            length = casacore::IPosition(2, 1, 1);
            data[0] = 0.0f; // Placeholder for 2D data
            spdlog::debug("GetCursorSpectralData: 2D image - returning single placeholder value");
            return true;
        } else {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Unsupported number of dimensions: {}", shape.size());
            return false;
        }
        
        // Use direct TensorStore read via readPixelFromTensorStore to bypass cache entirely
        casacore::Array<float> pixel_array;
        casacore::Slicer section(start, length);
        
        if (!zarr_image->readPixelFromTensorStore(pixel_array, section)) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: readPixelFromTensorStore failed for spectral profile");
            return false;
        }
        
        // Copy array data to output vector
        if (pixel_array.nelements() != num_channels) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Expected {} elements, got {}", 
                         num_channels, pixel_array.nelements());
            return false;
        }
        
        const float* pixel_data = pixel_array.data();
        for (int z = 0; z < num_channels; ++z) {
            data[z] = pixel_data[z];
        }
        
        spdlog::debug("GetCursorSpectralData: Successfully read {} channels one by one for point ({},{})", 
                     num_channels, cursor_x, cursor_y);
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetCursorSpectralData: Exception: {}", e.what());
        return false;
    }
}

bool ZarrLoader::UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& image_mutex) {
    spdlog::info("=== UseRegionSpectralData CALLED ===");
    spdlog::info("Region shape: [{} dimensions] = [{}]", region_shape.size(), 
                 region_shape.size() >= 2 ? fmt::format("{}, {}", region_shape[0], region_shape[1]) : "N/A");
    
    // Always use optimized path for point spectral (cursor) 
    if (region_shape.size() >= 2 && region_shape[0] == 1 && region_shape[1] == 1) {
        spdlog::info("UseRegionSpectralData: RETURNING TRUE for optimized point spectral data (1x1 region)");
        return true;
    }

    if (_image && region_shape.size() >= 2) {
        int region_size = region_shape[0] * region_shape[1];
        
        // Always use batch processing for region spectral data since we use direct read without cache
        spdlog::info("UseRegionSpectralData: RETURNING TRUE for region spectral data ({}x{} region, size: {} pixels) - using multi-channel batch processing", 
                     region_shape[0], region_shape[1], region_size);
        return true;
    }

    spdlog::info("UseRegionSpectralData: RETURNING FALSE - Using default image slicing for {}x{} region", 
                 region_shape[0], region_shape[1]);
    return false;
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& spectral_range, int stokes,
    const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin,
    std::mutex& image_mutex, std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) {
    
    spdlog::info("ZarrLoader::GetRegionSpectralData: BATCH PROCESSING CALLED - region_id={}, spectral_range={}:{}, stokes={}", 
                region_id, spectral_range.from, spectral_range.to, stokes);
    
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        if (!_image) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: No image available");
            progress = 1.0;  // Set progress to avoid infinite while loop in RegionHandler
            return false;
        }
        
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: Image is not a CartaZarrImage");
            progress = 1.0;  // Set progress to avoid infinite while loop in RegionHandler
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        int num_channels = (shape.size() > 2) ? shape[2] : 1;  // FIXED: Use shape[2] for frequency dimension

        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            progress = 1.0;  // Set progress to avoid infinite while loop in RegionHandler
            return false;
        }
        
        casacore::IPosition mask_shape = mask.shape();
        int x_min = std::max(0, static_cast<int>(origin[0]));
        int y_min = std::max(0, static_cast<int>(origin[1]));
        int x_max = std::min(img_width - 1, static_cast<int>(origin[0] + mask_shape[0] - 1));
        int y_max = std::min(img_height - 1, static_cast<int>(origin[1] + mask_shape[1] - 1));
        
        int region_width = x_max - x_min + 1;
        int region_height = y_max - y_min + 1;
        int region_area = region_width * region_height;

        int z_start = spectral_range.from;
        int z_end = spectral_range.to;
        int profile_size = z_end - z_start + 1;

        // Calculate beam area for flux density calculation (like Hdf5Loader)
        double beam_area = CalculateBeamArea();
        bool has_flux = !std::isnan(beam_area);
        
        spdlog::info("GetRegionSpectralData: beam_area={}, has_flux={}", beam_area, has_flux);
        
        // Additional debug: check image info
        auto image = GetImage();
        if (image) {
            auto& info = image->imageInfo();
            if (info.hasSingleBeam()) {
                spdlog::info("GetRegionSpectralData: Image has single beam");
            } else {
                spdlog::warn("GetRegionSpectralData: Image does NOT have single beam!");
            }
        } else {
            spdlog::warn("GetRegionSpectralData: Image pointer is null!");
        }

        // Initialize results vectors for all statistics types (like Hdf5Loader)
        std::vector<double> num_pixels_vec(profile_size, 0);
        std::vector<double> nan_count_vec(profile_size, 0);
        std::vector<double> sum_vec(profile_size, 0.0);
        std::vector<double> mean_vec(profile_size, NAN);
        std::vector<double> rms_vec(profile_size, NAN);
        std::vector<double> sigma_vec(profile_size, NAN);
        std::vector<double> sum_sq_vec(profile_size, 0.0);
        std::vector<double> min_vec(profile_size, std::numeric_limits<float>::max());
        std::vector<double> max_vec(profile_size, std::numeric_limits<float>::lowest());
        std::vector<double> extrema_vec(profile_size, NAN);
        std::vector<double> flux_vec(profile_size, NAN);
        
        // OPTIMIZED THREAD STRATEGY for I/O-bound operations
        // Fewer threads = less I/O contention, better throughput
        unsigned int hardware_cpus = std::thread::hardware_concurrency();
        int num_threads;
        if (hardware_cpus >= 11) {
            num_threads = 6;  // 11-14+ CPUs: use 6 threads
        } else if (hardware_cpus >= 7) {
            num_threads = 4;  // 7-10 CPUs: use 4 threads
        } else if (hardware_cpus >= 4) {
            num_threads = 2;  // 4-6 CPUs: use 2 threads
        } else {
            num_threads = 1;  // 1-3 CPUs: use 1 thread
        }
        
        // CONFIGURABLE: Number of channels per CPU thread (via --cpu_ch flag)
        // Get from ProgramSettings, default is 8 channels per CPU
        const int PARALLEL_THREADS = 4;  // Fixed: use 4 CPUs
        const int CHANNELS_PER_THREAD = carta::ProgramSettings::GetInstance().cpu_ch;
        const int BATCH_SIZE = PARALLEL_THREADS * CHANNELS_PER_THREAD;  // Dynamic batch size
        
        size_t memory_per_batch_mb = (BATCH_SIZE * region_area * sizeof(float)) / (1024 * 1024);
        double memory_per_batch_gb = memory_per_batch_mb / 1024.0;
        
        spdlog::info("GetRegionSpectralData: Processing {} channels, region={}x{} pixels ({} total pixels)", 
                    profile_size, region_width, region_height, region_area);
        spdlog::info("  Thread strategy: {} CPUs detected -> using {} threads for I/O", hardware_cpus, PARALLEL_THREADS);
#ifdef _OPENMP
        int omp_threads = omp_get_max_threads();
        spdlog::info("  OpenMP enabled: {} threads available for parallel statistics computation", omp_threads);
#else
        spdlog::warn("  OpenMP NOT enabled - statistics will run sequentially");
#endif
        spdlog::info("  Batch config: {} threads × {} channels/thread = {} channels/batch ({:.2f} GB/batch)", 
                    PARALLEL_THREADS, CHANNELS_PER_THREAD, BATCH_SIZE, memory_per_batch_gb);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Enable nested OpenMP parallelism for I/O threads + compute parallelism
#ifdef _OPENMP
        omp_set_max_active_levels(2);  // Enable 2-level parallelism: I/O threads + compute threads
        omp_set_num_threads(PARALLEL_THREADS);
        spdlog::info("Using OpenMP with {} I/O threads, nested parallelism enabled for compute", PARALLEL_THREADS);
#endif
        
        // Process channels in batches - each thread reads CHANNELS_PER_THREAD channels at once
        // Use uint8_t instead of bool for thread-safety (vector<bool> is not thread-safe even for different indices)
        std::vector<uint8_t> channel_success(profile_size, 0);
        
        // Process all channels in parallel batches
        for (int batch_start = z_start; batch_start <= z_end; batch_start += BATCH_SIZE) {
            int batch_end = std::min(batch_start + BATCH_SIZE - 1, z_end);
            auto batch_time_start = std::chrono::high_resolution_clock::now();
            
            spdlog::info("Processing batch: channels {}-{} ({} channels total)", 
                        batch_start, batch_end, batch_end - batch_start + 1);
            
#pragma omp parallel for schedule(static)
            for (int thread_idx = 0; thread_idx < PARALLEL_THREADS; ++thread_idx) {
                auto thread_start_time = std::chrono::high_resolution_clock::now();
                
                int thread_start_z = batch_start + thread_idx * CHANNELS_PER_THREAD;
                int thread_end_z = std::min(thread_start_z + CHANNELS_PER_THREAD - 1, batch_end);
                
                if (thread_start_z > batch_end) continue;
                
                int num_channels = thread_end_z - thread_start_z + 1;
                
                // Read CHANNELS_PER_THREAD channels at once (length = num_channels)
                casacore::IPosition start, length;
                if (shape.size() == 5) {
                    start = casacore::IPosition(5, 0, thread_start_z, stokes, x_min, y_min);
                    length = casacore::IPosition(5, 1, num_channels, 1, region_width, region_height);
                } else if (shape.size() == 4) {
                    start = casacore::IPosition(4, x_min, y_min, thread_start_z, stokes);
                    length = casacore::IPosition(4, region_width, region_height, num_channels, 1);
                } else if (shape.size() == 3) {
                    start = casacore::IPosition(3, x_min, y_min, thread_start_z);
                    length = casacore::IPosition(3, region_width, region_height, num_channels);
                } else if (shape.size() == 2) {
                    start = casacore::IPosition(2, x_min, y_min);
                    length = casacore::IPosition(2, region_width, region_height);
                }
                
                // Time the I/O operation
                auto io_start = std::chrono::high_resolution_clock::now();
                casacore::Array<float> batch_array;
                casacore::Slicer slicer(start, length);
                
                if (!zarr_image->readPixelFromTensorStore(batch_array, slicer)) {
                    #pragma omp critical
                    {
                        spdlog::error("  Thread {}: Failed readPixelFromTensorStore for channels {}-{} (start={}, length={})",
                                     thread_idx, thread_start_z, thread_end_z, start.toString(), length.toString());
                    }
                    for (int z = thread_start_z; z <= thread_end_z; ++z) {
                        channel_success[z - z_start] = 0;
                    }
                    continue;
                }
                auto io_end = std::chrono::high_resolution_clock::now();
                auto io_ms = std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start).count();
                
                // Process each channel in the batch with OpenMP parallelization
                auto compute_start = std::chrono::high_resolution_clock::now();
                const float* data_ptr = batch_array.data();
                size_t pixels_per_channel = region_width * region_height;
                
#pragma omp parallel for schedule(dynamic)
                for (int ch_offset = 0; ch_offset < num_channels; ++ch_offset) {
                    int z = thread_start_z + ch_offset;
                    int profile_index = z - z_start;
                    
                    if (profile_index < 0 || profile_index >= profile_size) {
                        #pragma omp critical
                        {
                            spdlog::error("Thread {}: Invalid profile_index {} for channel {} (z_start={}, profile_size={})",
                                         thread_idx, profile_index, z, z_start, profile_size);
                        }
                        continue;
                    }
                    
                    // Get pointer to this channel's data
                    const float* channel_data = data_ptr + (ch_offset * pixels_per_channel);
                    
                    // Collect valid pixels
                    std::vector<float> valid_pixels;
                    valid_pixels.reserve(pixels_per_channel / 2);
                    
                    for (size_t i = 0; i < pixels_per_channel; ++i) {
                        int y = i / region_width;
                        int x = i % region_width;
                        int mask_x = x + x_min - origin[0];
                        int mask_y = y + y_min - origin[1];
                        
                        if (mask_x >= 0 && mask_x < mask_shape[0] && mask_y >= 0 && mask_y < mask_shape[1]) {
                            casacore::IPosition mask_pos(2, mask_x, mask_y);
                            if (mask(mask_pos)) {
                                float value = channel_data[i];
                                if (std::isfinite(value)) {
                                    valid_pixels.push_back(value);
                                }
                            }
                        }
                    }
                    
                    uint64_t valid_count = valid_pixels.size();
                    if (valid_count > 0) {
                        // Use xtensor for SIMD-accelerated statistics
                        std::vector<size_t> shape_1d = {valid_pixels.size()};
                        auto valid_arr = xt::adapt(valid_pixels.data(), valid_pixels.size(), xt::no_ownership(), shape_1d);
                        
                        double sum = xt::sum(valid_arr)();
                        double sum_sq = xt::sum(xt::square(valid_arr))();
                        float min_val = xt::amin(valid_arr)();
                        float max_val = xt::amax(valid_arr)();
                        
                        num_pixels_vec[profile_index] = valid_count;
                        sum_vec[profile_index] = sum;
                        sum_sq_vec[profile_index] = sum_sq;
                        min_vec[profile_index] = min_val;
                        max_vec[profile_index] = max_val;
                    } else {
                        num_pixels_vec[profile_index] = 0;
                        sum_vec[profile_index] = 0.0;
                        sum_sq_vec[profile_index] = 0.0;
                        min_vec[profile_index] = std::numeric_limits<float>::quiet_NaN();
                        max_vec[profile_index] = std::numeric_limits<float>::quiet_NaN();
                    }
                    
                    channel_success[profile_index] = 1;
                }
                
                auto compute_end = std::chrono::high_resolution_clock::now();
                auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
                auto thread_end_time = std::chrono::high_resolution_clock::now();
                auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(thread_end_time - thread_start_time).count();
                
                // Calculate per-thread statistics
                size_t thread_mb = (num_channels * region_width * region_height * sizeof(float)) / (1024*1024);
                double io_speed_gbs = (io_ms > 0) ? (thread_mb / 1024.0) / (io_ms / 1000.0) : 0.0;
                
                #pragma omp critical
                {
                    spdlog::info("  Thread {}: channels {}-{} ({} channels) - I/O {} ms ({:.2f} GB/s), compute {} ms, total {} ms",
                                thread_idx, thread_start_z, thread_end_z, num_channels, 
                                io_ms, io_speed_gbs, compute_ms, total_ms);
                }
            }
            
            auto batch_time_end = std::chrono::high_resolution_clock::now();
            auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_time_end - batch_time_start).count();
            int actual_batch_size = batch_end - batch_start + 1;
            size_t batch_mb = (actual_batch_size * region_width * region_height * sizeof(float)) / (1024*1024);
            double batch_speed_gbs = (batch_mb / 1024.0) / (batch_ms / 1000.0);
            spdlog::info("Batch complete: {} ms for {} channels ({} MB, {:.2f} GB/s)", 
                        batch_ms, actual_batch_size, batch_mb, batch_speed_gbs);
        }
        
        // Check if all channels succeeded
        bool all_success = true;
        for (size_t i = 0; i < channel_success.size(); ++i) {
            if (!channel_success[i]) {
                spdlog::error("ZarrLoader::GetRegionSpectralData: Failed to process channel {} (profile_index={}, channel_success[{}]={})", 
                    z_start + i, i, i, channel_success[i]);
                all_success = false;
            }
        }
        
        // Debug: Show which channels succeeded
        spdlog::debug("Channel success status (z_start={}, total={}): [{}]", 
            z_start, channel_success.size(),
            [&]() {
                std::string status;
                for (size_t i = 0; i < channel_success.size(); ++i) {
                    if (i > 0) status += ", ";
                    status += std::to_string(z_start + i) + ":" + (channel_success[i] ? "OK" : "FAIL");
                }
                return status;
            }()
        );
        
        if (!all_success) {
            progress = 1.0;
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // All channels processed - calculate derived statistics (like Hdf5Loader)
        for (int z = 0; z < profile_size; ++z) {
            uint64_t num_pixels = num_pixels_vec[z];
            if (num_pixels > 0) {
                double sum = sum_vec[z];
                double sum_sq = sum_sq_vec[z];
                
                mean_vec[z] = sum / num_pixels;
                rms_vec[z] = std::sqrt(sum_sq / num_pixels);
                sigma_vec[z] = num_pixels > 1 ? std::sqrt((sum_sq - (sum * sum / num_pixels)) / (num_pixels - 1)) : 0;
                extrema_vec[z] = (std::abs(min_vec[z]) > std::abs(max_vec[z])) ? min_vec[z] : max_vec[z];
                
                // Calculate flux density if beam area is available
                if (has_flux) {
                    flux_vec[z] = sum / beam_area;
                }
            }
        }
        
        // Store all statistics in results map
        results[CARTA::StatsType::NumPixels] = num_pixels_vec;
        results[CARTA::StatsType::NanCount] = nan_count_vec;
        results[CARTA::StatsType::Sum] = sum_vec;
        results[CARTA::StatsType::Mean] = mean_vec;
        results[CARTA::StatsType::RMS] = rms_vec;
        results[CARTA::StatsType::Sigma] = sigma_vec;
        results[CARTA::StatsType::SumSq] = sum_sq_vec;
        results[CARTA::StatsType::Min] = min_vec;
        results[CARTA::StatsType::Max] = max_vec;
        results[CARTA::StatsType::Extrema] = extrema_vec;
        
        // Add flux density if available
        if (has_flux) {
            results[CARTA::StatsType::FluxDensity] = flux_vec;
        }
        
        progress = 1.0;
        
        // Log comprehensive statistics summary
        spdlog::info("GetRegionSpectralData: Successfully processed {} channels for region {}x{} - TOTAL TIME: {} ms ({:.2f} ms/channel)", 
                     profile_size, region_width, region_height, total_ms, static_cast<double>(total_ms) / profile_size);
        spdlog::info("GetRegionSpectralData: Returned {} statistics types: NumPixels, NanCount, Sum, Mean, RMS, Sigma, SumSq, Min, Max, Extrema{}",
                     results.size(), has_flux ? ", FluxDensity" : "");
        
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: Exception: {}", e.what());
        progress = 1.0;  // Set progress to avoid infinite while loop in RegionHandler
        return false;
    }
}

bool ZarrLoader::GetDownsampledRasterData(std::vector<float>& data, int z, int stokes,
    CARTA::ImageBounds& bounds, int mip, std::mutex& image_mutex) {
    return false;
}

bool ZarrLoader::GetChunk(std::vector<float>& data, int& data_width, int& data_height,
    int min_x, int min_y, int z, int stokes, std::mutex& image_mutex) {
    
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        if (!_image) {
            spdlog::error("ZarrLoader::GetChunk: No image available");
            return false;
        }

        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetChunk: Image is not a CartaZarrImage");
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        
        // Check that coordinates are within image bounds
        if (min_x < 0 || min_y < 0 || min_x >= img_width || min_y >= img_height) {
            spdlog::error("ZarrLoader::GetChunk: Coordinates out of bounds: min_x={}, min_y={} (image: {}x{})", 
                         min_x, min_y, img_width, img_height);
            return false;
        }
        
        data_width = std::min(CHUNK_SIZE, img_width - min_x);
        data_height = std::min(CHUNK_SIZE, img_height - min_y);
        
        std::cout << "[ZARR DEBUG] GetChunk called: min_x=" << min_x << ", min_y=" << min_y 
                  << ", z=" << z << ", stokes=" << stokes << std::endl;
        std::cout << "[ZARR DEBUG] Calculated chunk size: " << data_width << "x" << data_height << std::endl;
        
        if (data_width <= 0 || data_height <= 0) {
            spdlog::warn("ZarrLoader::GetChunk: Invalid calculated chunk size: {}x{} at ({},{}) (image: {}x{})", 
                        data_width, data_height, min_x, min_y, img_width, img_height);
            return false;
        }
        
        std::cout << "[ZARR DEBUG] GetChunk request: start=[" << min_x << "," << min_y << "," << z << "," << stokes 
                  << "] length=[" << data_width << "," << data_height << ",1,1]" << std::endl;
        
        casacore::IPosition start, length;
        if (shape.size() == 5) {
            // 5D ZARR: [time, freq, stokes, x, y] 
            start = casacore::IPosition(5, 0, z, stokes, min_x, min_y);
            length = casacore::IPosition(5, 1, 1, 1, data_width, data_height);
        } else if (shape.size() == 4) {
            // 4D: Use CARTA standard order [x, y, z, stokes]
            start = casacore::IPosition(4, min_x, min_y, z, stokes);
            length = casacore::IPosition(4, data_width, data_height, 1, 1);
        } else if (shape.size() == 3) {
            // 3D: Use CARTA standard order [x, y, z]
            start = casacore::IPosition(3, min_x, min_y, z);
            length = casacore::IPosition(3, data_width, data_height, 1);
        } else if (shape.size() == 2) {
            // 2D: Use CARTA standard order [x, y]
            start = casacore::IPosition(2, min_x, min_y);
            length = casacore::IPosition(2, data_width, data_height);
        } else {
            spdlog::error("ZarrLoader::GetChunk: Unsupported number of dimensions: {}", shape.size());
            return false;
        }
        
        casacore::Array<float> chunk_array;
        spdlog::info("GetChunk: Reading chunk with start={} length={}", 
                     fmt::join(start.asStdVector(), ","), fmt::join(length.asStdVector(), ","));
        if (!zarr_image->doGetSlice(chunk_array, casacore::Slicer(start, length))) {
            spdlog::error("ZarrLoader::GetChunk: doGetSlice failed");
            return false;
        }
        
        data.resize(data_width * data_height);
        
        casacore::Array<float>::const_iterator array_iter = chunk_array.begin();
        for (int i = 0; i < data_width * data_height; ++i) {
            data[i] = *array_iter;
            ++array_iter;
        }
        
        std::cout << "[ZARR DEBUG] GetChunk successful: read " << data.size() << " pixels" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("ZarrLoader::GetChunk exception: {}", e.what());
        return false;
    }
}

const casacore::IPosition ZarrLoader::GetStatsDataShape(FileInfo::Data ds) {
    return casacore::IPosition();
}

std::unique_ptr<casacore::ArrayBase> ZarrLoader::GetStatsData(FileInfo::Data ds) {
    return nullptr;
}

bool ZarrLoader::GetSlice(casacore::Array<float>& data, const StokesSlicer& stokes_slicer) {
    // ZARR-optimized GetSlice implementation following Frame.cc GetSlicerData pattern
    if (!_image) {
        spdlog::error("ZarrLoader::GetSlice: No image available");
        return false;
    }

    try {
        auto zarr_image = dynamic_cast<CartaZarrImage*>(_image.get());
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetSlice: Image is not a CartaZarrImage");
            return false;
        }

        casacore::Slicer slicer = stokes_slicer.slicer;
        StokesSource stokes_source = stokes_slicer.stokes_source;
        
        // Resize data array if needed
        if (data.shape() != slicer.length()) {
            data.resize(slicer.length());
        }

        spdlog::debug("ZarrLoader::GetSlice: Slicer start={} length={}", 
                     fmt::join(slicer.start().asStdVector(), ","), 
                     fmt::join(slicer.length().asStdVector(), ","));

        // For ZARR, directly use doGetSlice which goes through our optimized CartaZarrImage
        // This will utilize the caching strategies we implemented in CartaZarrImage
        bool success = zarr_image->doGetSlice(data, slicer);
        
        if (!success) {
            spdlog::error("ZarrLoader::GetSlice: doGetSlice failed for slicer start={} length={}", 
                         fmt::join(slicer.start().asStdVector(), ","),
                         fmt::join(slicer.length().asStdVector(), ","));
            return false;
        }

        spdlog::debug("ZarrLoader::GetSlice: Successfully read slice of size {}", data.size());
        return true;

    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetSlice: Exception: {}", e.what());
        return false;
    }
}

bool ZarrLoader::GetSpectralDataOptimized(std::vector<float>& data, int stokes, int x, int y, 
                                         int z_start, int z_end, std::mutex& image_mutex) {
    // Optimized method for reading large spectral ranges at once for ZARR files
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        if (!_image) {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: No image available");
            return false;
        }
        
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Image is not a CartaZarrImage");
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        int num_channels = (shape.size() > 2) ? shape[2] : 1;
        
        // Validate parameters
        if (x < 0 || y < 0 || x >= img_width || y >= img_height) {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Position out of bounds: ({},{}) (image: {}x{})", 
                         x, y, img_width, img_height);
            return false;
        }
        
        if (z_start < 0 || z_end >= num_channels || z_start > z_end) {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Invalid z range [{},{}] (max: {})", 
                         z_start, z_end, num_channels - 1);
            return false;
        }
        
        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            return false;
        }
        
        int spectral_length = z_end - z_start + 1;
        data.resize(spectral_length);
        
        // Use optimized reading for large spectral ranges
        casacore::IPosition start, length;
        
        if (shape.size() == 5) {
            // 5D ZARR: [time, freq, stokes, x, y]
            start = casacore::IPosition(5, 0, z_start, stokes, x, y);
            length = casacore::IPosition(5, 1, spectral_length, 1, 1, 1);
        } else if (shape.size() == 4) {
            // 4D: [x, y, freq, stokes]
            start = casacore::IPosition(4, x, y, z_start, stokes);
            length = casacore::IPosition(4, 1, 1, spectral_length, 1);
        } else if (shape.size() == 3) {
            // 3D: [x, y, freq]
            start = casacore::IPosition(3, x, y, z_start);
            length = casacore::IPosition(3, 1, 1, spectral_length);
        } else {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Unsupported number of dimensions: {}", shape.size());
            return false;
        }
        
        casacore::Array<float> spectral_array;
        casacore::Slicer slicer(start, length);
        
        spdlog::debug("GetSpectralDataOptimized: Reading spectral range [{},{}] ({} channels) at ({},{}) with start={} length={}", 
                     z_start, z_end, spectral_length, x, y,
                     fmt::join(start.asStdVector(), ","), fmt::join(length.asStdVector(), ","));
        
        // Try direct TensorStore read first for better performance
        if (zarr_image->readDirectFromTensorStore(spectral_array, slicer)) {
            spdlog::debug("GetSpectralDataOptimized: Direct TensorStore read successful for {} channels", spectral_length);
        } else if (zarr_image->doGetSlice(spectral_array, slicer)) {
            spdlog::debug("GetSpectralDataOptimized: Regular doGetSlice successful for {} channels", spectral_length);
        } else {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Both direct and regular read failed");
            return false;
        }
        
        if (spectral_array.nelements() != spectral_length) {
            spdlog::error("ZarrLoader::GetSpectralDataOptimized: Expected {} elements, got {}", 
                         spectral_length, spectral_array.nelements());
            return false;
        }
        
        // Copy data to output vector
        const float* array_data = spectral_array.data();
        for (int i = 0; i < spectral_length; ++i) {
            data[i] = array_data[i];
        }
        
        spdlog::debug("GetSpectralDataOptimized: Successfully read {} channels [{},{}] at position ({},{})", 
                     spectral_length, z_start, z_end, x, y);
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetSpectralDataOptimized: Exception: {}", e.what());
        return false;
    }
}

bool ZarrLoader::GetSpatialProfileX(std::vector<float>& data, int x_start, int x_end, int y, int z, int stokes, std::mutex& image_mutex) {
    // Read X spatial profile (horizontal line) directly from TensorStore
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        if (!_image) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: No image available");
            return false;
        }
        
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: Image is not a CartaZarrImage");
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        
        // Validate parameters
        if (y < 0 || y >= img_height) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: y={} out of bounds (height={})", y, img_height);
            return false;
        }
        
        if (x_start < 0 || x_end >= img_width || x_start > x_end) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: Invalid x range [{},{}] (width={})", x_start, x_end, img_width);
            return false;
        }
        
        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            return false;
        }
        
        int profile_length = x_end - x_start + 1;
        data.resize(profile_length);
        
        // Read horizontal line from TensorStore
        casacore::IPosition start, length;
        
        if (shape.size() == 5) {
            // 5D ZARR: [time, freq, stokes, l, m] where l=x(width), m=y(height)
            start = casacore::IPosition(5, 0, z, stokes, x_start, y);
            length = casacore::IPosition(5, 1, 1, 1, profile_length, 1);
        } else if (shape.size() == 4) {
            // 4D: [x, y, z, stokes]
            start = casacore::IPosition(4, x_start, y, z, stokes);
            length = casacore::IPosition(4, profile_length, 1, 1, 1);
        } else if (shape.size() == 3) {
            // 3D: [x, y, z]
            start = casacore::IPosition(3, x_start, y, z);
            length = casacore::IPosition(3, profile_length, 1, 1);
        } else if (shape.size() == 2) {
            // 2D: [x, y]
            start = casacore::IPosition(2, x_start, y);
            length = casacore::IPosition(2, profile_length, 1);
        } else {
            spdlog::error("ZarrLoader::GetSpatialProfileX: Unsupported dimensions: {}", shape.size());
            return false;
        }
        
        casacore::Array<float> profile_array;
        casacore::Slicer slicer(start, length);
        
        spdlog::debug("GetSpatialProfileX: Reading x=[{},{}] at y={}, z={}, stokes={}", x_start, x_end, y, z, stokes);
        
        // Use direct TensorStore read to bypass cache
        if (!zarr_image->readPixelFromTensorStore(profile_array, slicer)) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: readPixelFromTensorStore failed");
            return false;
        }
        
        if (profile_array.nelements() != profile_length) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: Expected {} elements, got {}", profile_length, profile_array.nelements());
            return false;
        }
        
        // Copy to output vector
        const float* array_data = profile_array.data();
        for (int i = 0; i < profile_length; ++i) {
            data[i] = array_data[i];
        }
        
        spdlog::debug("GetSpatialProfileX: Success - read {} pixels", profile_length);
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetSpatialProfileX: Exception: {}", e.what());
        return false;
    }
}

bool ZarrLoader::GetSpatialProfileY(std::vector<float>& data, int x, int y_start, int y_end, int z, int stokes, std::mutex& image_mutex) {
    // Read Y spatial profile (vertical line) directly from TensorStore
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        if (!_image) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: No image available");
            return false;
        }
        
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: Image is not a CartaZarrImage");
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        
        // Validate parameters
        if (x < 0 || x >= img_width) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: x={} out of bounds (width={})", x, img_width);
            return false;
        }
        
        if (y_start < 0 || y_end >= img_height || y_start > y_end) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: Invalid y range [{},{}] (height={})", y_start, y_end, img_height);
            return false;
        }
        
        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            return false;
        }
        
        int profile_length = y_end - y_start + 1;
        data.resize(profile_length);
        
        // Read vertical line from TensorStore
        casacore::IPosition start, length;
        
        if (shape.size() == 5) {
            // 5D ZARR: [time, freq, stokes, l, m] where l=x(width), m=y(height)
            start = casacore::IPosition(5, 0, z, stokes, x, y_start);
            length = casacore::IPosition(5, 1, 1, 1, 1, profile_length);
        } else if (shape.size() == 4) {
            // 4D: [x, y, z, stokes]
            start = casacore::IPosition(4, x, y_start, z, stokes);
            length = casacore::IPosition(4, 1, profile_length, 1, 1);
        } else if (shape.size() == 3) {
            // 3D: [x, y, z]
            start = casacore::IPosition(3, x, y_start, z);
            length = casacore::IPosition(3, 1, profile_length, 1);
        } else if (shape.size() == 2) {
            // 2D: [x, y]
            start = casacore::IPosition(2, x, y_start);
            length = casacore::IPosition(2, 1, profile_length);
        } else {
            spdlog::error("ZarrLoader::GetSpatialProfileY: Unsupported dimensions: {}", shape.size());
            return false;
        }
        
        casacore::Array<float> profile_array;
        casacore::Slicer slicer(start, length);
        
        spdlog::debug("GetSpatialProfileY: Reading y=[{},{}] at x={}, z={}, stokes={}", y_start, y_end, x, z, stokes);
        
        // Use direct TensorStore read to bypass cache
        if (!zarr_image->readPixelFromTensorStore(profile_array, slicer)) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: readPixelFromTensorStore failed");
            return false;
        }
        
        if (profile_array.nelements() != profile_length) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: Expected {} elements, got {}", profile_length, profile_array.nelements());
            return false;
        }
        
        // Copy to output vector
        const float* array_data = profile_array.data();
        for (int i = 0; i < profile_length; ++i) {
            data[i] = array_data[i];
        }
        
        spdlog::debug("GetSpatialProfileY: Success - read {} pixels", profile_length);
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetSpatialProfileY: Exception: {}", e.what());
        return false;
    }
}