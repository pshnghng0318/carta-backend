/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"
#include "CartaZarrImage.h"
#include "Logger/Logger.h"
#include "Util/Image.h"

#include <filesystem>
#include <memory>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <future>
#include <vector>

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

        std::vector<double> profile_data(profile_size, 0.0);
        
        // PARALLEL BATCH READING STRATEGY: Use 4 threads, each reads 4 channels
        // 4 threads × 4 channels = 16 channels per parallel batch
        const int PARALLEL_THREADS = 4;  // Number of concurrent threads
        const int CHANNELS_PER_THREAD = 4;  // Each thread reads 4 channels
        const int BATCH_SIZE = PARALLEL_THREADS * CHANNELS_PER_THREAD;  // 16 channels total
        
        spdlog::info("GetRegionSpectralData: Processing {} channels ({}x{} parallel batches)", 
                    profile_size, PARALLEL_THREADS, CHANNELS_PER_THREAD);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Lambda function to read multiple channels AND calculate statistics for one thread
        auto read_and_compute_batch = [&](int start_z, int end_z) -> std::pair<bool, std::vector<double>> {
            auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
            auto thread_start_time = std::chrono::high_resolution_clock::now();
            
            int num_channels = end_z - start_z + 1;
            
            // Step 1: Read data from TensorStore
            auto read_start = std::chrono::high_resolution_clock::now();
            casacore::IPosition start, length;
            if (shape.size() == 5) {
                start = casacore::IPosition(5, 0, start_z, stokes, x_min, y_min);
                length = casacore::IPosition(5, 1, num_channels, 1, region_width, region_height);
            } else if (shape.size() == 4) {
                start = casacore::IPosition(4, x_min, y_min, start_z, stokes);
                length = casacore::IPosition(4, region_width, region_height, num_channels, 1);
            } else if (shape.size() == 3) {
                start = casacore::IPosition(3, x_min, y_min, start_z);
                length = casacore::IPosition(3, region_width, region_height, num_channels);
            } else if (shape.size() == 2) {
                start = casacore::IPosition(2, x_min, y_min);
                length = casacore::IPosition(2, region_width, region_height);
            } else {
                spdlog::error("Unsupported dimensions: {}", shape.size());
                return {false, std::vector<double>()};
            }
            
            casacore::Array<float> batch_array;
            casacore::Slicer slicer(start, length);
            
            bool success = zarr_image->readPixelFromTensorStore(batch_array, slicer);
            if (!success) {
                return {false, std::vector<double>()};
            }
            
            auto read_end = std::chrono::high_resolution_clock::now();
            auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start).count();
            
            // Step 2: Calculate mean for each channel in this batch (PARALLELIZED!)
            auto compute_start = std::chrono::high_resolution_clock::now();
            const float* data_ptr = batch_array.data();
            casacore::IPosition array_shape = batch_array.shape();
            std::vector<double> channel_means(num_channels);
            
            // Pre-calculate mask access pattern to avoid repeated IPosition creation
            casacore::IPosition mask_pos(2);
            
            for (int ch_offset = 0; ch_offset < num_channels; ++ch_offset) {
                double sum = 0.0;
                int valid_count = 0;
                
                // Optimize: iterate in memory order for better cache locality
                for (int y = 0; y < region_height; ++y) {
                    int mask_y = y + y_min - origin[1];
                    bool y_in_mask = (mask_y >= 0 && mask_y < mask_shape[1]);
                    
                    for (int x = 0; x < region_width; ++x) {
                        int mask_x = x + x_min - origin[0];
                        
                        // Early exit if outside mask bounds
                        if (!y_in_mask || mask_x < 0 || mask_x >= mask_shape[0]) {
                            continue;
                        }
                        
                        // Reuse IPosition object instead of creating new one each time
                        mask_pos[0] = mask_x;
                        mask_pos[1] = mask_y;
                        if (!mask(mask_pos)) {
                            continue;
                        }
                        
                        // Calculate linear index (optimized for common case)
                        size_t linear_index;
                        if (array_shape.size() == 4 || array_shape.size() == 3) {
                            // Most common case: [x, y, ch] or [x, y, ch, stokes]
                            linear_index = x + y * region_width + ch_offset * region_width * region_height;
                        } else if (array_shape.size() == 5) {
                            linear_index = ch_offset * region_width * region_height + x * region_height + y;
                        } else {
                            linear_index = y * region_width + x;
                        }
                        
                        // Bounds check only in debug mode (assume correct access in release)
                        #ifdef DEBUG
                        if (linear_index >= batch_array.nelements()) {
                            continue;
                        }
                        #endif
                        
                        float value = data_ptr[linear_index];
                        if (std::isfinite(value)) {
                            sum += value;
                            valid_count++;
                        }
                    }
                }
                
                if (valid_count > 0) {
                    channel_means[ch_offset] = sum / static_cast<double>(valid_count);
                } else {
                    channel_means[ch_offset] = std::numeric_limits<double>::quiet_NaN();
                }
            }
            
            auto compute_end = std::chrono::high_resolution_clock::now();
            auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - thread_start_time).count();
            
            size_t batch_mb = (num_channels * region_width * region_height * sizeof(float)) / (1024*1024);
            double read_speed_gbs = (batch_mb / 1024.0) / (read_ms / 1000.0);
            
            spdlog::info("Thread {} channels {}-{}: read {} ms ({:.2f} GB/s), compute {} ms, total {} ms", 
                        thread_id, start_z, end_z, read_ms, read_speed_gbs, compute_ms, total_ms);
            
            return {success, std::move(channel_means)};
        };
        
        // Process channels in groups of BATCH_SIZE (16 channels = 4 threads × 4 channels)
        for (int batch_start = z_start; batch_start <= z_end; batch_start += BATCH_SIZE) {
            int batch_end = std::min(batch_start + BATCH_SIZE - 1, z_end);
            int actual_batch_size = batch_end - batch_start + 1;
            
            auto batch_start_time = std::chrono::high_resolution_clock::now();
            spdlog::info("Processing parallel batch: channels {} to {} ({} channels, {} threads)", 
                         batch_start, batch_end, actual_batch_size, PARALLEL_THREADS);
            
            // Launch parallel reads AND computation using std::async
            std::vector<std::future<std::pair<bool, std::vector<double>>>> futures;
            for (int thread_idx = 0; thread_idx < PARALLEL_THREADS; ++thread_idx) {
                int thread_start_z = batch_start + thread_idx * CHANNELS_PER_THREAD;
                int thread_end_z = std::min(thread_start_z + CHANNELS_PER_THREAD - 1, z_end);
                
                if (thread_start_z <= z_end) {
                    futures.push_back(std::async(std::launch::async, read_and_compute_batch, thread_start_z, thread_end_z));
                }
            }
            
            // Wait for all threads to complete and collect results
            for (int thread_idx = 0; thread_idx < futures.size(); ++thread_idx) {
                int thread_start_z = batch_start + thread_idx * CHANNELS_PER_THREAD;
                int thread_end_z = std::min(thread_start_z + CHANNELS_PER_THREAD - 1, z_end);
                
                auto result = futures[thread_idx].get();
                
                if (!result.first) {
                    spdlog::error("ZarrLoader::GetRegionSpectralData: Failed to process channels {}-{}", 
                                 thread_start_z, thread_end_z);
                    progress = 1.0;
                    return false;
                }
                
                // Copy computed means to profile_data
                const std::vector<double>& channel_means = result.second;
                for (int ch_offset = 0; ch_offset < channel_means.size(); ++ch_offset) {
                    int z_index = thread_start_z + ch_offset;
                    int profile_index = z_index - z_start;
                    profile_data[profile_index] = channel_means[ch_offset];
                }
            }
            
            auto batch_end_time = std::chrono::high_resolution_clock::now();
            auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end_time - batch_start_time).count();
            size_t total_mb = (actual_batch_size * region_width * region_height * sizeof(float)) / (1024*1024);
            double total_speed_gbs = (total_mb / 1024.0) / (batch_ms / 1000.0);
            spdlog::info("Parallel batch complete: {} ms for {} channels ({} MB total, {:.2f} GB/s aggregate)", 
                        batch_ms, actual_batch_size, total_mb, total_speed_gbs);
            
            // Handle 2D case where we only process one channel
            if (shape.size() == 2) {
                break;
            }
        }
        
        // All channels processed
        results[CARTA::StatsType::Mean] = profile_data;
        progress = 1.0;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        spdlog::info("GetRegionSpectralData: Successfully processed {} channels in batches for region {}x{} - TOTAL TIME: {} ms ({:.2f} ms/channel)", 
                     profile_size, region_width, region_height, total_ms, static_cast<double>(total_ms) / profile_size);
        
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