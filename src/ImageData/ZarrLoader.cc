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
        
        spdlog::debug("GetCursorSpectralData: Reading {} channels one by one for point ({},{}) stokes={}", 
                     num_channels, cursor_x, cursor_y, stokes);
        
        data.resize(num_channels);
        
        // Read each channel individually
        for (int z = 0; z < num_channels; ++z) {
            casacore::IPosition start, length;
            
            if (shape.size() == 5) {
                // 5D ZARR: [time, freq, stokes, x, y] - read single channel for single pixel
                start = casacore::IPosition(5, 0, z, stokes, cursor_x, cursor_y);
                length = casacore::IPosition(5, 1, 1, 1, 1, 1);
            } else if (shape.size() == 4) {
                // 4D: CARTA internal format [x, y, freq, stokes] - read single channel for single pixel
                start = casacore::IPosition(4, cursor_x, cursor_y, z, stokes);
                length = casacore::IPosition(4, 1, 1, 1, 1);
            } else if (shape.size() == 3) {
                // 3D: [x, y, freq] - read single channel for single pixel
                start = casacore::IPosition(3, cursor_x, cursor_y, z);
                length = casacore::IPosition(3, 1, 1, 1);
            } else if (shape.size() == 2) {
                // 2D: [x, y] - single channel
                start = casacore::IPosition(2, cursor_x, cursor_y);
                length = casacore::IPosition(2, 1, 1);
            } else {
                spdlog::error("ZarrLoader::GetCursorSpectralData: Unsupported number of dimensions: {}", shape.size());
                return false;
            }
            
            casacore::Array<float> pixel_array;
            if (!zarr_image->doGetSlice(pixel_array, casacore::Slicer(start, length))) {
                spdlog::error("ZarrLoader::GetCursorSpectralData: doGetSlice failed for channel {}", z);
                return false;
            }
            
            if (pixel_array.nelements() != 1) {
                spdlog::error("ZarrLoader::GetCursorSpectralData: Expected 1 element for channel {}, got {}", 
                             z, pixel_array.nelements());
                return false;
            }
            
            data[z] = *pixel_array.data();
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
        const int SMALL_REGION_THRESHOLD = 100; // pixels
        
        // Only use loader path for very small regions to prevent cache overflow
        if (region_size <= SMALL_REGION_THRESHOLD) {
            spdlog::info("UseRegionSpectralData: RETURNING TRUE for small region spectral data ({}x{} region, size: {} pixels)", 
                         region_shape[0], region_shape[1], region_size);
            return true;
        } else {
            // Use RegionHandler's segmented processing for larger regions to prevent cache overflow
            spdlog::info("UseRegionSpectralData: RETURNING FALSE for large region ({}x{} region, size: {} pixels) - using segmented processing", 
                         region_shape[0], region_shape[1], region_size);
            return false;
        }
    }

    spdlog::info("UseRegionSpectralData: RETURNING FALSE - Using default image slicing for {}x{} region", 
                 region_shape[0], region_shape[1]);
    return false;
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& spectral_range, int stokes,
    const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin,
    std::mutex& image_mutex, std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) {
    
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
        
        // SINGLE-CHANNEL STRATEGY: Process one channel at a time to avoid cache conflicts
        // spdlog::info("GetRegionSpectralData: Processing {} channels one by one", profile_size);
        
        for (int z = z_start; z <= z_end; ++z) {
            int profile_index = z - z_start;
            
            // Read single channel from TensorStore
            casacore::IPosition start, length;
            if (shape.size() == 5) {
                // 5D ZARR: [time, freq, stokes, x, y] 
                start = casacore::IPosition(5, 0, z, stokes, x_min, y_min);
                length = casacore::IPosition(5, 1, 1, 1, region_width, region_height);
            } else if (shape.size() == 4) {
                // 4D: Use CARTA standard order [x, y, freq, stokes]
                start = casacore::IPosition(4, x_min, y_min, z, stokes);
                length = casacore::IPosition(4, region_width, region_height, 1, 1);
            } else if (shape.size() == 3) {
                // 3D: Use CARTA standard order [x, y, freq]
                start = casacore::IPosition(3, x_min, y_min, z);
                length = casacore::IPosition(3, region_width, region_height, 1);
            } else if (shape.size() == 2) {
                // 2D: Use CARTA standard order [x, y] (only one channel)
                start = casacore::IPosition(2, x_min, y_min);
                length = casacore::IPosition(2, region_width, region_height);
            } else {
                spdlog::error("ZarrLoader::GetRegionSpectralData: Unsupported number of dimensions: {}", shape.size());
                progress = 1.0;
                return false;
            }
            
            casacore::Array<float> channel_array;
            casacore::Slicer channel_slicer(start, length);
            
            if (!zarr_image->readDirectFromTensorStore(channel_array, channel_slicer)) {
                spdlog::error("ZarrLoader::GetRegionSpectralData: Failed to read channel {} from TensorStore", z);
                progress = 1.0;
                return false;
            }
            
            // Process this single channel
            const float* data_ptr = channel_array.data();
            casacore::IPosition array_shape = channel_array.shape();
            
            // Calculate mean for this channel
            double sum = 0.0;
            int valid_count = 0;
            
            for (int y = 0; y < region_height; ++y) {
                for (int x = 0; x < region_width; ++x) {
                    int mask_x = x;
                    int mask_y = y;
                    
                    if (mask_x < mask_shape[0] && mask_y < mask_shape[1] && 
                        mask(casacore::IPosition(2, mask_x, mask_y))) {
                        
                        // Calculate linear index in single channel array
                        size_t linear_index;
                        if (array_shape.size() == 5) {
                            // [time=1, freq=1, stokes=1, y=region_height, x=region_width]
                            linear_index = y * region_width + x;
                        } else if (array_shape.size() == 4) {
                            // [x=region_width, y=region_height, freq=1, stokes=1]
                            linear_index = x + y * region_width;
                        } else if (array_shape.size() == 3) {
                            // [x=region_width, y=region_height, freq=1]
                            linear_index = x + y * region_width;
                        } else {
                            // 2D: [x=region_width, y=region_height]
                            linear_index = x + y * region_width;
                        }
                        
                        if (linear_index < channel_array.size()) {
                            float value = data_ptr[linear_index];
                            if (std::isfinite(value)) {
                                sum += value;
                                valid_count++;
                            }
                        }
                    }
                }
            }
            
            if (valid_count > 0) {
                profile_data[profile_index] = sum / valid_count;
            } else {
                profile_data[profile_index] = std::numeric_limits<double>::quiet_NaN();
            }
            
            // Only log for debugging first few and last few channels
            if (profile_index < 5 || profile_index >= profile_size - 5 || profile_index % 20 == 0) {
                spdlog::info("Channel {} - region {}x{}, valid {} pixels, sum = {}, mean = {}", 
                            z, region_width, region_height, valid_count, sum, profile_data[profile_index]);
            }
            
            // Update progress
            progress = static_cast<float>(profile_index + 1) / static_cast<float>(profile_size);
        }
        
        // All channels processed
        results[CARTA::StatsType::Mean] = profile_data;
        progress = 1.0;
        
        // spdlog::info("GetRegionSpectralData: Successfully processed {} channels individually for region {}x{}", 
        //              profile_size, region_width, region_height);
        
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