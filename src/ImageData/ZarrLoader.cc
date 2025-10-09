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
        
        spdlog::debug("GetCursorSpectralData: Reading Z profile for point ({},{}) stokes={}, {} channels", 
                     cursor_x, cursor_y, stokes, num_channels);
        
        data.resize(num_channels);
        
        for (int z = 0; z < num_channels; ++z) {
            casacore::IPosition start, length;
            
            if (shape.size() == 5) {
                // 5D ZARR: [time, freq, stokes, y, x]
                start = casacore::IPosition(5, 0, z, 0, cursor_y, cursor_x);
                length = casacore::IPosition(5, 1, 1, 1, 1, 1);
            } else if (shape.size() == 4) {
                // 4D: CARTA internal format [x, y, freq, stokes]
                start = casacore::IPosition(4, cursor_x, cursor_y, z, 0);
                length = casacore::IPosition(4, 1, 1, 1, 1);
            } else if (shape.size() == 3) {
                // 3D: [freq, y, x] 
                start = casacore::IPosition(3, z, cursor_y, cursor_x);
                length = casacore::IPosition(3, 1, 1, 1);
            } else if (shape.size() == 2) {
                // 2D: [y, x]
                start = casacore::IPosition(2, cursor_y, cursor_x);
                length = casacore::IPosition(2, 1, 1);
            } else {
                spdlog::error("ZarrLoader::GetCursorSpectralData: Unsupported number of dimensions: {}", shape.size());
                return false;
            }
            
            casacore::Array<float> pixel_array;
            spdlog::debug("GetCursorSpectralData: Reading channel {} with start={} length={}", 
                         z, fmt::join(start.asStdVector(), ","), fmt::join(length.asStdVector(), ","));
            if (!zarr_image->doGetSlice(pixel_array, casacore::Slicer(start, length))) {
                spdlog::error("ZarrLoader::GetCursorSpectralData: doGetSlice failed for channel {}", z);
                return false;
            }
            
            if (pixel_array.nelements() == 1) {
                data[z] = pixel_array.data()[0];
            } else {
                spdlog::error("ZarrLoader::GetCursorSpectralData: Expected 1 element, got {}", pixel_array.nelements());
                return false;
            }
        }
        
        spdlog::debug("GetCursorSpectralData: Successfully read {} channels for point ({},{})", 
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
    
    if (region_shape.size() >= 2 && region_shape[0] == 1 && region_shape[1] == 1) {
        spdlog::info("UseRegionSpectralData: RETURNING TRUE for optimized point spectral data (1x1 region)");
        return true;
    }

    if (_image && region_shape.size() >= 2) {
        casacore::IPosition img_shape = _image->shape();
        int img_width = img_shape[0];
        int img_height = img_shape[1];
        int region_size = region_shape[0] * region_shape[1];
        
        if (region_size <= 100) {
            spdlog::info("UseRegionSpectralData: RETURNING TRUE for optimized region spectral data ({}x{} region, size: {} pixels)", 
                         region_shape[0], region_shape[1], region_size);
            return true;
        } else {
            spdlog::info("UseRegionSpectralData: RETURNING FALSE - Region {}x{} too large (size: {} pixels), using default image slicing to avoid cache issues", 
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
            return false;
        }
        
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: Image is not a CartaZarrImage");
            return false;
        }
        
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        int num_channels = (shape.size() > 1) ? shape[1] : 1;

        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
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

        bool is_very_small_region = (region_area <= 100);
        
        if (is_very_small_region) {
            spdlog::info("SMALL REGION OPTIMIZATION: Using direct read for {}x{} region (area={}) to avoid cache overhead", 
                         region_width, region_height, region_area);
        } else {
            spdlog::debug("GetRegionSpectralData: Reading spectral data for region [{},{} to {},{}] ({}x{}) stokes={}, {} channels", 
                         x_min, y_min, x_max, y_max, region_width, region_height, stokes, num_channels);
        }
        
        int z_start = spectral_range.from;
        int z_end = spectral_range.to;
        int profile_size = z_end - z_start + 1;

        std::vector<double> profile_data(profile_size, 0.0);
        
        if (is_very_small_region) {
            
            casacore::IPosition start, length;
            if (shape.size() == 5) {
                // 5D ZARR: [time, freq, ?, y, x] -> 讀取 [time, freq_range, ?, y_range, x_range]
                start = casacore::IPosition(5, 0, z_start, 0, y_min, x_min);
                length = casacore::IPosition(5, 1, profile_size, 1, region_height, region_width);
            } else if (shape.size() == 4) {
                // 4D: [freq, ?, y, x]
                start = casacore::IPosition(4, z_start, 0, y_min, x_min);
                length = casacore::IPosition(4, profile_size, 1, region_height, region_width);
            } else if (shape.size() == 3) {
                // 3D: [freq, y, x]
                start = casacore::IPosition(3, z_start, y_min, x_min);
                length = casacore::IPosition(3, profile_size, region_height, region_width);
            } else if (shape.size() == 2) {
                // 2D: [y, x]
                start = casacore::IPosition(2, y_min, x_min);
                length = casacore::IPosition(2, region_height, region_width);
            } else {
                spdlog::error("ZarrLoader::GetRegionSpectralData: Unsupported number of dimensions: {}", shape.size());
                return false;
            }
            
            casacore::Array<float> full_region_array;
            casacore::Slicer full_slicer(start, length);
            
            if (zarr_image->readDirectFromTensorStore(full_region_array, full_slicer)) {
                spdlog::info("DIRECT READ SUCCESS: Read {}D block of size {} for small region", 
                             full_region_array.ndim(), full_region_array.size());
                
                casacore::IPosition array_shape = full_region_array.shape();
                spdlog::info("DIRECT READ: Array shape = [{}]", fmt::join(array_shape.asStdVector(), ", "));
                spdlog::info("DIRECT READ: Expected profile_size = {}, region_width = {}, region_height = {}", 
                            profile_size, region_width, region_height);
                spdlog::info("DIRECT READ: Mask shape = [{}], origin = [{}]", 
                            fmt::join(mask_shape.asStdVector(), ", "), fmt::join(origin.asStdVector(), ", "));
                
                spdlog::warn("DIRECT READ: ZARR shape is [time={}, freq={}, ?={}, y={}, x={}]", 
                            shape.size() > 0 ? shape[0] : 0,
                            shape.size() > 1 ? shape[1] : 0, 
                            shape.size() > 2 ? shape[2] : 0,
                            shape.size() > 3 ? shape[3] : 0,
                            shape.size() > 4 ? shape[4] : 0);
                
                if (full_region_array.size() > 0) {
                    const float* data_ptr = full_region_array.data();
                    auto first_value = *full_region_array.begin();
                    spdlog::warn("DIRECT READ: First pixel value = {} (array size = {})", first_value, full_region_array.size());
                    
                    if (array_shape.size() >= 2 && profile_size >= 3) {
                        size_t ch0_idx, ch1_idx, ch2_idx;
                        
                        if (array_shape.size() == 5) {
                            // [time=1, freq=profile_size, stokes=1, y=region_height, x=region_width]
                            ch0_idx = 0;
                            ch1_idx = region_height * region_width;
                            ch2_idx = 2 * region_height * region_width;
                        } else if (array_shape.size() == 4) {
                            // [freq=profile_size, stokes=1, y=region_height, x=region_width]
                            ch0_idx = 0;
                            ch1_idx = region_height * region_width;
                            ch2_idx = 2 * region_height * region_width;
                        } else {
                            ch0_idx = 0;
                            ch1_idx = region_width * region_height;
                            ch2_idx = 2 * region_width * region_height;
                        }
                        
                        spdlog::warn("DIRECT READ: Raw data check - Ch0[0]={}, Ch1[0]={}, Ch2[0]={}", 
                                    data_ptr[ch0_idx], data_ptr[ch1_idx], data_ptr[ch2_idx]);

                        spdlog::warn("DIRECT READ: Sample pixels - Ch0[1]={}, Ch0[center]={}, Ch0[last]={}",
                                    array_shape[0] > 1 ? data_ptr[1] : data_ptr[0],
                                    data_ptr[region_width * region_height / 2],
                                    data_ptr[region_width * region_height - 1]);

                        if (ch1_idx + region_width * region_height <= full_region_array.size()) {
                            spdlog::warn("DIRECT READ: Channel 1 samples - Ch1[1]={}, Ch1[center]={}, Ch1[last]={}",
                                        ch1_idx + 1 < full_region_array.size() ? data_ptr[ch1_idx + 1] : data_ptr[ch1_idx],
                                        ch1_idx + region_width * region_height / 2 < full_region_array.size() ? 
                                        data_ptr[ch1_idx + region_width * region_height / 2] : data_ptr[ch1_idx],
                                        ch1_idx + region_width * region_height - 1 < full_region_array.size() ? 
                                        data_ptr[ch1_idx + region_width * region_height - 1] : data_ptr[ch1_idx]);
                        }
                        
                        float min_val = *std::min_element(data_ptr, data_ptr + full_region_array.size());
                        float max_val = *std::max_element(data_ptr, data_ptr + full_region_array.size());
                        spdlog::warn("DIRECT READ: Data range - min={}, max={}", min_val, max_val);
                        
                        int non_zero_count = 0;
                        int finite_count = 0;
                        for (size_t i = 0; i < full_region_array.size(); ++i) {
                            if (std::isfinite(data_ptr[i])) {
                                finite_count++;
                                if (data_ptr[i] != 0.0f) {
                                    non_zero_count++;
                                }
                            }
                        }
                        spdlog::warn("DIRECT READ: Statistics - total={}, finite={}, non_zero={}", 
                                    full_region_array.size(), finite_count, non_zero_count);
                    }
                }
                
                for (int z = 0; z < profile_size; ++z) {
                    double sum = 0.0;
                    int valid_count = 0;
                    int total_checked = 0;
                    int mask_true_count = 0;

                    for (int y = 0; y < region_height; ++y) {
                        for (int x = 0; x < region_width; ++x) {
                            total_checked++;

                            bool mask_value = false;
                            if (x < mask_shape[0] && y < mask_shape[1]) {
                                mask_value = mask(casacore::IPosition(2, x, y));
                                if (mask_value) mask_true_count++;
                            }
                            
                            if (mask_value) {
                                float value;
                                
                                const float* data_ptr = full_region_array.data();
                                
                                if (array_shape.size() == 5) {
                                    size_t linear_index = 0 + z * 1 + 0 * 1 * profile_size + y * 1 * profile_size * 1 + x * 1 * profile_size * 1 * region_height;
                                    if (linear_index < full_region_array.size()) {
                                        value = data_ptr[linear_index];
                                    } else {
                                        spdlog::error("DIRECT READ: 5D Linear index {} out of bounds (array size: {})", linear_index, full_region_array.size());
                                        continue;
                                    }
                                } else if (array_shape.size() == 4) {
                                    // 4D: [freq=profile_size, ?=1, y=region_height, x=region_width]
                                    size_t linear_index = z * 1 * region_height * region_width + 0 * region_height * region_width + y * region_width + x;
                                    if (linear_index < full_region_array.size()) {
                                        value = data_ptr[linear_index];
                                    } else {
                                        spdlog::error("DIRECT READ: 4D Linear index {} out of bounds (array size: {})", linear_index, full_region_array.size());
                                        continue;
                                    }
                                } else if (array_shape.size() == 3) {
                                    // 3D: [freq=profile_size, y=region_height, x=region_width]
                                    size_t linear_index = z * region_height * region_width + y * region_width + x;
                                    if (linear_index < full_region_array.size()) {
                                        value = data_ptr[linear_index];
                                    } else {
                                        spdlog::error("DIRECT READ: 3D Linear index {} out of bounds (array size: {})", linear_index, full_region_array.size());
                                        continue;
                                    }
                                } else {
                                    // 2D: [y=region_height, x=region_width]
                                    size_t linear_index = y * region_width + x;
                                    if (linear_index < full_region_array.size()) {
                                        value = data_ptr[linear_index];
                                    } else {
                                        spdlog::error("DIRECT READ: 2D Linear index {} out of bounds (array size: {})", linear_index, full_region_array.size());
                                        continue;
                                    }
                                }
                                
                                if (std::isfinite(value)) {
                                    sum += value;
                                    valid_count++;

                                    if ((z <= 4 || z >= profile_size - 2) && valid_count <= 3) {
                                        size_t debug_linear_index;
                                        if (array_shape.size() == 5) {
                                            debug_linear_index = 0 + z * 1 + 0 * 1 * profile_size + y * 1 * profile_size * 1 + x * 1 * profile_size * 1 * region_height;
                                        } else if (array_shape.size() == 4) {
                                            debug_linear_index = z * 1 * region_height * region_width + 0 * region_height * region_width + y * region_width + x;
                                        } else if (array_shape.size() == 3) {
                                            debug_linear_index = z * region_height * region_width + y * region_width + x;
                                        } else {
                                            debug_linear_index = y * region_width + x;
                                        }
                                        spdlog::info("DIRECT READ: Pixel[{},{},{}] = {} (valid #{} for channel {}) linear_idx={}", 
                                                    x, y, z, value, valid_count, z, debug_linear_index);
                                    }
                                }
                            }
                        }
                    }
                    
                    if (valid_count > 0) {
                        profile_data[z] = sum / valid_count;
                    } else {
                        profile_data[z] = std::numeric_limits<double>::quiet_NaN();
                    }
                    
                    if (z < 5 || z >= profile_size - 5 || z % 20 == 0) {
                        spdlog::info("DIRECT READ: Channel {} - checked {} pixels, mask_true {} pixels, valid {} pixels, sum = {}, mean = {}", 
                                    z, total_checked, mask_true_count, valid_count, sum, profile_data[z]);
                    }
                }
                
                spdlog::warn("DIRECT READ: Profile calculation completed for ALL {} channels (z_start={}, z_end={})", 
                            profile_size, z_start, z_end);
                spdlog::warn("DIRECT READ: First 5 channel means: [{}]", 
                            fmt::join(profile_data.begin(), profile_data.begin() + std::min(5, (int)profile_data.size()), ", "));
                if (profile_data.size() > 10) {
                    spdlog::warn("DIRECT READ: Last 5 channel means: [{}]", 
                                fmt::join(profile_data.end() - 5, profile_data.end(), ", "));
                }
                
                results[CARTA::StatsType::Mean] = profile_data;
                progress = 1.0;
                
                spdlog::debug("DIRECT READ OPTIMIZATION: Successfully processed {}x{} region across {} channels", 
                             region_width, region_height, profile_size);
                return true;
            } else {
                spdlog::warn("Direct read failed for small region, falling back to per-channel method");
            }
        }

        for (int z = z_start; z <= z_end; ++z) {
            if (z >= num_channels) break;

            casacore::IPosition start, length;
            if (shape.size() == 5) {
                // 5D ZARR: [time, freq, ?, y, x]
                start = casacore::IPosition(5, 0, z, 0, y_min, x_min);
                length = casacore::IPosition(5, 1, 1, 1, region_height, region_width);
            } else if (shape.size() == 4) {
                // 4D: [freq, ?, y, x]
                start = casacore::IPosition(4, z, 0, y_min, x_min);
                length = casacore::IPosition(4, 1, 1, region_height, region_width);
            } else if (shape.size() == 3) {
                // 3D: [freq, y, x]
                start = casacore::IPosition(3, z, y_min, x_min);
                length = casacore::IPosition(3, 1, region_height, region_width);
            } else if (shape.size() == 2) {
                // 2D: [y, x]
                start = casacore::IPosition(2, y_min, x_min);
                length = casacore::IPosition(2, region_height, region_width);
            } else {
                spdlog::error("ZarrLoader::GetRegionSpectralData: Unsupported number of dimensions: {}", shape.size());
                return false;
            }

            casacore::Array<float> region_array;
            spdlog::debug("GetRegionSpectralData: Reading channel {} with start={} length={}", 
                         z, fmt::join(start.asStdVector(), ","), fmt::join(length.asStdVector(), ","));
            if (!zarr_image->doGetSlice(region_array, casacore::Slicer(start, length))) {
                spdlog::error("ZarrLoader::GetRegionSpectralData: doGetSlice failed for channel {}", z);
                continue;
            }

            double sum = 0.0;
            int valid_count = 0;
            
            auto region_iter = region_array.begin();
            for (int y = 0; y < region_height; ++y) {
                for (int x = 0; x < region_width; ++x) {
                    int mask_x = x;
                    int mask_y = y;
                    if (mask_x < mask_shape[0] && mask_y < mask_shape[1] && mask(casacore::IPosition(2, mask_x, mask_y))) {
                        float value = *region_iter;
                        if (std::isfinite(value)) {
                            sum += value;
                            valid_count++;
                        }
                    }
                    ++region_iter;
                }
            }

            if (valid_count > 0) {
                profile_data[z - z_start] = sum / valid_count;
            } else {
                profile_data[z - z_start] = std::numeric_limits<double>::quiet_NaN();
            }
            
            progress = static_cast<float>(z - z_start + 1) / profile_size;
        }

        results[CARTA::StatsType::Mean] = profile_data;
        
        spdlog::debug("GetRegionSpectralData: Successfully calculated spectral profile for region {}x{} across {} channels", 
                     region_width, region_height, profile_size);
        
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: Exception: {}", e.what());
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
            // 5D: Use CARTA standard order [x, y, z, stokes, time] 
            start = casacore::IPosition(5, min_x, min_y, z, stokes, 0);
            length = casacore::IPosition(5, data_width, data_height, 1, 1, 1);
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
