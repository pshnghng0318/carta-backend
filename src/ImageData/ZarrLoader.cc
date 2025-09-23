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
    // 不在構造函數中創建 CartaZarrImage，避免重複創建
    // 只設定預設值，等到 AllocateImage 時才真正創建
    _num_dims = 0;
    _has_pixel_mask = false;
    
    spdlog::info("ZarrLoader created for: {}", _filename);
}

void ZarrLoader::AllocateImage(const std::string& hdu) {
    // 為文件瀏覽器創建基本的 CartaZarrImage
    try {
        _image = std::shared_ptr<casacore::ImageInterface<float>>(new CartaZarrImage(_filename));
        
        if (_image) {
            // 設定 _num_dims 和 _has_pixel_mask 供 HasData 使用
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
    // 使用基類的預設實現，讓 CARTA 實時計算統計數據
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
            // 對於統計數據，讓基類處理（返回 false，促使實時計算）
            return FileLoader::HasData(ds);
    }
}

bool ZarrLoader::HasMip(int mip) const {
    return false; // 暫時不支援 MipMap
}

bool ZarrLoader::UseTileCache() const {
    return true; // 使用 tile cache
}

bool ZarrLoader::GetCursorSpectralData(std::vector<float>& data, int stokes, int cursor_x, int count_x,
    int cursor_y, int count_y, std::mutex& image_mutex) {
    
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        // 確保 CartaZarrImage 已創建
        if (!_image) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: No image available");
            return false;
        }
        
        // 將 CartaZarrImage 轉換為具體類型  
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Image is not a CartaZarrImage");
            return false;
        }
        
        // 獲取圖像形狀
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        int num_channels = (shape.size() > 1) ? shape[1] : 1;  // ZARR: 頻道在第1維
        
        // 檢查座標邊界
        if (cursor_x < 0 || cursor_y < 0 || cursor_x >= img_width || cursor_y >= img_height) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Cursor out of bounds: ({},{}) (image: {}x{})", 
                         cursor_x, cursor_y, img_width, img_height);
            return false;
        }
        
        // 檢查 stokes 邊界
        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            return false;
        }
        
        spdlog::debug("GetCursorSpectralData: Reading Z profile for point ({},{}) stokes={}, {} channels", 
                     cursor_x, cursor_y, stokes, num_channels);
        
        // 為每個頻道分配存儲空間
        data.resize(num_channels);
        
        // 逐個頻道讀取單個像素值
        for (int z = 0; z < num_channels; ++z) {
            casacore::IPosition start, length;
            
            if (shape.size() == 5) {
                // 5D ZARR: [time, freq, ?, y, x]
                start = casacore::IPosition(5, 0, z, 0, cursor_y, cursor_x);
                length = casacore::IPosition(5, 1, 1, 1, 1, 1);
            } else if (shape.size() == 4) {
                // 4D: [freq, ?, y, x] (沒有時間維度)
                start = casacore::IPosition(4, z, 0, cursor_y, cursor_x);
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
            
            // 讀取單個像素
            casacore::Array<float> pixel_array;
            if (!zarr_image->doGetSlice(pixel_array, casacore::Slicer(start, length))) {
                spdlog::error("ZarrLoader::GetCursorSpectralData: doGetSlice failed for channel {}", z);
                return false;
            }
            
            // 提取單個像素值
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
    
    // 對於點區域（1x1），使用優化的讀取方法
    if (region_shape.size() >= 2 && region_shape[0] == 1 && region_shape[1] == 1) {
        spdlog::info("UseRegionSpectralData: RETURNING TRUE for optimized point spectral data (1x1 region)");
        return true;
    }
    
    // 對於 PV generator，優先使用點區域來避免緩存過大的區域
    // 只對非常小的區域（例如小於100像素）使用區域優化讀取
    if (_image && region_shape.size() >= 2) {
        casacore::IPosition img_shape = _image->shape();
        int img_width = img_shape[0];
        int img_height = img_shape[1];
        int region_size = region_shape[0] * region_shape[1];
        
        // 只對非常小的區域使用區域優化（避免"Cache dimensions too big"錯誤）
        if (region_size <= 100) {  // 限制為100像素以下的小區域
            spdlog::info("UseRegionSpectralData: RETURNING TRUE for optimized region spectral data ({}x{} region, size: {} pixels)", 
                         region_shape[0], region_shape[1], region_size);
            return true;
        } else {
            spdlog::info("UseRegionSpectralData: RETURNING FALSE - Region {}x{} too large (size: {} pixels), using default image slicing to avoid cache issues", 
                         region_shape[0], region_shape[1], region_size);
            return false;
        }
    }
    
    // 對於較大的區域，使用默認的圖像切片方法
    spdlog::info("UseRegionSpectralData: RETURNING FALSE - Using default image slicing for {}x{} region", 
                 region_shape[0], region_shape[1]);
    return false;
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& spectral_range, int stokes,
    const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin,
    std::mutex& image_mutex, std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) {
    
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        // 確保 CartaZarrImage 已創建
        if (!_image) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: No image available");
            return false;
        }
        
        // 將 CartaZarrImage 轉換為具體類型
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: Image is not a CartaZarrImage");
            return false;
        }
        
        // 獲取圖像形狀
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        int num_channels = (shape.size() > 1) ? shape[1] : 1;  // ZARR: 頻道在第1維
        
        // 檢查 stokes 邊界 (如果有的話，通常不適用於這種ZARR格式)
        if (shape.size() > 3 && (stokes < 0 || stokes >= shape[3])) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: Stokes {} out of bounds (max: {})", stokes, shape[3] - 1);
            return false;
        }
        
        // 從 mask 和 origin 計算區域邊界
        casacore::IPosition mask_shape = mask.shape();
        int x_min = std::max(0, static_cast<int>(origin[0]));
        int y_min = std::max(0, static_cast<int>(origin[1]));
        int x_max = std::min(img_width - 1, static_cast<int>(origin[0] + mask_shape[0] - 1));
        int y_max = std::min(img_height - 1, static_cast<int>(origin[1] + mask_shape[1] - 1));
        
        int region_width = x_max - x_min + 1;
        int region_height = y_max - y_min + 1;
        int region_area = region_width * region_height;
        
        // **PV 生成器優化**: 檢測非常小的區域（PV generator box regions）
        // PV generator 的每個 box region 通常是 1×1 到 20×3 像素
        // 對於這類小區域，使用直接讀取以避免觸發緩存系統
        bool is_very_small_region = (region_area <= 100);  // 100 像素以下的區域
        
        if (is_very_small_region) {
            spdlog::info("SMALL REGION OPTIMIZATION: Using direct read for {}x{} region (area={}) to avoid cache overhead", 
                         region_width, region_height, region_area);
        } else {
            spdlog::debug("GetRegionSpectralData: Reading spectral data for region [{},{} to {},{}] ({}x{}) stokes={}, {} channels", 
                         x_min, y_min, x_max, y_max, region_width, region_height, stokes, num_channels);
        }
        
        // 計算要讀取的頻道範圍
        int z_start = spectral_range.from;
        int z_end = spectral_range.to;
        int profile_size = z_end - z_start + 1;
        
        // 初始化結果（假設只處理 MEAN 統計）
        std::vector<double> profile_data(profile_size, 0.0);
        
        if (is_very_small_region) {
            // **直接讀取優化**: 對於非常小的區域，使用一次性讀取所有頻道的方法
            // 這可以避免觸發 CartaZarrImage 的緩存機制，減少內存使用和提高性能
            
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
            
            // 直接從 TensorStore 讀取整個 4D 塊 (x, y, z, stokes)
            casacore::Array<float> full_region_array;
            casacore::Slicer full_slicer(start, length);
            
            // 使用 CartaZarrImage 的 readDirectFromTensorStore 方法
            if (zarr_image->readDirectFromTensorStore(full_region_array, full_slicer)) {
                spdlog::info("DIRECT READ SUCCESS: Read {}D block of size {} for small region", 
                             full_region_array.ndim(), full_region_array.size());
                
                // 處理數據 - 需要考慮ZARR的實際維度順序 [time, freq, ?, y, x]
                casacore::IPosition array_shape = full_region_array.shape();
                spdlog::info("DIRECT READ: Array shape = [{}]", fmt::join(array_shape.asStdVector(), ", "));
                spdlog::info("DIRECT READ: Expected profile_size = {}, region_width = {}, region_height = {}", 
                            profile_size, region_width, region_height);
                spdlog::info("DIRECT READ: Mask shape = [{}], origin = [{}]", 
                            fmt::join(mask_shape.asStdVector(), ", "), fmt::join(origin.asStdVector(), ", "));
                
                // 重要：記錄ZARR的原始維度順序以便調試
                spdlog::warn("DIRECT READ: ZARR shape is [time={}, freq={}, ?={}, y={}, x={}]", 
                            shape.size() > 0 ? shape[0] : 0,
                            shape.size() > 1 ? shape[1] : 0, 
                            shape.size() > 2 ? shape[2] : 0,
                            shape.size() > 3 ? shape[3] : 0,
                            shape.size() > 4 ? shape[4] : 0);
                
                // 檢查第一個像素值和一些其他值
                if (full_region_array.size() > 0) {
                    const float* data_ptr = full_region_array.data();
                    auto first_value = *full_region_array.begin();
                    spdlog::warn("DIRECT READ: First pixel value = {} (array size = {})", first_value, full_region_array.size());
                    
                    // 檢查前幾個通道的第一個像素值和多個位置的像素值
                    if (array_shape.size() >= 2 && profile_size >= 3) {
                        // 對於ZARR格式，頻道維度的位置需要根據實際的數組形狀來確定
                        size_t ch0_idx, ch1_idx, ch2_idx;
                        
                        if (array_shape.size() == 5) {
                            // [time=1, freq=profile_size, ?=1, y=region_height, x=region_width]
                            ch0_idx = 0;  // 第0頻道第一個像素: [0,0,0,0,0]
                            ch1_idx = region_height * region_width;  // 第1頻道第一個像素: [0,1,0,0,0]
                            ch2_idx = 2 * region_height * region_width;  // 第2頻道第一個像素: [0,2,0,0,0]
                        } else if (array_shape.size() == 4) {
                            // [freq=profile_size, ?=1, y=region_height, x=region_width]
                            ch0_idx = 0;
                            ch1_idx = region_height * region_width;
                            ch2_idx = 2 * region_height * region_width;
                        } else {
                            // 回退到之前的邏輯
                            ch0_idx = 0;
                            ch1_idx = region_width * region_height;
                            ch2_idx = 2 * region_width * region_height;
                        }
                        
                        spdlog::warn("DIRECT READ: Raw data check - Ch0[0]={}, Ch1[0]={}, Ch2[0]={}", 
                                    data_ptr[ch0_idx], data_ptr[ch1_idx], data_ptr[ch2_idx]);
                        
                        // 檢查更多像素值以確認數據完整性
                        spdlog::warn("DIRECT READ: Sample pixels - Ch0[1]={}, Ch0[center]={}, Ch0[last]={}",
                                    array_shape[0] > 1 ? data_ptr[1] : data_ptr[0],
                                    data_ptr[region_width * region_height / 2],
                                    data_ptr[region_width * region_height - 1]);
                        
                        // 檢查第1頻道的更多位置
                        if (ch1_idx + region_width * region_height <= full_region_array.size()) {
                            spdlog::warn("DIRECT READ: Channel 1 samples - Ch1[1]={}, Ch1[center]={}, Ch1[last]={}",
                                        ch1_idx + 1 < full_region_array.size() ? data_ptr[ch1_idx + 1] : data_ptr[ch1_idx],
                                        ch1_idx + region_width * region_height / 2 < full_region_array.size() ? 
                                        data_ptr[ch1_idx + region_width * region_height / 2] : data_ptr[ch1_idx],
                                        ch1_idx + region_width * region_height - 1 < full_region_array.size() ? 
                                        data_ptr[ch1_idx + region_width * region_height - 1] : data_ptr[ch1_idx]);
                        }
                        
                        // 檢查數據範圍
                        float min_val = *std::min_element(data_ptr, data_ptr + full_region_array.size());
                        float max_val = *std::max_element(data_ptr, data_ptr + full_region_array.size());
                        spdlog::warn("DIRECT READ: Data range - min={}, max={}", min_val, max_val);
                        
                        // 統計非零值的數量
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
                    
                    // **修復**: CARTA數組是 Fortran 順序 (column-major)，所以 x 是最快變化的維度
                    // 循環順序應該是 x 在內層，y 在外層，以匹配 casacore::Array 的存儲順序
                    for (int y = 0; y < region_height; ++y) {
                        for (int x = 0; x < region_width; ++x) {
                            total_checked++;
                            
                            // 檢查 mask
                            bool mask_value = false;
                            if (x < mask_shape[0] && y < mask_shape[1]) {
                                mask_value = mask(casacore::IPosition(2, x, y));
                                if (mask_value) mask_true_count++;
                            }
                            
                            if (mask_value) {
                                float value;
                                
                                // **重要修復**: 針對ZARR格式的維度順序進行索引計算
                                // ZARR: [time, freq, ?, y, x] -> casacore可能重排為不同順序
                                const float* data_ptr = full_region_array.data();
                                
                                if (array_shape.size() == 5) {
                                    // 5D: [time=1, freq=profile_size, ?=1, y=region_height, x=region_width]
                                    // 線性索引：time + freq*1 + ?*1*profile_size + y*1*profile_size*1 + x*1*profile_size*1*region_height
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
                                    // 2D: [y=region_height, x=region_width] - 只有一個頻道
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
                                    
                                    // 記錄前幾個有效像素值進行調試
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
                    
                    // 為前幾個頻道和最後幾個頻道記錄詳細信息
                    if (z < 5 || z >= profile_size - 5 || z % 20 == 0) {
                        spdlog::info("DIRECT READ: Channel {} - checked {} pixels, mask_true {} pixels, valid {} pixels, sum = {}, mean = {}", 
                                    z, total_checked, mask_true_count, valid_count, sum, profile_data[z]);
                    }
                }
                
                // 添加總結性日誌顯示所有頻道處理完成
                spdlog::warn("DIRECT READ: Profile calculation completed for ALL {} channels (z_start={}, z_end={})", 
                            profile_size, z_start, z_end);
                spdlog::warn("DIRECT READ: First 5 channel means: [{}]", 
                            fmt::join(profile_data.begin(), profile_data.begin() + std::min(5, (int)profile_data.size()), ", "));
                if (profile_data.size() > 10) {
                    spdlog::warn("DIRECT READ: Last 5 channel means: [{}]", 
                                fmt::join(profile_data.end() - 5, profile_data.end(), ", "));
                }
                
                // 設置結果
                results[CARTA::StatsType::Mean] = profile_data;
                progress = 1.0;
                
                spdlog::debug("DIRECT READ OPTIMIZATION: Successfully processed {}x{} region across {} channels", 
                             region_width, region_height, profile_size);
                return true;
            } else {
                spdlog::warn("Direct read failed for small region, falling back to per-channel method");
                // 如果直接讀取失敗，繼續使用原來的逐頻道方法
            }
        }
        
        // **原來的逐頻道處理方法** (用於較大區域或直接讀取失敗時)
        for (int z = z_start; z <= z_end; ++z) {
            if (z >= num_channels) break;
            
            // 設定要讀取的區域 - 針對ZARR格式的維度順序
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
            
            // 讀取區域數據
            casacore::Array<float> region_array;
            if (!zarr_image->doGetSlice(region_array, casacore::Slicer(start, length))) {
                spdlog::error("ZarrLoader::GetRegionSpectralData: doGetSlice failed for channel {}", z);
                continue;
            }
            
            // 應用 mask 並計算統計
            double sum = 0.0;
            int valid_count = 0;
            
            auto region_iter = region_array.begin();
            for (int y = 0; y < region_height; ++y) {
                for (int x = 0; x < region_width; ++x) {
                    // 檢查 mask（相對於 mask 的坐標）
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
            
            // 計算平均值
            if (valid_count > 0) {
                profile_data[z - z_start] = sum / valid_count;
            } else {
                profile_data[z - z_start] = std::numeric_limits<double>::quiet_NaN();
            }
            
            // 更新進度
            progress = static_cast<float>(z - z_start + 1) / profile_size;
        }
        
        // 設置結果 - 假設只處理 MEAN 統計
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
    return false; // 暫時不實現
}

bool ZarrLoader::GetChunk(std::vector<float>& data, int& data_width, int& data_height,
    int min_x, int min_y, int z, int stokes, std::mutex& image_mutex) {
    
    std::lock_guard<std::mutex> lock(image_mutex);
    
    try {
        // 確保 CartaZarrImage 已創建
        if (!_image) {
            spdlog::error("ZarrLoader::GetChunk: No image available");
            return false;
        }
        
        // 將 CartaZarrImage 轉換為具體類型
        auto zarr_image = std::dynamic_pointer_cast<CartaZarrImage>(_image);
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetChunk: Image is not a CartaZarrImage");
            return false;
        }
        
        // 獲取圖像形狀
        casacore::IPosition shape = _image->shape();
        int img_width = shape[0];
        int img_height = shape[1];
        
        // Check that coordinates are within image bounds
        if (min_x < 0 || min_y < 0 || min_x >= img_width || min_y >= img_height) {
            spdlog::error("ZarrLoader::GetChunk: Coordinates out of bounds: min_x={}, min_y={} (image: {}x{})", 
                         min_x, min_y, img_width, img_height);
            return false;
        }
        
        // 計算實際要讀取的區域大小 - 這應該由我們的函數來決定，不是輸入參數
        // 遵循 HDF5Loader 的模式：使用 CHUNK_SIZE 常數
        data_width = std::min(CHUNK_SIZE, img_width - min_x);
        data_height = std::min(CHUNK_SIZE, img_height - min_y);
        
        // 輸出調試信息
        std::cout << "[ZARR DEBUG] GetChunk called: min_x=" << min_x << ", min_y=" << min_y 
                  << ", z=" << z << ", stokes=" << stokes << std::endl;
        std::cout << "[ZARR DEBUG] Calculated chunk size: " << data_width << "x" << data_height << std::endl;
        
        if (data_width <= 0 || data_height <= 0) {
            spdlog::warn("ZarrLoader::GetChunk: Invalid calculated chunk size: {}x{} at ({},{}) (image: {}x{})", 
                        data_width, data_height, min_x, min_y, img_width, img_height);
            return false;
        }
        
        // 輸出調試信息，與你看到的格式匹配
        std::cout << "[ZARR DEBUG] GetChunk request: start=[" << min_x << "," << min_y << "," << z << "," << stokes 
                  << "] length=[" << data_width << "," << data_height << ",1,1]" << std::endl;
        
        // 設定要讀取的區域 - 針對ZARR格式的維度順序
        casacore::IPosition start, length;
        if (shape.size() == 5) {
            // 5D ZARR: [time, freq, ?, y, x]
            start = casacore::IPosition(5, 0, z, 0, min_y, min_x);
            length = casacore::IPosition(5, 1, 1, 1, data_height, data_width);
        } else if (shape.size() == 4) {
            // 4D: [freq, ?, y, x]
            start = casacore::IPosition(4, z, 0, min_y, min_x);
            length = casacore::IPosition(4, 1, 1, data_height, data_width);
        } else if (shape.size() == 3) {
            // 3D: [freq, y, x]
            start = casacore::IPosition(3, z, min_y, min_x);
            length = casacore::IPosition(3, 1, data_height, data_width);
        } else if (shape.size() == 2) {
            // 2D: [y, x]
            start = casacore::IPosition(2, min_y, min_x);
            length = casacore::IPosition(2, data_height, data_width);
        } else {
            spdlog::error("ZarrLoader::GetChunk: Unsupported number of dimensions: {}", shape.size());
            return false;
        }
        
        // 使用 doGetSlice 讀取數據
        casacore::Array<float> chunk_array;
        if (!zarr_image->doGetSlice(chunk_array, casacore::Slicer(start, length))) {
            spdlog::error("ZarrLoader::GetChunk: doGetSlice failed");
            return false;
        }
        
        // 將數據複製到輸出向量
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
    return casacore::IPosition(); // 返回空的 IPosition
}

std::unique_ptr<casacore::ArrayBase> ZarrLoader::GetStatsData(FileInfo::Data ds) {
    return nullptr; // 暫時不實現
}
