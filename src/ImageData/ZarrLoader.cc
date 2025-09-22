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
        int num_channels = (shape.size() > 2) ? shape[2] : 1;
        
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
            
            if (shape.size() == 4) {
                // 4D: [x, y, freq, stokes]
                start = casacore::IPosition(4, cursor_x, cursor_y, z, stokes);
                length = casacore::IPosition(4, 1, 1, 1, 1);
            } else if (shape.size() == 3) {
                // 3D: [x, y, freq] 
                start = casacore::IPosition(3, cursor_x, cursor_y, z);
                length = casacore::IPosition(3, 1, 1, 1);
            } else if (shape.size() == 2) {
                // 2D: [x, y]
                start = casacore::IPosition(2, cursor_x, cursor_y);
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
    // 對於點區域（1x1），使用優化的讀取方法
    // 這樣可以避免讀取每個頻道的完整圖像
    if (region_shape.size() >= 2 && region_shape[0] == 1 && region_shape[1] == 1) {
        spdlog::debug("UseRegionSpectralData: Using optimized point spectral data for 1x1 region");
        return true;
    }
    
    // 對於較大的區域，仍使用默認的圖像切片方法
    spdlog::debug("UseRegionSpectralData: Using default image slicing for {}x{} region", 
                 region_shape[0], region_shape[1]);
    return false;
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& spectral_range, int stokes,
    const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin,
    std::mutex& image_mutex, std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) {
    return false; // 暫時不實現
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
        
        // 設定要讀取的區域
        casacore::IPosition start, length;
        if (shape.size() == 4) {
            start = casacore::IPosition(4, min_x, min_y, z, stokes);
            length = casacore::IPosition(4, data_width, data_height, 1, 1);
        } else if (shape.size() == 3) {
            start = casacore::IPosition(3, min_x, min_y, z);
            length = casacore::IPosition(3, data_width, data_height, 1);
        } else if (shape.size() == 2) {
            start = casacore::IPosition(2, min_x, min_y);
            length = casacore::IPosition(2, data_width, data_height);
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
