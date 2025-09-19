/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"
#include "CartaZarrImage.h"
#include "Logger/Logger.h"

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
    // 使用基礎類別的 HasData 邏輯
    return FileLoader::HasData(ds);
}

bool ZarrLoader::HasMip(int mip) const {
    return false; // 暫時不支援 MipMap
}

bool ZarrLoader::UseTileCache() const {
    return true; // 使用 tile cache
}

bool ZarrLoader::GetCursorSpectralData(std::vector<float>& data, int stokes, int cursor_x, int count_x,
    int cursor_y, int count_y, std::mutex& image_mutex) {
    return false; // 暫時不實現
}

bool ZarrLoader::UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& image_mutex) {
    return false; // 暫時不實現
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
        
        // 計算實際要讀取的區域
        int end_x = std::min(min_x + data_width, img_width);
        int end_y = std::min(min_y + data_height, img_height);
        int actual_width = end_x - min_x;
        int actual_height = end_y - min_y;
        
        if (actual_width <= 0 || actual_height <= 0) {
            spdlog::warn("ZarrLoader::GetChunk: Invalid region [{},{}] to [{},{}]", 
                        min_x, min_y, end_x, end_y);
            return false;
        }
        
        // 輸出調試信息，與你看到的格式匹配
        std::cout << "[ZARR DEBUG] GetChunk request: start=[" << min_x << "," << min_y << "," << z << "," << stokes 
                  << "] length=[" << actual_width << "," << actual_height << ",1,1]" << std::endl;
        
        // 設定要讀取的區域
        casacore::IPosition start, length;
        if (shape.size() == 4) {
            start = casacore::IPosition(4, min_x, min_y, z, stokes);
            length = casacore::IPosition(4, actual_width, actual_height, 1, 1);
        } else if (shape.size() == 3) {
            start = casacore::IPosition(3, min_x, min_y, z);
            length = casacore::IPosition(3, actual_width, actual_height, 1);
        } else if (shape.size() == 2) {
            start = casacore::IPosition(2, min_x, min_y);
            length = casacore::IPosition(2, actual_width, actual_height);
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
        data.resize(actual_width * actual_height);
        
        casacore::Array<float>::const_iterator array_iter = chunk_array.begin();
        for (int i = 0; i < actual_width * actual_height; ++i) {
            data[i] = *array_iter;
            ++array_iter;
        }
        
        // 更新輸出參數
        data_width = actual_width;
        data_height = actual_height;
        
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
