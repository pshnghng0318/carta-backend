/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"

#include <spdlog/spdlog.h>

namespace carta {

ZarrLoader::ZarrLoader(const std::string& filename) : FileLoader(filename) {}

void ZarrLoader::AllocateImage(const std::string& hdu) {
    if (!_image) {
        try {
            _image.reset(new CartaZarrImage(_filename));
            
            _image_shape = _image->shape();
            _num_dims = _image_shape.size();
            _coord_sys = std::shared_ptr<casacore::CoordinateSystem>(
                static_cast<casacore::CoordinateSystem*>(_image->coordinates().clone()));
            _data_type = _image->dataType();
            _has_pixel_mask = _image->hasPixelMask();
            
            spdlog::debug("ZarrLoader: Allocated image with shape {}", _image_shape.toString());
            
        } catch (const casacore::AipsError& err) {
            spdlog::error("ZarrLoader: Failed to allocate image: {}", err.getMesg());
            throw;
        }
    }
}

bool ZarrLoader::HasData(FileInfo::Data data_type) const {
    switch (data_type) {
        case FileInfo::Data::Image:
            return true;
        case FileInfo::Data::XY:
            return _image && _image->shape().size() >= 2;
        default:
            // ZARR doesn't have pre-computed statistics like HDF5
            return false;
    }
}

bool ZarrLoader::HasMip(int mip_level) const {
    // ZARR loader doesn't support mipmaps (yet)
    return false;
}

bool ZarrLoader::UseTileCache() const {
    // Enable tile cache for efficient tile-based rendering
    return true;
}

CartaZarrImage* ZarrLoader::GetZarrImage() {
    return dynamic_cast<CartaZarrImage*>(_image.get());
}

bool ZarrLoader::GetChunk(std::vector<float>& data, int& data_width, int& data_height,
                          int min_x, int min_y, int channel, int stokes, 
                          std::mutex& image_mutex) {
    std::lock_guard<std::mutex> lock(image_mutex);
    
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        spdlog::error("ZarrLoader::GetChunk: No valid ZARR image");
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZarrLoader::GetChunk: Reader not initialized");
        return false;
    }
    
    return reader->GetChunk(data, data_width, data_height, min_x, min_y, channel, stokes);
}

bool ZarrLoader::GetCursorSpectralData(std::vector<float>& data, int stokes, 
                                        int cursor_x, int count_x,
                                        int cursor_y, int count_y, 
                                        std::mutex& image_mutex) {
    std::lock_guard<std::mutex> lock(image_mutex);
    
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        return false;
    }
    
    const auto& shape = reader->GetShape();
    if (shape.size() < 3) {
        spdlog::warn("ZarrLoader::GetCursorSpectralData: Image has < 3 dimensions");
        return false;
    }
    
    int num_channels = shape[2];  // Frequency axis
    data.resize(num_channels * count_x * count_y);
    
    // Read spectral data for each channel
    try {
        for (int chan = 0; chan < num_channels; ++chan) {
            casacore::IPosition start(4, cursor_x, cursor_y, chan, stokes);
            casacore::IPosition length(4, count_x, count_y, 1, 1);
            casacore::Slicer section(start, length);
            
            casacore::Array<float> channel_data;
            if (!reader->ReadSlice(section, channel_data)) {
                return false;
            }
            
            // Copy to output
            size_t offset = chan * count_x * count_y;
            std::copy(channel_data.begin(), channel_data.end(), data.begin() + offset);
        }
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("ZarrLoader::GetCursorSpectralData exception: {}", ex.what());
        return false;
    }
}

bool ZarrLoader::GetSpatialProfileX(std::vector<float>& profile, int start_x, int end_x, 
                                     int cursor_y, int channel, int stokes, 
                                     std::mutex& image_mutex) {
    std::lock_guard<std::mutex> lock(image_mutex);
    
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        spdlog::error("ZarrLoader::GetSpatialProfileX: No valid ZARR image");
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZarrLoader::GetSpatialProfileX: Reader not initialized");
        return false;
    }
    
    try {
        // Read a horizontal line at cursor_y
        int width = end_x - start_x + 1;
        casacore::IPosition start(4, start_x, cursor_y, channel, stokes);
        casacore::IPosition length(4, width, 1, 1, 1);
        casacore::Slicer section(start, length);
        
        casacore::Array<float> line_data;
        if (!reader->ReadSlice(section, line_data)) {
            spdlog::error("ZarrLoader::GetSpatialProfileX: ReadSlice failed");
            return false;
        }
        
        // Copy to profile vector
        profile.resize(width);
        std::copy(line_data.begin(), line_data.end(), profile.begin());
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("ZarrLoader::GetSpatialProfileX exception: {}", ex.what());
        return false;
    }
}

bool ZarrLoader::GetSpatialProfileY(std::vector<float>& profile, int cursor_x, 
                                     int start_y, int end_y, int channel, int stokes, 
                                     std::mutex& image_mutex) {
    std::lock_guard<std::mutex> lock(image_mutex);
    
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        spdlog::error("ZarrLoader::GetSpatialProfileY: No valid ZARR image");
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZarrLoader::GetSpatialProfileY: Reader not initialized");
        return false;
    }
    
    try {
        // Read a vertical line at cursor_x
        int height = end_y - start_y + 1;
        casacore::IPosition start(4, cursor_x, start_y, channel, stokes);
        casacore::IPosition length(4, 1, height, 1, 1);
        casacore::Slicer section(start, length);
        
        casacore::Array<float> line_data;
        if (!reader->ReadSlice(section, line_data)) {
            spdlog::error("ZarrLoader::GetSpatialProfileY: ReadSlice failed");
            return false;
        }
        
        // Copy to profile vector
        profile.resize(height);
        std::copy(line_data.begin(), line_data.end(), profile.begin());
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("ZarrLoader::GetSpatialProfileY exception: {}", ex.what());
        return false;
    }
}

} // namespace carta
