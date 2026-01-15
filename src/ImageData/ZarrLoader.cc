/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <spdlog/spdlog.h>

#include "Util/Image.h"

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
    return false;
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
    // Return false to use Frame.cc's incremental reading path with progress updates.
    // Frame.cc will call GetSlicerData which uses ReadSlice with multi-channel chunks.
    // This provides progress feedback for large spectral cubes.
    //
    // Note: ReadSpectralProfile is available for other use cases where batch reading 
    // is preferred (e.g., point region spectral profiles).
    spdlog::debug("ZarrLoader::GetCursorSpectralData: Using Frame.cc fallback path for progress updates");
    return false;
}

bool ZarrLoader::UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& image_mutex) {
    if (region_shape.size() < 2) {
        return false;
    }
    if ((region_shape(0) <= 0) || (region_shape(1) <= 0)) {
        return false;
    }
    if ((region_shape(0) == 1) && (region_shape(1) == 1)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(image_mutex);
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        return false;
    }

    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        return false;
    }

    return true;
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& spectral_range, int stokes,
    const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin, std::mutex& image_mutex,
    std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) {
    std::shared_ptr<ZarrDataReader> reader;
    {
        std::lock_guard<std::mutex> lock(image_mutex);
        auto* zarr_image = GetZarrImage();
        if (!zarr_image) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: No valid ZARR image");
            return false;
        }
        reader = zarr_image->GetReader();
    }

    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: Reader not initialized");
        return false;
    }

    bool all_z = spectral_range.from == 0 && (spectral_range.to == ALL_Z || spectral_range.to == _dims.depth - 1);
    AxisRange z_range(spectral_range.from, spectral_range.to);
    if (all_z) {
        z_range.to = _dims.depth - 1;
    }

    auto region_stats_id = FileInfo::RegionStatsId(region_id, stokes);
    casacore::IPosition mask_shape(mask.shape());
    if (_region_stats.count(region_stats_id) && _region_stats[region_stats_id].IsValid(origin, mask_shape) && all_z &&
        _region_stats[region_stats_id].IsCompleted()) {
        results = _region_stats[region_stats_id].stats;
        progress = 1.0;
        return true;
    }

    int width = mask_shape(0);
    int height = mask_shape(1);
    int depth = z_range.to - z_range.from + 1;
    if ((width <= 0) || (height <= 0) || (depth <= 0)) {
        return false;
    }

    double beam_area = CalculateBeamArea();
    bool has_flux = !std::isnan(beam_area);

    if (_region_stats.find(region_stats_id) == _region_stats.end()) {
        _region_stats.emplace(
            std::piecewise_construct, std::forward_as_tuple(region_id, stokes), std::forward_as_tuple(origin, mask_shape, depth, has_flux));
    } else if (!_region_stats[region_stats_id].IsValid(origin, mask_shape)) {
        _region_stats[region_stats_id] = FileInfo::RegionSpectralStats(origin, mask_shape, depth, has_flux);
    }

    auto& stats = _region_stats[region_stats_id].stats;
    auto& num_pixels = stats[CARTA::StatsType::NumPixels];
    auto& nan_count = stats[CARTA::StatsType::NanCount];
    auto& sum = stats[CARTA::StatsType::Sum];
    auto& mean = stats[CARTA::StatsType::Mean];
    auto& rms = stats[CARTA::StatsType::RMS];
    auto& sigma = stats[CARTA::StatsType::Sigma];
    auto& sum_sq = stats[CARTA::StatsType::SumSq];
    auto& min = stats[CARTA::StatsType::Min];
    auto& max = stats[CARTA::StatsType::Max];
    auto& extrema = stats[CARTA::StatsType::Extrema];
    double* flux = has_flux ? stats[CARTA::StatsType::FluxDensity].data() : nullptr;

    size_t z_start = _region_stats[region_stats_id].latest_x;
    if (z_start >= static_cast<size_t>(depth)) {
        results = stats;
        progress = 1.0;
        return true;
    }

    for (size_t z = 0; z < static_cast<size_t>(depth); ++z) {
        if ((z_start == 0) || (num_pixels[z] == 0)) {
            min[z] = std::numeric_limits<float>::max();
            max[z] = std::numeric_limits<float>::lowest();
            num_pixels[z] = 0;
            nan_count[z] = 0;
            sum[z] = 0;
            sum_sq[z] = 0;
        }
    }

    auto calculate_stats = [&]() {
        for (size_t z = 0; z < static_cast<size_t>(depth); ++z) {
            if (num_pixels[z]) {
                double sum_z = sum[z];
                double sum_sq_z = sum_sq[z];
                uint64_t num_pixels_z = num_pixels[z];

                mean[z] = sum_z / num_pixels_z;
                rms[z] = sqrt(sum_sq_z / num_pixels_z);
                sigma[z] = num_pixels_z > 1 ? sqrt((sum_sq_z - (sum_z * sum_z / num_pixels_z)) / (num_pixels_z - 1)) : 0;
                extrema[z] = (abs(min[z]) > abs(max[z]) ? min[z] : max[z]);
                if (has_flux) {
                    flux[z] = sum_z / beam_area;
                }
            } else {
                for (auto& kv : stats) {
                    switch (kv.first) {
                        case CARTA::StatsType::NanCount:
                        case CARTA::StatsType::NumPixels:
                            break;
                        default:
                            kv.second[z] = NAN;
                            break;
                    }
                }
            }
        }
    };

    constexpr size_t kTargetChunkBytes = 8 * 1024 * 1024;
    constexpr size_t kMaxZBatch = 4096;
    size_t bytes_per_z = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float);
    size_t delta_z = bytes_per_z > 0 ? kTargetChunkBytes / bytes_per_z : 1;
    if (delta_z < 1) {
        delta_z = 1;
    }
    if (delta_z > kMaxZBatch) {
        delta_z = kMaxZBatch;
    }
    size_t max_z = std::min(static_cast<size_t>(depth), z_start + delta_z);
    size_t chunk_depth = max_z - z_start;

    casacore::IPosition start(_num_dims, 0);
    casacore::IPosition length(_num_dims, 1);
    start(0) = origin(0);
    start(1) = origin(1);
    start(2) = z_range.from + z_start;
    length(0) = width;
    length(1) = height;
    length(2) = chunk_depth;
    if (_num_dims > 3) {
        start(3) = stokes;
        length(3) = 1;
    }

    casacore::Array<float> chunk_data;
    {
        std::lock_guard<std::mutex> lock(image_mutex);
        if (!reader->ReadSlice(casacore::Slicer(start, length), chunk_data)) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: ReadSlice failed");
            return false;
        }
    }

    bool delete_data_ptr(false);
    const float* data_ptr = chunk_data.getStorage(delete_data_ptr);
    if (!data_ptr) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: chunk_data storage is null");
        return false;
    }
    size_t plane_stride = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t z = 0; z < chunk_depth; ++z) {
        size_t z_index = z_start + z;
        size_t z_offset = z * plane_stride;
        for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
            size_t base = z_offset + y * static_cast<size_t>(width);
            for (size_t x = 0; x < static_cast<size_t>(width); ++x) {
                if (!mask.getAt(casacore::IPosition(2, x, y))) {
                    continue;
                }
                double v = data_ptr[base + x];
                if (std::isfinite(v)) {
                    num_pixels[z_index] += 1;
                    sum[z_index] += v;
                    sum_sq[z_index] += v * v;
                    min[z_index] = std::min(min[z_index], v);
                    max[z_index] = std::max(max[z_index], v);
                } else {
                    nan_count[z_index] += 1;
                }
            }
        }
    }
    chunk_data.freeStorage(data_ptr, delete_data_ptr);

    calculate_stats();

    results = stats;
    if (max_z == static_cast<size_t>(depth)) {
        progress = 1.0;
    } else {
        progress = static_cast<float>(max_z) / static_cast<float>(depth);
    }

    _region_stats[region_stats_id].latest_x = max_z;

    if (progress >= 1.0) {
        if (region_id <= TEMP_REGION_ID) {
            _region_stats.erase(region_stats_id);
        } else {
            _region_stats[region_stats_id].completed = true;
        }
    }

    return true;
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
