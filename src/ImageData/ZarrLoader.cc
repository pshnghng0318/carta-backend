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
    return reader && reader->IsInitialized();
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& z_range, int stokes,
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

    bool all_z = z_range.from == 0 && (z_range.to == ALL_Z || z_range.to == _dims.depth - 1);
    AxisRange spec_range(z_range.from, z_range.to);
    if (all_z) {
        spec_range.to = _dims.depth - 1;
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
    int depth = spec_range.to - spec_range.from + 1;
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

    auto& region_stats = _region_stats[region_stats_id];
    auto& stats = region_stats.stats;
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

    size_t z_start = region_stats.latest_z;
    if (z_start >= static_cast<size_t>(depth)) {
        results = stats;
        progress = 1.0;
        return true;
    }

    // Initialize all stats to NaN only on first batch
    if (z_start == 0) {
        for (size_t z = 0; z < static_cast<size_t>(depth); ++z) {
            num_pixels[z] = 0;
            nan_count[z] = 0;
            min[z] = NAN;
            max[z] = NAN;
            sum[z] = NAN;
            sum_sq[z] = NAN;
            mean[z] = NAN;
            rms[z] = NAN;
            sigma[z] = NAN;
            extrema[z] = NAN;
            if (has_flux) {
                flux[z] = NAN;
            }
        }
    }

    constexpr size_t target_batch_bytes = 64 * 1024 * 1024;
    size_t chunk_depth = 1;
    int freq_chunk = 0;
    auto chunk_shape = reader->GetChunkShape();
    if (chunk_shape.size() > 1) {
        freq_chunk = chunk_shape[1];
        if (freq_chunk < 0) {
            freq_chunk = depth;
        }
    }
    if (freq_chunk > 0) {
        chunk_depth = static_cast<size_t>(freq_chunk);
    }

    size_t bytes_per_chunk_depth =
        static_cast<size_t>(width) * static_cast<size_t>(height) * chunk_depth * sizeof(float);
    size_t chunks_per_batch = bytes_per_chunk_depth > 0 ? target_batch_bytes / bytes_per_chunk_depth : 1;
    chunks_per_batch = std::max<size_t>(chunks_per_batch, 1);

    size_t batch_depth = chunks_per_batch * chunk_depth;
    size_t absolute_z = static_cast<size_t>(spec_range.from) + z_start;
    size_t offset = chunk_depth > 0 ? absolute_z % chunk_depth : 0;
    if (offset != 0 && chunk_depth > 0) {
        size_t remainder = chunk_depth - offset;
        batch_depth = remainder + (chunks_per_batch - 1) * chunk_depth;
    }
    if (batch_depth == 0 && chunk_depth > 0) {
        batch_depth = chunk_depth;
    }

    size_t max_z = std::min(static_cast<size_t>(depth), z_start + batch_depth);
    batch_depth = max_z - z_start;

    casacore::IPosition start(_num_dims, 0);
    casacore::IPosition length(_num_dims, 1);
    start(0) = origin(0);
    start(1) = origin(1);
    start(2) = spec_range.from + z_start;
    length(0) = width;
    length(1) = height;
    length(2) = batch_depth;
    if (_num_dims > 3) {
        start(3) = stokes;
        length(3) = 1;
    }

    casacore::Array<float> batch_data;
    {
        std::lock_guard<std::mutex> lock(image_mutex);
        if (!reader->ReadSlice(casacore::Slicer(start, length), batch_data)) {
            spdlog::error("ZarrLoader::GetRegionSpectralData: ReadSlice failed");
            return false;
        }
    }

    bool delete_data_ptr(false);
    const float* data_ptr = batch_data.getStorage(delete_data_ptr);
    if (!data_ptr) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: batch_data storage is null");
        return false;
    }

    // Pre-cache mask to avoid repeated casacore::IPosition creation per pixel in hot loop
    // Use char instead of bool for faster access (no bit packing overhead)
    size_t w = static_cast<size_t>(width);
    size_t h = static_cast<size_t>(height);
    auto& mask_cache = region_stats.mask_cache;
    if (region_stats.mask_width != w || region_stats.mask_height != h || mask_cache.empty()) {
        region_stats.mask_width = w;
        region_stats.mask_height = h;
        mask_cache.assign(w * h, 0);
        casacore::IPosition pos(2);
        for (size_t y = 0; y < h; ++y) {
            pos(1) = y;
            for (size_t x = 0; x < w; ++x) {
                pos(0) = x;
                mask_cache[(y * w) + x] = mask.getAt(pos) ? 1 : 0;
            }
        }
    }

    size_t plane_stride = w * h;

    // Parallelize over z-axis: each channel's stats are independent
#pragma omp parallel for schedule(dynamic)
    for (size_t z = 0; z < batch_depth; ++z) {
        size_t z_index = z_start + z;
        size_t z_offset = z * plane_stride;

        double local_sum = 0.0;
        double local_sum_sq = 0.0;
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();
        uint64_t local_count = 0;
        uint64_t local_nan = 0;

        for (size_t y = 0; y < h; ++y) {
            size_t row_offset = z_offset + (y * w);
            size_t mask_row = y * w;
            for (size_t x = 0; x < w; ++x) {
                if (!mask_cache[mask_row + x]) {
                    continue;
                }
                double v = static_cast<double>(data_ptr[row_offset + x]);
                if (std::isfinite(v)) {
                    local_count++;
                    local_sum += v;
                    local_sum_sq += v * v;
                    local_min = std::min(v, local_min);
                    local_max = std::max(v, local_max);
                } else {
                    local_nan++;
                }
            }
        }

        // Write results and compute derived stats (each z_index is unique, no race condition)
        num_pixels[z_index] = local_count;
        nan_count[z_index] = local_nan;
        sum[z_index] = local_sum;
        sum_sq[z_index] = local_sum_sq;

        if (local_count > 0) {
            min[z_index] = local_min;
            max[z_index] = local_max;
            mean[z_index] = local_sum / local_count;
            rms[z_index] = sqrt(local_sum_sq / local_count);
            sigma[z_index] = local_count > 1 ? sqrt((local_sum_sq - (local_sum * local_sum / local_count)) / (local_count - 1)) : 0;
            extrema[z_index] = (std::abs(local_min) > std::abs(local_max) ? local_min : local_max);
            if (has_flux) {
                flux[z_index] = local_sum / beam_area;
            }
        } else {
            min[z_index] = NAN;
            max[z_index] = NAN;
            mean[z_index] = NAN;
            rms[z_index] = NAN;
            sigma[z_index] = NAN;
            extrema[z_index] = NAN;
            if (has_flux) {
                flux[z_index] = NAN;
            }
        }
    }
    batch_data.freeStorage(data_ptr, delete_data_ptr);

    results = stats;
    if (max_z == static_cast<size_t>(depth)) {
        progress = 1.0;
    } else {
        progress = static_cast<float>(max_z) / static_cast<float>(depth);
    }

    region_stats.latest_z = max_z;

    if (progress >= 1.0) {
        if (region_id <= TEMP_REGION_ID) {
            _region_stats.erase(region_stats_id);
        } else {
            region_stats.completed = true;
        }
    }

    return true;
}

void ZarrLoader::ClearRegionSpectralCache(int region_id) {
    if (region_id == ALL_REGIONS) {
        _region_stats.clear();
        return;
    }

    for (auto it = _region_stats.begin(); it != _region_stats.end();) {
        if (it->first.region_id == region_id) {
            it = _region_stats.erase(it);
        } else {
            ++it;
        }
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
