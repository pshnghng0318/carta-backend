/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>

#include <spdlog/spdlog.h>
#include <Main/ProgramSettings.h>
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
                          std::mutex& /*image_mutex*/) {
    
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

bool ZarrLoader::GetCursorSpectralData(std::vector<float>& data, const AxisRange& z_range, int stokes, int cursor_x, int count_x,
                                       int cursor_y, int count_y, std::mutex& /*image_mutex*/, float& progress) {
    if (count_x <= 0 || count_y <= 0) {
        return false;
    }

    auto* zarr_image_cursor = GetZarrImage();
    if (!zarr_image_cursor) {
        spdlog::error("ZarrLoader::GetCursorSpectralData: No valid ZARR image");
        return false;
    }
    std::shared_ptr<ZarrDataReader> reader = zarr_image_cursor->GetReader();

    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZarrLoader::GetCursorSpectralData: Reader not initialized");
        return false;
    }

    int total_depth = _dims.depth;
    if (total_depth <= 0) {
        return false;
    }

    AxisRange spec_range(z_range.from, z_range.to);
    if (spec_range.to == ALL_Z || spec_range.to >= total_depth) {
        spec_range.to = total_depth - 1;
    }

    int requested_depth = spec_range.to - spec_range.from + 1;
    if (requested_depth <= 0) {
        return false;
    }

    size_t expected_size = static_cast<size_t>(requested_depth * count_x * count_y);
    {
        std::lock_guard<std::mutex> guard(_cursor_profile_mutex);
        bool cache_match = _cursor_profile_cache.valid && (_cursor_profile_cache.stokes == stokes) &&
            (_cursor_profile_cache.cursor_x == cursor_x) && (_cursor_profile_cache.cursor_y == cursor_y) &&
            (_cursor_profile_cache.count_x == count_x) && (_cursor_profile_cache.count_y == count_y) &&
            (_cursor_profile_cache.z_from == spec_range.from) &&
            (_cursor_profile_cache.z_to == spec_range.to) &&
            (_cursor_profile_cache.data.size() == expected_size);

        if (progress > 0.0F && cache_match) {
            data = _cursor_profile_cache.data;
        } else {
            data.assign(expected_size, NAN);
            _cursor_profile_cache.valid = true;
            _cursor_profile_cache.stokes = stokes;
            _cursor_profile_cache.cursor_x = cursor_x;
            _cursor_profile_cache.cursor_y = cursor_y;
            _cursor_profile_cache.count_x = count_x;
            _cursor_profile_cache.count_y = count_y;
            _cursor_profile_cache.z_from = spec_range.from;
            _cursor_profile_cache.z_to = spec_range.to;
            _cursor_profile_cache.data = data;
        }
    }

    size_t z_start_in_data = static_cast<size_t>(progress * requested_depth);
    if (z_start_in_data >= static_cast<size_t>(requested_depth)) {
        progress = 1.0;
        return true;
    }

    // Determine batch size (similar to GetRegionSpectralData)
    size_t freq_chunk = requested_depth;
    auto chunk_shape = reader->GetChunkShape();
    if (chunk_shape.size() > 1) {
        freq_chunk = chunk_shape[1];
        if (freq_chunk <= 0) {
            freq_chunk = requested_depth;
        }
    }

    // Cap batch to enable more frequent progress updates (aim for ~4-8 updates)
    size_t max_batch_for_updates = std::max<size_t>(freq_chunk, requested_depth / 8);
    
    auto align_batch = [&](size_t batch_depth) {
        if (batch_depth == 0) {
            batch_depth = 1;
        }
        if (freq_chunk > 0) {
            batch_depth = (batch_depth / freq_chunk) * freq_chunk;
            if (batch_depth == 0) {
                batch_depth = freq_chunk;
            }
        }
        // Cap to max_batch_for_updates to ensure frequent progress updates
        batch_depth = std::min(batch_depth, max_batch_for_updates);
        return batch_depth;
    };

    size_t plane_size = static_cast<size_t>(count_x) * static_cast<size_t>(count_y);
    size_t z_batch = 0;
    {
        std::lock_guard<std::mutex> guard(_cursor_batch_mutex);
        if (_cursor_batch_state.plane_size != plane_size) {
            _cursor_batch_state = CursorBatchState();
            _cursor_batch_state.plane_size = plane_size;
        }

        if (_cursor_batch_state.sample_count >= 2 && _cursor_batch_state.avg_ms_per_channel > 0.0) {
            double target_ms = TARGET_PARTIAL_REGION_TIME;
            size_t desired_batch = static_cast<size_t>(std::round(target_ms / _cursor_batch_state.avg_ms_per_channel));
            desired_batch = std::max<size_t>(1, desired_batch);
            z_batch = align_batch(desired_batch);
        } else if (_cursor_batch_state.batch_depth == 0) {
            // For single pixel profiles, 64MB is huge (16M channels). Use smaller batches to allow progress updates.
            constexpr size_t target_profile_batch_bytes = 1 * 1024 * 1024; // 1MB for profiles
            // constexpr size_t target_batch_bytes = 64 * 1024 * 1024;
            size_t target_batch_bytes = carta::ProgramSettings::GetInstance().batch_MB * 1024 * 1024;
            size_t target_bytes = (plane_size == 1) ? target_profile_batch_bytes : target_batch_bytes;
            z_batch = target_bytes / (plane_size * sizeof(float));
            z_batch = align_batch(z_batch);
            if (plane_size == 1 && requested_depth > freq_chunk && z_batch >= static_cast<size_t>(requested_depth)) {
                size_t max_initial_batch = std::max<size_t>(freq_chunk, static_cast<size_t>(requested_depth / 4));
                z_batch = std::max<size_t>(1, align_batch(max_initial_batch));
            }
        } else {
            z_batch = _cursor_batch_state.batch_depth;
        }
    }

    z_batch = std::min<size_t>(z_batch, requested_depth - z_start_in_data);
    spdlog::debug("ZarrLoader::GetCursorSpectralData: z_batch={}, freq_chunk={}, requested_depth={}, z_start_in_data={}",
        z_batch, freq_chunk, requested_depth, z_start_in_data);

    casacore::IPosition start(_num_dims, 0);
    casacore::IPosition length(_num_dims, 1);
    start(0) = cursor_x;
    start(1) = cursor_y;
    start(2) = static_cast<int>(spec_range.from + z_start_in_data);
    length(0) = count_x;
    length(1) = count_y;
    length(2) = z_batch;
    if (_num_dims > 3) {
        start(3) = stokes;
        length(3) = 1;
    }

    casacore::Array<float> batch_data;
    auto t_batch_start = std::chrono::high_resolution_clock::now();
    {
        if (!reader->ReadSlice(batch_data, casacore::Slicer(start, length))) {
            spdlog::error("ZarrLoader::GetCursorSpectralData: ReadSlice failed");
            return false;
        }
    }

    bool delete_data_ptr(false);
    const float* data_ptr = batch_data.getStorage(delete_data_ptr);
    if (!data_ptr) {
        spdlog::error("ZarrLoader::GetCursorSpectralData: batch_data storage is null");
        return false;
    }

    // Copy batch data to the main data vector at the correct position
    std::copy(data_ptr, data_ptr + batch_data.nelements(), data.begin() + z_start_in_data * count_x * count_y);
    batch_data.freeStorage(data_ptr, delete_data_ptr);

    auto t_batch_end = std::chrono::high_resolution_clock::now();
    double dt_ms = std::chrono::duration<double, std::milli>(t_batch_end - t_batch_start).count();

    size_t next_batch = z_batch;
    if (dt_ms > 0.0) {
        double scale = TARGET_PARTIAL_REGION_TIME / dt_ms;
        scale = std::clamp(scale, 0.5, 2.0);
        next_batch = static_cast<size_t>(std::max<double>(1.0, std::round(z_batch * scale)));
        next_batch = align_batch(next_batch);
    }

    {
        std::lock_guard<std::mutex> guard(_cursor_batch_mutex);
        _cursor_batch_state.plane_size = plane_size;
        if (z_batch > 0 && dt_ms > 0.0) {
            double ms_per_channel = dt_ms / static_cast<double>(z_batch);
            if (_cursor_batch_state.sample_count < 4) {
                _cursor_batch_state.avg_ms_per_channel =
                    (_cursor_batch_state.avg_ms_per_channel * _cursor_batch_state.sample_count + ms_per_channel) /
                    static_cast<double>(_cursor_batch_state.sample_count + 1);
                _cursor_batch_state.sample_count++;
            } else {
                _cursor_batch_state.avg_ms_per_channel = (0.8 * _cursor_batch_state.avg_ms_per_channel) + (0.2 * ms_per_channel);
            }
        }
        _cursor_batch_state.batch_depth = next_batch;
        _cursor_batch_state.last_elapsed_ms = dt_ms;
    }

    {
        std::lock_guard<std::mutex> guard(_cursor_profile_mutex);
        if (_cursor_profile_cache.valid && _cursor_profile_cache.data.size() == data.size()) {
            _cursor_profile_cache.data = data;
        }
    }

    z_start_in_data += z_batch;
    progress = static_cast<float>(z_start_in_data) / requested_depth;
    progress = std::min<double>(progress, 1.0);

    return true;
}

bool ZarrLoader::UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& /*image_mutex*/) {
    if (region_shape.size() < 2) {
        return false;
    }
    if ((region_shape(0) <= 0) || (region_shape(1) <= 0)) {
        return false;
    }
    // Allow point regions to use loader path for progress-aware reads.

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
    return GetRegionSpectralData(region_id, z_range, stokes, mask, origin, image_mutex, results, progress, nullptr);
}

bool ZarrLoader::GetRegionSpectralData(int region_id, const AxisRange& z_range, int stokes,
    const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin, std::mutex& /*image_mutex*/,
    std::map<CARTA::StatsType, std::vector<double>>& results, float& progress,
    std::function<bool()> cancellation_check) {
    auto t_grsd_start = std::chrono::high_resolution_clock::now();
    spdlog::debug("ZarrLoader::GetRegionSpectralData: region_id={}, z=[{},{}], stokes={}, progress={:.3f}",
        region_id, z_range.from, z_range.to, stokes, progress);
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: No valid ZARR image");
        return false;
    }
    std::shared_ptr<ZarrDataReader> reader = zarr_image->GetReader();

    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZarrLoader::GetRegionSpectralData: Reader not initialized");
        return false;
    }

    if (cancellation_check && cancellation_check()) {
        return false;
    }

    bool all_z = z_range.from == 0 && (z_range.to == ALL_Z || z_range.to == _dims.depth - 1);
    AxisRange spec_range(z_range.from, z_range.to);
    if (all_z) {
        spec_range.to = _dims.depth - 1;
    }

    auto region_stats_id = FileInfo::RegionStatsId(region_id, stokes);
    casacore::IPosition mask_shape(mask.shape());
    std::shared_ptr<FileInfo::RegionSpectralStats> existing_stats_ptr;
    if (_region_stats.count(region_stats_id)) {
        existing_stats_ptr = _region_stats[region_stats_id];
    }

    if (existing_stats_ptr && existing_stats_ptr->IsValid(origin, mask_shape) && all_z &&
        existing_stats_ptr->IsCompleted()) {
        results = existing_stats_ptr->stats;
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

    std::shared_ptr<FileInfo::RegionSpectralStats> stats_ptr;
    if (_region_stats.find(region_stats_id) == _region_stats.end()) {
        stats_ptr = std::make_shared<FileInfo::RegionSpectralStats>(origin, mask_shape, depth, has_flux);
        _region_stats.emplace(region_stats_id, stats_ptr);
    } else {
        stats_ptr = _region_stats[region_stats_id];
        if (!stats_ptr->IsValid(origin, mask_shape)) {
            stats_ptr = std::make_shared<FileInfo::RegionSpectralStats>(origin, mask_shape, depth, has_flux);
            _region_stats[region_stats_id] = stats_ptr;
        }
    }

    auto& region_stats = *stats_ptr;
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

    // constexpr size_t target_batch_bytes = 64 * 1024 * 1024;
    size_t target_batch_bytes = carta::ProgramSettings::GetInstance().batch_MB * 1024 * 1024;
    spdlog::debug("ZarrLoader::GetRegionSpectralData: target_batch_bytes={} MB", target_batch_bytes / (1024 * 1024));
    size_t chunk_depth = 1;
    int freq_chunk = 0;
    auto t_chunk_shape = std::chrono::high_resolution_clock::now();
    auto chunk_shape = reader->GetChunkShape();
    spdlog::debug("ZarrLoader::GetRegionSpectralData: GetChunkShape={} took {:.3f} ms",
        chunk_shape.toString(),
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_chunk_shape).count());
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
    
    // Cap batch to enable more frequent progress updates (aim for ~4-8 updates)
    size_t max_batch_for_updates = std::max<size_t>(chunk_depth, static_cast<size_t>(depth) / 8);
    batch_depth = std::min(batch_depth, max_batch_for_updates);

    size_t max_z = std::min(static_cast<size_t>(depth), z_start + batch_depth);
    batch_depth = max_z - z_start;

    // Pre-cache mask before the parallel section (casacore API is not thread-safe)
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

    // Split the batch across OMP threads: each thread does ReadSlice + stats for its sub-batch.
    // e.g. batch_depth=100, omp_threads=8 → ~13 channels per thread, all threads run concurrently.
    const int omp_threads = carta::ProgramSettings::GetInstance().omp_thread_count;
    const size_t sub_depth =
        std::max<size_t>(1, (batch_depth + static_cast<size_t>(omp_threads) - 1) / static_cast<size_t>(omp_threads));
    const int actual_parts = static_cast<int>((batch_depth + sub_depth - 1) / sub_depth);
    spdlog::debug("ZarrLoader::GetRegionSpectralData: batch_depth={}, omp_threads={}, sub_depth={}, actual_parts={}",
        batch_depth, omp_threads, sub_depth, actual_parts);

    std::atomic<bool> read_ok{true};
    std::atomic<bool> cancelled{false};
    auto t_stats = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static) num_threads(actual_parts)
    for (int part = 0; part < actual_parts; ++part) {
        if (!read_ok.load() || cancelled.load()) continue;

        size_t z_thread_start = static_cast<size_t>(part) * sub_depth;
        size_t z_thread_end   = std::min(z_thread_start + sub_depth, batch_depth);
        size_t thread_depth   = z_thread_end - z_thread_start;
        if (thread_depth == 0) continue;

        casacore::IPosition t_start(_num_dims, 0);
        casacore::IPosition t_length(_num_dims, 1);
        t_start(0)  = origin(0);
        t_start(1)  = origin(1);
        t_start(2)  = static_cast<int>(spec_range.from + z_start + z_thread_start);
        t_length(0) = width;
        t_length(1) = height;
        t_length(2) = static_cast<int>(thread_depth);
        if (_num_dims > 3) {
            t_start(3)  = stokes;
            t_length(3) = 1;
        }

        casacore::Array<float> thread_data;
        {
            auto t_read = std::chrono::high_resolution_clock::now();
            bool slice_ok = reader->ReadSlice(thread_data, casacore::Slicer(t_start, t_length));
            spdlog::debug("ZarrLoader::GetRegionSpectralData [part {}/{}]: ReadSlice [{}x{}x{}] took {:.3f} ms",
                part, actual_parts, width, height, thread_depth,
                std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_read).count());
            if (!slice_ok) {
                spdlog::error("ZarrLoader::GetRegionSpectralData [part {}]: ReadSlice failed", part);
                read_ok.store(false);
                continue;
            }
        }

        if (cancellation_check && cancellation_check()) {
            cancelled.store(true);
            continue;
        }

        bool delete_thread_ptr = false;
        const float* data_ptr = thread_data.getStorage(delete_thread_ptr);
        if (!data_ptr) {
            spdlog::error("ZarrLoader::GetRegionSpectralData [part {}]: data storage is null", part);
            read_ok.store(false);
            continue;
        }

        for (size_t z = 0; z < thread_depth; ++z) {
            size_t z_index = z_start + z_thread_start + z;
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

            // Each z_index is unique across threads — no race condition on these writes
            num_pixels[z_index] = local_count;
            nan_count[z_index]  = local_nan;
            sum[z_index]        = local_sum;
            sum_sq[z_index]     = local_sum_sq;

            if (local_count > 0) {
                min[z_index]    = local_min;
                max[z_index]    = local_max;
                mean[z_index]   = local_sum / local_count;
                rms[z_index]    = sqrt(local_sum_sq / local_count);
                sigma[z_index]  = local_count > 1
                    ? sqrt((local_sum_sq - (local_sum * local_sum / local_count)) / (local_count - 1)) : 0;
                extrema[z_index] = (std::abs(local_min) > std::abs(local_max) ? local_min : local_max);
                if (has_flux) {
                    flux[z_index] = local_sum / beam_area;
                }
            } else {
                min[z_index]     = NAN;
                max[z_index]     = NAN;
                mean[z_index]    = NAN;
                rms[z_index]     = NAN;
                sigma[z_index]   = NAN;
                extrema[z_index] = NAN;
                if (has_flux) {
                    flux[z_index] = NAN;
                }
            }
        }

        thread_data.freeStorage(data_ptr, delete_thread_ptr);
    }

    if (cancelled.load()) {
        spdlog::info("ZarrLoader::GetRegionSpectralData: Cancelled during parallel processing");
        return false;
    }
    if (!read_ok.load()) {
        return false;
    }
    spdlog::debug("ZarrLoader::GetRegionSpectralData: parallel ReadSlice+stats for {} channels ({} parts) took {:.3f} ms",
        batch_depth, actual_parts,
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_stats).count());

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

    spdlog::debug("ZarrLoader::GetRegionSpectralData: batch complete in {:.3f} ms total, progress={:.3f}",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_grsd_start).count(), progress);
    return true;
}

void ZarrLoader::ClearRegionSpectralCache(int region_id) {
    spdlog::debug("ZarrLoader::ClearRegionSpectralCache: region_id={}, cache_size={}", region_id, _region_stats.size());
    if (region_id == ALL_REGIONS) {
        _region_stats.clear();
        spdlog::debug("ZarrLoader::ClearRegionSpectralCache: cleared all {} entries", _region_stats.size());
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
                                     std::mutex& /*image_mutex*/) {
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
        if (!reader->ReadSlice(line_data, section)) {
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
                                     std::mutex& /*image_mutex*/) {
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
        if (!reader->ReadSlice(line_data, section)) {
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
