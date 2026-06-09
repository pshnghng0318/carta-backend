/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrLoader.h"
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <queue>
#include <thread>
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
        spdlog::error("ZL::GetChunk: No valid ZARR image");
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZL::GetChunk: Reader not initialized");
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
        spdlog::error("ZL::GetCursorSpectralData: No valid ZARR image");
        return false;
    }
    std::shared_ptr<ZarrDataReader> reader = zarr_image_cursor->GetReader();

    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZL::GetCursorSpectralData: Reader not initialized");
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
    // size_t max_batch_for_updates = std::max<size_t>(freq_chunk, requested_depth / 8);
    size_t max_batch_for_updates = std::max<size_t>(freq_chunk, requested_depth);
    
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
    spdlog::debug("ZL::GetCursorSpectralData: z_batch={}, freq_chunk={}, requested_depth={}, z_start_in_data={}",
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
            spdlog::error("ZL::GetCursorSpectralData: ReadSlice failed");
            return false;
        }
    }

    bool delete_data_ptr(false);
    const float* data_ptr = batch_data.getStorage(delete_data_ptr);
    if (!data_ptr) {
        spdlog::error("ZL::GetCursorSpectralData: batch_data storage is null");
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
    // auto t_grsd_start = std::chrono::high_resolution_clock::now();
    spdlog::debug("ZL::GetRegionSpectralData: region_id={}, z=[{},{}], stokes={}, progress={:.3f}",
        region_id, z_range.from, z_range.to, stokes, progress);
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        spdlog::error("ZL::GetRegionSpectralData: No valid ZARR image");
        return false;
    }
    std::shared_ptr<ZarrDataReader> reader = zarr_image->GetReader();

    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZL::GetRegionSpectralData: Reader not initialized");
        return false;
    }

    if (cancellation_check && cancellation_check()) {
        return false;
    }

    // spec range = z range
    bool all_z = z_range.from == 0 && (z_range.to == ALL_Z || z_range.to == _dims.depth - 1);
    AxisRange spec_range(z_range.from, z_range.to);
    if (all_z) {
        spec_range.to = _dims.depth - 1;
    }

    // check region stats cache
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

    // set up region size and depth
    int width = mask_shape(0);
    int height = mask_shape(1);
    int depth = spec_range.to - spec_range.from + 1;
    if ((width <= 0) || (height <= 0) || (depth <= 0)) {
        return false;
    }

    // Look up or create the per-region cumulative stats entry
    // If the region origin or mask shape changed, the old entry is invalid and reset it
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

    // Bind references to each stat vector
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

    // latest_z is the resume point
    size_t z_start = region_stats.latest_z;
    if (z_start >= static_cast<size_t>(depth)) {
        results = stats;
        progress = 1.0;
        return true;
    }

    // Initialize stat arrays to neutral values
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

    // Determine batch size based on chunk size and target batch bytes
    size_t target_batch_bytes = carta::ProgramSettings::GetInstance().batch_MB * 1024 * 1024;
    spdlog::debug("ZL::GetRegionSpectralData: target_batch_bytes={} MB", target_batch_bytes / (1024 * 1024));
    size_t chunk_depth = 1;
    int freq_chunk = 0;
    // auto t_chunk_shape = std::chrono::high_resolution_clock::now();
    auto chunk_shape = reader->GetChunkShape();
    // spdlog::debug("ZarrLoader: GetChunkShape={} took {:.3f} ms",
    //     chunk_shape.toString(),
    //     std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_chunk_shape).count());
    if (chunk_shape.size() > 1) {
        freq_chunk = chunk_shape[1];
        if (freq_chunk < 0) {
            freq_chunk = depth;
        }
    }
    if (freq_chunk > 0) {
        chunk_depth = static_cast<size_t>(freq_chunk);
    }

    // Calculate how many channels per batch based on chunk depth and target batch size, then align to chunk boundaries
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
    
    //// Cap batch to enable more frequent progress updates (aim for ~4-8 updates)
    // size_t max_batch_for_updates = std::max<size_t>(chunk_depth, static_cast<size_t>(depth) / 8);
    // batch_depth = std::min(batch_depth, max_batch_for_updates);
    size_t max_batch_for_updates = std::max<size_t>(chunk_depth, static_cast<size_t>(depth));

    size_t max_z = std::min(static_cast<size_t>(depth), z_start + batch_depth);
    batch_depth = max_z - z_start;

    // Prefetch depth declared here so read_depth formula can reference it.
    const int    num_threads = std::max(1, omp_get_max_threads());  // consumer thread count
    const size_t queue_size  = static_cast<size_t>(std::max(1, (num_threads) / 2)); // producer queue size

    // read_depth per queue_size for TensorStore I/O
    size_t read_depth = chunk_depth; // default: one z-chunk per future
    if (chunk_shape.size() >= 5 && chunk_depth > 0 && bytes_per_chunk_depth > 0) {
        // chunk_shape is in raw TensorStore order [T, F, S, L, M].
        size_t chunk_l = static_cast<size_t>(chunk_shape[3]); // L: horizontal
        size_t chunk_m = static_cast<size_t>(chunk_shape[4]); // M: vertical
        if (chunk_l > 0 && chunk_m > 0) {
            // Number of spatial chunks the region spans (ceiling division).
            size_t covered_l = (static_cast<size_t>(width)  + chunk_l - 1) / chunk_l;
            size_t covered_m = (static_cast<size_t>(height) + chunk_m - 1) / chunk_m;
            size_t covered_spatial = std::max<size_t>(1, covered_l * covered_m);

            // Uncompressed bytes per 5D chunk (T=1, S=1 slices assumed).
            size_t chunk_bytes_5d = chunk_depth * chunk_l * chunk_m * sizeof(float);
            // Total bytes touched per z-chunk step across the whole region.
            size_t bytes_per_z_step = chunk_bytes_5d * covered_spatial;

            // Solve for how many z-chunks to pack into one future.
            size_t target_per_future = target_batch_bytes / std::max<size_t>(1, queue_size);
            size_t z_chunks_per_future = target_per_future / std::max<size_t>(1, bytes_per_z_step);
            z_chunks_per_future = std::max<size_t>(1, z_chunks_per_future);
            read_depth = z_chunks_per_future * chunk_depth;
        }
    }
    read_depth = std::min(read_depth, batch_depth); // never exceed batch boundary
    spdlog::debug("ZL:: read granularity: read_depth={} ch (chunk_depth={} ch, queue_size={})",
        read_depth, chunk_depth, queue_size);

    // Cache mask
    // casacore::ArrayLattice::getAt() is not thread-safe. Convert the mask to a
    // flat vector<char> here (single-threaded) so consumer threads can read it
    // without locks during the parallel stats phase.
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

    // Async prefetch producer-consumer pipeline
    //   producer_thread:
    //     buf = make_shared<Array>()
    //     result = reader->SubmitRead(buf, slicer)        // non-blocking: issues I/O immediately
    //     pkt_queue.push({buf, result, z_local, z_depth}) // blocks I/O if queue full (back-pressure)
    //   consumer_threads[0..N-1]  (N = num_threads)
    //     pkt_queue.pop(item)                             // pop item
    //     item.result()                                   // tensorstore read results
    //     serial stats(*item.buf, item.z_local_start, item.z_depth) // write to stats

    struct InFlightRead {
        std::shared_ptr<casacore::Array<float>> buf;  // heap-stable buffer (never moves)
        std::function<bool()>                   result;
        size_t z_local_start;
        size_t z_depth;
    };

    // Bounded blocking queue ─────────────────────────────────────────────────
    struct BoundedQueue {
        std::queue<InFlightRead>  q;
        std::mutex                mtx;
        std::condition_variable   cv_push;
        std::condition_variable   cv_pop;
        size_t                    cap;
        bool                      done = false;

        explicit BoundedQueue(size_t capacity) : cap(capacity) {}

        void push(InFlightRead&& item) {
            std::unique_lock<std::mutex> lk(mtx);
            cv_push.wait(lk, [&]{ return q.size() < cap || done; });
            if (done) return;
            q.push(std::move(item));
            cv_pop.notify_one();
        }

        bool pop(InFlightRead& item) {
            std::unique_lock<std::mutex> lk(mtx);
            cv_pop.wait(lk, [&]{ return !q.empty() || done; });
            if (q.empty()) return false;
            item = std::move(q.front());
            q.pop();
            cv_push.notify_one();
            return true;
        }

        void close() {
            { std::lock_guard<std::mutex> lk(mtx); done = true; }
            cv_push.notify_all();
            cv_pop.notify_all();
        }
    };
    // ──────────────────────────────────────────────────────────────────────────

    BoundedQueue pkt_queue(queue_size);
    std::atomic<bool> read_ok{true};
    std::atomic<bool> cancelled{false};

    spdlog::debug("ZL:: async prefetch pipeline (batch_depth={}, chunk_depth={}, read_depth={}, queue_size={})",
        batch_depth, chunk_depth, read_depth, queue_size);
    auto t_stats = std::chrono::high_resolution_clock::now();

    // ── Producer thread: issues SubmitRead (non-blocking) → puts future in queue ──
    std::thread producer_thread([&]() {
        size_t z_local = 0;
        while (z_local < batch_depth && read_ok.load() && !cancelled.load()) {
            size_t this_depth = std::min(read_depth, batch_depth - z_local);

            casacore::IPosition t_start(_num_dims, 0);
            casacore::IPosition t_length(_num_dims, 1);
            t_start(0)  = origin(0);
            t_start(1)  = origin(1);
            t_start(2)  = static_cast<int>(spec_range.from + z_start + z_local);
            t_length(0) = width;
            t_length(1) = height;
            t_length(2) = static_cast<int>(this_depth);
            if (_num_dims > 3) { t_start(3) = stokes; t_length(3) = 1; }

            // Heap-stable buffer: shared_ptr ensures the raw pointer captured by
            // TensorStore is never invalidated by a move or copy.
            auto buf = std::make_shared<casacore::Array<float>>();
            auto result = reader->SubmitRead(buf, casacore::Slicer(t_start, t_length));
            if (!result) {
                spdlog::error("ZL:: producer SubmitRead failed at z_local={}", z_local);
                read_ok.store(false);
                break;
            }

            // Put future (not data) in queue; blocks if queue is full (back-pressure).
            pkt_queue.push({std::move(buf), std::move(result), z_local, this_depth});
            z_local += this_depth;
        }
        pkt_queue.close();
    });

    // ── Consumer threads (N = num_threads) ─────────────────────────────────────
    // Loop: pkt_queue.pop(item) [blocks if empty] → item.result() [blocks until TensorStore
    // completes I/O + decompression] → iterate z in [z_local_start, z_local_start + z_depth)
    // and write stats[z_start + z_local_start + z]. Each item has a unique z_local_start
    // → writes go to disjoint indices → no mutex needed.
    auto consumer_fn = [&]() {
        while (true) {
            InFlightRead item;
            if (!pkt_queue.pop(item)) break;

            if (cancellation_check && cancellation_check()) {
                cancelled.store(true);
                pkt_queue.close();
                break;
            }

            // Block until TensorStore completes this read (I/O + decompression).
            if (!item.result()) {
                spdlog::error("ZL:: consumer: result() failed at z_local={}", item.z_local_start);
                read_ok.store(false);
                pkt_queue.close();
                break;
            }

            bool del_ptr = false;
            const float* data_ptr = item.buf->getStorage(del_ptr);
            if (!data_ptr) {
                spdlog::error("ZL:: consumer: data storage null");
                read_ok.store(false);
                pkt_queue.close();
                break;
            }

            const size_t zdepth = item.z_depth;
            const size_t z_off0 = item.z_local_start;

            for (size_t z = 0; z < zdepth; ++z) {
                size_t z_index  = z_start + z_off0 + z;
                size_t z_offset = z * plane_stride;

                double local_sum = 0.0, local_sum_sq = 0.0;
                double local_min = std::numeric_limits<double>::max();
                double local_max = std::numeric_limits<double>::lowest();
                uint64_t local_count = 0, local_nan = 0;

                for (size_t y = 0; y < h; ++y) {
                    size_t row_offset = z_offset + y * w;
                    size_t mask_row   = y * w;
                    for (size_t x = 0; x < w; ++x) {
                        if (!mask_cache[mask_row + x]) continue;
                        double v = static_cast<double>(data_ptr[row_offset + x]);
                        if (std::isfinite(v)) {
                            ++local_count;
                            local_sum    += v;
                            local_sum_sq += v * v;
                            local_min = std::min(v, local_min);
                            local_max = std::max(v, local_max);
                        } else {
                            ++local_nan;
                        }
                    }
                }

                num_pixels[z_index] = local_count;
                nan_count[z_index]  = local_nan;
                sum[z_index]        = local_sum;
                sum_sq[z_index]     = local_sum_sq;

                if (local_count > 0) {
                    min[z_index]     = local_min;
                    max[z_index]     = local_max;
                    mean[z_index]    = local_sum / local_count;
                    rms[z_index]     = sqrt(local_sum_sq / local_count);
                    sigma[z_index]   = local_count > 1
                        ? sqrt((local_sum_sq - (local_sum * local_sum / local_count)) / (local_count - 1)) : 0.0;
                    extrema[z_index] = (std::abs(local_min) > std::abs(local_max)) ? local_min : local_max;
                    if (has_flux) flux[z_index] = local_sum / beam_area;
                } else {
                    min[z_index] = max[z_index] = mean[z_index] = rms[z_index] =
                        sigma[z_index] = extrema[z_index] = NAN;
                    if (has_flux) flux[z_index] = NAN;
                }
            }

            item.buf->freeStorage(data_ptr, del_ptr);
        }
    };

    std::vector<std::thread> consumer_threads;
    consumer_threads.reserve(num_threads);
    for (int ci = 0; ci < num_threads; ++ci) {
        consumer_threads.emplace_back(consumer_fn);
    }

    // Main thread: wait for producer and all consumers
    producer_thread.join();
    for (auto& ct : consumer_threads) ct.join();

    if (cancelled.load()) {
        spdlog::info("ZL::GetRegionSpectralData: Cancelled during processing");
        return false;
    }
    if (!read_ok.load()) {
        return false;
    }
    spdlog::debug("ZL:: async prefetch pipeline read+stats for {} ch took {:.3f} ms",
        batch_depth,
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

    // spdlog::debug("ZL::GetRegionSpectralData: batch complete in {:.3f} ms total, progress={:.3f}",
    //     std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_grsd_start).count(), progress);
    return true;
}

void ZarrLoader::ClearRegionSpectralCache(int region_id) {
    spdlog::debug("ZL:: Clear Cache: region_id={}, cache_size={}", region_id, _region_stats.size());
    if (region_id == ALL_REGIONS) {
        _region_stats.clear();
        spdlog::debug("ZL:: cleared all {} entries", _region_stats.size());
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
        spdlog::error("ZL::GetSpatialProfileX: No valid ZARR image");
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZL::GetSpatialProfileX: Reader not initialized");
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
            spdlog::error("ZL::GetSpatialProfileX: ReadSlice failed");
            return false;
        }
        
        // Copy to profile vector
        profile.resize(width);
        std::copy(line_data.begin(), line_data.end(), profile.begin());
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("ZL::GetSpatialProfileX exception: {}", ex.what());
        return false;
    }
}

bool ZarrLoader::GetSpatialProfileY(std::vector<float>& profile, int cursor_x, 
                                     int start_y, int end_y, int channel, int stokes, 
                                     std::mutex& /*image_mutex*/) {
    auto* zarr_image = GetZarrImage();
    if (!zarr_image) {
        spdlog::error("ZL::GetSpatialProfileY: No valid ZARR image");
        return false;
    }
    
    auto reader = zarr_image->GetReader();
    if (!reader || !reader->IsInitialized()) {
        spdlog::error("ZL::GetSpatialProfileY: Reader not initialized");
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
            spdlog::error("ZL::GetSpatialProfileY: ReadSlice failed");
            return false;
        }
        
        // Copy to profile vector
        profile.resize(height);
        std::copy(line_data.begin(), line_data.end(), profile.begin());
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("ZL::GetSpatialProfileY exception: {}", ex.what());
        return false;
    }
}

} // namespace carta
