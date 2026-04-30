/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrDataReader.h"

// Standard library includes MUST come before TensorStore to ensure types are defined
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>
#include <type_traits>
#include <vector>

#include <malloc.h>
#include <omp.h>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

// TensorStore includes - isolated to implementation file
#include "contiguous_layout.h"
#include "Main/ProgramSettings.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/context.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"

namespace carta {

// Renamed to avoid macro collision
#ifndef K_TILE_SIZE
constexpr int K_TILE_SIZE = 256;
#endif

// Cache size for TensorStore shared context.
constexpr size_t kDefaultCacheSizeMB = 1;
constexpr size_t kDefaultCpuCount = 4;
// Number of parallel read partitions for spatial splits
// constexpr int kParallelReadParts = 1;
constexpr size_t kDimSize5D = 5;
// constexpr int kDefaultStripeHeight = 256;
// constexpr int kStripeChunkMultiplier = 8; // Read multiple chunks per stripe to reduce overhead

// Maximum data size per column batch in MiB for ReadChannelSliceV2
// Single chunk size is taken as minimum to avoid reading same chunk multiple times
constexpr size_t kColumnBatchMaxDataMiB = 16;

//-----------------------------------------------------------------------------
// Pimpl Implementation Helper
//-----------------------------------------------------------------------------
struct ZarrDataReader::Impl {
    tensorstore::TensorStore<> store;

    // Cached .zmetadata
    nlohmann::json zmetadata;
    bool has_zmetadata = false;

    // Returns the configured OMP thread count from ProgramSettings (or a safe default)
    static int GetOmpThreadCount() {
        int omp_threads = 4;
        try {
            omp_threads = carta::ProgramSettings::GetInstance().omp_thread_count;
        } catch (...) {
            omp_threads = 4;
        }
        if (omp_threads <= 0) omp_threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : kDefaultCpuCount;
        return omp_threads;
    }

    // Get shared TensorStore context (created once, reused by all instances)
    // This saves memory (single cache pool) and reduces initialization overhead
    static tensorstore::Context GetSharedContext() {
        static tensorstore::Context shared_context = []() {
            int omp_threads = GetOmpThreadCount();

            int file_io_conc = 2;

            // Runtime changes from Program Settings
            try {
                file_io_conc = carta::ProgramSettings::GetInstance().file_io;
                if (file_io_conc >= std::thread::hardware_concurrency()) {
                    file_io_conc = std::thread::hardware_concurrency() - omp_threads;
                }
                if (file_io_conc < 1) {
                    file_io_conc = 1;
                }
            } catch (...) {
                file_io_conc = 2;
            }

            int data_copy_conc = omp_threads;
            
            if (file_io_conc + data_copy_conc > std::thread::hardware_concurrency()) {
                spdlog::warn("file_io_concurrency ({}) + data_copy_concurrency ({}) exceeds hardware concurrency ({}), adjusting data_copy_concurrency",
                    file_io_conc, data_copy_conc, std::thread::hardware_concurrency());
                data_copy_conc = std::thread::hardware_concurrency() - file_io_conc;
            }
            if (data_copy_conc < 1) data_copy_conc = 1;

            size_t cache_pool_MB = static_cast<size_t>(carta::ProgramSettings::GetInstance().cache_pool);
            if (cache_pool_MB < 0) cache_pool_MB = kDefaultCacheSizeMB;

            nlohmann::json context_spec = {
                {"cache_pool", {{"total_bytes_limit", cache_pool_MB * 1024 * 1024}}},
                {"data_copy_concurrency", {{"limit", data_copy_conc}}},
                {"file_io_concurrency", {{"limit", file_io_conc}}}
            };

            auto context_result = tensorstore::Context::FromJson(context_spec);
            if (context_result.ok()) {
                spdlog::info("Created shared TensorStore context with omp_threads={}, data_copy_concurrency={}, file_io_concurrency={}, cache={}MB", omp_threads, data_copy_conc, file_io_conc, cache_pool_MB);
                return context_result.value();
            }
            spdlog::warn("Failed to create custom context, using default");
            return tensorstore::Context::Default();
        }();
        return shared_context;
    }

    // Early dimension validation before creating expensive TensorStore context
    // Returns: 0 = success/skip, -1 = failed (wrong dimension)
    static int ValidateEarlyDimension(const std::string& array_path) {
        std::filesystem::path zarray_path = std::filesystem::path(array_path) / ".zarray";
        if (!std::filesystem::exists(zarray_path)) {
            return 0; // Skip check if file doesn't exist
        }
        try {
            std::ifstream zarray_file(zarray_path);
            nlohmann::json zarray;
            zarray_file >> zarray;
            if (zarray.contains("shape")) {
                size_t ndim = zarray["shape"].size();
                if (ndim != kDimSize5D) {
                    spdlog::error("XRADIO schema requires 5D array, got {}D (early check)", ndim);
                    return -1;
                }
            }
        } catch (const std::exception& ex) {
            spdlog::debug("Early dimension check skipped: {}", ex.what());
        }
        return 0;
    }

    // Load .zmetadata file if present
    bool LoadZmetadata(const std::string& filename) {
        std::filesystem::path zmetadata_path = std::filesystem::path(filename) / ".zmetadata";
        if (!std::filesystem::exists(zmetadata_path)) {
            return false;
        }
        try {
            std::ifstream file(zmetadata_path);
            file >> zmetadata;
            has_zmetadata = true;
            spdlog::info("Loaded .zmetadata from {}", filename);
            return true;
        } catch (const std::exception& ex) {
            spdlog::warn("Failed to parse .zmetadata: {}", ex.what());
            has_zmetadata = false;
            return false;
        }
    }

    // Validate _ARRAY_DIMENSIONS attribute
    void ValidateArrayDimensions(const std::string& fallback_zattrs_str) {
        try {
            nlohmann::json zattrs;
            if (has_zmetadata) {
                // Direct access from cached metadata - no string serialization/parsing overhead
                std::string key = "SKY/.zattrs";
                if (zmetadata.contains("metadata") && zmetadata["metadata"].contains(key)) {
                    zattrs = zmetadata["metadata"][key];
                }
            } else if (!fallback_zattrs_str.empty() && fallback_zattrs_str != "{}") {
                zattrs = nlohmann::json::parse(fallback_zattrs_str);
            }

            if (zattrs.empty() || !zattrs.contains("_ARRAY_DIMENSIONS")) {
                return;
            }

            std::vector<std::string> dims = zattrs["_ARRAY_DIMENSIONS"].get<std::vector<std::string>>();
            if (dims.size() != kDimSize5D) {
                return;
            }

            // Check for deviations from standard XRADIO order [time, freq, pol, l, m]
            bool standard_order = (dims[0] == "time") && (dims[1] == "frequency") && (dims[2] == "polarization") &&
                                  (dims[3] == "l" || dims[3] == "u") && (dims[4] == "m" || dims[4] == "v");
            if (!standard_order) {
                spdlog::warn("Effectively assuming [t, f, p, l, m] but _ARRAY_DIMENSIONS are: {}", fmt::join(dims, ", "));
            }
        } catch (const std::exception& ex) {
            spdlog::debug("_ARRAY_DIMENSIONS validation skipped: {}", ex.what());
        } catch (...) {
            spdlog::debug("_ARRAY_DIMENSIONS validation skipped: unknown error");
        }
    }

    // Map Coordinates Helper - XRADIO schema: always 5D [T, F, S, L, M]
    // Zarr is C-order (row-major): M (index 4) changes fastest
    // CARTA is Fortran-order (column-major): X (index 0) changes fastest
    // L is the horizontal axis (X), M is the vertical axis (Y)
    static std::vector<tensorstore::Index> MapToZarrCoords(const casacore::IPosition& start) {
        std::vector<tensorstore::Index> res(kDimSize5D, 0);
        res[0] = 0; // T (always 0 for CARTA)
        if (start.size() > 2) {
            res[1] = start[2];
        } // F
        if (start.size() > 3) {
            res[2] = start[3];
        } // S
        if (start.size() > 0) {
            res[3] = start[0];
        } // L (CARTA X)
        if (start.size() > 1) {
            res[4] = start[1];
        } // M (CARTA Y)
        return res;
    }
};

//-----------------------------------------------------------------------------
// ZarrDataReader Implementation
//-----------------------------------------------------------------------------

ZarrDataReader::ZarrDataReader(const std::string& filename) : _filename(filename), _impl(std::make_unique<Impl>()) {}

ZarrDataReader::~ZarrDataReader() = default;

// Accessors implementations
bool ZarrDataReader::IsInitialized() const {
    return _initialized;
}
const casacore::IPosition& ZarrDataReader::GetShape() const {
    return _shape;
}
const casacore::IPosition& ZarrDataReader::GetOriginalZarrShape() const {
    return _original_shape;
}
const casacore::IPosition& ZarrDataReader::GetChunkShape() const {
    return _chunk_shape;
}
const std::string& ZarrDataReader::GetFilename() const {
    return _filename;
}
int ZarrDataReader::NumDimensions() const {
    return _shape.size();
}

std::string ZarrDataReader::FindArrayPath() const {
    std::filesystem::path base_path(_filename);

    std::vector<std::string> common_array_names = {"SKY", "APERTURE"};
    for (const auto& array_name : common_array_names) {
        auto potential_path = base_path / array_name;
        if (std::filesystem::exists(potential_path / ".zarray")) {
            spdlog::debug("Found Zarr array in subdirectory: {}", potential_path.string());
            return potential_path.string();
        }
    }

    if (std::filesystem::exists(base_path / ".zarray")) {
        return _filename;
    }

    spdlog::error("No .zarray file found in {} or its subdirectories", _filename);
    return "";
}

bool ZarrDataReader::Initialize() {
    if (_initialized) {
        return true;
    }

    try {
        std::string array_path = FindArrayPath();
        if (array_path.empty()) {
            return false;
        }

        // Early dimension check before creating expensive TensorStore context
        if (Impl::ValidateEarlyDimension(array_path) < 0) {
            return false;
        }

        // Try to read .zmetadata
        _impl->LoadZmetadata(_filename);

        nlohmann::json spec_json = {{"driver", "zarr"}, {"kvstore", {{"driver", "file"}, {"path", array_path}}}};

        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            spdlog::error("Failed to create TensorStore spec: {}", spec_result.status().ToString());
            return false;
        }

        // Use shared context for all ZarrDataReader instances (saves memory and init time)
        auto open_future =
            tensorstore::Open(spec_result.value(), Impl::GetSharedContext(), tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

        auto open_result = open_future.result();
        if (!open_result.ok()) {
            spdlog::error("Failed to open TensorStore: {}", open_result.status().ToString());
            return false;
        }

        _impl->store = std::move(open_result).value();

        auto domain = _impl->store.domain();
        auto ts_shape = domain.shape();

        // Pre-allocate vector for original shape
        std::vector<int> orig_shape_vec;
        orig_shape_vec.reserve(ts_shape.size());
        for (size_t i = 0; i < ts_shape.size(); ++i) {
            orig_shape_vec.push_back(static_cast<int>(ts_shape[i]));
        }
        _original_shape = casacore::IPosition(orig_shape_vec);

        // Validate _ARRAY_DIMENSIONS if present (uses cached zmetadata if available)
        std::string fallback_zattrs = _impl->has_zmetadata ? "" : GetZattrsString("");
        _impl->ValidateArrayDimensions(fallback_zattrs);

        // XRADIO schema: always 5D [T, F, P, L, M]
        if (ts_shape.size() != kDimSize5D) {
            spdlog::error("XRADIO schema requires 5D array, got {}D", ts_shape.size());
            return false;
        }

        // Map to CARTA shape [X, Y, F, S]
        // L (index 3) is the horizontal axis -> X
        // M (index 4) is the vertical axis -> Y
        // Pre-allocate with exact size needed
        std::vector<int> carta_shape;
        carta_shape.reserve(4);
        carta_shape.push_back(orig_shape_vec[3]); // L -> X
        carta_shape.push_back(orig_shape_vec[4]); // M -> Y
        carta_shape.push_back(orig_shape_vec[1]); // F -> F
        carta_shape.push_back(orig_shape_vec[2]); // P -> S
        _shape = casacore::IPosition(carta_shape);

        auto chunk_layout_result = _impl->store.chunk_layout();
        if (chunk_layout_result.ok()) {
            auto read_chunk_shape = chunk_layout_result.value().read_chunk_shape();
            if (!read_chunk_shape.empty()) {
                std::vector<int> chunk_shape_vec;
                chunk_shape_vec.reserve(read_chunk_shape.size());
                for (auto chunk_dim_size : read_chunk_shape) {
                    chunk_shape_vec.push_back(static_cast<int>(chunk_dim_size));
                }
                _chunk_shape = casacore::IPosition(chunk_shape_vec);
                spdlog::debug("ZarrDataReader chunk shape (read): {}", _chunk_shape.toString());
            }
        } else {
            spdlog::debug("ZarrDataReader: chunk_layout unavailable: {}", chunk_layout_result.status().ToString());
        }

        _initialized = true;
        spdlog::debug("ZarrDataReader initialized: ZARR shape={}, CARTA shape={}", _original_shape.toString(), _shape.toString());

        return true;

    } catch (const std::exception& ex) {
        spdlog::error("Exception initializing ZarrDataReader: {}", ex.what());
        return false;
    }
}

bool ZarrDataReader::ReadSlice(casacore::Array<float>& buffer, const casacore::Slicer& section) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }

    const auto& start = section.start();
    const auto& stop = section.end();
    const auto& length = section.length();

    spdlog::debug("ZarrDataReader::ReadSlice: start={}, stop={}, length={}", start.toString(), stop.toString(), length.toString());

    auto t_rs_0 = std::chrono::high_resolution_clock::now();
    buffer.resize(length);
    spdlog::debug("ZarrDataReader::ReadSlice [1/5] buffer.resize took {:.3f} ms",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_rs_0).count());

    const int width_x = length[0];
    const int height_y = length[1];
    const int num_freq = length[2];
    const int num_stokes = length[3];

    // Time index is always 0 for now.
    const tensorstore::Index time_idx = 0;

    // Let TensorStore manage its own internal parallelism (data_copy_concurrency,
    // file_io_concurrency). Issue a single read for the whole slice.
    float* dst_ptr = buffer.data();

    auto t_rs_1 = std::chrono::high_resolution_clock::now();
    auto slice_result = _impl->store
        | tensorstore::Dims(0).IndexSlice(time_idx)
        | tensorstore::Dims(0).ClosedInterval(start[2], stop[2])   // F
        | tensorstore::Dims(1).ClosedInterval(start[3], stop[3])   // P
        | tensorstore::Dims(2).ClosedInterval(start[0], stop[0])   // L
        | tensorstore::Dims(3).ClosedInterval(start[1], stop[1]);  // M
    spdlog::debug("ZarrDataReader::ReadSlice [2/5] store slicing (F={}-{}, L={}-{}, M={}-{}) took {:.3f} ms",
        start[2], stop[2], start[0], stop[0], start[1], stop[1],
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_rs_1).count());

    if (!slice_result.ok()) {
        spdlog::error("ReadSlice slice failed: {}", slice_result.status().ToString());
        return false;
    }

    // Reorder [F, P, L, M] → [L, M, F, P] to match CARTA Fortran-order layout [x, y, z, stokes]
    auto t_rs_2 = std::chrono::high_resolution_clock::now();
    auto reorder_result = std::move(slice_result).value() | tensorstore::Dims(2, 3, 0, 1).Transpose();
    spdlog::debug("ZarrDataReader::ReadSlice [3/5] Dims Transpose took {:.3f} ms",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_rs_2).count());

    if (!reorder_result.ok()) {
        spdlog::error("ReadSlice reorder failed: {}", reorder_result.status().ToString());
        return false;
    }

    auto t_rs_3 = std::chrono::high_resolution_clock::now();
    std::array<tensorstore::Index, 4> out_shape = {
        static_cast<tensorstore::Index>(width_x),
        static_cast<tensorstore::Index>(height_y),
        static_cast<tensorstore::Index>(num_freq),
        static_cast<tensorstore::Index>(num_stokes)};
    auto batch_array = tensorstore::SharedArray<float>(
        tensorstore::internal::UnownedToShared(dst_ptr), out_shape, tensorstore::fortran_order);
    spdlog::debug("ZarrDataReader::ReadSlice [4/5] SharedArray [{}x{}x{}x{}] construction took {:.3f} ms",
        width_x, height_y, num_freq, num_stokes,
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_rs_3).count());

    auto t_rs_4 = std::chrono::high_resolution_clock::now();
    auto read_status = tensorstore::Read(reorder_result.value(), batch_array).result();
    spdlog::debug("ZarrDataReader::ReadSlice [5/5] tensorstore::Read [{}x{}x{}x{}] took {:.3f} ms",
        width_x, height_y, num_freq, num_stokes,
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_rs_4).count());
    if (!read_status.ok()) {
        spdlog::error("ReadSlice read failed: {}", read_status.status().ToString());
        return false;
    }

    // Force glibc to return freed memory from per-thread arenas back to OS.
    malloc_trim(0);

    return true;
}

bool ZarrDataReader::GetChunk(std::vector<float>& data, int& data_width, int& data_height, int min_x, int min_y, int channel, int stokes) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }

    try {
        int width = _shape[0];
        int height = _shape[1];

        constexpr int K_CHUNK_SIZE_PX = K_TILE_SIZE * 2;
        data_width = std::min(K_CHUNK_SIZE_PX, width - min_x);
        data_height = std::min(K_CHUNK_SIZE_PX, height - min_y);

        if (data_width <= 0 || data_height <= 0) {
            return false;
        }

        size_t chunk_size = static_cast<size_t>(data_width) * data_height;
        data.resize(chunk_size);

        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(_impl->store);
        if (!typed_store_result.ok()) {
            spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }

        // Slice T/F/S dimensions to get a 2D L×M plane
        std::array<tensorstore::Index, 3> slice_indices = {
            0, static_cast<tensorstore::Index>(channel), static_cast<tensorstore::Index>(stokes)};
        auto plane_result = typed_store_result.value() | tensorstore::Dims(0, 1, 2).IndexSlice(slice_indices);
        if (!plane_result.ok()) {
            spdlog::error("GetChunk: Error slicing T/F/S dimensions: {}", plane_result.status().ToString());
            return false;
        }

        // Only split into parallel strips if there are enough chunks along M
        // to avoid redundant decompression of the same chunk.
        const int chunk_shape_m = (_chunk_shape.size() == kDimSize5D) ? _chunk_shape[4] : 512;
        const int num_chunks_in_tile_m = (data_height + chunk_shape_m - 1) / chunk_shape_m;

        spdlog::debug("GetChunk: tile={}x{}, chunk_m={}, chunks_in_tile={}",
            data_width, data_height, chunk_shape_m, num_chunks_in_tile_m);

        // Let TensorStore manage its own internal parallelism. Issue a single read.
        auto chunk_l_result = plane_result.value() | tensorstore::Dims(0).ClosedInterval(min_x, min_x + data_width - 1);
        if (!chunk_l_result.ok()) {
            spdlog::error("GetChunk: Error slicing L dimension: {}", chunk_l_result.status().ToString());
            return false;
        }

        auto chunk_store_result = chunk_l_result.value() | tensorstore::Dims(1).ClosedInterval(min_y, min_y + data_height - 1);
        if (!chunk_store_result.ok()) {
            spdlog::error("GetChunk: Error slicing M dimension: {}", chunk_store_result.status().ToString());
            return false;
        }

        std::array<tensorstore::Index, 2> output_shape = {
            static_cast<tensorstore::Index>(data_width), static_cast<tensorstore::Index>(data_height)};

        auto output_array =
            tensorstore::SharedArray<float>(tensorstore::internal::UnownedToShared(data.data()), output_shape, tensorstore::fortran_order);

        auto read_status = tensorstore::Read(chunk_store_result.value(), output_array).result();
        if (!read_status.ok()) {
            spdlog::error("TensorStore GetChunk read failed: {}", read_status.status().ToString());
            return false;
        }

        // Force glibc to return freed memory from per-thread arenas back to OS.
        malloc_trim(0);

        return true;

    } catch (const std::exception& ex) {
        spdlog::error("Exception in GetChunk: {}", ex.what());
        return false;
    }
}

bool ZarrDataReader::ReadSpectralProfile(int x, int y, int stokes, std::vector<float>& data) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }

    try {
        int num_channels = _shape[2]; // Frequency axis in CARTA shape [X, Y, F, S]

        spdlog::debug("ReadSpectralProfile: x={}, y={}, stokes={}, channels={}", x, y, stokes, num_channels);

        // XRADIO schema: always 5D [T, F, S, L, M]
        // Read ALL channels at once (entire F axis)
        std::vector<tensorstore::Index> zarr_start = {
            0,                                       // T (always 0)
            0,                                       // F start (all channels)
            static_cast<tensorstore::Index>(stokes), // S
            static_cast<tensorstore::Index>(x),      // L = X
            static_cast<tensorstore::Index>(y)       // M = Y
        };
        std::vector<tensorstore::Index> zarr_shape = {
            1,                                             // T
            static_cast<tensorstore::Index>(num_channels), // F (all channels)
            1,                                             // S
            1,                                             // L
            1                                              // M
        };

        // Bounds validation
        for (size_t dim = 0; dim < kDimSize5D; ++dim) {
            tensorstore::Index end_idx = zarr_start[dim] + zarr_shape[dim] - 1;
            if (zarr_start[dim] < 0 || end_idx >= _original_shape[dim]) {
                spdlog::error("ReadSpectralProfile: Bounds error! dim={}, start={}, end={}, array_size={}", dim, zarr_start[dim], end_idx,
                    _original_shape[dim]);
                return false;
            }
        }

        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(_impl->store);
        if (!typed_store_result.ok()) {
            spdlog::error("ReadSpectralProfile: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }

        std::array<tensorstore::Index, 4> slice_indices = {
            0, static_cast<tensorstore::Index>(stokes), static_cast<tensorstore::Index>(x), static_cast<tensorstore::Index>(y)};
        auto spectrum_result = typed_store_result.value() | tensorstore::Dims(0, 2, 3, 4).IndexSlice(slice_indices);
        if (!spectrum_result.ok()) {
            spdlog::error("ReadSpectralProfile: Error slicing dimensions: {}", spectrum_result.status().ToString());
            return false;
        }

        data.resize(num_channels);
        float* dst_ptr = data.data();
        if (!dst_ptr) {
            spdlog::error("ReadSpectralProfile: data pointer is null after resize");
            return false;
        }

        std::array<tensorstore::Index, 1> output_shape = {static_cast<tensorstore::Index>(num_channels)};

        auto output_array = tensorstore::SharedArray<float>(tensorstore::internal::UnownedToShared(dst_ptr), output_shape);

        auto read_status = tensorstore::Read(spectrum_result.value(), output_array).result();
        if (!read_status.ok()) {
            spdlog::error("ReadSpectralProfile: TensorStore read failed: {}", read_status.status().ToString());
            return false;
        }

        spdlog::debug("ReadSpectralProfile: Successfully read {} channels", num_channels);

        return true;

    } catch (const std::exception& ex) {
        spdlog::error("Exception in ReadSpectralProfile: {}", ex.what());
        return false;
    }
}

//-----------------------------------------------------------------------------
// Metadata Helpers
//-----------------------------------------------------------------------------

std::vector<double> ZarrDataReader::ReadVector(const std::string& array_name) {
    if (!_initialized) {
        return {};
    }

    try {
        std::filesystem::path base_path(_filename);
        std::filesystem::path target_path = base_path / array_name;

        // Open the array using TensorStore
        nlohmann::json spec_json = {{"driver", "zarr"}, {"kvstore", {{"driver", "file"}, {"path", target_path.string()}}}};

        // Reuse main context to share cache and reduce memory allocations
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            // Try searching in subdirs if main path fails
            return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(), tensorstore::Context::Default(), tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

        auto open_result = open_future.result();
        if (!open_result.ok()) {
            return {};
        }

        auto store = open_result.value();
        auto domain = store.domain();

        // Ensure 1D
        if (domain.rank() != 1) {
            spdlog::warn("ReadVector: Array {} is not 1D (rank={})", array_name, domain.rank());
            return {};
        }

        // Read data
        // We cast to double for uniformity
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<double>>(store);
        if (!typed_store_result.ok()) {
            return {};
        }

        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        if (!read_result.ok()) {
            return {};
        }

        auto array = read_result.value();
        size_t size = array.num_elements();
        std::vector<double> result(size);

        // Copy data
        const double* ptr = array.data();
        std::copy(ptr, ptr + size, result.begin());

        return result;

    } catch (const std::exception& ex) {
        spdlog::warn("Error reading vector {}: {}", array_name, ex.what());
        return {};
    }
}

std::vector<std::string> ZarrDataReader::ReadStringVector(const std::string& array_name) {
    if (!_initialized) {
        return {};
    }

    try {
        std::filesystem::path base_path(_filename);
        std::filesystem::path target_path = base_path / array_name;

        nlohmann::json spec_json = {{"driver", "zarr"}, {"kvstore", {{"driver", "file"}, {"path", target_path.string()}}}};

        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(), tensorstore::Context::Default(), tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

        auto open_result = open_future.result();
        if (!open_result.ok()) {
            return {};
        }

        auto store = open_result.value();
        auto domain = store.domain();

        if (domain.rank() != 1) {
            spdlog::warn("ReadStringVector: Array {} is not 1D (rank={})", array_name, domain.rank());
            return {};
        }

        // Try reading as string (TensorStore supports casting some types to string,
        // but for Zarr fixed-length strings it usually maps to std::string or view)
        // Explicitly casting to string store
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<std::string>>(store);

        if (typed_store_result.ok()) {
            // It's already a string-compatible type
            auto read_result = tensorstore::Read(typed_store_result.value()).result();
            if (!read_result.ok()) {
                return {};
            }
            auto array = read_result.value();
            size_t size = array.num_elements();
            std::vector<std::string> result(size);

            // Copy data
            // Since we ensured rank 1, we can iterate by index
            for (tensorstore::Index i = 0; i < array.domain().shape()[0]; ++i) {
                result[i] = array(i);
            }
            return result;
        } else {
            // Fallback: maybe it's bytes or incompatible?
            // If fits2xradio wrote it as JSON strings or unicode, StaticCast should verify compatibility.
            spdlog::warn("ReadStringVector: could not cast array {} to string", array_name);
            return {};
        }

    } catch (const std::exception& ex) {
        spdlog::warn("Error reading string vector {}: {}", array_name, ex.what());
        return {};
    }
}

std::vector<double> ZarrDataReader::ReadFlattenedVector(const std::string& array_name) {
    if (!_initialized) {
        return {};
    }

    try {
        std::filesystem::path base_path(_filename);
        std::filesystem::path target_path = base_path / array_name;

        // Open the array using TensorStore
        nlohmann::json spec_json = {{"driver", "zarr"}, {"kvstore", {{"driver", "file"}, {"path", target_path.string()}}}};

        // Reuse main context to share cache and reduce memory allocations
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(), tensorstore::Context::Default(), tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

        auto open_result = open_future.result();
        if (!open_result.ok()) {
            return {};
        }

        auto store = open_result.value();

        // No rank check - we want to flatten whatever it is

        // Read data
        // We cast to double for uniformity
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<double>>(store);
        if (!typed_store_result.ok()) {
            return {};
        }

        // Read into memory
        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        if (!read_result.ok()) {
            return {};
        }

        auto array = read_result.value();
        size_t size = array.num_elements();
        std::vector<double> result(size);

        // Copy data assuming C-order flattening is desired or at least some consistent order.
        // If contiguity is not guaranteed we'd need to iterate.
        // But typically Read() returns a contiguous array.

        // If not contiguous, we can try to iterate, but iteration helpers might be complex.
        // Let's rely on data() for now. If it's null (non-contiguous), we fail.
        const double* ptr = array.data();
        if (ptr) {
            std::copy(ptr, ptr + size, result.begin());
        } else {
            spdlog::warn("ReadFlattenedVector: Array {} resulted in non-contiguous memory, simpler copy failed.", array_name);
            return {};
        }

        return result;

    } catch (const std::exception& ex) {
        spdlog::warn("Error reading flattened vector {}: {}", array_name, ex.what());
        return {};
    }
}

std::string ZarrDataReader::GetAttributeString(const std::string& array_name, const std::string& attr_name) {
    if (!_initialized) {
        return "";
    }

    // Read .zattrs for the array
    std::filesystem::path base_path(_filename);
    std::filesystem::path attrs_path = base_path / array_name / ".zattrs";

    if (!std::filesystem::exists(attrs_path)) {
        return "";
    }

    try {
        std::ifstream fstr(attrs_path);
        nlohmann::json json_obj;
        fstr >> json_obj;

        if (json_obj.contains(attr_name)) {
            if (json_obj[attr_name].is_string()) {
                return json_obj[attr_name].get<std::string>();
            }
            if (json_obj[attr_name].is_array() && !json_obj[attr_name].empty() && json_obj[attr_name][0].is_string()) {
                return json_obj[attr_name][0].get<std::string>(); // e.g. units: ["rad"]
            }
        }
    } catch (...) {
    }

    return "";
}

std::string ZarrDataReader::GetZattrsString(const std::string& array_name) {
    if (_impl && _impl->has_zmetadata) {
        std::string key = array_name.empty() ? ".zattrs" : array_name + "/.zattrs";
        if (_impl->zmetadata.contains("metadata") && _impl->zmetadata["metadata"].contains(key)) {
            const auto& val = _impl->zmetadata["metadata"][key];
            if (val.is_string())
                return val.get<std::string>();
            return val.dump();
        }
    }

    std::filesystem::path base_path(_filename);
    if (!array_name.empty()) {
        base_path /= array_name;
    }

    // Check for .zattrs in the resolved path
    std::filesystem::path zattrs_path = base_path / ".zattrs";
    if (std::filesystem::exists(zattrs_path)) {
        std::ifstream file(zattrs_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    return "{}";
}

std::string ZarrDataReader::GetZarrayString(const std::string& array_name) {
    if (_impl && _impl->has_zmetadata) {
        std::string key = array_name.empty() ? ".zarray" : array_name + "/.zarray";
        if (_impl->zmetadata.contains("metadata") && _impl->zmetadata["metadata"].contains(key)) {
            const auto& val = _impl->zmetadata["metadata"][key];
            if (val.is_string())
                return val.get<std::string>();
            return val.dump();
        }
    }

    std::filesystem::path base_path(_filename);
    if (!array_name.empty()) {
        base_path /= array_name;
    }

    std::filesystem::path zarray_path = base_path / ".zarray";
    if (!std::filesystem::exists(zarray_path)) {
        return "{}";
    }

    std::ifstream file(zarray_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::map<std::string, std::string> ZarrDataReader::GetZattrMap(const std::string& array_name) {
    std::map<std::string, std::string> result;
    try {
        std::string json_str = GetZattrsString(array_name);
        nlohmann::json json_obj = nlohmann::json::parse(json_str);

        for (const auto& [key, val] : json_obj.items()) {
            if (val.is_string()) {
                result[key] = val.get<std::string>();
            } else {
                result[key] = val.dump();
            }
        }
    } catch (...) {
    }

    return result;
}

} // namespace carta
