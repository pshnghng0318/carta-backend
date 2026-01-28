/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrDataReader.h"

// Standard library includes MUST come before TensorStore to ensure types are defined
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

// TensorStore includes - isolated to implementation file
#include "contiguous_layout.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/static_cast.h" 
#include "tensorstore/tensorstore.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/util/result.h"
#include "tensorstore/context.h"


namespace carta {

// Renamed to avoid macro collision
#ifndef K_TILE_SIZE
constexpr int K_TILE_SIZE = 256;
#endif

// Reduced from 128MB to 16MB to lower memory footprint.
// TensorStore still handles chunk caching internally, and 16MB is sufficient
// for typical tile operations (256x256 float tiles = 256KB each).
constexpr size_t kDefaultCacheSizeMB = 16;
constexpr size_t kDefaultCpuCount = 8;
constexpr size_t kDimSize5D = 5;
constexpr int kDefaultStripeHeight = 256;
constexpr int kStripeChunkMultiplier = 8;  // Read multiple chunks per stripe to reduce overhead

// Maximum data size per column batch in MiB for ReadChannelSliceV2
// Single chunk size is taken as minimum to avoid reading same chunk multiple times
constexpr size_t kColumnBatchMaxDataMiB = 16;

//-----------------------------------------------------------------------------
// Pimpl Implementation Helper
//-----------------------------------------------------------------------------
struct ZarrDataReader::Impl {
    tensorstore::Context context;
    tensorstore::TensorStore<> store;
    
    // Cached .zmetadata
    nlohmann::json zmetadata;
    bool has_zmetadata = false;
    
    // Create Context
    void CreateContext() {
        unsigned int num_cpus = std::thread::hardware_concurrency();
        if (num_cpus == 0) {
            num_cpus = kDefaultCpuCount;
        }
        
        nlohmann::json context_spec = {
            {"cache_pool", {
                {"total_bytes_limit", kDefaultCacheSizeMB * 1024 * 1024}
            }},
            {"data_copy_concurrency", {
                {"limit", num_cpus}
            }},
            {"file_io_concurrency", {
                {"limit", num_cpus}
            }}
        };
        
        auto context_result = tensorstore::Context::FromJson(context_spec);
        if (context_result.ok()) {
            context = context_result.value();
            spdlog::debug("Created TensorStore context with {} CPU cores", num_cpus);
        } else {
            spdlog::warn("Failed to create custom context: {}", context_result.status().ToString());
            context = tensorstore::Context::Default();
        }
    }
    
    // Map Coordinates Helper - XRADIO schema: always 5D [T, F, S, L, M]
    // Zarr is C-order (row-major): M (index 4) changes fastest
    // CARTA is Fortran-order (column-major): X (index 0) changes fastest
    // L is the horizontal axis (X), M is the vertical axis (Y)
    static std::vector<tensorstore::Index> MapToZarrCoords(const casacore::IPosition& start) {
        std::vector<tensorstore::Index> res(kDimSize5D, 0);
        res[0] = 0; // T (always 0 for CARTA)
        if (start.size() > 2) { res[1] = start[2]; } // F
        if (start.size() > 3) { res[2] = start[3]; } // S
        if (start.size() > 0) { res[3] = start[0]; } // L (CARTA X)
        if (start.size() > 1) { res[4] = start[1]; } // M (CARTA Y)
        return res;
    }
};

//-----------------------------------------------------------------------------
// ZarrDataReader Implementation
//-----------------------------------------------------------------------------

ZarrDataReader::ZarrDataReader(const std::string& filename) 
    : _filename(filename), _impl(std::make_unique<Impl>()) {}

ZarrDataReader::~ZarrDataReader() = default;

// Accessors implementations
bool ZarrDataReader::IsInitialized() const { return _initialized; }
const casacore::IPosition& ZarrDataReader::GetShape() const { return _shape; }
const casacore::IPosition& ZarrDataReader::GetOriginalZarrShape() const { return _original_shape; }
const casacore::IPosition& ZarrDataReader::GetChunkShape() const { return _chunk_shape; }
const std::string& ZarrDataReader::GetFilename() const { return _filename; }
int ZarrDataReader::NumDimensions() const { return _shape.size(); }

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
        
        // Try to read .zmetadata
        std::filesystem::path zmetadata_path = std::filesystem::path(_filename) / ".zmetadata";
        if (std::filesystem::exists(zmetadata_path)) {
            try {
                std::ifstream file(zmetadata_path);
                file >> _impl->zmetadata;
                _impl->has_zmetadata = true;
                spdlog::info("Loaded .zmetadata from {}", _filename);
            } catch (const std::exception& e) {
                spdlog::warn("Failed to parse .zmetadata: {}", e.what());
                _impl->has_zmetadata = false;
            }
        }
        
        _impl->CreateContext();
        
        nlohmann::json spec_json = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", array_path}
            }}
        };
        
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
            spdlog::error("Failed to create TensorStore spec: {}", spec_result.status().ToString());
            return false;
        }
        
        auto open_future = tensorstore::Open(
            spec_result.value(),
            _impl->context,
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        );
        
        auto open_result = open_future.result();
        if (!open_result.ok()) {
            spdlog::error("Failed to open TensorStore: {}", open_result.status().ToString());
            return false;
        }
        
        _impl->store = std::move(open_result).value();
        
        auto domain = _impl->store.domain();
        auto ts_shape = domain.shape();
        
        std::vector<int> orig_shape_vec;
        for (size_t i = 0; i < ts_shape.size(); ++i) {
            orig_shape_vec.push_back(static_cast<int>(ts_shape[i]));
        }
        _original_shape = casacore::IPosition(orig_shape_vec);
        
        // Validate _ARRAY_DIMENSIONS if present
        try {
            // Read .zattrs manually to check _ARRAY_DIMENSIONS
            // (tensorstore might expose it via Schema but simple JSON read is reliable)
            std::string zattrs_str = GetZattrsString(""); 
            if (!zattrs_str.empty() && zattrs_str != "{}") {
                nlohmann::json zattrs = nlohmann::json::parse(zattrs_str);
                if (zattrs.contains("_ARRAY_DIMENSIONS")) {
                    std::vector<std::string> dims = zattrs["_ARRAY_DIMENSIONS"].get<std::vector<std::string>>();
                    if (dims.size() == 5) {
                         // Check for deviations from standard XRADIO order [time, freq, pol, l, m]
                         // Note: names might vary slightly, but we expect l, m at end
                         bool standard_order = (dims[3] == "l" || dims[3] == "u" || dims[3] == "X") &&
                                               (dims[4] == "m" || dims[4] == "v" || dims[4] == "Y");
                         if (!standard_order) {
                             spdlog::warn("Effectively assuming [t, f, p, l, m] but _ARRAY_DIMENSIONS are: {}", fmt::join(dims, ", "));
                         }
                    }
                }
            }
        } catch (...) {
            // Ignore metadata read errors during init, handled elsewhere or non-critical
        }

        // XRADIO schema: always 5D [T, F, S, L, M]
        if (ts_shape.size() != kDimSize5D) {
            spdlog::error("XRADIO schema requires 5D array, got {}D", ts_shape.size());
            return false;
        }
        
        // Map to CARTA shape [X, Y, F, S]
        // L (index 3) is the horizontal axis -> X
        // M (index 4) is the vertical axis -> Y
        std::vector<int> carta_shape;
        carta_shape.push_back(orig_shape_vec[3]); // L -> X
        carta_shape.push_back(orig_shape_vec[4]); // M -> Y
        carta_shape.push_back(orig_shape_vec[1]); // F
        carta_shape.push_back(orig_shape_vec[2]); // S
        _shape = casacore::IPosition(carta_shape);
        
        auto chunk_layout_result = _impl->store.chunk_layout();
        if (chunk_layout_result.ok()) {
            auto read_chunk_shape = chunk_layout_result.value().read_chunk_shape();
            if (!read_chunk_shape.empty()) {
                std::vector<int> chunk_shape_vec;
                chunk_shape_vec.reserve(read_chunk_shape.size());
                for (auto size : read_chunk_shape) {
                    chunk_shape_vec.push_back(static_cast<int>(size));
                }
                _chunk_shape = casacore::IPosition(chunk_shape_vec);
                spdlog::debug("ZarrDataReader chunk shape (read): {}", _chunk_shape.toString());
            }
        } else {
            spdlog::debug("ZarrDataReader: chunk_layout unavailable: {}", chunk_layout_result.status().ToString());
        }

        _initialized = true;
        spdlog::debug("ZarrDataReader initialized: ZARR shape={}, CARTA shape={}", 
                    _original_shape.toString(), _shape.toString());
        
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
    
    std::lock_guard<std::mutex> lock(_read_mutex);

    const auto& start = section.start();
    const auto& stop = section.end();
    const auto& length = section.length();

    spdlog::debug("ZarrDataReader::ReadSlice: start={}, stop={}, length={}", 
                start.toString(), stop.toString(), length.toString());

    buffer.resize(length);
    
    const int width_x = length[0];
    const int height_y = length[1];
    const int num_freq = length[2];
    const int num_stokes = length[3];
    
    // Time index is always 0 for now.
    const tensorstore::Index time_idx = 0;
    
    // Get L chunk size for batching
    const int chunk_shape_l = (_chunk_shape.size() == kDimSize5D) ? _chunk_shape[3] : 512;
    
    // Calculate columns per batch
    const size_t max_bytes = kColumnBatchMaxDataMiB * 1024 * 1024;
    const size_t bytes_per_column = static_cast<size_t>(height_y) * num_freq * num_stokes * sizeof(float);
    int cols_per_batch = static_cast<int>(max_bytes / bytes_per_column);
    cols_per_batch = std::max(cols_per_batch, chunk_shape_l);
    // Align to chunk boundary
    cols_per_batch = (cols_per_batch / chunk_shape_l) * chunk_shape_l;
    
    float* dst_ptr = buffer.data();
    int x_offset = 0;
    
    while (x_offset < width_x) {
        int batch_width = std::min(cols_per_batch, width_x - x_offset);
        
        // Align first batch to chunk boundary
        if (x_offset == 0 && (start[0] % chunk_shape_l) != 0) {
            int to_boundary = chunk_shape_l - (start[0] % chunk_shape_l);
            batch_width = std::min(to_boundary, width_x);
        }
        
        // Create slice: [T=0, F range, P range, L batch, M full]
        // XRADIO 5D: [T, F, P, L, M] -> CARTA 4D: [X, Y, F, S]
        auto slice_result = _impl->store
            | tensorstore::Dims(0).IndexSlice(time_idx)
            | tensorstore::Dims(0).ClosedInterval(start[2], stop[2])       // F (now dim 0 after T slice)
            | tensorstore::Dims(1).ClosedInterval(start[3], stop[3])   // P
            | tensorstore::Dims(2).ClosedInterval(start[0] + x_offset, start[0] + x_offset + batch_width - 1)  // L
            | tensorstore::Dims(3).ClosedInterval(start[1], stop[1]);         // M

        if (!slice_result.ok()) {
            spdlog::error("ReadSlice batch slice failed: {}", slice_result.status().ToString());
            return false;
        }

        // Transpose [F, P, L, M] -> [L, M, F, P]
        auto reorder_result = std::move(slice_result).value() | tensorstore::Dims(2, 3, 0, 1).Transpose();
        if (!reorder_result.ok()) {
            spdlog::error("ReadSlice reorder failed: {}", reorder_result.status().ToString());
            return false;
        }

        // Destination pointer: x_offset into Fortran-order buffer [X, Y, F, S]
        float* batch_dst = dst_ptr + x_offset;
        
        // Strides for Fortran order in 4D [X, Y, F, S]
        std::array<tensorstore::Index, 4> batch_shape = {
            static_cast<tensorstore::Index>(batch_width),
            static_cast<tensorstore::Index>(height_y),
            static_cast<tensorstore::Index>(num_freq),
            static_cast<tensorstore::Index>(num_stokes)
        };
        
        std::array<tensorstore::Index, 4> batch_byte_strides = {
            static_cast<tensorstore::Index>(sizeof(float)),
            static_cast<tensorstore::Index>(width_x * sizeof(float)),
            static_cast<tensorstore::Index>(width_x * height_y * sizeof(float)),
            static_cast<tensorstore::Index>(width_x * height_y * num_freq * sizeof(float))
        };
        
        tensorstore::StridedLayout<4> batch_layout(batch_shape, batch_byte_strides);
        tensorstore::ArrayView<float, 4> batch_view(batch_dst, batch_layout);
        auto batch_array = tensorstore::UnownedToShared(batch_view);

        auto read_status = tensorstore::Read(reorder_result.value(), batch_array).result();
        if (!read_status.ok()) {
            spdlog::error("ReadSlice batch read failed: {}", read_status.status().ToString());
            return false;
        }
        
        x_offset += batch_width;
    }
    
    return true;
}

bool ZarrDataReader::GetChunk(std::vector<float>& data, int& data_width, int& data_height,
                              int min_x, int min_y, int channel, int stokes) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(_read_mutex);
    
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
        
        // XRADIO always 5D [T, F, S, L, M]
        // L -> X (min_x, data_width), M -> Y (min_y, data_height)
        std::vector<tensorstore::Index> zarr_start = {
            0,
            static_cast<tensorstore::Index>(channel),
            static_cast<tensorstore::Index>(stokes),
            static_cast<tensorstore::Index>(min_x),     // L = X
            static_cast<tensorstore::Index>(min_y)      // M = Y
        };
        std::vector<tensorstore::Index> zarr_shape = {
            1,
            1,
            1,
            static_cast<tensorstore::Index>(data_width),   // L = X
            static_cast<tensorstore::Index>(data_height)   // M = Y
        };
        
        // Bounds validation
        for (size_t dim = 0; dim < kDimSize5D; ++dim) {
            tensorstore::Index end_idx = zarr_start[dim] + zarr_shape[dim] - 1;
            if (zarr_start[dim] < 0 || end_idx >= _original_shape[dim]) {
                spdlog::error("GetChunk: Bounds error! dim={}, start={}, end={}, array_size={}",
                              dim, zarr_start[dim], end_idx, _original_shape[dim]);
                return false;
            }
        }
        
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(_impl->store);
        if (!typed_store_result.ok()) {
            spdlog::error("ZarrDataReader: Error casting to float store: {}",
                          typed_store_result.status().ToString());
            return false;
        }
        
        std::array<tensorstore::Index, 3> slice_indices = {
            0,
            static_cast<tensorstore::Index>(channel),
            static_cast<tensorstore::Index>(stokes)
        };
        auto plane_result =
            typed_store_result.value() | tensorstore::Dims(0, 1, 2).IndexSlice(slice_indices);
        if (!plane_result.ok()) {
            spdlog::error("GetChunk: Error slicing T/F/S dimensions: {}",
                          plane_result.status().ToString());
            return false;
        }
        
        auto chunk_l_result =
            plane_result.value() | tensorstore::Dims(0).ClosedInterval(min_x, min_x + data_width - 1);
        if (!chunk_l_result.ok()) {
            spdlog::error("GetChunk: Error slicing L dimension: {}",
                          chunk_l_result.status().ToString());
            return false;
        }
        
        auto chunk_store_result = chunk_l_result.value() |
            tensorstore::Dims(1).ClosedInterval(min_y, min_y + data_height - 1);
        if (!chunk_store_result.ok()) {
            spdlog::error("GetChunk: Error slicing M dimension: {}",
                          chunk_store_result.status().ToString());
            return false;
        }
        
        data.resize(chunk_size);
        float* dst_ptr = data.data();
        if (!dst_ptr) {
            spdlog::error("GetChunk: data pointer is null after resize");
            return false;
        }
        
        std::array<tensorstore::Index, 2> output_shape = {
            static_cast<tensorstore::Index>(data_width),
            static_cast<tensorstore::Index>(data_height)
        };
        
        auto output_array = tensorstore::SharedArray<float>(
            tensorstore::internal::UnownedToShared(dst_ptr), output_shape,
            tensorstore::fortran_order);
        
        auto read_status = tensorstore::Read(chunk_store_result.value(), output_array).result();
        if (!read_status.ok()) {
            spdlog::error("TensorStore GetChunk failed: {}", read_status.status().ToString());
            return false;
        }
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("Exception in GetChunk: {}", ex.what());
        return false;
    }
}

bool ZarrDataReader::ReadSpectralProfile(int x, int y, int stokes, 
                                          std::vector<float>& data) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(_read_mutex);
    
    try {
        int num_channels = _shape[2];  // Frequency axis in CARTA shape [X, Y, F, S]
        
        spdlog::debug("ReadSpectralProfile: x={}, y={}, stokes={}, channels={}", 
                     x, y, stokes, num_channels);
        
        // XRADIO schema: always 5D [T, F, S, L, M]
        // Read ALL channels at once (entire F axis)
        std::vector<tensorstore::Index> zarr_start = {
            0,                                         // T (always 0)
            0,                                         // F start (all channels)
            static_cast<tensorstore::Index>(stokes),   // S
            static_cast<tensorstore::Index>(x),        // L = X
            static_cast<tensorstore::Index>(y)         // M = Y
        };
        std::vector<tensorstore::Index> zarr_shape = {
            1,                                         // T
            static_cast<tensorstore::Index>(num_channels),  // F (all channels)
            1,                                         // S
            1,                                         // L
            1                                          // M
        };
        
        // Bounds validation
        for (size_t dim = 0; dim < kDimSize5D; ++dim) {
            tensorstore::Index end_idx = zarr_start[dim] + zarr_shape[dim] - 1;
            if (zarr_start[dim] < 0 || end_idx >= _original_shape[dim]) {
                spdlog::error("ReadSpectralProfile: Bounds error! dim={}, start={}, end={}, array_size={}",
                              dim, zarr_start[dim], end_idx, _original_shape[dim]);
                return false;
            }
        }
        
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(_impl->store);
        if (!typed_store_result.ok()) {
            spdlog::error("ReadSpectralProfile: Error casting to float store: {}",
                          typed_store_result.status().ToString());
            return false;
        }
        
        std::array<tensorstore::Index, 4> slice_indices = {
            0,
            static_cast<tensorstore::Index>(stokes),
            static_cast<tensorstore::Index>(x),
            static_cast<tensorstore::Index>(y)
        };
        auto spectrum_result =
            typed_store_result.value() | tensorstore::Dims(0, 2, 3, 4).IndexSlice(slice_indices);
        if (!spectrum_result.ok()) {
            spdlog::error("ReadSpectralProfile: Error slicing dimensions: {}",
                          spectrum_result.status().ToString());
            return false;
        }
        
        data.resize(num_channels);
        float* dst_ptr = data.data();
        if (!dst_ptr) {
            spdlog::error("ReadSpectralProfile: data pointer is null after resize");
            return false;
        }
        
        std::array<tensorstore::Index, 1> output_shape = {
            static_cast<tensorstore::Index>(num_channels)
        };
        
        auto output_array = tensorstore::SharedArray<float>(
            tensorstore::internal::UnownedToShared(dst_ptr), output_shape);
        
        auto read_status = tensorstore::Read(spectrum_result.value(), output_array).result();
        if (!read_status.ok()) {
            spdlog::error("ReadSpectralProfile: TensorStore read failed: {}", 
                         read_status.status().ToString());
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
    if (!_initialized) { return {};
}
    std::lock_guard<std::mutex> lock(_read_mutex);

    try {
        std::filesystem::path base_path(_filename);
        std::filesystem::path target_path = base_path / array_name;
        
        // Open the array using TensorStore
        nlohmann::json spec_json = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", target_path.string()}
            }}
        };
        
        // Reuse main context to share cache and reduce memory allocations
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
             // Try searching in subdirs if main path fails
             return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(),
            tensorstore::Context::Default(),
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        );
        
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
    if (!_initialized) { return {}; }
    std::lock_guard<std::mutex> lock(_read_mutex);

    try {
        std::filesystem::path base_path(_filename);
        std::filesystem::path target_path = base_path / array_name;
        
        nlohmann::json spec_json = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", target_path.string()}
            }}
        };
        
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
             return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(),
            tensorstore::Context::Default(),
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        );
        
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
    if (!_initialized) { return {}; }
    std::lock_guard<std::mutex> lock(_read_mutex);

    try {
        std::filesystem::path base_path(_filename);
        std::filesystem::path target_path = base_path / array_name;
        
        // Open the array using TensorStore
        nlohmann::json spec_json = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", target_path.string()}
            }}
        };
        
        // Reuse main context to share cache and reduce memory allocations
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
             return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(),
            tensorstore::Context::Default(),
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        );
        
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
    if (!_initialized) { return "";
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
    } catch (...) {}
    
    return "";
}

std::string ZarrDataReader::GetZattrsString(const std::string& array_name) {
    if (_impl && _impl->has_zmetadata) {
        std::string key = array_name.empty() ? ".zattrs" : array_name + "/.zattrs";
        if (_impl->zmetadata.contains("metadata") && _impl->zmetadata["metadata"].contains(key)) {
            const auto& val = _impl->zmetadata["metadata"][key];
            if (val.is_string()) return val.get<std::string>();
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
            if (val.is_string()) return val.get<std::string>();
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
    } catch (...) {}
    
    return result;
}

} // namespace carta
