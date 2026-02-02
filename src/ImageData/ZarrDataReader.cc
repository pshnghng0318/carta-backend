/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrDataReader.h"

// Standard library includes MUST come before TensorStore to ensure types are defined
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

// TensorStore includes - isolated to implementation file
#include "contiguous_layout.h"
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

// Reduced from 128MB to 16MB to lower memory footprint.
// TensorStore still handles chunk caching internally, and 16MB is sufficient
// for typical tile operations (256x256 float tiles = 256KB each).
constexpr size_t kDefaultCacheSizeMB = 16;
constexpr size_t kDefaultCpuCount = 8;
constexpr size_t kDimSize5D = 5;
constexpr int kDefaultStripeHeight = 256;
constexpr int kStripeChunkMultiplier = 8; // Read multiple chunks per stripe to reduce overhead

// Maximum data size per column batch in MiB for ReadChannelSliceV2
// Single chunk size is taken as minimum to avoid reading same chunk multiple times
constexpr size_t kColumnBatchMaxDataMiB = 16;

//-----------------------------------------------------------------------------
// Anonymous namespace for helper functions used by ReadSlice/ReadMaskedSlice/ReadMaskSlice
//-----------------------------------------------------------------------------
namespace {

// Helper: Build XRADIO slice ranges for a data or mask store
// XRADIO 5D: [T, F, P, L, M] -> CARTA 4D: [X, Y, F, S]
// Returns tensorstore::Result for the sliced and time-indexed store
template <typename Store>
tensorstore::Result<tensorstore::TensorStore<typename Store::Element>> BuildXRadioSlice(Store& store, const casacore::IPosition& start,
    const casacore::IPosition& stop, int x_offset, int batch_width, tensorstore::Index time_idx) {
    auto slice_result = store | tensorstore::Dims(0).IndexSlice(time_idx) | tensorstore::Dims(0).ClosedInterval(start[2], stop[2]) // F
                        | tensorstore::Dims(1).ClosedInterval(start[3], stop[3])                                                   // P
                        | tensorstore::Dims(2).ClosedInterval(start[0] + x_offset, start[0] + x_offset + batch_width - 1)          // L
                        | tensorstore::Dims(3).ClosedInterval(start[1], stop[1]);                                                  // M
    return slice_result;
}

// Helper: Transpose [F, P, L, M] -> [L, M, F, P] for CARTA order
template <typename Store>
tensorstore::Result<tensorstore::TensorStore<typename Store::Element>> TransposeToCartaOrder(tensorstore::Result<Store>&& slice_result) {
    if (!slice_result.ok()) {
        return slice_result.status();
    }
    return std::move(slice_result).value() | tensorstore::Dims(2, 3, 0, 1).Transpose();
}

// Helper: Construct StridedLayout, ArrayView, and UnownedToShared for 4D buffer
// with configurable element size stride (for float or int8_t)
template <typename T>
tensorstore::SharedArray<T, 4> Build4DArrayView(
    T* data_ptr, int batch_width, int height_y, int num_freq, int num_stokes, int full_width_x) {
    std::array<tensorstore::Index, 4> shape = {static_cast<tensorstore::Index>(batch_width), static_cast<tensorstore::Index>(height_y),
        static_cast<tensorstore::Index>(num_freq), static_cast<tensorstore::Index>(num_stokes)};

    std::array<tensorstore::Index, 4> byte_strides = {static_cast<tensorstore::Index>(sizeof(T)),
        static_cast<tensorstore::Index>(full_width_x * sizeof(T)), static_cast<tensorstore::Index>(full_width_x * height_y * sizeof(T)),
        static_cast<tensorstore::Index>(full_width_x * height_y * num_freq * sizeof(T))};

    tensorstore::StridedLayout<4> layout(shape, byte_strides);
    return tensorstore::SharedArray<T, 4>(tensorstore::internal::UnownedToShared(data_ptr), layout);
}

// Helper: Construct StridedLayout for contiguous 4D buffer (used for mask reads where full_width == batch_width)
template <typename T>
tensorstore::SharedArray<T, 4> Build4DContiguousArrayView(T* data_ptr, int batch_width, int height_y, int num_freq, int num_stokes) {
    std::array<tensorstore::Index, 4> shape = {static_cast<tensorstore::Index>(batch_width), static_cast<tensorstore::Index>(height_y),
        static_cast<tensorstore::Index>(num_freq), static_cast<tensorstore::Index>(num_stokes)};

    std::array<tensorstore::Index, 4> byte_strides = {static_cast<tensorstore::Index>(sizeof(T)),
        static_cast<tensorstore::Index>(batch_width * sizeof(T)), static_cast<tensorstore::Index>(batch_width * height_y * sizeof(T)),
        static_cast<tensorstore::Index>(batch_width * height_y * num_freq * sizeof(T))};

    tensorstore::StridedLayout<4> layout(shape, byte_strides);
    return tensorstore::SharedArray<T, 4>(tensorstore::internal::UnownedToShared(data_ptr), layout);
}

// Helper: Perform tensorstore::Read and return status
template <typename Store, typename Array>
absl::Status PerformTensorStoreRead(Store& store, Array& array) {
    auto read_result = tensorstore::Read(store, array).result();
    if (!read_result.ok()) {
        return read_result.status();
    }
    return absl::OkStatus();
}

// Helper: Open a Zarr array using TensorStore and return the result
// Returns empty result on failure (check .ok() on the returned value)
tensorstore::Result<tensorstore::TensorStore<>> OpenZarrArray(
    const std::string& array_path, const tensorstore::Context& context = tensorstore::Context::Default()) {
    nlohmann::json spec_json = {{"driver", "zarr"}, {"kvstore", {{"driver", "file"}, {"path", array_path}}}};

    auto spec_result = tensorstore::Spec::FromJson(spec_json);
    if (!spec_result.ok()) {
        return spec_result.status();
    }

    auto open_future = tensorstore::Open(spec_result.value(), context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

    return open_future.result();
}

// Helper: Apply mask to data buffer by setting masked values to NaN
void ApplyMaskToData(float* dst_ptr, int x_offset, int batch_width, int height_y, int num_freq, int num_stokes, int width_x,
    const int8_t* mask_int8_buffer, float nan_value) {
    const size_t batch_xy = static_cast<size_t>(batch_width) * height_y;
    const size_t batch_xyf = batch_xy * num_freq;
    const size_t image_xy = static_cast<size_t>(width_x) * height_y;
    const size_t image_xyf = image_xy * num_freq;

    for (int stokes_idx = 0; stokes_idx < num_stokes; ++stokes_idx) {
        for (int freq_idx = 0; freq_idx < num_freq; ++freq_idx) {
            for (int y_idx = 0; y_idx < height_y; ++y_idx) {
                const size_t mask_base = (static_cast<size_t>(y_idx) * batch_width) + (static_cast<size_t>(freq_idx) * batch_xy) +
                                         (static_cast<size_t>(stokes_idx) * batch_xyf);

                float* data_row = dst_ptr + x_offset + (y_idx * width_x) + (freq_idx * image_xy) + (stokes_idx * image_xyf);

                for (int x_idx = 0; x_idx < batch_width; ++x_idx) {
                    if (mask_int8_buffer[mask_base + x_idx] != 0) {
                        data_row[x_idx] = nan_value;
                    }
                }
            }
        }
    }
}

} // anonymous namespace

//-----------------------------------------------------------------------------
// Pimpl Implementation Helper
//-----------------------------------------------------------------------------
struct ZarrDataReader::Impl {
    tensorstore::TensorStore<> store;

    // Cached .zmetadata
    nlohmann::json zmetadata;
    bool has_zmetadata = false;

    // Cached mask store for efficient repeated mask reads
    // Note: Mask may be stored as int8 (0/1) or bool, so we use dynamic type
    tensorstore::TensorStore<int8_t> mask_store;
    bool mask_store_checked = false; // True after first check (avoid repeated filesystem lookups)
    bool has_mask_store = false;     // True if mask_store is valid and ready
    std::string cached_mask_path;    // Cached active mask path

    // Get shared TensorStore context (created once, reused by all instances)
    // This saves memory (single cache pool) and reduces initialization overhead
    static tensorstore::Context GetSharedContext() {
        static tensorstore::Context shared_context = []() {
            static const unsigned int num_cpus = []() {
                auto cpu_count = std::thread::hardware_concurrency();
                return cpu_count ? cpu_count : kDefaultCpuCount;
            }();

            nlohmann::json context_spec = {{"cache_pool", {{"total_bytes_limit", kDefaultCacheSizeMB * 1024 * 1024}}},
                {"data_copy_concurrency", {{"limit", num_cpus}}}, {"file_io_concurrency", {{"limit", num_cpus}}}};

            auto context_result = tensorstore::Context::FromJson(context_spec);
            if (context_result.ok()) {
                spdlog::info("Created shared TensorStore context with {} CPU cores, {}MB cache", num_cpus, kDefaultCacheSizeMB);
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

    std::string GetZarrMetadataFile(const std::string& filename, const std::string& array_name, const std::string& meta_filename) {
        if (has_zmetadata) {
            std::string key = array_name.empty() ? meta_filename : array_name + "/" + meta_filename;
            if (zmetadata.contains("metadata") && zmetadata["metadata"].contains(key)) {
                const auto& val = zmetadata["metadata"][key];
                return val.is_string() ? val.get<std::string>() : val.dump();
            }
        }

        std::filesystem::path base_path(filename);
        if (!array_name.empty()) {
            base_path /= array_name;
        }

        std::filesystem::path meta_path = base_path / meta_filename;
        if (!std::filesystem::exists(meta_path)) {
            return "{}";
        }

        std::ifstream file(meta_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
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
const casacore::IPosition& ZarrDataReader::GetChunkShape() const {
    return _chunk_shape;
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

        auto open_result = OpenZarrArray(array_path, Impl::GetSharedContext());
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

    std::lock_guard<std::mutex> lock(_read_mutex);

    const auto& start = section.start();
    const auto& stop = section.end();
    const auto& length = section.length();

    spdlog::debug("ZarrDataReader::ReadSlice: start={}, stop={}, length={}", start.toString(), stop.toString(), length.toString());

    buffer.resize(length);

    const int width_x = length[0];
    const int height_y = length[1];
    const int num_freq = length[2];
    const int num_stokes = length[3];

    const tensorstore::Index time_idx = 0;

    const int chunk_shape_l = (_chunk_shape.size() == kDimSize5D) ? _chunk_shape[3] : 512;

    const size_t max_bytes = kColumnBatchMaxDataMiB * 1024 * 1024;
    const size_t bytes_per_column = static_cast<size_t>(height_y) * num_freq * num_stokes * sizeof(float);
    int cols_per_batch = static_cast<int>(max_bytes / bytes_per_column);
    cols_per_batch = std::max(cols_per_batch, chunk_shape_l);
    cols_per_batch = (cols_per_batch / chunk_shape_l) * chunk_shape_l;

    float* dst_ptr = buffer.data();
    int x_offset = 0;

    while (x_offset < width_x) {
        int batch_width = std::min(cols_per_batch, width_x - x_offset);

        if (x_offset == 0 && (start[0] % chunk_shape_l) != 0) {
            int to_boundary = chunk_shape_l - (start[0] % chunk_shape_l);
            batch_width = std::min(to_boundary, width_x);
        }

        auto slice_result = BuildXRadioSlice(_impl->store, start, stop, x_offset, batch_width, time_idx);
        if (!slice_result.ok()) {
            spdlog::error("ReadSlice batch slice failed: {}", slice_result.status().ToString());
            return false;
        }

        auto reorder_result = TransposeToCartaOrder(std::move(slice_result));
        if (!reorder_result.ok()) {
            spdlog::error("ReadSlice reorder failed: {}", reorder_result.status().ToString());
            return false;
        }

        float* batch_dst = dst_ptr + x_offset;
        auto batch_array = Build4DArrayView(batch_dst, batch_width, height_y, num_freq, num_stokes, width_x);

        auto read_status = PerformTensorStoreRead(reorder_result.value(), batch_array);
        if (!read_status.ok()) {
            spdlog::error("ReadSlice batch read failed: {}", read_status.ToString());
            return false;
        }

        x_offset += batch_width;
    }

    return true;
}

bool ZarrDataReader::ReadMaskedSlice(casacore::Array<float>& buffer, const casacore::Slicer& section) {
    if (!EnsureMaskStore()) {
        return ReadSlice(buffer, section);
    }

    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }

    std::lock_guard<std::mutex> lock(_read_mutex);

    const auto& start = section.start();
    const auto& stop = section.end();
    const auto& length = section.length();

    spdlog::debug("ZarrDataReader::ReadSlice: start={}, stop={}, length={}", start.toString(), stop.toString(), length.toString());

    buffer.resize(length);

    const int width_x = length[0];
    const int height_y = length[1];
    const int num_freq = length[2];
    const int num_stokes = length[3];

    const tensorstore::Index time_idx = 0;

    const int chunk_shape_l = (_chunk_shape.size() == kDimSize5D) ? _chunk_shape[3] : 512;

    const size_t max_bytes = kColumnBatchMaxDataMiB * 1024 * 1024;
    const size_t bytes_per_column = static_cast<size_t>(height_y) * num_freq * num_stokes * sizeof(float);
    int cols_per_batch = static_cast<int>(max_bytes / bytes_per_column);
    cols_per_batch = std::max(cols_per_batch, chunk_shape_l);
    cols_per_batch = (cols_per_batch / chunk_shape_l) * chunk_shape_l;

    float* dst_ptr = buffer.data();
    int x_offset = 0;

    std::vector<int8_t> mask_int8_buffer;
    mask_int8_buffer.reserve(static_cast<size_t>(cols_per_batch) * height_y * num_freq * num_stokes);

    const float nan_value = std::numeric_limits<float>::quiet_NaN();

    while (x_offset < width_x) {
        int batch_width = std::min(cols_per_batch, width_x - x_offset);

        if (x_offset == 0 && (start[0] % chunk_shape_l) != 0) {
            int to_boundary = chunk_shape_l - (start[0] % chunk_shape_l);
            batch_width = std::min(to_boundary, width_x);
        }

        auto slice_result = BuildXRadioSlice(_impl->store, start, stop, x_offset, batch_width, time_idx);
        if (!slice_result.ok()) {
            spdlog::error("ReadSlice batch slice failed: {}", slice_result.status().ToString());
            return false;
        }

        auto reorder_result = TransposeToCartaOrder(std::move(slice_result));
        if (!reorder_result.ok()) {
            spdlog::error("ReadSlice reorder failed: {}", reorder_result.status().ToString());
            return false;
        }

        float* batch_dst = dst_ptr + x_offset;
        auto batch_array = Build4DArrayView(batch_dst, batch_width, height_y, num_freq, num_stokes, width_x);

        auto read_status = PerformTensorStoreRead(reorder_result.value(), batch_array);
        if (!read_status.ok()) {
            spdlog::error("ReadSlice batch read failed: {}", read_status.ToString());
            return false;
        }

        const size_t batch_elements = static_cast<size_t>(batch_width) * height_y * num_freq * num_stokes;
        mask_int8_buffer.resize(batch_elements);

        auto mask_slice_result = BuildXRadioSlice(_impl->mask_store, start, stop, x_offset, batch_width, time_idx);
        if (!mask_slice_result.ok()) {
            spdlog::error("ReadMaskedSlice mask slice failed: {}", mask_slice_result.status().ToString());
            return false;
        }

        auto mask_reorder_result = TransposeToCartaOrder(std::move(mask_slice_result));
        if (!mask_reorder_result.ok()) {
            spdlog::error("ReadMaskedSlice mask reorder failed: {}", mask_reorder_result.status().ToString());
            return false;
        }

        auto mask_array = Build4DContiguousArrayView(mask_int8_buffer.data(), batch_width, height_y, num_freq, num_stokes);

        auto mask_read_status = PerformTensorStoreRead(mask_reorder_result.value(), mask_array);
        if (!mask_read_status.ok()) {
            spdlog::error("ReadMaskedSlice mask read failed: {}", mask_read_status.ToString());
            return false;
        }

        ApplyMaskToData(dst_ptr, x_offset, batch_width, height_y, num_freq, num_stokes, width_x, mask_int8_buffer.data(), nan_value);

        x_offset += batch_width;
    }

    return true;
}

bool ZarrDataReader::GetChunk(std::vector<float>& data, int& data_width, int& data_height, int min_x, int min_y, int channel, int stokes) {
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
            0, static_cast<tensorstore::Index>(channel), static_cast<tensorstore::Index>(stokes),
            static_cast<tensorstore::Index>(min_x), // L = X
            static_cast<tensorstore::Index>(min_y)  // M = Y
        };
        std::vector<tensorstore::Index> zarr_shape = {
            1, 1, 1,
            static_cast<tensorstore::Index>(data_width), // L = X
            static_cast<tensorstore::Index>(data_height) // M = Y
        };

        // Bounds validation
        for (size_t dim = 0; dim < kDimSize5D; ++dim) {
            tensorstore::Index end_idx = zarr_start[dim] + zarr_shape[dim] - 1;
            if (zarr_start[dim] < 0 || end_idx >= _original_shape[dim]) {
                spdlog::error(
                    "GetChunk: Bounds error! dim={}, start={}, end={}, array_size={}", dim, zarr_start[dim], end_idx, _original_shape[dim]);
                return false;
            }
        }

        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(_impl->store);
        if (!typed_store_result.ok()) {
            spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }

        std::array<tensorstore::Index, 3> slice_indices = {
            0, static_cast<tensorstore::Index>(channel), static_cast<tensorstore::Index>(stokes)};
        auto plane_result = typed_store_result.value() | tensorstore::Dims(0, 1, 2).IndexSlice(slice_indices);
        if (!plane_result.ok()) {
            spdlog::error("GetChunk: Error slicing T/F/S dimensions: {}", plane_result.status().ToString());
            return false;
        }

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

        data.resize(chunk_size);
        float* dst_ptr = data.data();
        if (!dst_ptr) {
            spdlog::error("GetChunk: data pointer is null after resize");
            return false;
        }

        std::array<tensorstore::Index, 2> output_shape = {
            static_cast<tensorstore::Index>(data_width), static_cast<tensorstore::Index>(data_height)};

        auto output_array =
            tensorstore::SharedArray<float>(tensorstore::internal::UnownedToShared(dst_ptr), output_shape, tensorstore::fortran_order);

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

//-----------------------------------------------------------------------------
// Metadata Helpers
//-----------------------------------------------------------------------------

std::vector<double> ZarrDataReader::ReadVector(const std::string& array_name) {
    if (!_initialized) {
        return {};
    }
    std::lock_guard<std::mutex> lock(_read_mutex);

    try {
        std::filesystem::path target_path = std::filesystem::path(_filename) / array_name;

        auto open_result = OpenZarrArray(target_path.string());
        if (!open_result.ok()) {
            return {};
        }

        auto store = open_result.value();
        if (store.domain().rank() != 1) {
            spdlog::warn("ReadVector: Array {} is not 1D (rank={})", array_name, store.domain().rank());
            return {};
        }

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
    std::lock_guard<std::mutex> lock(_read_mutex);

    try {
        std::filesystem::path target_path = std::filesystem::path(_filename) / array_name;

        auto open_result = OpenZarrArray(target_path.string());
        if (!open_result.ok()) {
            return {};
        }

        auto store = open_result.value();
        if (store.domain().rank() != 1) {
            spdlog::warn("ReadStringVector: Array {} is not 1D (rank={})", array_name, store.domain().rank());
            return {};
        }

        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<std::string>>(store);
        if (!typed_store_result.ok()) {
            spdlog::warn("ReadStringVector: could not cast array {} to string", array_name);
            return {};
        }

        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        if (!read_result.ok()) {
            return {};
        }

        auto array = read_result.value();
        size_t size = array.num_elements();
        std::vector<std::string> result(size);
        for (tensorstore::Index i = 0; i < array.domain().shape()[0]; ++i) {
            result[i] = array(i);
        }
        return result;

    } catch (const std::exception& ex) {
        spdlog::warn("Error reading string vector {}: {}", array_name, ex.what());
        return {};
    }
}

std::vector<double> ZarrDataReader::ReadFlattenedVector(const std::string& array_name) {
    if (!_initialized) {
        return {};
    }
    std::lock_guard<std::mutex> lock(_read_mutex);

    try {
        std::filesystem::path target_path = std::filesystem::path(_filename) / array_name;

        auto open_result = OpenZarrArray(target_path.string());
        if (!open_result.ok()) {
            return {};
        }

        auto store = open_result.value();
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

        const double* ptr = array.data();
        if (ptr) {
            std::copy(ptr, ptr + size, result.begin());
        } else {
            spdlog::warn("ReadFlattenedVector: Array {} resulted in non-contiguous memory", array_name);
            return {};
        }

        return result;

    } catch (const std::exception& ex) {
        spdlog::warn("Error reading flattened vector {}: {}", array_name, ex.what());
        return {};
    }
}

std::string ZarrDataReader::GetZattrsString(const std::string& array_name) {
    return _impl->GetZarrMetadataFile(_filename, array_name, ".zattrs");
}

std::string ZarrDataReader::GetZarrayString(const std::string& array_name) {
    return _impl->GetZarrMetadataFile(_filename, array_name, ".zarray");
}

//-----------------------------------------------------------------------------
// Mask Support
//-----------------------------------------------------------------------------

bool ZarrDataReader::HasMask() const {
    return !GetActiveMaskPath().empty();
}

std::string ZarrDataReader::GetActiveMaskPath() const {
    if (!_initialized) {
        return "";
    }

    // Check SKY/.zattrs and APERTURE/.zattrs for "active_mask" attribute
    for (const auto& array_name : {"SKY", "APERTURE"}) {
        try {
            std::string zattrs_str = const_cast<ZarrDataReader*>(this)->GetZattrsString(array_name);
            if (zattrs_str.empty() || zattrs_str == "{}") {
                continue;
            }

            nlohmann::json zattrs = nlohmann::json::parse(zattrs_str);
            if (zattrs.contains("active_mask") && zattrs["active_mask"].is_string()) {
                std::string mask_path = zattrs["active_mask"].get<std::string>();
                if (!mask_path.empty()) {
                    // Verify the mask array exists
                    std::filesystem::path full_path = std::filesystem::path(_filename) / mask_path;
                    if (std::filesystem::exists(full_path / ".zarray")) {
                        spdlog::debug("ZarrDataReader: Found active_mask '{}' in {}", mask_path, array_name);
                        return mask_path;
                    }
                }
            }
        } catch (const std::exception& ex) {
            spdlog::debug("ZarrDataReader: Error checking active_mask in {}: {}", array_name, ex.what());
        }
    }

    return "";
}

bool ZarrDataReader::EnsureMaskStore() {
    // Already checked - return cached result
    if (_impl->mask_store_checked) {
        return _impl->has_mask_store;
    }

    // Mark as checked to avoid repeated filesystem lookups
    _impl->mask_store_checked = true;
    _impl->has_mask_store = false;

    // Find active mask path
    std::string mask_path = GetActiveMaskPath();
    if (mask_path.empty()) {
        spdlog::debug("EnsureMaskStore: No active mask found");
        return false;
    }

    try {
        std::filesystem::path full_mask_path = std::filesystem::path(_filename) / mask_path;

        auto open_result = OpenZarrArray(full_mask_path.string(), Impl::GetSharedContext());
        if (!open_result.ok()) {
            spdlog::warn("EnsureMaskStore: Failed to open mask array: {}", open_result.status().ToString());
            return false;
        }

        auto domain = open_result.value().domain();
        if (domain.rank() != kDimSize5D) {
            spdlog::warn("EnsureMaskStore: Mask array has unexpected rank {} (expected 5)", domain.rank());
            return false;
        }

        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<int8_t>>(open_result.value());
        if (!typed_store_result.ok()) {
            spdlog::warn("EnsureMaskStore: Error casting to int8 store: {}", typed_store_result.status().ToString());
            return false;
        }

        _impl->mask_store = typed_store_result.value();
        _impl->cached_mask_path = mask_path;
        _impl->has_mask_store = true;

        spdlog::info("EnsureMaskStore: Cached mask store for '{}'", mask_path);
        return true;

    } catch (const std::exception& ex) {
        spdlog::warn("EnsureMaskStore: Exception: {}", ex.what());
        return false;
    }
}

bool ZarrDataReader::ReadMaskSlice(casacore::Array<bool>& buffer, const casacore::Slicer& section) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }

    std::lock_guard<std::mutex> lock(_read_mutex);

    if (!EnsureMaskStore()) {
        buffer.resize(section.length());
        buffer = true;
        return false;
    }

    try {
        const auto& start = section.start();
        const auto& stop = section.end();
        const auto& length = section.length();

        const int width_x = length[0];
        const int height_y = length[1];
        const int num_freq = length[2];
        const int num_stokes = length[3];
        const size_t num_elements = static_cast<size_t>(width_x) * height_y * num_freq * num_stokes;

        const tensorstore::Index time_idx = 0;

        auto slice_result = BuildXRadioSlice(_impl->mask_store, start, stop, 0, width_x, time_idx);
        if (!slice_result.ok()) {
            spdlog::error("ReadMaskSlice: Slice failed: {}", slice_result.status().ToString());
            buffer.resize(section.length());
            buffer = true;
            return false;
        }

        auto reorder_result = TransposeToCartaOrder(std::move(slice_result));
        if (!reorder_result.ok()) {
            spdlog::error("ReadMaskSlice: Reorder failed: {}", reorder_result.status().ToString());
            buffer.resize(section.length());
            buffer = true;
            return false;
        }

        std::vector<int8_t> int8_buffer(num_elements);
        int8_t* int8_ptr = int8_buffer.data();

        auto output_array = Build4DContiguousArrayView(int8_ptr, width_x, height_y, num_freq, num_stokes);

        auto read_status = PerformTensorStoreRead(reorder_result.value(), output_array);
        if (!read_status.ok()) {
            spdlog::error("ReadMaskSlice: Read failed: {}", read_status.ToString());
            buffer.resize(section.length());
            buffer = true;
            return false;
        }

        buffer.resize(length);
        bool* bool_ptr = buffer.data();
        for (size_t i = 0; i < num_elements; ++i) {
            bool_ptr[i] = (int8_ptr[i] == 0);
        }

        spdlog::debug("ReadMaskSlice: Successfully read mask slice with shape {}", length.toString());
        return true;

    } catch (const std::exception& ex) {
        spdlog::error("ReadMaskSlice: Exception: {}", ex.what());
        buffer.resize(section.length());
        buffer = true;
        return false;
    }
}

} // namespace carta
