/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "ZarrDataReader.h"

// Standard library includes MUST come before TensorStore to ensure types are defined
#include <filesystem>
#include <thread>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

// TensorStore includes - isolated to implementation file
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/static_cast.h" 
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/context.h"
// Include driver headers if necessary for Read
#include "tensorstore/driver/read.h" 

#include "Util/Image.h"

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

//-----------------------------------------------------------------------------
// Pimpl Implementation Helper
//-----------------------------------------------------------------------------
struct ZarrDataReader::Impl {
    tensorstore::Context context;
    tensorstore::TensorStore<> store;
    
    // Create Context
    void CreateContext() {
        unsigned int num_cpus = std::thread::hardware_concurrency();
        if (num_cpus == 0) {
            num_cpus = kDefaultCpuCount;
        }
        
        nlohmann::json context_spec = {
            {"cache_pool", {
                {"total_bytes_limit", 0}
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
const std::string& ZarrDataReader::GetFilename() const { return _filename; }
int ZarrDataReader::NumDimensions() const { return _shape.size(); }

std::string ZarrDataReader::FindArrayPath() const {
    std::filesystem::path base_path(_filename);
    
    std::vector<std::string> common_array_names = {"SKY", "DATA", "ARRAY", "0"};
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
        
        _initialized = true;
        spdlog::info("ZarrDataReader initialized: ZARR shape={}, CARTA shape={}", 
                    _original_shape.toString(), _shape.toString());
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("Exception initializing ZarrDataReader: {}", ex.what());
        return false;
    }
}

bool ZarrDataReader::ReadSlice(const casacore::Slicer& section, casacore::Array<float>& buffer) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(_read_mutex);
    
    try {
        const auto& start = section.start();
        const auto& length = section.length();

        spdlog::debug("ReadSlice: start={}, length={}", start.toString(), length.toString());
        
        auto zarr_start = Impl::MapToZarrCoords(start);
        spdlog::debug("ReadSlice: zarr_start=[{},{},{},{},{}]", 
                      zarr_start[0], zarr_start[1], zarr_start[2], zarr_start[3], zarr_start[4]);
        
        // Build shape for transform - XRADIO always 5D [T, F, S, L, M]
        // CARTA length is [X, Y, F, S], L -> X, M -> Y
        std::vector<tensorstore::Index> zarr_shape(kDimSize5D, 1);
        zarr_shape[0] = 1;            // time
        if (length.size() > 2) { zarr_shape[1] = length[2]; }    // freq
        if (length.size() > 3) { zarr_shape[2] = length[3]; }    // pol
        if (length.size() > 0) { zarr_shape[3] = length[0]; }    // L (CARTA X)
        if (length.size() > 1) { zarr_shape[4] = length[1]; }    // M (CARTA Y)
        
        spdlog::debug("ReadSlice: zarr_shape=[{},{},{},{},{}]", 
                      zarr_shape[0], zarr_shape[1], zarr_shape[2], zarr_shape[3], zarr_shape[4]);
        spdlog::debug("ReadSlice: original_shape={}", _original_shape.toString());
        
        // Bounds validation - check that requested slice is within the original array bounds
        for (size_t dim = 0; dim < kDimSize5D; ++dim) {
            tensorstore::Index end_idx = zarr_start[dim] + zarr_shape[dim] - 1;
            if (zarr_start[dim] < 0 || end_idx >= _original_shape[dim]) {
                spdlog::error("ReadSlice: Bounds error! dim={}, start={}, end={}, array_size={}",
                              dim, zarr_start[dim], end_idx, _original_shape[dim]);
                return false;
            }
        }
        spdlog::debug("ReadSlice: Bounds check passed");
        
        tensorstore::TensorStore<> sliced_store = _impl->store;
        
        for (size_t dim = 0; dim < zarr_start.size(); ++dim) {
            auto transform_result = sliced_store | 
                tensorstore::Dims(static_cast<tensorstore::DimensionIndex>(dim))
                    .ClosedInterval(zarr_start[dim], zarr_start[dim] + zarr_shape[dim] - 1);
            
            if (!transform_result.ok()) {
                spdlog::error("ZarrDataReader: Error creating slice transform: {}", transform_result.status().ToString());
                return false;
            }
            sliced_store = transform_result.value();
        }
        spdlog::debug("ReadSlice: Slice transform created");
        
        // Cast to typed float store
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(sliced_store);
        if (!typed_store_result.ok()) {
            spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }
        spdlog::debug("ReadSlice: Float store cast succeeded");

        // OPTIMIZATION: Reduce copies from 3 to 2
        // Read WITHOUT transpose to get contiguous C-order data, then transpose directly to output
        // This avoids one MakeCopy while keeping correct data access
        
        int width_x = length[0];   // target X size (CARTA L)
        int height_y = length[1];  // target Y size (CARTA M)
        int num_freq = (length.size() > 2) ? length[2] : 1;
        int num_stokes = (length.size() > 3) ? length[3] : 1;
        size_t total_elements = static_cast<size_t>(width_x) * height_y * num_freq * num_stokes;
        spdlog::debug("ReadSlice: Dimensions: {}x{}, freq={}, stokes={}, total={}", 
                      width_x, height_y, num_freq, num_stokes, total_elements);
        
        // Pre-allocate output buffer
        buffer.resize(length);
        float* dst_ptr = buffer.data();
        if (!dst_ptr) {
            spdlog::error("ReadSlice: buffer.data() returned null after resize");
            return false;
        }
        spdlog::debug("ReadSlice: Buffer allocated");
        
        // Read directly - TensorStore returns contiguous C-order data for Zarr
        spdlog::debug("ReadSlice: Starting TensorStore read...");
        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        spdlog::debug("ReadSlice: TensorStore read completed");
        
        if (!read_result.ok()) {
            spdlog::error("TensorStore read failed: {}", read_result.status().ToString());
            return false;
        }
        
        auto& result_array = read_result.value();
        spdlog::debug("ReadSlice: Result array has {} elements, rank={}", 
                      result_array.num_elements(), result_array.rank());
        
        // Ensure contiguous C-order layout
        // usage of MakeCopy returns Array directly here
        auto dense_array = tensorstore::MakeCopy(result_array, tensorstore::c_order);
        const float* src_ptr = dense_array.data();
        
        if (!src_ptr) {
            spdlog::error("ReadSlice: data pointer is null");
            return false;
        }
        
        if (static_cast<size_t>(result_array.num_elements()) != total_elements) {
            spdlog::error("ZarrDataReader: Read size mismatch. Expected {}, got {}", 
                          total_elements, result_array.num_elements());
            return false;
        }
        spdlog::debug("ReadSlice: Data pointer validated, starting transpose...");
        
        // Source is C-order [T, F, S, L, M] with M fastest (after slicing: [1, num_freq, num_stokes, width_x, height_y])
        // Destination is Fortran-order [X, Y, F, S] with X fastest
        // T dimension is always 1, so we can ignore it
        // Need to: swap L<->X axes AND transpose from C-order to Fortran-order
        
        // Fast path: For spectral profile (1x1 spatial), just reorder F,S dimensions
        if (width_x == 1 && height_y == 1) {
            // src is [F, S] in C-order (S fastest within each F)
            // dst is [F, S] in Fortran-order (F fastest)
            for (int is = 0; is < num_stokes; ++is) {
                for (int ifreq = 0; ifreq < num_freq; ++ifreq) {
                    // src index: ifreq * num_stokes + is (C-order: S fastest)
                    // dst index: is * num_freq + ifreq (Fortran: F fastest) - but casacore uses [F,S] order
                    // Actually for 1x1, src[f,s] -> dst[f,s], just copy directly
                    size_t src_idx = static_cast<size_t>(ifreq) * num_stokes + is;
                    size_t dst_idx = static_cast<size_t>(is) * num_freq + ifreq;
                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            }
        } else {
            // General case: Full transpose
            // src is C-order [F, S, L, M] with M fastest (ignoring T=1)
            // dst is Fortran-order [X, Y, F, S] with X fastest
            // Mapping: L->X, M->Y
            size_t plane_size = static_cast<size_t>(width_x) * height_y;
            
            for (int is = 0; is < num_stokes; ++is) {
                for (int ifreq = 0; ifreq < num_freq; ++ifreq) {
                    // Source plane offset in C-order: (ifreq * num_stokes + is) * plane_size
                    size_t src_plane_offset = (static_cast<size_t>(ifreq) * num_stokes + is) * plane_size;
                    // Dest plane offset in Fortran-order: (is * num_freq + ifreq) * plane_size
                    size_t dst_plane_offset = (static_cast<size_t>(is) * num_freq + ifreq) * plane_size;
                    
                    const float* src_plane = src_ptr + src_plane_offset;
                    float* dst_plane = dst_ptr + dst_plane_offset;
                    
                    // Within each plane:
                    // src is [L, M] C-order: M fastest, index = ix * height_y + iy
                    // dst is [X, Y] Fortran-order: X fastest, index = ix + iy * width_x
                    for (int iy = 0; iy < height_y; ++iy) {
                        for (int ix = 0; ix < width_x; ++ix) {
                            size_t src_idx = static_cast<size_t>(ix) * height_y + iy;
                            size_t dst_idx = static_cast<size_t>(ix) + iy * width_x;
                            dst_plane[dst_idx] = src_plane[src_idx];
                        }
                    }
                }
            }
        }

        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("Exception in ReadSlice: {}", ex.what());
        return false;
    }
}

bool ZarrDataReader::ReadChannel(int channel, int stokes, std::vector<float>& data) {
    if (!_initialized) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(_read_mutex);
    
    try {
        int width = _shape[0];
        int height = _shape[1];
        size_t channel_size = static_cast<size_t>(width) * height;
        
        // XRADIO always 5D [T, F, S, L, M]
        // L -> X (width), M -> Y (height)
        std::vector<tensorstore::Index> zarr_start = {
            0,
            static_cast<tensorstore::Index>(channel),
            static_cast<tensorstore::Index>(stokes),
            0,  // L start
            0   // M start
        };
        std::vector<tensorstore::Index> zarr_shape = {
            1,
            1,
            1,
            static_cast<tensorstore::Index>(width),   // L = X = width
            static_cast<tensorstore::Index>(height)   // M = Y = height
        };
        
        // Bounds validation
        for (size_t dim = 0; dim < kDimSize5D; ++dim) {
            tensorstore::Index end_idx = zarr_start[dim] + zarr_shape[dim] - 1;
            if (zarr_start[dim] < 0 || end_idx >= _original_shape[dim]) {
                spdlog::error("ReadChannel: Bounds error! dim={}, start={}, end={}, array_size={}",
                              dim, zarr_start[dim], end_idx, _original_shape[dim]);
                return false;
            }
        }
        
        tensorstore::TensorStore<> sliced_store = _impl->store;
        
        for (size_t dim = 0; dim < zarr_start.size(); ++dim) {
            auto transform_result = sliced_store | 
                tensorstore::Dims(static_cast<tensorstore::DimensionIndex>(dim))
                    .ClosedInterval(zarr_start[dim], zarr_start[dim] + zarr_shape[dim] - 1);
                    
            if (!transform_result.ok()) {
                return false;
            }
            sliced_store = transform_result.value();
        }
        
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(sliced_store);
        if (!typed_store_result.ok()) {
             spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }

        // Read and transpose directly to output
        data.resize(channel_size);
        float* dst_ptr = data.data();
        
        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        
        if (!read_result.ok()) {
            spdlog::error("TensorStore ReadChannel failed: {}", read_result.status().ToString());
            return false;
        }

        auto& result_array = read_result.value();
        
        // Use data() directly if contiguous, otherwise fallback to MakeCopy
        // Ensure contiguous C-order layout
        auto dense_array = tensorstore::MakeCopy(result_array, tensorstore::c_order);
        const float* src_ptr = dense_array.data();
        
        if (!src_ptr) {
            spdlog::error("ReadChannel: data pointer is null");
            return false;
        }
        
        // XY transpose: src is [L, M] C-order (M fastest), dst is [X, Y] Fortran (X fastest)
        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                size_t src_idx = static_cast<size_t>(ix) * height + iy;
                size_t dst_idx = static_cast<size_t>(ix) + iy * width;
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
        }
        
        return true;
        
    } catch (const std::exception& ex) {
        spdlog::error("Exception in ReadChannel: {}", ex.what());
        return false;
    }
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
        
        tensorstore::TensorStore<> sliced_store = _impl->store;
        
        for (size_t dim = 0; dim < zarr_start.size(); ++dim) {
             auto transform_result = sliced_store | 
                tensorstore::Dims(static_cast<tensorstore::DimensionIndex>(dim))
                    .ClosedInterval(zarr_start[dim], zarr_start[dim] + zarr_shape[dim] - 1);
            
            if (!transform_result.ok()) {
                return false;
            }
            sliced_store = transform_result.value();
        }
        
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(sliced_store);
        if (!typed_store_result.ok()) {
             spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }

        // Read and transpose directly to output
        data.resize(chunk_size);
        float* dst_ptr = data.data();
        
        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        
        if (!read_result.ok()) {
            spdlog::error("TensorStore GetChunk failed: {}", read_result.status().ToString());
            return false;
        }
        
        auto& result_array = read_result.value();
        
        // Use data() directly if contiguous, otherwise fallback to MakeCopy
        // Ensure contiguous C-order layout
        auto dense_array = tensorstore::MakeCopy(result_array, tensorstore::c_order);
        const float* src_ptr = dense_array.data();
        
        if (!src_ptr) {
            spdlog::error("GetChunk: data pointer is null");
            return false;
        }
        
        // XY transpose: src is [L, M] C-order (M fastest), dst is [X, Y] Fortran (X fastest)
        for (int iy = 0; iy < data_height; ++iy) {
            for (int ix = 0; ix < data_width; ++ix) {
                size_t src_idx = static_cast<size_t>(ix) * data_height + iy;
                size_t dst_idx = static_cast<size_t>(ix) + iy * data_width;
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
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
        
        tensorstore::TensorStore<> sliced_store = _impl->store;
        
        for (size_t dim = 0; dim < zarr_start.size(); ++dim) {
            auto transform_result = sliced_store | 
                tensorstore::Dims(static_cast<tensorstore::DimensionIndex>(dim))
                    .ClosedInterval(zarr_start[dim], zarr_start[dim] + zarr_shape[dim] - 1);
            
            if (!transform_result.ok()) {
                spdlog::error("ReadSpectralProfile: Error creating slice transform: {}", 
                             transform_result.status().ToString());
                return false;
            }
            sliced_store = transform_result.value();
        }
        
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(sliced_store);
        if (!typed_store_result.ok()) {
            spdlog::error("ReadSpectralProfile: Error casting to float store: {}", 
                         typed_store_result.status().ToString());
            return false;
        }
        
        // Read the data
        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        
        if (!read_result.ok()) {
            spdlog::error("ReadSpectralProfile: TensorStore read failed: {}", 
                         read_result.status().ToString());
            return false;
        }
        
        auto result_array = read_result.value();
        
        spdlog::debug("ReadSpectralProfile: result has {} elements, rank={}", 
                     result_array.num_elements(), result_array.rank());
        
        if (static_cast<size_t>(result_array.num_elements()) != static_cast<size_t>(num_channels)) {
            spdlog::error("ReadSpectralProfile: Size mismatch. Expected {}, got {}", 
                         num_channels, result_array.num_elements());
            return false;
        }
        
        // Make a contiguous C-order copy to ensure linear memory access
        auto contiguous_result = tensorstore::MakeCopy(result_array, tensorstore::c_order);
        
        const float* src_ptr = contiguous_result.data();
        if (!src_ptr) {
            spdlog::error("ReadSpectralProfile: Null data pointer after MakeCopy");
            return false;
        }
        
        // Copy to output vector
        data.resize(num_channels);
        std::copy(src_ptr, src_ptr + num_channels, data.begin());
        
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
    if (!_initialized) return {};
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

std::string ZarrDataReader::GetAttributeString(const std::string& array_name, const std::string& attr_name) {
    if (!_initialized) return "";
    
    // Read .zattrs for the array
    std::filesystem::path base_path(_filename);
    std::filesystem::path attrs_path = base_path / array_name / ".zattrs";
    
    if (!std::filesystem::exists(attrs_path)) {
        return "";
    }
    
    try {
        std::ifstream fstr(attrs_path);
        nlohmann::json jsonObj;
        fstr >> jsonObj;
        
        if (jsonObj.contains(attr_name)) {
            if (jsonObj[attr_name].is_string()) {
                return jsonObj[attr_name].get<std::string>();
            } else if (jsonObj[attr_name].is_array() && !jsonObj[attr_name].empty() && jsonObj[attr_name][0].is_string()) {
                return jsonObj[attr_name][0].get<std::string>(); // e.g. units: ["rad"]
            }
        }
    } catch (...) {}
    
    return "";
}

std::string ZarrDataReader::GetZattrsString(const std::string& array_name) {
    if (!_initialized) return "{}";
    
    std::filesystem::path base_path(_filename);
    std::filesystem::path attrs_path = base_path;
    
    if (!array_name.empty()) {
        attrs_path = attrs_path / array_name;
    }
    
    attrs_path = attrs_path / ".zattrs";
    
    if (!std::filesystem::exists(attrs_path)) {
        return "{}";
    }
    
    try {
        std::ifstream fstr(attrs_path);
        std::stringstream buffer;
        buffer << fstr.rdbuf();
        return buffer.str();
    } catch (...) {
        return "{}";
    }
}

std::map<std::string, std::string> ZarrDataReader::GetZattrMap(const std::string& array_name) {
    std::map<std::string, std::string> result;
    try {
        std::string json_str = GetZattrsString(array_name);
        nlohmann::json jsonObj = nlohmann::json::parse(json_str);
        
        for (auto& [key, val] : jsonObj.items()) {
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
