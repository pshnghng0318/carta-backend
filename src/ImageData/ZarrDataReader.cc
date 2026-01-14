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

constexpr size_t kDefaultCacheSizeMB = 128;
constexpr size_t kDefaultCpuCount = 8;
constexpr size_t kDimSize5D = 5;
constexpr size_t kDimSize4D = 4;
constexpr size_t kDimSize3D = 3;

//-----------------------------------------------------------------------------
// Pimpl Implementation Helper
//-----------------------------------------------------------------------------
struct ZarrDataReader::Impl {
    tensorstore::Context context;
    tensorstore::TensorStore<> store;
    bool is_5d = false;
    
    // Create Context
    void CreateContext() {
        unsigned int num_cpus = std::thread::hardware_concurrency();
        if (num_cpus == 0) {
            num_cpus = kDefaultCpuCount;
        }
        
        nlohmann::json context_spec = {
            {"cache_pool", {
                {"total_bytes_limit", static_cast<size_t>(kDefaultCacheSizeMB) << 20}
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
    
    // Map Coordinates Helper
    std::vector<tensorstore::Index> MapToZarrCoords(
        const casacore::IPosition& start, const casacore::IPosition& length) const {
        
        std::vector<tensorstore::Index> result;
        
        if (is_5d && start.size() >= kDimSize4D) {
            // CARTA [x, y, freq, stokes] -> ZARR [time, freq, pol, l, m]
            // time=0 (always first time slice)
            result.push_back(0);                    // time
            result.push_back(start[2]);             // freq
            result.push_back(start[3]);             // pol
            result.push_back(start[0]);             // l (x)
            result.push_back(start[1]);             // m (y)
        } else {
            // Direct mapping for 2D/3D/4D
            for (size_t i = 0; i < start.size(); ++i) {
                result.push_back(start[i]);
            }
        }
        
        return result;
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
        
        _impl->is_5d = (ts_shape.size() == kDimSize5D);
        
        if (_impl->is_5d) {
            std::vector<int> carta_shape;
            carta_shape.push_back(orig_shape_vec[3]); // l (x)
            carta_shape.push_back(orig_shape_vec[4]); // m (y)
            carta_shape.push_back(orig_shape_vec[1]); // freq
            carta_shape.push_back(orig_shape_vec[2]); // pol
            _shape = casacore::IPosition(carta_shape);
        } else {
            _shape = _original_shape;
        }
        
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
        
        auto zarr_start = _impl->MapToZarrCoords(start, length);
        
        // Build shape for transform
        std::vector<tensorstore::Index> zarr_shape;
        if (_impl->is_5d && length.size() >= kDimSize4D) {
            zarr_shape.push_back(1);            // time
            zarr_shape.push_back(length[2]);    // freq
            zarr_shape.push_back(length[3]);    // pol
            zarr_shape.push_back(length[0]);    // l (x)
            zarr_shape.push_back(length[1]);    // m (y)
        } else {
            for (size_t i = 0; i < length.size(); ++i) {
                zarr_shape.push_back(length[i]);
            }
        }
        
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
        
        // Cast to typed float store
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(sliced_store);
        if (!typed_store_result.ok()) {
            spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }
        
        // Read into new array
        auto read_future = tensorstore::Read(typed_store_result.value());
        auto read_result = read_future.result();
        
        if (!read_result.ok()) {
            spdlog::error("TensorStore read failed: {}", read_result.status().ToString());
            return false;
        }
        
        auto result_array = read_result.value();
        
        // Copy to casacore array
        buffer.resize(length);
        
        // Use data() and num_elements()
        if (static_cast<size_t>(result_array.num_elements()) != length.product()) {
             spdlog::error("ZarrDataReader: Read size mismatch");
             return false;
        }
        
        const float* src_ptr = result_array.data();
        size_t src_size = result_array.num_elements();
        
        // Use buffer iterator to be safe against non-continguous casacore arrays
        std::copy(src_ptr, src_ptr + src_size, buffer.begin());

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
        
        std::vector<tensorstore::Index> zarr_start;
        std::vector<tensorstore::Index> zarr_shape;
        
        if (_impl->is_5d) {
            zarr_start = {0, channel, stokes, 0, 0};
            zarr_shape = {1, 1, 1, width, height};
        } else if (_original_shape.size() == kDimSize4D) {
            zarr_start = {0, 0, channel, stokes};
            zarr_shape = {width, height, 1, 1};
        } else if (_original_shape.size() == kDimSize3D) {
             zarr_start = {0, 0, channel};
             zarr_shape = {width, height, 1};
        } else {
             zarr_start = {0, 0};
             zarr_shape = {width, height};
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
        
        // Explicitly cast to typed store
        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<float>>(sliced_store);
        if (!typed_store_result.ok()) {
             spdlog::error("ZarrDataReader: Error casting to float store: {}", typed_store_result.status().ToString());
            return false;
        }
        
        auto read_future = tensorstore::Read(typed_store_result.value());
        auto read_result = read_future.result();
        
        if (!read_result.ok()) {
            spdlog::error("TensorStore ReadChannel failed: {}", read_result.status().ToString());
            return false;
        }

        auto result_array = read_result.value();
        
        data.resize(channel_size);
        const float* src_ptr = result_array.data();
        size_t src_size = result_array.num_elements();
        
        std::copy(src_ptr, src_ptr + src_size, data.begin());
        
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
        
        std::vector<tensorstore::Index> zarr_start;
        std::vector<tensorstore::Index> zarr_shape;
        
        if (_impl->is_5d) {
             zarr_start = {0, channel, stokes, min_x, min_y};
             zarr_shape = {1, 1, 1, data_width, data_height};
        } else if (_original_shape.size() == kDimSize4D) {
             zarr_start = {min_x, min_y, channel, stokes};
             zarr_shape = {data_width, data_height, 1, 1};
        } else if (_original_shape.size() == kDimSize3D) {
             zarr_start = {min_x, min_y, channel};
             zarr_shape = {data_width, data_height, 1};
        } else {
             zarr_start = {min_x, min_y};
             zarr_shape = {data_width, data_height};
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

        auto read_future = tensorstore::Read(typed_store_result.value());
        auto read_result = read_future.result();
        
        if (!read_result.ok()) {
            spdlog::error("TensorStore GetChunk failed: {}", read_result.status().ToString());
            return false;
        }
        
        auto result_array = read_result.value();
        
        data.resize(chunk_size);
        const float* src_ptr = result_array.data();
        size_t src_size = result_array.num_elements();
        
        std::copy(src_ptr, src_ptr + src_size, data.begin());
        
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
        
        auto context_result = tensorstore::Context::Default(); // Use default context for metadata
        auto spec_result = tensorstore::Spec::FromJson(spec_json);
        if (!spec_result.ok()) {
             // Try searching in subdirs if main path fails
             return {};
        }

        auto open_future = tensorstore::Open(
            spec_result.value(),
            context_result,
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
