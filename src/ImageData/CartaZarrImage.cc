/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"
#include "Logger/Logger.h"

#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/lattices/Lattices/ArrayLattice.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

// TensorStore includes
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"

using namespace casacore;

namespace carta {

CartaZarrImage::CartaZarrImage(const std::string& filename) : _name(filename) {
    // Initialize TensorStore context
    _context = tensorstore::Context::Default();
    
    // Try to read actual Zarr metadata
    try {
        std::filesystem::path zarr_path(filename);
        std::filesystem::path zarray_path = zarr_path / ".zarray";
        
        if (std::filesystem::exists(zarray_path)) {
            std::ifstream zarray_file(zarray_path);
            nlohmann::json zarray_json;
            zarray_file >> zarray_json;
            
            // Read shape from .zarray metadata
            if (zarray_json.contains("shape")) {
                auto shape_array = zarray_json["shape"];
                std::vector<int> shape_vec;
                for (auto& dim : shape_array) {
                    shape_vec.push_back(dim.get<int>());
                }
                if (!shape_vec.empty()) {
                    _shape = IPosition(shape_vec);
                    spdlog::info("Zarr image shape: {}", _shape.toString());
                } else {
                    _shape = IPosition(2, 1, 1);  // Default fallback
                }
            } else {
                _shape = IPosition(2, 1, 1);  // Default fallback
            }
        } else {
            _shape = IPosition(2, 1, 1);  // Default fallback
        }
        
        // Initialize TensorStore for this Zarr file
        initializeTensorStore();
        
    } catch (std::exception& e) {
        spdlog::warn("Failed to read Zarr metadata for {}: {}", filename, e.what());
        _shape = IPosition(2, 1, 1);  // Default fallback
    }
    
    setupCoordinateSystem();
}

void CartaZarrImage::setupCoordinateSystem() {
    // Try to read coordinate system from .zattrs file
    try {
        std::filesystem::path zarr_path(_name.c_str());
        std::filesystem::path zattrs_path = zarr_path / ".zattrs";
        
        if (std::filesystem::exists(zattrs_path)) {
            std::ifstream zattrs_file(zattrs_path);
            nlohmann::json zattrs_json;
            zattrs_file >> zattrs_json;
            
            spdlog::info("Found .zattrs file for Zarr image: {}", _name);
            
            // Parse WCS-like coordinate information
            if (parseWCSFromZattrs(zattrs_json)) {
                spdlog::info("Successfully parsed coordinate system from .zattrs");
                return;
            }
        }
    } catch (std::exception& e) {
        spdlog::warn("Failed to read .zattrs for {}: {}", _name, e.what());
    }
    
    // Fallback to minimal coordinate system
    createMinimalCoordinateSystem();
}

bool CartaZarrImage::parseWCSFromZattrs(const nlohmann::json& zattrs) {
    try {
        // Check if we have WCS information
        bool has_wcs = zattrs.contains("CTYPE1") && zattrs.contains("CTYPE2");
        if (!has_wcs) {
            return false;
        }
        
        // Get dimension names if available
        std::vector<std::string> axis_names;
        if (zattrs.contains("_ARRAY_DIMENSIONS")) {
            for (const auto& dim : zattrs["_ARRAY_DIMENSIONS"]) {
                axis_names.push_back(dim.get<std::string>());
            }
        }
        
        // Create direction coordinate (RA/DEC)
        if (zattrs.contains("CTYPE1") && zattrs.contains("CTYPE2")) {
            std::string ctype1 = zattrs["CTYPE1"].get<std::string>();
            std::string ctype2 = zattrs["CTYPE2"].get<std::string>();
            
            // For now, use a simple direction coordinate
            // TODO: Parse full WCS parameters properly
            casacore::DirectionCoordinate dir_coord;
            _coord_sys.addCoordinate(dir_coord);
            
            spdlog::info("Added direction coordinate: {} {}", ctype1, ctype2);
        }
        
        // Add spectral coordinate if present
        if (zattrs.contains("CTYPE3") && _shape.size() >= 3) {
            std::string ctype3 = zattrs["CTYPE3"].get<std::string>();
            
            if (ctype3 == "FREQ" || ctype3.find("FREQ") != std::string::npos) {
                double crval3 = zattrs.contains("CRVAL3") ? zattrs["CRVAL3"].get<double>() : 1.4e9;
                double crpix3 = zattrs.contains("CRPIX3") ? zattrs["CRPIX3"].get<double>() - 1.0 : 0.0;
                double cdelt3 = zattrs.contains("CDELT3") ? zattrs["CDELT3"].get<double>() : 1e6;
                
                casacore::SpectralCoordinate spec_coord(casacore::MFrequency::TOPO, crval3, cdelt3, crpix3);
                _coord_sys.addCoordinate(spec_coord);
                
                spdlog::info("Added spectral coordinate: {}", ctype3);
            }
        }
        
        return _coord_sys.nCoordinates() > 0;
        
    } catch (std::exception& e) {
        spdlog::warn("Error parsing WCS from .zattrs: {}", e.what());
        return false;
    }
}

void CartaZarrImage::createMinimalCoordinateSystem() {
    // Create minimal coordinate system for file browser support
    DirectionCoordinate dir_coord;
    SpectralCoordinate spec_coord;
    
    _coord_sys.addCoordinate(dir_coord);
    _coord_sys.addCoordinate(spec_coord);
}

void CartaZarrImage::initializeTensorStore() {
    try {
        // Create TensorStore spec for Zarr
        auto spec_result = tensorstore::Spec::FromJson({
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", _name}
            }}
        });
        
        if (!spec_result.ok()) {
            spdlog::error("Failed to create TensorStore spec for {}: {}", _name, spec_result.status().ToString());
            return;
        }
        
        // Open the TensorStore
        auto open_result = tensorstore::Open(
            spec_result.value(),
            _context,
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read
        ).result();
        
        if (!open_result.ok()) {
            spdlog::error("Failed to open TensorStore for {}: {}", _name, open_result.status().ToString());
            return;
        }
        
        _tensorstore = std::move(open_result).value();
        _tensorstore_initialized = true;
        
        spdlog::info("Successfully initialized TensorStore for {}", _name);
        
        // Verify shape matches what we read from .zarray
        auto ts_shape = _tensorstore.domain().shape();
        spdlog::info("TensorStore shape: [{}]", 
                    [&ts_shape]() {
                        std::string result;
                        for (size_t i = 0; i < ts_shape.size(); ++i) {
                            if (i > 0) result += ", ";
                            result += std::to_string(ts_shape[i]);
                        }
                        return result;
                    }());
        
    } catch (std::exception& e) {
        spdlog::error("Exception in initializeTensorStore for {}: {}", _name, e.what());
        _tensorstore_initialized = false;
    }
}

String CartaZarrImage::imageType() const {
    return "zarr";
}

DataType CartaZarrImage::dataType() const {
    return TpFloat;
}

Bool CartaZarrImage::doGetSlice(Array<float>& buffer, const Slicer& section) {
    if (!_tensorstore_initialized) {
        spdlog::error("TensorStore not initialized for {}", _name);
        return false;
    }
    
    try {
        // Convert casacore Slicer to TensorStore transform
        const IPosition& start = section.start();
        const IPosition& length = section.length();
        const IPosition& stride = section.stride();
        
        spdlog::debug("doGetSlice: start={}, length={}, stride={}", 
                     start.toString(), length.toString(), stride.toString());
        
        // Create TensorStore transform based on the section
        tensorstore::Result<tensorstore::IndexTransform<>> transform = 
            tensorstore::IdentityTransform(_tensorstore.domain());
        
        // Apply slicing for each dimension
        for (int dim = 0; dim < start.size() && dim < _tensorstore.rank(); ++dim) {
            if (length[dim] == 1) {
                // Single index slice
                auto slice_result = transform.value() | tensorstore::Dims(dim).IndexSlice(start[dim]);
                if (!slice_result.ok()) {
                    spdlog::error("Failed to apply IndexSlice: {}", slice_result.status().ToString());
                    return false;
                }
                transform = std::move(slice_result);
            } else {
                // Range slice
                tensorstore::Index ts_start = start[dim];
                tensorstore::Index ts_stop = start[dim] + length[dim];
                auto slice_result = transform.value() | tensorstore::Dims(dim).BoxSlice(
                    tensorstore::Box<>({ts_start}, {ts_stop - ts_start}));
                if (!slice_result.ok()) {
                    spdlog::error("Failed to apply BoxSlice: {}", slice_result.status().ToString());
                    return false;
                }
                transform = std::move(slice_result);
            }
        }
        
        if (!transform.ok()) {
            spdlog::error("Failed to create transform: {}", transform.status().ToString());
            return false;
        }
        
        // Apply transform to TensorStore
        auto constrained_store = _tensorstore | transform.value();
        if (!constrained_store.ok()) {
            spdlog::error("Failed to apply transform: {}", constrained_store.status().ToString());
            return false;
        }
        
        // Read data from TensorStore
        auto read_result = tensorstore::Read<tensorstore::zero_origin>(constrained_store.value()).result();
        if (!read_result.ok()) {
            spdlog::error("Failed to read data: {}", read_result.status().ToString());
            return false;
        }
        
        auto zarr_array = std::move(read_result.value());
        
        // Convert TensorStore array to casacore Array
        IPosition buffer_shape = length;
        buffer.resize(buffer_shape);
        
        // Copy data from TensorStore array to casacore array
        const float* src_data = reinterpret_cast<const float*>(zarr_array.data());
        float* dest_data = buffer.data();
        
        size_t num_elements = buffer.nelements();
        std::copy(src_data, src_data + num_elements, dest_data);
        
        spdlog::debug("Successfully read {} elements from Zarr file", num_elements);
        return true;
        
    } catch (std::exception& e) {
        spdlog::error("Exception in doGetSlice for {}: {}", _name, e.what());
        return false;
    }
}

void CartaZarrImage::doPutSlice(const Array<float>& buffer, const IPosition& where, const IPosition& stride) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::doPutSlice not implemented");
}

Bool CartaZarrImage::doGetMaskSlice(Array<Bool>& buffer, const Slicer& section) {
    // Stub implementation - not needed for file browser
    return false;
}

Bool CartaZarrImage::isMasked() const {
    return false;
}

Bool CartaZarrImage::isPersistent() const {
    return true;
}

Bool CartaZarrImage::isWritable() const {
    return false;
}

String CartaZarrImage::name(Bool stripPath) const {
    if (stripPath) {
        auto pos = _name.find_last_of('/');
        if (pos != std::string::npos) {
            return _name.substr(pos + 1);
        }
    }
    return _name;
}

IPosition CartaZarrImage::shape() const {
    return _shape;
}

void CartaZarrImage::resize(const TiledShape& newShape) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::resize not implemented");
}

Bool CartaZarrImage::ok() const {
    return _tensorstore_initialized;  // Only return true if TensorStore is properly initialized
}

void CartaZarrImage::flush() {
    // Stub implementation - not needed for file browser
}

void CartaZarrImage::tempClose() {
    // Stub implementation - not needed for file browser
}

void CartaZarrImage::reopen() {
    // Stub implementation - not needed for file browser
}

Bool CartaZarrImage::hasPixelMask() const {
    return false;
}

const Lattice<Bool>& CartaZarrImage::pixelMask() const {
    // Return a dummy lattice - use static to avoid recreation
    static ArrayLattice<Bool> dummy_lattice(Array<Bool>(IPosition(2, 1, 1)));
    return dummy_lattice;
}

Lattice<Bool>& CartaZarrImage::pixelMask() {
    // Return a dummy lattice - use static to avoid recreation
    static ArrayLattice<Bool> dummy_lattice(Array<Bool>(IPosition(2, 1, 1)));
    return dummy_lattice;
}

const LatticeRegion* CartaZarrImage::getRegionPtr() const {
    return nullptr;
}

ImageInterface<float>* CartaZarrImage::cloneII() const {
    return new CartaZarrImage(_name);
}

const CoordinateSystem& CartaZarrImage::coordinates() const {
    return _coord_sys;
}

} // namespace carta
