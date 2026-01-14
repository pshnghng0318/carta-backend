/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"

#include <filesystem>
#include <fstream>

#include <casacore/casa/OS/Path.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/StokesCoordinate.h>
#include <casacore/casa/Quanta/Unit.h>
#include <casacore/tables/DataMan/TiledFileAccess.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using namespace casacore;

namespace carta {

CartaZarrImage::CartaZarrImage(const std::string& filename)
    : ImageInterface<float>(),
      _reader(std::make_shared<ZarrDataReader>(filename)),
      _name(filename),
      _is_copy(false) {
    
    // Initialize the reader
    if (!_reader->Initialize()) {
        throw AipsError("Failed to initialize ZarrDataReader for: " + filename);
    }
    
    // Get shape from reader
    _shape = _reader->GetShape();
    
    // Set up tiled shape for cursor operations
    _tiled_shape = TiledShape(_shape, TiledFileAccess::makeTileShape(_shape));
    
    // Set up coordinate system
    SetupCoordinateSystem();
    
    spdlog::info("CartaZarrImage created: {} with shape {}", filename, _shape.toString());
}

CartaZarrImage::CartaZarrImage(const CartaZarrImage& other)
    : ImageInterface<float>(other),
      _reader(other._reader),  // Share reader!
      _shape(other._shape),
      _name(other._name),
      _tiled_shape(other._tiled_shape),
      _is_copy(true) {
    
    spdlog::debug("CartaZarrImage copy constructor: sharing reader for {}", _name);
}

CartaZarrImage::~CartaZarrImage() {
    if (!_is_copy) {
        spdlog::debug("CartaZarrImage destructor: cleaning up {}", _name);
    }
}

// ============================================================================
// ImageInterface implementation
// ============================================================================

String CartaZarrImage::imageType() const {
    return "CartaZarrImage";
}

String CartaZarrImage::name(Bool stripPath) const {
    if (stripPath) {
        Path path(_name);
        return path.baseName();
    }
    return _name;
}

IPosition CartaZarrImage::shape() const {
    return _shape;
}

Bool CartaZarrImage::ok() const {
    return _reader && _reader->IsInitialized();
}

DataType CartaZarrImage::dataType() const {
    return TpFloat;
}

Bool CartaZarrImage::doGetSlice(Array<float>& buffer, const Slicer& section) {
    // Simple delegation to reader - matching CartaFitsImage pattern
    if (!_reader || !_reader->IsInitialized()) {
        spdlog::error("ZarrDataReader not initialized");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(_slice_mutex);
    return _reader->ReadSlice(section, buffer);
}

void CartaZarrImage::doPutSlice(const Array<float>& buffer, 
                                 const IPosition& where, 
                                 const IPosition& stride) {
    throw AipsError("CartaZarrImage::doPutSlice - image is not writable");
}

const LatticeRegion* CartaZarrImage::getRegionPtr() const {
    return nullptr;
}

ImageInterface<float>* CartaZarrImage::cloneII() const {
    return new CartaZarrImage(*this);
}

void CartaZarrImage::resize(const TiledShape& newShape) {
    throw AipsError("CartaZarrImage::resize - image is not writable");
}

uInt CartaZarrImage::advisedMaxPixels() const {
    return _tiled_shape.tileShape().product();
}

IPosition CartaZarrImage::doNiceCursorShape(uInt maxPixels) const {
    return _tiled_shape.tileShape();
}

// ============================================================================
// Mask-related methods (ZARR typically doesn't have masks)
// ============================================================================

Bool CartaZarrImage::isMasked() const {
    return false;
}

Bool CartaZarrImage::hasPixelMask() const {
    return false;
}

const Lattice<Bool>& CartaZarrImage::pixelMask() const {
    throw AipsError("CartaZarrImage::pixelMask - no pixel mask");
}

Lattice<Bool>& CartaZarrImage::pixelMask() {
    throw AipsError("CartaZarrImage::pixelMask - no pixel mask");
}

Bool CartaZarrImage::doGetMaskSlice(Array<Bool>& buffer, const Slicer& section) {
    // Return all true (no mask)
    buffer.resize(section.length());
    buffer = true;
    return false;
}

// ============================================================================
// Additional methods
// ============================================================================

Bool CartaZarrImage::isPersistent() const {
    return true;
}

Bool CartaZarrImage::isWritable() const {
    return false;
}

void CartaZarrImage::flush() {
    // Read-only, nothing to flush
}

void CartaZarrImage::tempClose() {
    // TensorStore handles its own caching
}

void CartaZarrImage::reopen() {
    // TensorStore handles its own caching
}

// ============================================================================
// Coordinate system setup
// ============================================================================

void CartaZarrImage::SetupCoordinateSystem() {
    try {
        // Try to parse WCS from ZARR metadata first
        if (ParseWCSFromMetadata()) {
            return;
        }
        
        // Fall back to default coordinate system
        CreateDefaultCoordinateSystem();
        
    } catch (const std::exception& e) {
        spdlog::warn("Error setting up coordinate system: {}, using default", e.what());
        CreateDefaultCoordinateSystem();
    }
}


// Use explicit namespace or typedef to avoid ambiguity if strict
using casacore::String;

bool CartaZarrImage::ParseWCSFromMetadata() {
    try {
        std::map<std::string, std::string> root_attrs = _reader->GetZattrMap(""); 
        
        nlohmann::json zattrs = nlohmann::json::parse(_reader->GetZattrsString(""));
        nlohmann::json wcs_dict;
        
        // --- 1. Locate WCS Direction Info ---
        if (zattrs.contains("direction")) {
            wcs_dict = zattrs["direction"];
        } else if (zattrs.contains("SKY") && zattrs["SKY"].contains("direction_info")) {
            wcs_dict = zattrs["SKY"]["direction_info"];
        } else if (zattrs.contains("direction_info")) {
            wcs_dict = zattrs["direction_info"];
        } else {
            return false;
        }
        
        if (wcs_dict.empty()) {
            return false;
        }

        // Extract Reference Info
        if (!wcs_dict.contains("reference") || !wcs_dict["reference"].contains("data")) {
            return false;
        }
        
        auto ref_data = wcs_dict["reference"]["data"];
        double crval1_rad = ref_data[0];
        double crval2_rad = ref_data[1];
        
        std::string projection_str = "SIN";
        if (wcs_dict.contains("projection")) {
            projection_str = wcs_dict["projection"].get<std::string>();
        }
        
        // Extract PC Matrix (default identity)
        Matrix<double> pc_matrix(2, 2);
        pc_matrix = 0.0;
        pc_matrix(0,0) = 1.0; pc_matrix(1,1) = 1.0;
        
        if (wcs_dict.contains("pc") && wcs_dict["pc"].contains("_value")) {
            auto pc_val = wcs_dict["pc"]["_value"];
            if (pc_val.size() >= 2 && pc_val[0].size() >= 2) {
                 pc_matrix(0,0) = pc_val[0][0];
                 pc_matrix(0,1) = pc_val[0][1];
                 pc_matrix(1,0) = pc_val[1][0];
                 pc_matrix(1,1) = pc_val[1][1];
            }
        }
        
        // --- 2. Read Coordinate Arrays for Increments (CDELT) and Reference Pixels (CRPIX) ---
        std::vector<double> l_arr = _reader->ReadVector("l");
        std::vector<double> m_arr = _reader->ReadVector("m");
        
        if (l_arr.empty() || m_arr.empty()) {
            spdlog::warn("Could not read 'l' or 'm' arrays for WCS");
            return false;
        }
        
        // Calculate CDELT
        double cdelt1_rad = l_arr.size() > 1 ? (l_arr[1] - l_arr[0]) : 1.0;
        double cdelt2_rad = m_arr.size() > 1 ? (m_arr[1] - m_arr[0]) : 1.0;
        
        // Calculate CRPIX
        auto find_zero_index = [](const std::vector<double>& arr) -> double {
            constexpr double kTolerance = 1e-9;
            for (size_t i = 0; i < arr.size(); ++i) {
                if (std::abs(arr[i]) < kTolerance) {
                    return static_cast<double>(i); // 0-based index for casacore
                }
            }
            return 0.0; // Default to first pixel
        };
        
        double crpix1 = find_zero_index(l_arr);
        double crpix2 = find_zero_index(m_arr);
        
        // Build DirectionCoordinate
        MDirection::Types direction_type_enum = MDirection::J2000;
        
        if (wcs_dict["reference"].contains("attrs")) {
            auto ref_attrs = wcs_dict["reference"]["attrs"];
            if (ref_attrs.contains("frame")) {
                std::string frame = ref_attrs["frame"];
                MDirection::getType(direction_type_enum, String(frame)); 
            }
        }
        
        Projection projection(Projection::SIN);
        
        DirectionCoordinate dir_coord(direction_type_enum, projection, 
                                      crval1_rad, crval2_rad,
                                      cdelt1_rad, cdelt2_rad,
                                      pc_matrix,
                                      crpix1, crpix2);

        // --- 3. Spectral Coordinate ---
        std::string freq_name = "frequency";
        std::vector<double> freq_arr = _reader->ReadVector(freq_name);
        if (freq_arr.empty()) {
            freq_name = "freq";
            freq_arr = _reader->ReadVector(freq_name);
        }
        
        CoordinateSystem coord_sys;
        coord_sys.addCoordinate(dir_coord);
        
        if (!freq_arr.empty()) {
            double crval_freq = freq_arr[0];
            double cdelt_freq = freq_arr.size() > 1 ? (freq_arr[1] - freq_arr[0]) : 1.0;
            double crpix_freq = 0.0; // 0-based index
            
            // Parse attributes for rest frequency and frame
            double rest_freq = 0.0;
            MFrequency::Types freq_frame = MFrequency::TOPO;
            
            try {
                std::string attrs_str = _reader->GetZattrsString(freq_name);
                if (attrs_str != "{}" && !attrs_str.empty()) {
                    nlohmann::json attrs = nlohmann::json::parse(attrs_str);
                    
                    // Rest Frequency
                    if (attrs.contains("rest_frequency")) {
                        auto& rf = attrs["rest_frequency"];
                        if (rf.is_object() && rf.contains("data")) {
                            rest_freq = rf["data"];
                        } else if (rf.is_number()) {
                            rest_freq = rf;
                        }
                    }
                    
                    // Spectral Frame (SPECSYS -> observer/frame)
                    // Python: freq_ref["attrs"]["observer"].upper()
                    nlohmann::json ref_obj;
                    bool has_ref = false;
                    if (attrs.contains("reference_value")) {
                        ref_obj = attrs["reference_value"];
                        has_ref = true;
                    } else if (attrs.contains("reference_frequency")) {
                        ref_obj = attrs["reference_frequency"];
                        has_ref = true;
                    }
                    
                    if (has_ref && ref_obj.contains("attrs")) {
                        auto& ref_attrs = ref_obj["attrs"];
                        std::string frame_str;
                        if (ref_attrs.contains("observer")) {
                            frame_str = ref_attrs["observer"];
                        } else if (ref_attrs.contains("frame")) {
                            frame_str = ref_attrs["frame"];
                        }
                        
                        if (!frame_str.empty()) {
                            MFrequency::getType(freq_frame, String(frame_str));
                        }
                    }
                }
            } catch (const std::exception& ex) {
                spdlog::warn("Error parsing frequency attributes: {}", ex.what());
            }
            
            SpectralCoordinate spec_coord(freq_frame, 
                                          crval_freq, cdelt_freq, crpix_freq, 
                                          rest_freq);
            coord_sys.addCoordinate(spec_coord);
        } else if (_shape.size() > 2) {
             constexpr double kDefaultFreq = 1.4e9;
             constexpr double kDefaultWidth = 1e6;
             SpectralCoordinate spec_coord(MFrequency::TOPO, kDefaultFreq, kDefaultWidth, 0.0);
             coord_sys.addCoordinate(spec_coord);
        }
        
        // --- 4. Stokes Coordinate ---
        if (_shape.size() > 3) {
             int stokes_size = _shape[3];
             Vector<int> stokes(stokes_size);
             stokes = Stokes::I; // Default
             
             StokesCoordinate stokes_coord(stokes);
             coord_sys.addCoordinate(stokes_coord);
        }
        
        setCoordinateInfo(coord_sys);
        
        // --- 5. BUNIT (Units) ---
        // Try getting units from main array or "SKY"
        std::string units_str = _reader->GetAttributeString("", "units");
        if (units_str.empty()) {
            units_str = _reader->GetAttributeString("SKY", "units");
        }
        
        if (!units_str.empty()) {
            try {
                setUnits(Unit(String(units_str)));
                spdlog::info("Set BUNIT to {}", units_str);
            } catch (const std::exception& e) {
                spdlog::warn("Invalid units string '{}': {}", units_str, e.what());
            }
        }
        
        spdlog::info("Successfully parsed WCS from Zarr metadata");
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("Exception parsing WCS from metadata: {}", e.what());
        return false;
    }
}

void CartaZarrImage::CreateDefaultCoordinateSystem() {
    CoordinateSystem coord_sys;
    
    try {
        int ndim = _shape.size();
        
        if (ndim >= 2) {
            // Create default DirectionCoordinate for first 2 axes
            DirectionCoordinate dir_coord;
            coord_sys.addCoordinate(dir_coord);
        }
        
        if (ndim >= 3) {
            // Add SpectralCoordinate for 3rd axis
            SpectralCoordinate spec_coord;
            coord_sys.addCoordinate(spec_coord);
        }
        
        if (ndim >= 4) {
            // Add StokesCoordinate for 4th axis
            int stokes_size = _shape[3];
            Vector<int> stokes_types(stokes_size);
            for (int i = 0; i < stokes_size; ++i) {
                stokes_types(i) = Stokes::I;  // Default to Stokes I
            }
            StokesCoordinate stokes_coord(stokes_types);
            coord_sys.addCoordinate(stokes_coord);
        }
        
        setCoordinateInfo(coord_sys);
        
    } catch (const std::exception& e) {
        spdlog::error("Error creating default coordinate system: {}", e.what());
        
        // Emergency fallback - just DirectionCoordinate
        CoordinateSystem fallback;
        DirectionCoordinate dir_coord;
        fallback.addCoordinate(dir_coord);
        setCoordinateInfo(fallback);
    }
}

} // namespace carta
