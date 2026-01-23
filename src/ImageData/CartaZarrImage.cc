/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"

#include <algorithm>
#include <cmath>

#include <spdlog/fmt/fmt.h>

#include <casacore/casa/OS/Path.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/StokesCoordinate.h>
#include <casacore/casa/Quanta/Unit.h>
#include <casacore/images/Images/ImageFITSConverter.h>
#include <casacore/images/Images/ImageInfo.h>
#include <casacore/tables/DataMan/TiledFileAccess.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using namespace casacore;

namespace carta {

CartaZarrImage::CartaZarrImage(const std::string& filename)
    : _reader(std::make_shared<ZarrDataReader>(filename)),
      _name(filename),
      _is_copy(false) {
    
    // Constants
    static constexpr double kDefaultSpectralFreq = 1.4e9;
    static constexpr double kDefaultSpectralWidth = 1e6;
    
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


Vector<String> CartaZarrImage::FitsHeaderStrings() {
    std::vector<String> headers;

    static constexpr size_t kFitsKeywordMaxLen = 8;
    static constexpr size_t kFitsHeaderLen = 80;

    // Format headers in standard FITS format: 80 characters, 8-char keyword name
    auto add_string_header = [&headers](const std::string& key, const std::string& value) {
        if (value.empty()) {
            return;
        }
        std::string key_value = fmt::format("{:<8}= '{}'", key, value);
        headers.emplace_back(fmt::format("{:<80}", key_value));
    };

    auto add_double_header = [&headers](const std::string& key, double value) {
        std::string key_value = fmt::format("{:<8}= {:#.13G}", key, value);
        headers.emplace_back(fmt::format("{:<80}", key_value));
    };

    auto add_int_header = [&headers](const std::string& key, int value) {
        std::string key_value = fmt::format("{:<8}= {}", key, value);
        headers.emplace_back(fmt::format("{:<80}", key_value));
    };


    auto to_upper_ascii = [](std::string& text) {
        std::transform(text.begin(), text.end(), text.begin(),
            [](unsigned char character) { return std::toupper(character); });
    };

    auto make_ctype = [](const std::string& axis, const std::string& proj) {
        // FITS CTYPE format: 4-char axis type + "-" + 3-char projection
        // e.g., "RA---SIN", "DEC--SIN", "GLON-TAN"
        std::string axis_str(axis);
        if (!proj.empty()) {
            // Pad with dashes to reach 4 chars, then add "-" + projection
            while (axis_str.size() < 4) {
                axis_str += '-';
            }
            return axis_str + "-" + proj;
        }
        return axis_str;
    };


    // BITPIX
    // TODO: currently XRADIO only supports float32 and float64
    static constexpr int kBitpixFloat32 = -32;
    static constexpr int kBitpixFloat64 = -64;
    int bitpix = kBitpixFloat32;
    try {
        nlohmann::json zarray = nlohmann::json::parse(_reader->GetZarrayString("SKY"));
        if (zarray.contains("dtype") && zarray["dtype"].is_string()) {
            std::string dtype = zarray["dtype"].get<std::string>();
            if (dtype == "<f4") {
                bitpix = kBitpixFloat32;
            } else if (dtype == "<f8") {
                bitpix = kBitpixFloat64;
            }
        }
    } catch (...) {
    }

    add_string_header("SIMPLE", "T");
    add_int_header("BITPIX", bitpix);

    // Add image shape information
    int ndim = _shape.size();
    if (ndim > 0) {
        add_int_header("NAXIS", ndim);
        for (int i = 0; i < ndim; ++i) {
            add_int_header("NAXIS" + std::to_string(i + 1), _shape[i]);
        }
    }

    constexpr double kRadToDeg = 180.0 / M_PI;

    // Parse direction information from .zattrs
    try {
        nlohmann::json zattrs = nlohmann::json::parse(_reader->GetZattrsString(""));
        if (zattrs.contains("direction")) {
            const auto& direction = zattrs["direction"];
            // Set default CTYPE prefixes
            std::string ctype1_prefix = "RA";
            std::string ctype2_prefix = "DEC";

            // CRVAL1, CRVAL2, RADESYS and EQUINOX
            if (direction.contains("reference") && direction["reference"].contains("data")) {
                const auto& ref_data = direction["reference"]["data"];
                if (ref_data.size() >= 2) {
                    add_double_header("CRVAL1", ref_data[0].get<double>() * kRadToDeg);
                    add_double_header("CRVAL2", ref_data[1].get<double>() * kRadToDeg);
                }
                // Extract RADESYS and EQUINOX from frame and infer CTYPE prefixes
                if (direction["reference"].contains("attrs")) {
                    const auto& ref_attrs = direction["reference"]["attrs"];
                    if (ref_attrs.contains("frame")) {
                        std::string frame = ref_attrs["frame"].get<std::string>();
                        to_upper_ascii(frame);
                        add_string_header("RADESYS", frame);

                        if (frame == "GALACTIC") {
                            ctype1_prefix = "GLON";
                            ctype2_prefix = "GLAT";
                        } else if (frame == "ECLIPTIC") {
                            ctype1_prefix = "ELON";
                            ctype2_prefix = "ELAT";
                        } else if (frame == "SUPERGALACTIC") {
                            ctype1_prefix = "SLON";
                            ctype2_prefix = "SLAT";
                        }
                    }
                    if (ref_attrs.contains("equinox")) {
                        std::string equinox = ref_attrs["equinox"].get<std::string>();
                        to_upper_ascii(equinox);
                        add_string_header("EQUINOX", equinox);
                    }
                }
            }

            // CTYPE1 and CTYPE2
            std::string projection_str;
            if (direction.contains("projection") && direction["projection"].is_string()) {
                projection_str = direction["projection"].get<std::string>();
            }
            add_string_header("CTYPE1", make_ctype(ctype1_prefix, projection_str));
            add_string_header("CTYPE2", make_ctype(ctype2_prefix, projection_str));

            // LATPOLE and LONPOLE
            if (direction.contains("latpole") && direction["latpole"].contains("data")) {
                const auto& lat_data = direction["latpole"]["data"];
                add_double_header("LATPOLE", lat_data.get<double>() * kRadToDeg);
            }
            if (direction.contains("lonpole") && direction["lonpole"].contains("data")) {
                const auto& lon_data = direction["lonpole"]["data"];
                add_double_header("LONPOLE", lon_data.get<double>() * kRadToDeg);
            }

            // TODO: XRADIO has not finalized projection_parameters yet.

            // PC1_1, PC1_2, PC2_1 and PC2_2
            if (direction.contains("pc") && direction["pc"].contains("_value")) {
                const auto& pc_val = direction["pc"]["_value"];
                if (pc_val.size() >= 2 && pc_val[0].size() >= 2) {
                    add_double_header("PC1_1", pc_val[0][0].get<double>());
                    add_double_header("PC1_2", pc_val[0][1].get<double>());
                    add_double_header("PC2_1", pc_val[1][0].get<double>());
                    add_double_header("PC2_2", pc_val[1][1].get<double>());
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Error parsing direction metadata for FITS headers: {}", e.what());
    }

    // Direction increments and units from coordinate arrays (radians -> degrees)
    try {
    std::vector<double> l_arr = _reader->ReadVector("l");
    std::vector<double> m_arr = _reader->ReadVector("m");
    if (l_arr.size() > 1) {
        double cdelt1_rad = l_arr[1] - l_arr[0];
        add_double_header("CDELT1", cdelt1_rad * kRadToDeg);
        if (cdelt1_rad != 0.0) {
            add_double_header("CRPIX1", (-l_arr[0] / cdelt1_rad) + 1.0);  // +1 for FITS 1-indexed
        }
    }
    if (m_arr.size() > 1) {
        double cdelt2_rad = m_arr[1] - m_arr[0];
        add_double_header("CDELT2", cdelt2_rad * kRadToDeg);
        if (cdelt2_rad != 0.0) {
            add_double_header("CRPIX2", (-m_arr[0] / cdelt2_rad) + 1.0);  // +1 for FITS 1-indexed
        }
    }
    add_string_header("CUNIT1", "deg");
    add_string_header("CUNIT2", "deg");
    } catch (const std::exception& e) {
        spdlog::warn("Error parsing direction increments and units for FITS headers: {}", e.what());
    }

    // Spectral axis
    try {
        std::vector<double> freq_arr = _reader->ReadVector("frequency");
        nlohmann::json zattrs = nlohmann::json::parse(_reader->GetZattrsString("frequency"));
        // If no reference_frequency or reference_value, it has no frequency axis
        // TODO: drop support for reference_value, only support reference_frequency
        if (!freq_arr.empty()) {
            add_string_header("CTYPE3", "FREQ");
            add_double_header("CRPIX3", 1.0);  // FITS 1-indexed
            add_double_header("CRVAL3", freq_arr[0]);
            if (freq_arr.size() > 1) {
                add_double_header("CDELT3", freq_arr[1] - freq_arr[0]);
            }
            std::string ref_name = "reference_frequency";
            if (zattrs.contains("reference_value")) {
                ref_name = "reference_value";
            }
            if (zattrs.contains(ref_name) && zattrs[ref_name].contains("attrs")) {
                const auto& ref_freq = zattrs[ref_name]["attrs"];
                if (ref_freq.contains("units")) {
                    add_string_header("CUNIT3", ref_freq["units"].is_array() ? ref_freq["units"][0].get<std::string>() : ref_freq["units"].get<std::string>());
                }
                if (ref_freq.contains("observer")) {
                    std::string specsys = ref_freq["observer"].get<std::string>();
                    to_upper_ascii(specsys);
                    add_string_header("SPECSYS", specsys);
                }
            } else {
                // No frequency axis
                // See XRADIO `xradio.image._util.common._default_freq_info` for details
                if (zattrs.contains("units")) {
                    add_string_header("CUNIT3", zattrs["units"].get<std::string>());
                }
                if (zattrs.contains("frame")) {
                    std::string specsys = zattrs["frame"].get<std::string>();
                    to_upper_ascii(specsys);
                    add_string_header("SPECSYS", specsys);
                }
            }
            if (zattrs.contains("rest_frequency")) {
                const auto& rest_freq = zattrs["rest_frequency"];
                if (rest_freq.is_object() && rest_freq.contains("data")) {
                    add_double_header("RESTFRQ", rest_freq["data"].get<double>());
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Error parsing frequency metadata for FITS headers: {}", e.what());
    }

    // Stokes axis
    try {
        std::vector<std::string> pol_strs = _reader->ReadStringVector("polarization");
        add_string_header("CTYPE4", "STOKES");
        add_double_header("CRPIX4", 1.0);  // FITS 1-indexed
        if (!pol_strs.empty() && (pol_strs.size() != 1 || Stokes::type(pol_strs[0]) != Stokes::I)) {
            int stokes_first = Stokes::type(pol_strs[0]);
            add_double_header("CRVAL4", static_cast<double>(stokes_first));
            if (pol_strs.size() > 1) {
                int stokes_second = Stokes::type(pol_strs[1]);
                int delta = stokes_second - stokes_first;
                // Verify uniform Stokes spacing for FITS linear axis representation
                for (size_t i = 2; i < pol_strs.size(); ++i) {
                    int expected = stokes_first + (static_cast<int>(i) * delta);
                    int actual = Stokes::type(pol_strs[i]);
                    if (actual != expected) {
                        spdlog::warn("Non-uniform Stokes spacing: expected {} at index {}, got {}", expected, i, actual);
                        break;
                    }
                }
                add_double_header("CDELT4", static_cast<double>(delta));
            } else {
                add_double_header("CDELT4", 1.0);
            }
            add_string_header("CUNIT4", "");
        } else {
            // No polarization axis
            add_double_header("CDELT4", 1.0);
            add_double_header("CRVAL4", 1.0);
            add_string_header("CUNIT4", "");
        }
    } catch (const std::exception& e) {
        spdlog::warn("Error parsing polarization metadata for FITS headers: {}", e.what());
    }
    
    // Information from SKY
    try {
        nlohmann::json zattrs = nlohmann::json::parse(_reader->GetZattrsString("SKY"));
        if (zattrs.contains("units")) {
            add_string_header("BUNIT", zattrs["units"].get<std::string>());
        }
        if (zattrs.contains("image_type")) {
            add_string_header("BTYPE", zattrs["image_type"].get<std::string>());
        }
        if (zattrs.contains("object_name")) {
            add_string_header("OBJECT", zattrs["object_name"].get<std::string>());
        }
        if (zattrs.contains("observer")) {
            add_string_header("OBSERVER", zattrs["observer"].get<std::string>());
        }
        if (zattrs.contains("obsdate")) {
            const auto& obsdate = zattrs["obsdate"];
            if (obsdate.contains("attrs") && obsdate.contains("data")) {
                const auto& attrs = obsdate["attrs"];
                if (attrs.contains("scale")) {
                    std::string scale = attrs["scale"].get<std::string>();
                    to_upper_ascii(scale);
                    add_string_header("TIMESYS", scale);
                }
                if (attrs.contains("format") && attrs["format"] == "MJD") {
                    double mjd = obsdate["data"].get<double>();
                    add_double_header("MJD-OBS", mjd);
                }
            }
        }
        if (zattrs.contains("telescope")) {
            const auto& telescope = zattrs["telescope"];
            if (telescope.contains("name")) {
                add_string_header("TELESCOP", telescope["name"].get<std::string>());
            }
            if (telescope.contains("direction") && telescope.contains("distance")) {
                const auto& direction = telescope["direction"];
                const auto& distance = telescope["distance"];
                if (direction.contains("data") && direction["data"].contains("_value") && distance.contains("data") && distance["data"].contains("_value")) {
                    const auto& lonlat = direction["data"]["_value"];
                    double lon = lonlat[0].get<double>();
                    double lat = lonlat[1].get<double>();
                    double radius = distance["data"]["_value"][0].get<double>();
                    double obsgeo_x = radius * cos(lat) * cos(lon);
                    double obsgeo_y = radius * cos(lat) * sin(lon);
                    double obsgeo_z = radius * sin(lat);
                    add_double_header("OBSGEO-X", obsgeo_x);
                    add_double_header("OBSGEO-Y", obsgeo_y);
                    add_double_header("OBSGEO-Z", obsgeo_z);
                }
            }
        }
        if (zattrs.contains("user") && zattrs["user"].is_object()) {
            const auto& user = zattrs["user"];
            for (const auto& [k, v] : user.items()) {
                std::string key = k;
                to_upper_ascii(key);
                if (key.size() > kFitsKeywordMaxLen) {
                    key = key.substr(0, kFitsKeywordMaxLen);
                }
                if (v.is_string()) {
                    add_string_header(key, v.get<std::string>());
                } else if (v.is_number_float() || v.is_number_integer() || v.is_number_unsigned()) {
                    add_double_header(key, v.get<double>());
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Error parsing SKY metadata for FITS headers: {}", e.what());
    }

    // Beam parameters, if available (radians -> degrees)
    std::vector<double> beam_data = _reader->ReadFlattenedVector("BEAM");
    if (beam_data.size() >= 3) {
        static constexpr size_t kBeamParamCount = 3;
        static constexpr double kBeamCompareEpsilon = 1e-12;

        const size_t n_beams = beam_data.size() / kBeamParamCount;
        bool single_beam = true;
        const double ref_bmaj = beam_data[0];
        const double ref_bmin = beam_data[1];
        const double ref_bpa = beam_data[2];

        for (size_t i = 1; i < n_beams; ++i) {
            const size_t base = i * kBeamParamCount;
            if (std::abs(beam_data[base + 0] - ref_bmaj) > kBeamCompareEpsilon ||
                std::abs(beam_data[base + 1] - ref_bmin) > kBeamCompareEpsilon ||
                std::abs(beam_data[base + 2] - ref_bpa) > kBeamCompareEpsilon) {
                single_beam = false;
                break;
            }
        }

        _is_single_beam = single_beam;

        if (single_beam) {
            _beam = casacore::GaussianBeam(
                Quantity(ref_bmaj, "rad"),
                Quantity(ref_bmin, "rad"),
                Quantity(ref_bpa, "rad")
            );
            add_double_header("BMAJ", ref_bmaj * kRadToDeg);
            add_double_header("BMIN", ref_bmin * kRadToDeg);
            add_double_header("BPA", ref_bpa * kRadToDeg);
        } else {
            add_string_header("CASAMBM", "T");
        }
    }

    // END keyword required for FITS header
    headers.emplace_back(fmt::format("{:<80}", "END"));

    Vector<String> header_vector(headers.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        header_vector[i] = headers[i];
    }
    return header_vector;
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
        // Get FITS header strings generated from Zarr metadata
        Vector<String> header_strings = FitsHeaderStrings();
        
        if (!header_strings.empty()) {
            // Use casacore's ImageFITSConverter to build coordinate system from FITS headers
            // This is the same approach used by CartaHdf5Image
            int stokes_fits_value(1);
            Record unused_headers;
            LogSink sink;  // null sink to suppress confusing FITS log messages
            LogIO log(sink);
            unsigned int which_rep(0);
            bool drop_stokes(true);
            
            CoordinateSystem coord_sys = ImageFITSConverter::getCoordinateSystem(
                stokes_fits_value, unused_headers, header_strings, log, which_rep, _shape, drop_stokes);
            
            setCoordinateInfo(coord_sys);
            
            // Set image units from unused headers
            setUnits(ImageFITSConverter::getBrightnessUnit(unused_headers, log));
            
            // Set image info (beam, image type, etc.)
            ImageInfo image_info = ImageFITSConverter::getImageInfo(unused_headers);
            if (stokes_fits_value != -1) {
                ImageInfo::ImageTypes type = ImageInfo::imageTypeFromFITS(stokes_fits_value);
                if (type != ImageInfo::Undefined) {
                    image_info.setImageType(type);
                }
            }
            setImageInfo(image_info);
            
            // Set misc info
            Record misc_info;
            ImageFITSConverter::extractMiscInfo(misc_info, unused_headers);
            setMiscInfo(misc_info);
            
            spdlog::info("Successfully set up coordinate system from FITS headers");
            return;
        }
    } catch (const AipsError& e) {
        spdlog::warn("Error setting up coordinate system from FITS headers: {}", e.getMesg());
    } catch (const std::exception& e) {
        spdlog::warn("Error setting up coordinate system: {}, using default", e.what());
    }
    
    // Fall back to default coordinate system
    CreateDefaultCoordinateSystem();
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


void CartaZarrImage::SetBeams() {
    if (_is_single_beam) {
        ImageInfo info = imageInfo();
        info.setRestoringBeam(_beam);
        setImageInfo(info);
    } else {
        std::vector<double> beam_data = _reader->ReadFlattenedVector("BEAM");
        if (beam_data.size() < 3) {
            return;
        }

        // _shape is CARTA shape [X, Y, Channel, Stokes]
        const int n_chan = (_shape.size() > 2) ? _shape[2] : 1;
        const int n_stokes = (_shape.size() > 3) ? _shape[3] : 1;
        _beam_set.resize(n_chan, n_stokes);

        const size_t expected = static_cast<size_t>(n_chan) * static_cast<size_t>(n_stokes) * 3;
        const size_t usable = std::min(beam_data.size(), expected);
        const size_t usable_groups = usable / 3;

        for (size_t beam_group = 0; beam_group < usable_groups; ++beam_group) {
            const int chan = static_cast<int>(beam_group / static_cast<size_t>(n_stokes));
            const int stokes = static_cast<int>(beam_group % static_cast<size_t>(n_stokes));
            if (chan >= n_chan) {
                break;
            }

            const size_t offset = beam_group * 3;
            const double bmaj = beam_data[offset + 0];
            const double bmin = beam_data[offset + 1];
            const double bpa = beam_data[offset + 2];
            casacore::GaussianBeam beam(
                casacore::Quantity(bmaj, "rad"),
                casacore::Quantity(bmin, "rad"),
                casacore::Quantity(bpa, "rad")
            );
            _beam_set.setBeam(chan, stokes, beam);
        }

        ImageInfo info = imageInfo();
        info.setBeams(_beam_set);
        setImageInfo(info);
    }
}

void CartaZarrImage::ParseBeamFromMetadata() {
    try {
        // Read BEAM array (flattened)
        // fits2xradio writes "BEAM" as [time, freq, pol, param] or similar ND array
        // We just need the first beam (single beam support for now)
        std::vector<double> beam_data = _reader->ReadFlattenedVector("BEAM");
        if (beam_data.size() < 3) {
            return;
        }

        // Check units
        std::string units = _reader->GetAttributeString("BEAM", "units");
        if (units.empty()) {
            units = "rad"; // Default to rad as per fits2xradio
        }

        // Taking first beam (index 0, 1, 2 for major, minor, pa)
        double bmaj = beam_data[0];
        double bmin = beam_data[1];
        double bpa = beam_data[2];

        // Construct GaussianBeam
        GaussianBeam beam(
            Quantity(bmaj, units),
            Quantity(bmin, units),
            Quantity(bpa, units)
        );

        ImageInfo info = imageInfo();
        info.setRestoringBeam(beam);
        setImageInfo(info);
        
        spdlog::info("Parsed restoring beam from Zarr BEAM array: {}, {}, {} {}", bmaj, bmin, bpa, units);

    } catch (const std::exception& e) {
        spdlog::warn("Error parsing beam from metadata: {}", e.what());
    }
}

} // namespace carta
