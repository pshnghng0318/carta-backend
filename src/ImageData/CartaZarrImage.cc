/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include <spdlog/fmt/fmt.h>

#include <casacore/casa/OS/Path.h>
#include <casacore/casa/Quanta/Unit.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/StokesCoordinate.h>
#include <casacore/images/Images/ImageFITSConverter.h>
#include <casacore/images/Images/ImageInfo.h>
#include <casacore/tables/DataMan/TiledFileAccess.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

using namespace casacore;

namespace carta {

CartaZarrImage::CartaZarrImage(const std::string& filename)
    : _reader(std::make_shared<ZarrDataReader>(filename)), _name(filename), _is_copy(false) {
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

    // Set beams (must be after SetupCoordinateSystem)
    SetBeams();

    spdlog::info("CartaZarrImage created: {} with shape {}", filename, _shape.toString());
}

CartaZarrImage::CartaZarrImage(const CartaZarrImage& other)
    : ImageInterface<float>(other),
      _reader(other._reader), // Share reader!
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
    // Return cached headers if available
    if (!_fits_header_strings.empty()) {
        return _fits_header_strings;
    }

    if (!_reader || !_reader->IsInitialized()) {
        return _fits_header_strings;
    }

    std::vector<std::string> headers;

    static constexpr size_t kFitsKeywordMaxLen = 8;
    static constexpr double kRadToDeg = 180.0 / M_PI;

    // Helper to add headers safely
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
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char character) { return std::toupper(character); });
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

    auto get_ptr = [](const nlohmann::json& obj, const char* ptr) -> const nlohmann::json* {
        try {
            return &obj.at(nlohmann::json::json_pointer(ptr));
        } catch (...) {
            return nullptr;
        }
    };

    // Helper for safe execution of blocks
    auto safe_exec = [](auto func, const std::string& context) {
        try {
            func();
        } catch (const std::exception& e) {
            spdlog::warn("Error parsing {}: {}", context, e.what());
        } catch (...) {
            spdlog::warn("Unknown error parsing {}", context);
        }
    };

    // 1. BITPIX
    static constexpr int kBitpixFloat32 = -32;
    static constexpr int kBitpixFloat64 = -64;
    int bitpix = kBitpixFloat32;
    safe_exec([&]() {
        // TODO: currently XRADIO only supports float32 and float64
        nlohmann::json zarray = nlohmann::json::parse(_reader->GetZarrayString("SKY"));
        const auto* dtype = get_ptr(zarray, "/dtype");
        if (dtype && dtype->is_string()) {
            std::string dtype_str = dtype->get<std::string>();
            if (dtype_str == "<f4") {
                bitpix = kBitpixFloat32;
            } else if (dtype_str == "<f8") {
                bitpix = kBitpixFloat64;
            }
        }
    }, "BITPIX");

    add_string_header("SIMPLE", "T");
    add_int_header("BITPIX", bitpix);

    // 2. NAXIS
    int ndim = _shape.size();
    if (ndim > 0) {
        add_int_header("NAXIS", ndim);
        for (int i = 0; i < ndim; ++i) {
            add_int_header("NAXIS" + std::to_string(i + 1), _shape[i]);
        }
    }

    // 3. Direction Information (.zattrs)
    {
        nlohmann::json zattrs;
        bool zattrs_valid = false;
        try {
            zattrs = nlohmann::json::parse(_reader->GetZattrsString(""));
            zattrs_valid = true;
        } catch (const std::exception& e) {
            spdlog::warn("Error parsing root .zattrs: {}", e.what());
        }

        if (zattrs_valid) {
            const auto* direction = get_ptr(zattrs, "/direction");
            if (direction && direction->is_object()) {
                // Shared state for CTYPE/RADESYS
                std::string ctype1_prefix = "RA";
                std::string ctype2_prefix = "DEC";

                // CRVAL
                safe_exec([&]() {
                    const auto* ref_data = get_ptr(zattrs, "/direction/reference/data");
                    if (ref_data && ref_data->is_array() && ref_data->size() >= 2) {
                        add_double_header("CRVAL1", (*ref_data)[0].get<double>() * kRadToDeg);
                        add_double_header("CRVAL2", (*ref_data)[1].get<double>() * kRadToDeg);
                    }
                }, "CRVAL");

                // RADESYS & Prefix
                safe_exec([&]() {
                    const auto* frame = get_ptr(zattrs, "/direction/reference/attrs/frame");
                    if (frame && frame->is_string()) {
                        std::string frame_str = frame->get<std::string>();
                        to_upper_ascii(frame_str);
                        add_string_header("RADESYS", frame_str);

                        if (frame_str == "GALACTIC") {
                            ctype1_prefix = "GLON";
                            ctype2_prefix = "GLAT";
                        } else if (frame_str == "ECLIPTIC") {
                            ctype1_prefix = "ELON";
                            ctype2_prefix = "ELAT";
                        } else if (frame_str == "SUPERGALACTIC") {
                            ctype1_prefix = "SLON";
                            ctype2_prefix = "SLAT";
                        }
                    }
                }, "RADESYS");

                // EQUINOX
                safe_exec([&]() {
                    const auto* equinox = get_ptr(zattrs, "/direction/reference/attrs/equinox");
                    if (equinox) {
                        if (equinox->is_number()) {
                            add_double_header("EQUINOX", equinox->get<double>());
                        } else if (equinox->is_string()) {
                            std::string val = equinox->get<std::string>();
                            if (!val.empty()) {
                                size_t start_pos = 0;
                                // Handle J2000/B1950 styles
                                if (std::toupper(val[0]) == 'J' || std::toupper(val[0]) == 'B') {
                                    start_pos = 1;
                                }
                                try {
                                    add_double_header("EQUINOX", std::stod(val.substr(start_pos)));
                                } catch (...) {
                                    // Ignore parsing failure
                                }
                            }
                        }
                    }
                }, "EQUINOX");

                // CTYPE
                safe_exec([&]() {
                    std::string projection_str;
                    const auto* projection = get_ptr(zattrs, "/direction/projection");
                    if (projection && projection->is_string()) {
                        projection_str = projection->get<std::string>();
                    }
                    add_string_header("CTYPE1", make_ctype(ctype1_prefix, projection_str));
                    add_string_header("CTYPE2", make_ctype(ctype2_prefix, projection_str));
                }, "CTYPE");

                // LATPOLE/LONPOLE
                safe_exec([&]() {
                    const auto* lat_data = get_ptr(zattrs, "/direction/latpole/data");
                    if (lat_data && lat_data->is_number()) {
                        add_double_header("LATPOLE", lat_data->get<double>() * kRadToDeg);
                    }
                    const auto* lon_data = get_ptr(zattrs, "/direction/lonpole/data");
                    if (lon_data && lon_data->is_number()) {
                        add_double_header("LONPOLE", lon_data->get<double>() * kRadToDeg);
                    }
                }, "LATPOLE/LONPOLE");

                // TODO: XRADIO has not finalized projection_parameters yet.

                // PC Matrix
                safe_exec([&]() {
                    const auto* pc_val = get_ptr(zattrs, "/direction/pc/_value");
                    if (pc_val && pc_val->is_array() && pc_val->size() >= 2 && (*pc_val)[0].is_array() && (*pc_val)[0].size() >= 2) {
                        add_double_header("PC1_1", (*pc_val)[0][0].get<double>());
                        add_double_header("PC1_2", (*pc_val)[0][1].get<double>());
                        add_double_header("PC2_1", (*pc_val)[1][0].get<double>());
                        add_double_header("PC2_2", (*pc_val)[1][1].get<double>());
                    }
                }, "PC Matrix");
            }
        }
    }

    // 4. Direction Increments
    safe_exec([&]() {
        std::vector<double> l_arr = _reader->ReadVector("l");
        std::vector<double> m_arr = _reader->ReadVector("m");
        if (l_arr.size() > 1) {
            double cdelt1_rad = l_arr[1] - l_arr[0];
            add_double_header("CDELT1", cdelt1_rad * kRadToDeg);
            if (cdelt1_rad != 0.0) {
                add_double_header("CRPIX1", (-l_arr[0] / cdelt1_rad) + 1.0); // +1 for FITS 1-indexed
            }
        }
        if (m_arr.size() > 1) {
            double cdelt2_rad = m_arr[1] - m_arr[0];
            add_double_header("CDELT2", cdelt2_rad * kRadToDeg);
            if (cdelt2_rad != 0.0) {
                add_double_header("CRPIX2", (-m_arr[0] / cdelt2_rad) + 1.0); // +1 for FITS 1-indexed
            }
        }
        add_string_header("CUNIT1", "deg");
        add_string_header("CUNIT2", "deg");
    }, "Direction Increments");

    // 5. Spectral Axis
    safe_exec([&]() {
        std::vector<double> freq_arr = _reader->ReadVector("frequency");
        nlohmann::json zattrs = nlohmann::json::parse(_reader->GetZattrsString("frequency"));
        
        if (!freq_arr.empty()) {
            add_string_header("CTYPE3", "FREQ");
            add_double_header("CRPIX3", 1.0); // FITS 1-indexed
            add_double_header("CRVAL3", freq_arr[0]);
            if (freq_arr.size() > 1) {
                add_double_header("CDELT3", freq_arr[1] - freq_arr[0]);
            }
            
            // TODO: drop support for reference_value, only support reference_frequency
            const auto* ref_attrs = get_ptr(zattrs, "/reference_frequency/attrs");
            if (!ref_attrs) {
                ref_attrs = get_ptr(zattrs, "/reference_value/attrs");
            }
            
            if (ref_attrs && ref_attrs->is_object()) {
                const auto* ref_units = get_ptr(*ref_attrs, "/units");
                if (ref_units) {
                    if (ref_units->is_array() && !ref_units->empty() && (*ref_units)[0].is_string()) {
                        add_string_header("CUNIT3", (*ref_units)[0].get<std::string>());
                    } else if (ref_units->is_string()) {
                        add_string_header("CUNIT3", ref_units->get<std::string>());
                    }
                }
                const auto* observer = get_ptr(*ref_attrs, "/observer");
                if (observer && observer->is_string()) {
                    std::string specsys = observer->get<std::string>();
                    to_upper_ascii(specsys);
                    add_string_header("SPECSYS", specsys);
                }
            } else {
                // No frequency axis or simple units
                const auto* units = get_ptr(zattrs, "/units");
                if (units && units->is_string()) {
                    add_string_header("CUNIT3", units->get<std::string>());
                }
                const auto* frame = get_ptr(zattrs, "/frame");
                if (frame && frame->is_string()) {
                    std::string specsys = frame->get<std::string>();
                    to_upper_ascii(specsys);
                    add_string_header("SPECSYS", specsys);
                }
            }
            
            const auto* rest_freq = get_ptr(zattrs, "/rest_frequency/data");
            if (rest_freq && rest_freq->is_number()) {
                add_double_header("RESTFRQ", rest_freq->get<double>());
            }
        }
    }, "Spectral Axis");

    // 6. Stokes Axis
    safe_exec([&]() {
        std::vector<std::string> pol_strs = _reader->ReadStringVector("polarization");
        add_string_header("CTYPE4", "STOKES");
        add_double_header("CRPIX4", 1.0); // FITS 1-indexed
        
        if (!pol_strs.empty() && (pol_strs.size() != 1 || casacore::Stokes::type(pol_strs[0]) != casacore::Stokes::I)) {
            int stokes_first = casacore::Stokes::type(pol_strs[0]);
            add_double_header("CRVAL4", static_cast<double>(stokes_first));
            if (pol_strs.size() > 1) {
                int stokes_second = casacore::Stokes::type(pol_strs[1]);
                int delta = stokes_second - stokes_first;
                // Verify uniform Stokes spacing for FITS linear axis representation
                for (size_t i = 2; i < pol_strs.size(); ++i) {
                    int expected = stokes_first + (static_cast<int>(i) * delta);
                    int actual = casacore::Stokes::type(pol_strs[i]);
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
    }, "Stokes Axis");

    // 7. SKY Metadata (BUNIT, OBJECT, etc.)
    {
        nlohmann::json zattrs_sky;
        bool zattrs_sky_valid = false;
        try {
            zattrs_sky = nlohmann::json::parse(_reader->GetZattrsString("SKY"));
            zattrs_sky_valid = true;
        } catch (...) {
            spdlog::warn("Error parsing SKY zattrs");
        }

        if (zattrs_sky_valid) {
            // BUNIT
            safe_exec([&]() {
                const auto* units = get_ptr(zattrs_sky, "/units");
                if (units && units->is_string()) {
                    add_string_header("BUNIT", units->get<std::string>());
                }
            }, "BUNIT");

            // BTYPE
            safe_exec([&]() {
                const auto* image_type = get_ptr(zattrs_sky, "/image_type");
                if (image_type && image_type->is_string()) {
                    add_string_header("BTYPE", image_type->get<std::string>());
                }
            }, "BTYPE");

            // OBJECT
            safe_exec([&]() {
                const auto* object_name = get_ptr(zattrs_sky, "/object_name");
                if (object_name && object_name->is_string()) {
                    add_string_header("OBJECT", object_name->get<std::string>());
                }
            }, "OBJECT");

            // OBSERVER
            safe_exec([&]() {
                const auto* observer = get_ptr(zattrs_sky, "/observer");
                if (observer && observer->is_string()) {
                    add_string_header("OBSERVER", observer->get<std::string>());
                }
            }, "OBSERVER");

            // TIMESYS & MJD-OBS
            safe_exec([&]() {
                const auto* obs_scale = get_ptr(zattrs_sky, "/obsdate/attrs/scale");
                if (obs_scale && obs_scale->is_string()) {
                    std::string scale = obs_scale->get<std::string>();
                    to_upper_ascii(scale);
                    add_string_header("TIMESYS", scale);
                }
                const auto* obs_format = get_ptr(zattrs_sky, "/obsdate/attrs/format");
                const auto* obs_data = get_ptr(zattrs_sky, "/obsdate/data");
                if (obs_format && obs_data && obs_format->is_string() && obs_data->is_number()) {
                    if (obs_format->get<std::string>() == "MJD") {
                        add_double_header("MJD-OBS", obs_data->get<double>());
                    }
                }
            }, "TIMESYS/MJD-OBS");

            // TELESCOPE & OBSGEO
            safe_exec([&]() {
                const auto* telescope_name = get_ptr(zattrs_sky, "/telescope/name");
                if (telescope_name && telescope_name->is_string()) {
                    add_string_header("TELESCOP", telescope_name->get<std::string>());
                }
                const auto* telescope_dir = get_ptr(zattrs_sky, "/telescope/direction/data/_value");
                const auto* telescope_dist = get_ptr(zattrs_sky, "/telescope/distance/data/_value");
                if (telescope_dir && telescope_dist && telescope_dir->is_array() && telescope_dir->size() >= 2 && telescope_dist->is_array() &&
                    !telescope_dist->empty()) {
                    double lon = (*telescope_dir)[0].get<double>();
                    double lat = (*telescope_dir)[1].get<double>();
                    double radius = (*telescope_dist)[0].get<double>();
                    double obsgeo_x = radius * cos(lat) * cos(lon);
                    double obsgeo_y = radius * cos(lat) * sin(lon);
                    double obsgeo_z = radius * sin(lat);
                    add_double_header("OBSGEO-X", obsgeo_x);
                    add_double_header("OBSGEO-Y", obsgeo_y);
                    add_double_header("OBSGEO-Z", obsgeo_z);
                }
            }, "TELESCOPE/OBSGEO");

            // User metadata
            safe_exec([&]() {
                const auto* user = get_ptr(zattrs_sky, "/user");
                if (user && user->is_object()) {
                    for (const auto& [k, v] : user->items()) {
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
            }, "User Metadata");
        }
    }

    // 8. Beam Parameters
    safe_exec([&]() {
        std::vector<double> beam_data = _reader->ReadFlattenedVector("BEAM");
        if (beam_data.size() >= 3) {
            static constexpr size_t kBeamParamCount = 3;
            static constexpr double kBeamCompareEpsilon = 1e-12;

            const size_t n_beams = beam_data.size() / kBeamParamCount;
            bool single_beam = true;
            const double ref_bmaj = beam_data[0];
            const double ref_bmin = beam_data[1];
            const double ref_bpa = beam_data[2];

            // Heuristic: Multi-beam data usually varies immediately or across the band.
            // Check first few, middle, and last to detect variation without full iteration.
            // This optimizes for the common case where data is effectively single-beam but stored as an array.
            std::vector<size_t> check_indices;
            const size_t check_limit = std::min(n_beams, size_t(10));
            for (size_t i = 1; i < check_limit; ++i) {
                check_indices.push_back(i);
            }
            if (n_beams > check_limit) {
                check_indices.push_back(n_beams / 2); // Middle
                check_indices.push_back(n_beams - 1); // Last
            }

            for (size_t idx : check_indices) {
                const size_t base = idx * kBeamParamCount;
                if (std::abs(beam_data[base + 0] - ref_bmaj) > kBeamCompareEpsilon ||
                    std::abs(beam_data[base + 1] - ref_bmin) > kBeamCompareEpsilon ||
                    std::abs(beam_data[base + 2] - ref_bpa) > kBeamCompareEpsilon) {
                    single_beam = false;
                    break;
                }
            }

            if (single_beam) {
                _is_single_beam = true;
                _beam = casacore::GaussianBeam(
                    casacore::Quantity(ref_bmaj, "rad"), casacore::Quantity(ref_bmin, "rad"), casacore::Quantity(ref_bpa, "rad"));

                add_double_header("BMAJ", ref_bmaj * kRadToDeg);
                add_double_header("BMIN", ref_bmin * kRadToDeg);
                add_double_header("BPA", ref_bpa * kRadToDeg);
            } else {
                _is_single_beam = false;
                add_string_header("CASAMBM", "T");
            }
        }
    }, "Beam Parameters");

    // END keyword required for FITS header
    headers.emplace_back(fmt::format("{:<80}", "END"));

    // Convert to casacore::Vector<String> and cache
    _fits_header_strings.resize(headers.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        _fits_header_strings[i] = headers[i];
    }
    return _fits_header_strings;
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

void CartaZarrImage::doPutSlice(const Array<float>& buffer, const IPosition& where, const IPosition& stride) {
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
            int stokes_fits_value(1);
            Record unused_headers;
            LogSink sink; // null sink to suppress confusing FITS log messages
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
                stokes_types(i) = Stokes::I; // Default to Stokes I
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
        spdlog::debug("CartaZarrImage::SetBeams - Setting single beam from _beam: {} x {} @ {}", _beam.getMajor().getValue("arcsec"),
            _beam.getMinor().getValue("arcsec"), _beam.getPA().getValue("deg"));
        ImageInfo info = imageInfo();
        info.setRestoringBeam(_beam);
        setImageInfo(info);
    } else {
        spdlog::debug("CartaZarrImage::SetBeams - Setting multiple beams from Zarr BEAM array");
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

        spdlog::debug(
            "CartaZarrImage::SetBeams - Processing {} beam entries for shape {}x{} (chan x stokes)", usable_groups, n_chan, n_stokes);

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
            casacore::GaussianBeam beam(casacore::Quantity(bmaj, "rad"), casacore::Quantity(bmin, "rad"), casacore::Quantity(bpa, "rad"));
            _beam_set.setBeam(chan, stokes, beam);
        }

        ImageInfo info = imageInfo();
        info.setBeams(_beam_set);
        setImageInfo(info);
    }
}

} // namespace carta
