/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_
#define CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_

#include <casacore/images/Images/ImageInterface.h>
#include <casacore/coordinates/Coordinates/CoordinateSystem.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/StokesCoordinate.h>
#include <casacore/lattices/Lattices/TiledShape.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <ctime>

// TensorStore includes  
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"

namespace carta {

class CartaZarrImage : public casacore::ImageInterface<float> {
public:
    explicit CartaZarrImage(const std::string& filename);
    virtual ~CartaZarrImage() = default;

    // ImageInterface implementation
    casacore::String imageType() const override;
    casacore::DataType dataType() const override;
    casacore::Bool doGetSlice(casacore::Array<float>& buffer, const casacore::Slicer& section) override;
    void doPutSlice(const casacore::Array<float>& buffer, const casacore::IPosition& where, const casacore::IPosition& stride) override;
    casacore::Bool doGetMaskSlice(casacore::Array<casacore::Bool>& buffer, const casacore::Slicer& section) override;
    casacore::Bool isMasked() const override;
    casacore::Bool isPersistent() const override;
    casacore::Bool isWritable() const override;
    casacore::String name(casacore::Bool stripPath = false) const override;
    casacore::IPosition shape() const override;
    void resize(const casacore::TiledShape& newShape) override;
    casacore::Bool ok() const override;
    void flush() override;
    void tempClose() override;
    void reopen() override;
    
    // Performance optimization methods - critical for preventing 1D slice processing
    casacore::uInt advisedMaxPixels() const override;
    casacore::IPosition doNiceCursorShape(casacore::uInt maxPixels) const override;

    // LatticeBase implementation
    casacore::Bool hasPixelMask() const override;
    const casacore::Lattice<casacore::Bool>& pixelMask() const override;
    casacore::Lattice<casacore::Bool>& pixelMask() override;

    // MaskedLattice implementation
    const casacore::LatticeRegion* getRegionPtr() const override;
    casacore::ImageInterface<float>* cloneII() const override;
    
    // Coordinates - required by ImageInterface
    const casacore::CoordinateSystem& coordinates() const;
    
    // Public method for direct TensorStore access (needed for optimized small region reads)
    casacore::Bool readDirectFromTensorStore(casacore::Array<float>& buffer, const casacore::Slicer& section);
    casacore::Bool readPixelFromTensorStore(casacore::Array<float>& buffer, const casacore::Slicer& section);

private:
    casacore::CoordinateSystem _coord_sys;
    casacore::IPosition _shape;
    casacore::IPosition _original_zarr_shape;  // Original ZARR shape for TensorStore access
    casacore::String _name;
    int _ndim = 2;  // Number of dimensions, default to 2D
    
    // TensorStore members
    tensorstore::Context _context;
    tensorstore::TensorStore<> _tensorstore;  // Explicitly specify float data type
    bool _tensorstore_initialized = false;
    
    // Data type information
    casacore::DataType _actual_data_type = casacore::DataType::TpFloat; // Default to float
    
    // Channel cache for fast access - now supports region-based caching
    std::vector<float> _channel_cache;     // Cached data for current region
    bool _channel_cache_loaded = false;    // Whether cache is loaded
    int _cached_channel = 0;               // Which channel is cached (default: first channel)
    int _cache_width = 0;                  // Width of cached data
    int _cache_height = 0;                 // Height of cached data
    int _cache_start_x = 0;                // Start X coordinate of cached region
    int _cache_start_y = 0;                // Start Y coordinate of cached region
    int _cache_num_freq = 1;               // Number of frequency channels in 4D cache
    int _cache_num_stokes = 1;             // Number of stokes parameters in 4D cache
    bool _is_full_channel_cache = false;   // Whether cache contains full channel or just a region
    
    // File modification time tracking for metadata caching
    std::time_t _file_last_modified = 0;
    bool _coordinate_system_initialized = false;
    bool _frequency_type_cached = false;
    casacore::MFrequency::Types _cached_frequency_type;
    
    void setupCoordinateSystem();
    bool hasFileChanged();
    bool parseWCSFromZattrs(const nlohmann::json& zattrs);
    bool parseWCSFromCoordinateArrays(const std::filesystem::path& ra_path, const std::filesystem::path& dec_path, const std::filesystem::path& freq_path);
    bool parseWCSFromMetadata(const nlohmann::json& zattrs);
    bool buildDirectionCoordinateFromArrays(double ra_rad, double dec_rad, double freq_hz, 
                                           double ra_cdelt_deg, double dec_cdelt_deg, double freq_cdelt_hz,
                                           size_t height, size_t width, size_t depth);
    void createMinimalCoordinateSystem();
    void initializeTensorStore();
    
    // Get frequency reference frame from ZARR metadata
    casacore::MFrequency::Types GetFrequencyType();
    casacore::MFrequency::Types ParseFrequencyFrame(const std::string& frame_str);
    casacore::MFrequency::Types ParseFrequencyFrameCode(int frame_code);
    
    // Channel cache methods - now support region-based caching
    bool loadChannelCache(int freq_channel = 0, int stokes_channel = 0);
    bool loadRegionCache(int freq_channel, int stokes_channel, int start_x, int start_y, int width, int height);
    bool load4DRegionCache(int start_x, int start_y, int width, int height, int num_freq, int num_stokes);
    bool getSliceFromCache(casacore::Array<float>& buffer, const casacore::Slicer& section);
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_
