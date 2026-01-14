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
#include <mutex>
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
    // Copy constructor for efficient SubImage creation (shares TensorStore and cache)
    CartaZarrImage(const CartaZarrImage& other);
    virtual ~CartaZarrImage();

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
    
    // Statistics structure for cached channel rendering data
    struct ChannelStats {
        int freq_channel = -1;
        int stokes_channel = -1;
        size_t valid_pixels = 0;
        double min_val = 0.0;
        double max_val = 0.0;
        double sum = 0.0;
        double sum_sq = 0.0;
        std::vector<int> histogram_bins;  // Histogram data cached here
        int num_bins = 0;                 // Number of histogram bins
        double bin_width = 0.0;           // Width of each histogram bin
        double bin_center = 0.0;          // Center value for bins
        bool valid = false;
    };
    
    // Get statistics for a specific channel (returns nullptr if not cached)
    const ChannelStats* GetCachedChannelStats(int freq_channel, int stokes_channel) const {
        if (!_all_channel_stats) return nullptr;
        int channel_id = freq_channel * 1000 + stokes_channel;
        auto it = _all_channel_stats->find(channel_id);
        return (it != _all_channel_stats->end() && it->second.valid) ? &it->second : nullptr;
    }

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
    // Using shared_ptr to prevent cache duplication on SubImage creation
    std::shared_ptr<std::vector<float>> _channel_cache;     // Cached data for current region
    bool _channel_cache_loaded = false;    // Whether cache is loaded
    int _cached_channel = 0;               // Which channel is cached (default: first channel)
    int _num_cached_channels = 1;          // Number of channels in cache (for multi-channel caching)
    int _cache_width = 0;                  // Width of cached data
    int _cache_height = 0;                 // Height of cached data
    int _cache_start_x = 0;                // Start X coordinate of cached region
    int _cache_start_y = 0;                // Start Y coordinate of cached region
    int _cache_num_freq = 1;               // Number of frequency channels in 4D cache
    int _cache_num_stokes = 1;             // Number of stokes parameters in 4D cache
    int _cache_freq_start = 0;             // Starting frequency index in 4D cache
    int _cache_stokes_start = 0;           // Starting stokes index in 4D cache
    bool _is_full_channel_cache = false;   // Whether cache contains full channel or just a region
    
    // Copy tracking - whether this is a shallow copy sharing TensorStore and cache
    bool _is_copy = false;
    
    // Rendering statistics cache for all channels (key = freq_channel * 1000 + stokes_channel)
    // Using shared_ptr to prevent cache duplication on SubImage creation
    std::shared_ptr<std::map<int, ChannelStats>> _all_channel_stats;
    
    // Mutex for thread-safe cache access
    mutable std::mutex _cache_mutex;
    
    // Current coordinate reordering state (detected in doGetSlice, used in getSliceFromCache)
    bool _current_coordinates_reordered = false;
    
    // File modification time tracking for metadata caching
    std::time_t _file_last_modified = 0;
    bool _coordinate_system_initialized = false;
    bool _frequency_type_cached = false;
    casacore::MFrequency::Types _cached_frequency_type;
    
    void setupCoordinateSystem();
    bool hasFileChanged();
    bool parseWCSFromZattrs(const nlohmann::json& zattrs);
    bool parseWCSFromCoordinateArrays(const std::filesystem::path& ra_path, const std::filesystem::path& dec_path, const std::filesystem::path& freq_path);
    bool parseWCSFromLMArrays(const std::filesystem::path& l_path, const std::filesystem::path& m_path, const std::filesystem::path& freq_path);
    bool parseWCSFromMetadata(const nlohmann::json& zattrs);
    bool buildDirectionCoordinateFromArrays(double ra_rad, double dec_rad, double freq_hz, 
                                           double ra_cdelt_deg, double dec_cdelt_deg, double freq_cdelt_hz,
                                           size_t height, size_t width, size_t depth);
    bool buildDirectionCoordinateFromLM(double ref_ra_rad, double ref_dec_rad, double freq_hz,
                                       double l_cdelt_deg, double m_cdelt_deg, double freq_cdelt_hz,
                                       double crpix_l, double crpix_m,
                                       size_t nl, size_t nm, size_t depth);
    void createMinimalCoordinateSystem();
    void initializeTensorStore();
    
    // Get frequency reference frame from ZARR metadata
    casacore::MFrequency::Types GetFrequencyType();
    casacore::MFrequency::Types ParseFrequencyFrame(const std::string& frame_str);
    casacore::MFrequency::Types ParseFrequencyFrameCode(int frame_code);
    
    // Get direction reference system from ZARR metadata (handles ICRS, FK5, FK4)
    casacore::MDirection::Types GetDirectionType();
    
    // Get projection type from ZARR metadata (handles SIN, CAR, TAN, etc.)
    casacore::Projection GetProjectionType();
    
    // Channel cache methods - now support region-based caching
    bool loadChannelCache(int freq_channel = 0, int stokes_channel = 0);
    bool loadRegionCache(int freq_channel, int stokes_channel, int start_x, int start_y, int width, int height);
    bool load4DRegionCache(int start_x, int start_y, int width, int height, int num_freq, int num_stokes, int freq_start = 0, int stokes_start = 0);
    bool getSliceFromCache(casacore::Array<float>& buffer, const casacore::Slicer& section);
    bool computeAndCacheHistogram(int freq_channel, int stokes_channel, casacore::Array<float>& buffer, const casacore::Slicer& section);
    
    // Brightness unit and beam information reading
    std::string readBrightnessUnit();
    void setupImageInfo();
    
    // Cached versions to avoid repeated loading
    void readBrightnessUnitIfNeeded();
    void setupImageInfoIfNeeded();
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_