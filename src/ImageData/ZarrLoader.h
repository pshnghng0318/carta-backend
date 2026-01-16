/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_ZARRLOADER_H_
#define CARTA_SRC_IMAGEDATA_ZARRLOADER_H_

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "CartaZarrImage.h"
#include "FileLoader.h"

namespace carta {

/**
 * @brief FileLoader implementation for ZARR format files.
 * 
 * This loader follows the same pattern as FitsLoader and CasaLoader,
 * delegating most operations to CartaZarrImage. Supports TileCache
 * integration for efficient tile-based rendering.
 */
class ZarrLoader : public FileLoader {
public:
    ZarrLoader(const std::string& filename);
    ~ZarrLoader() override = default;

    // FileLoader interface
    bool HasData(FileInfo::Data data_type) const override;
    bool HasMip(int mip_level) const override;
    bool UseTileCache() const override;
    
    // Tile/chunk access for TileCache integration
    bool GetChunk(std::vector<float>& data, int& data_width, int& data_height,
                  int min_x, int min_y, int channel, int stokes, 
                  std::mutex& image_mutex) override;
    
    // Spectral data access
    bool GetCursorSpectralData(std::vector<float>& data, const AxisRange& z_range, int stokes, int cursor_x, int count_x,
        int cursor_y, int count_y, std::mutex& image_mutex, float& progress) override;
    bool UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& image_mutex) override;
    bool GetRegionSpectralData(int region_id, const AxisRange& z_range, int stokes,
        const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin, std::mutex& image_mutex,
        std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) override;
    void ClearRegionSpectralCache(int region_id) override;

    // Spatial profile methods required by Frame.cc for ZARR optimization
    bool GetSpatialProfileX(std::vector<float>& profile, int start_x, int end_x, 
                            int cursor_y, int channel, int stokes, 
                            std::mutex& image_mutex);
    bool GetSpatialProfileY(std::vector<float>& profile, int cursor_x, 
                            int start_y, int end_y, int channel, int stokes, 
                            std::mutex& image_mutex);

private:
    void AllocateImage(const std::string& hdu) override;
    
    // Helper to get typed image
    CartaZarrImage* GetZarrImage();

    struct CursorBatchState {
        size_t plane_size = 0;
        size_t batch_depth = 0;
        double last_elapsed_ms = 0.0;
        size_t sample_count = 0;
        double avg_ms_per_channel = 0.0;
    };

    struct CursorProfileCache {
        bool valid = false;
        int stokes = 0;
        int cursor_x = 0;
        int cursor_y = 0;
        int count_x = 0;
        int count_y = 0;
        int z_from = 0;
        int z_to = 0;
        std::vector<float> data;
    };

    std::map<FileInfo::RegionStatsId, FileInfo::RegionSpectralStats> _region_stats;
    std::mutex _cursor_batch_mutex;
    CursorBatchState _cursor_batch_state;
    std::mutex _cursor_profile_mutex;
    CursorProfileCache _cursor_profile_cache;
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_ZARRLOADER_H_
