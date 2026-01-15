/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_ZARRLOADER_H_
#define CARTA_SRC_IMAGEDATA_ZARRLOADER_H_

#include <map>
#include <string>

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
    bool GetCursorSpectralData(std::vector<float>& data, int stokes, int cursor_x, 
                               int count_x, int cursor_y, int count_y, 
                               std::mutex& image_mutex) override;
    bool UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& image_mutex) override;
    bool GetRegionSpectralData(int region_id, const AxisRange& z_range, int stokes,
        const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin, std::mutex& image_mutex,
        std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) override;

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

    std::map<FileInfo::RegionStatsId, FileInfo::RegionSpectralStats> _region_stats;
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_ZARRLOADER_H_
