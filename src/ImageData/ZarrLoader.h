/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_ZARRLOADER_H_
#define CARTA_SRC_IMAGEDATA_ZARRLOADER_H_

#include "FileLoader.h"
#include <string>

// Forward declaration
class CartaZarrImage;

namespace carta {

class ZarrLoader : public FileLoader {
public:
    ZarrLoader(const std::string& filename);
    ~ZarrLoader() = default;

    void AllocateImage(const std::string& hdu = "") override;
    
    // FileLoader virtual function implementations
    bool HasData(FileInfo::Data ds) const override;
    bool HasMip(int mip) const override;
    bool UseTileCache() const override;
    
    bool GetCursorSpectralData(std::vector<float>& data, int stokes, int cursor_x, int count_x,
        int cursor_y, int count_y, std::mutex& image_mutex) override;
    
    bool UseRegionSpectralData(const casacore::IPosition& region_shape, std::mutex& image_mutex) override;
    
    bool GetRegionSpectralData(int region_id, const AxisRange& spectral_range, int stokes,
        const casacore::ArrayLattice<casacore::Bool>& mask, const casacore::IPosition& origin,
        std::mutex& image_mutex, std::map<CARTA::StatsType, std::vector<double>>& results, float& progress) override;
        
    bool GetDownsampledRasterData(std::vector<float>& data, int z, int stokes,
        CARTA::ImageBounds& bounds, int mip, std::mutex& image_mutex) override;
        
    bool GetChunk(std::vector<float>& data, int& data_width, int& data_height,
        int min_x, int min_y, int z, int stokes, std::mutex& image_mutex) override;
        
    // Override GetSlice for ZARR-specific optimizations
    bool GetSlice(casacore::Array<float>& data, const StokesSlicer& stokes_slicer);
    
    // Optimized method for reading large spectral ranges at once
    bool GetSpectralDataOptimized(std::vector<float>& data, int stokes, int x, int y, 
                                 int z_start, int z_end, std::mutex& image_mutex);
        
    const casacore::IPosition GetStatsDataShape(FileInfo::Data ds) override;
    std::unique_ptr<casacore::ArrayBase> GetStatsData(FileInfo::Data ds) override;

private:
    // Helper methods for Zarr detection
    bool HasZarrArrayMetadata(const std::string& path) const;
    bool HasZmetadataFile(const std::string& path) const;
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_ZARRLOADER_H_
