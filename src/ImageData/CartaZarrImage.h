/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_
#define CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_

#include <memory>
#include <mutex>
#include <string>

#include <casacore/coordinates/Coordinates/CoordinateSystem.h>
#include <casacore/images/Images/ImageInterface.h>
#include <casacore/lattices/Lattices/TiledShape.h>

#include "ZarrDataReader.h"

namespace carta {

/**
 * @brief ZARR image class implementing casacore::ImageInterface.
 * 
 * This class provides a clean interface to ZARR files, following the same
 * pattern as CartaFitsImage and CartaHdf5Image. Uses ZarrDataReader for
 * all TensorStore operations.
 */
class CartaZarrImage : public casacore::ImageInterface<float> {
public:
    explicit CartaZarrImage(const std::string& filename);
    CartaZarrImage(const CartaZarrImage& other);  // Copy constructor - shares reader
    ~CartaZarrImage() override;

    // ImageInterface implementation
    casacore::String imageType() const override;
    casacore::String name(casacore::Bool stripPath = false) const override;
    casacore::IPosition shape() const override;
    casacore::Bool ok() const override;
    casacore::DataType dataType() const override;
    
    casacore::Bool doGetSlice(casacore::Array<float>& buffer, 
                              const casacore::Slicer& section) override;
    void doPutSlice(const casacore::Array<float>& buffer, 
                    const casacore::IPosition& where, 
                    const casacore::IPosition& stride) override;
    
    const casacore::LatticeRegion* getRegionPtr() const override;
    casacore::ImageInterface<float>* cloneII() const override;
    void resize(const casacore::TiledShape& newShape) override;
    
    casacore::uInt advisedMaxPixels() const override;
    casacore::IPosition doNiceCursorShape(casacore::uInt maxPixels) const override;

    // Mask-related (ZARR images typically don't have masks)
    casacore::Bool isMasked() const override;
    casacore::Bool hasPixelMask() const override;
    const casacore::Lattice<casacore::Bool>& pixelMask() const override;
    casacore::Lattice<casacore::Bool>& pixelMask() override;
    casacore::Bool doGetMaskSlice(casacore::Array<casacore::Bool>& buffer, 
                                   const casacore::Slicer& section) override;
    
    // Additional methods
    casacore::Bool isPersistent() const override;
    casacore::Bool isWritable() const override;
    void flush() override;
    void tempClose() override;
    void reopen() override;

    // Access to underlying reader (for ZarrLoader)
    std::shared_ptr<ZarrDataReader> GetReader() const { return _reader; }

private:
    std::shared_ptr<ZarrDataReader> _reader;  // Shared across copies
    
    casacore::IPosition _shape;
    casacore::String _name;
    casacore::TiledShape _tiled_shape;
    
    bool _is_copy = false;
    mutable std::mutex _slice_mutex;
    
    // Coordinate system setup
    void SetupCoordinateSystem();
    void CreateDefaultCoordinateSystem();
    bool ParseWCSFromMetadata();
    void ParseBeamFromMetadata();
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_
