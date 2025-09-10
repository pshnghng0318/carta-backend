/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_
#define CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_

#include <casacore/images/Images/ImageInterface.h>
#include <casacore/coordinates/Coordinates/CoordinateSystem.h>
#include <casacore/lattices/Lattices/TiledShape.h>
#include <memory>

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

    // LatticeBase implementation
    casacore::Bool hasPixelMask() const override;
    const casacore::Lattice<casacore::Bool>& pixelMask() const override;
    casacore::Lattice<casacore::Bool>& pixelMask() override;

private:
    casacore::CoordinateSystem _coord_sys;
    casacore::IPosition _shape;
    casacore::String _name;
    
    void setupCoordinateSystem();
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_CARTAZARRIMAGE_H_
