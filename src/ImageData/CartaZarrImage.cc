/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "CartaZarrImage.h"
#include "Logger/Logger.h"

#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/lattices/Lattices/ArrayLattice.h>

using namespace casacore;

namespace carta {

CartaZarrImage::CartaZarrImage(const std::string& filename) : _name(filename) {
    // Minimal setup for file browser support
    _shape = IPosition(2, 1, 1);  // Default 1x1 shape
    setupCoordinateSystem();
}

void CartaZarrImage::setupCoordinateSystem() {
    // Create minimal coordinate system for file browser support
    DirectionCoordinate dir_coord;
    SpectralCoordinate spec_coord;
    
    _coord_sys.addCoordinate(dir_coord);
    _coord_sys.addCoordinate(spec_coord);
}

String CartaZarrImage::imageType() const {
    return "zarr";
}

DataType CartaZarrImage::dataType() const {
    return TpFloat;
}

Bool CartaZarrImage::doGetSlice(Array<float>& buffer, const Slicer& section) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::doGetSlice not implemented");
    return false;
}

void CartaZarrImage::doPutSlice(const Array<float>& buffer, const IPosition& where, const IPosition& stride) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::doPutSlice not implemented");
}

Bool CartaZarrImage::doGetMaskSlice(Array<Bool>& buffer, const Slicer& section) {
    // Stub implementation - not needed for file browser
    return false;
}

Bool CartaZarrImage::isMasked() const {
    return false;
}

Bool CartaZarrImage::isPersistent() const {
    return true;
}

Bool CartaZarrImage::isWritable() const {
    return false;
}

String CartaZarrImage::name(Bool stripPath) const {
    if (stripPath) {
        auto pos = _name.find_last_of('/');
        if (pos != std::string::npos) {
            return _name.substr(pos + 1);
        }
    }
    return _name;
}

IPosition CartaZarrImage::shape() const {
    return _shape;
}

void CartaZarrImage::resize(const TiledShape& newShape) {
    // Stub implementation - not needed for file browser
    spdlog::warn("CartaZarrImage::resize not implemented");
}

Bool CartaZarrImage::ok() const {
    return true;  // Return true for basic file browser support
}

void CartaZarrImage::flush() {
    // Stub implementation - not needed for file browser
}

void CartaZarrImage::tempClose() {
    // Stub implementation - not needed for file browser
}

void CartaZarrImage::reopen() {
    // Stub implementation - not needed for file browser
}

Bool CartaZarrImage::hasPixelMask() const {
    return false;
}

const Lattice<Bool>& CartaZarrImage::pixelMask() const {
    // Return a dummy lattice
    static auto dummy_lattice = std::make_shared<ArrayLattice<Bool>>(Array<Bool>(IPosition(2, 1, 1)));
    return *dummy_lattice;
}

Lattice<Bool>& CartaZarrImage::pixelMask() {
    // Return a dummy lattice
    static auto dummy_lattice = std::make_shared<ArrayLattice<Bool>>(Array<Bool>(IPosition(2, 1, 1)));
    return *dummy_lattice;
}

} // namespace carta
