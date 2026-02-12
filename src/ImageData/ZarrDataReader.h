/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_SRC_IMAGEDATA_ZARRDATAREADER_H_
#define CARTA_SRC_IMAGEDATA_ZARRDATAREADER_H_

#include <map> // Added for std::map
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Slicer.h>

namespace carta {

/**
 * @brief Encapsulates TensorStore operations for Zarr file access.
 *
 * This class provides a clean interface for reading Zarr data, handling
 * coordinate transformations between CARTA format [x,y,z,stokes] and
 * ZARR native format (e.g., [time,freq,pol,l,m] for 5D data).
 *
 * Uses Pimpl idiom to hide TensorStore dependencies from the header.
 */
class ZarrDataReader {
public:
    explicit ZarrDataReader(const std::string& filename);
    ~ZarrDataReader(); // Non-default destructor needed for Pimpl with unique_ptr

    // Prevent copying
    ZarrDataReader(const ZarrDataReader&) = delete;
    ZarrDataReader& operator=(const ZarrDataReader&) = delete;

    /**
     * @brief Initialize TensorStore connection to the Zarr file.
     * @return true if initialization succeeded
     */
    bool Initialize();

    /**
     * @brief Read a slice of data directly from TensorStore.
     * @param section The slicer defining the region to read (CARTA coordinates)
     * @param buffer Output array to fill with data
     * @return true if read succeeded
     */
    bool ReadSlice(casacore::Array<float>& buffer, const casacore::Slicer& section);

    /**
     * @brief Read an entire channel (2D spatial plane) from TensorStore.
     * @param channel The frequency/spectral channel index
     * @param stokes The stokes/polarization index
     * @param data Output vector to fill with data
     * @return true if read succeeded
     */
    bool ReadChannel(int channel, int stokes, std::vector<float>& data);

    /**
     * @brief Read a chunk of data for TileCache integration.
     * @param data Output vector to fill with chunk data
     * @param data_width Output: actual width of returned data
     * @param data_height Output: actual height of returned data
     * @param min_x Starting X coordinate
     * @param min_y Starting Y coordinate
     * @param channel Frequency/spectral channel index
     * @param stokes Stokes/polarization index
     * @return true if read succeeded
     */
    bool GetChunk(std::vector<float>& data, int& data_width, int& data_height, int min_x, int min_y, int channel, int stokes);

    /**
     * @brief Read spectral profile at a single spatial position (all channels).
     *
     * This method reads the entire frequency axis in a single TensorStore request,
     * which is much more efficient than calling ReadSlice for each channel.
     *
     * @param x The X coordinate (L axis in XRADIO schema)
     * @param y The Y coordinate (M axis in XRADIO schema)
     * @param stokes The stokes/polarization index
     * @param data Output vector with all channel values
     * @return true if read succeeded
     */
    bool ReadSpectralProfile(int x, int y, int stokes, std::vector<float>& data);

    // Read a 1D array (e.g. coordinates l, m, frequency) by relative path
    // Returns empty vector on failure
    std::vector<double> ReadVector(const std::string& array_name);

    // Read a 1D string array (e.g. polarization)
    std::vector<std::string> ReadStringVector(const std::string& array_name);

    // Read an array of any rank by relative path and return as flat vector
    // Returns empty vector on failure
    std::vector<double> ReadFlattenedVector(const std::string& array_name);

    // Read attribute string (e.g. unit) from an array
    std::string GetAttributeString(const std::string& array_name, const std::string& attr_name);

    // Read flat JSON from .zattrs as a map (helper for generic metadata)
    // Note: Returns a simplified map for string values; complex parsing still done in Image class
    std::map<std::string, std::string> GetZattrMap(const std::string& array_name = "");

    // Helper to get raw JSON content of .zattrs for full parsing
    std::string GetZattrsString(const std::string& array_name = "");

    // Helper to get raw JSON content of .zarray for full parsing
    std::string GetZarrayString(const std::string& array_name = "");

    // Accessors
    bool IsInitialized() const;
    const casacore::IPosition& GetShape() const;
    const casacore::IPosition& GetOriginalZarrShape() const;
    const casacore::IPosition& GetChunkShape() const;
    const std::string& GetFilename() const;
    int NumDimensions() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;

    // Cached metadata for quick access without dereferencing impl
    casacore::IPosition _shape;
    casacore::IPosition _original_shape;
    casacore::IPosition _chunk_shape;
    std::string _filename;
    bool _initialized = false;

    mutable std::mutex _read_mutex;

    /**
     * @brief Find the Zarr array path (handles hierarchical zarr files)
     */
    std::string FindArrayPath() const;
};

} // namespace carta

#endif // CARTA_SRC_IMAGEDATA_ZARRDATAREADER_H_
