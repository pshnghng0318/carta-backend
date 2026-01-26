/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-backend
   Copyright 2018- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include <fstream>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "ImageData/CartaZarrImage.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

namespace fs = std::filesystem;

class CartaZarrImageTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "carta_zarr_test";
        fs::create_directories(test_dir);
    }

    void TearDown() override {
        fs::remove_all(test_dir);
    }

    void CreateMockZarr(const fs::path& path) {
        fs::create_directories(path);
        
        // Root .zattrs
        nlohmann::json root_zattrs = {
            {"direction", {
                {"reference", {
                    {"data", {0.1, 0.2}},
                    {"attrs", {{"frame", "FK5"}, {"equinox", "J2000"}}}
                }},
                {"projection", "SIN"},
                {"latpole", {{"data", 0.3}}},
                {"lonpole", {{"data", 0.4}}},
                {"pc", {{"_value", {{1.0, 0.0}, {0.0, 1.0}}}}}
            }}
        };
        std::ofstream root_zattrs_file(path / ".zattrs");
        root_zattrs_file << root_zattrs.dump();
        root_zattrs_file.close();

        // SKY array
        fs::create_directories(path / "SKY");
        nlohmann::json sky_zarray = {
            {"zarr_format", 2},
            {"shape", {1, 10, 4, 100, 200}}, // [T, F, S, L, M]
            {"chunks", {1, 10, 4, 100, 200}},
            {"dtype", "<f4"},
            {"compressor", nullptr},
            {"fill_value", 0},
            {"order", "C"},
            {"filters", nullptr}
        };
        std::ofstream sky_zarray_file(path / "SKY" / ".zarray");
        sky_zarray_file << sky_zarray.dump();
        sky_zarray_file.close();
        
        nlohmann::json sky_zattrs = {
            {"_ARRAY_DIMENSIONS", {"time", "freq", "pol", "l", "m"}},
            {"units", "Jy/beam"},
            {"image_type", "Intensity"},
            {"object_name", "MOCK_OBJ"},
            {"observer", "MOCK_OBS"}
        };
        std::ofstream sky_zattrs_file(path / "SKY" / ".zattrs");
        sky_zattrs_file << sky_zattrs.dump();
        sky_zattrs_file.close();

        // Coordinate arrays (minimal 1D arrays for ZarrDataReader::ReadVector)
        auto create_vec_array = [&](const std::string& name, const std::vector<double>& data) {
            fs::create_directories(path / name);
            nlohmann::json z_arr = {
                {"zarr_format", 2},
                {"shape", {data.size()}},
                {"chunks", {data.size()}},
                {"dtype", "<f8"},
                {"compressor", nullptr},
                {"fill_value", 0},
                {"order", "C"},
                {"filters", nullptr}
            };
            std::ofstream zarray_file(path / name / ".zarray");
            zarray_file << z_arr.dump();
            zarray_file.close();

            // Write chunk "0"
            std::ofstream chunk_file(path / name / "0", std::ios::binary);
            chunk_file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
            chunk_file.close();
        };

        create_vec_array("l", {0.0, 0.001});
        create_vec_array("m", {0.0, 0.001});
        create_vec_array("frequency", {1.4e9, 1.41e9});
    }

    fs::path test_dir;
};

TEST_F(CartaZarrImageTest, GetFitsHeaderStringsTest) {
    fs::path zarr_path = test_dir / "mock.zarr";
    CreateMockZarr(zarr_path);

    std::unique_ptr<carta::CartaZarrImage> image;
    ASSERT_NO_THROW(image = std::make_unique<carta::CartaZarrImage>(zarr_path.string()));

    auto headers = image->FitsHeaderStrings();
    ASSERT_FALSE(headers.empty());

    // Helper to parse FITS header strings into a map
    // Format is approximately "KEY     = VALUE / comment" or "KEY     = 'VALUE'"
    std::map<std::string, std::string> header_map;
    for (const auto& h : headers) {
        // Key is first 8 chars, trimmed
        if (h.size() < 8) continue;
        std::string key = h.substr(0, 8);
        // Trim trailing spaces
        key.erase(key.find_last_not_of(" ") + 1);
        
        // Find '='
        size_t eq_pos = h.find('=');
        if (eq_pos != std::string::npos) {
            std::string val_part = h.substr(eq_pos + 1);
            // Trim leading/trailing spaces
            size_t first = val_part.find_first_not_of(" ");
            if (first == std::string::npos) {
                header_map[key] = "";
            } else {
                size_t last = val_part.find_last_not_of(" ");
                // Also remove potential comments starting with /? 
                // ZarrDataReader implementation doesn't seem to add comments usually, 
                // but standard FITS might have them.
                // The current implementation: fmt::format("{:<8}= '{}'", key, value) or "{:<8}= {}"
                // It does NOT add comments.
                // However, we should handle quotes for strings.
                std::string val = val_part.substr(first, (last - first + 1));
                if (val.size() >= 2 && val.front() == '\'' && val.back() == '\'') {
                    val = val.substr(1, val.size() - 2);
                }
                header_map[key] = val;
            }
        }
        spdlog::debug("Header parsed: [{}] -> [{}]", key, header_map[key]);
    }

    // 1. Basic Standard FITS Keys
    EXPECT_EQ(header_map["SIMPLE"], "T");
    EXPECT_TRUE(header_map.find("BITPIX") != header_map.end());
    EXPECT_EQ(header_map["NAXIS"], "4"); // CARTA shape [X, Y, F, S]
    EXPECT_EQ(header_map["NAXIS1"], "100"); // L -> X
    EXPECT_EQ(header_map["NAXIS2"], "200"); // M -> Y
    EXPECT_EQ(header_map["NAXIS3"], "10");  // F
    EXPECT_EQ(header_map["NAXIS4"], "4");   // S (from zarr shape in CreateMockZarr)

    // 2. Coordinate System (Direction)
    EXPECT_EQ(header_map["CTYPE1"], "RA---SIN");
    EXPECT_EQ(header_map["CTYPE2"], "DEC--SIN");
    EXPECT_EQ(header_map["RADESYS"], "FK5");
    EXPECT_EQ(header_map["EQUINOX"], "J2000");

    // 3. Direction Values (Radians -> Degrees)
    constexpr double kRadToDeg = 180.0 / M_PI;
    constexpr double kTol = 1e-6;

    // CRVAL1 = 0.1 rad, CRVAL2 = 0.2 rad
    EXPECT_NEAR(std::stod(header_map["CRVAL1"]), 0.1 * kRadToDeg, kTol);
    EXPECT_NEAR(std::stod(header_map["CRVAL2"]), 0.2 * kRadToDeg, kTol);

    // LATPOLE = 0.3 rad, LONPOLE = 0.4 rad
    EXPECT_NEAR(std::stod(header_map["LATPOLE"]), 0.3 * kRadToDeg, kTol);
    EXPECT_NEAR(std::stod(header_map["LONPOLE"]), 0.4 * kRadToDeg, kTol);

    // CDELT1, CDELT2
    // vector l: {0.0, 0.001}, m: {0.0, 0.001} (radians check mock)
    // Actually mock L/M are just "minimal 1D arrays".
    // CDELT = vec[1] - vec[0]
    double expected_cdelt = 0.001 * kRadToDeg;
    EXPECT_NEAR(std::stod(header_map["CDELT1"]), expected_cdelt, kTol);
    EXPECT_NEAR(std::stod(header_map["CDELT2"]), expected_cdelt, kTol);

    // CRPIX1, CRPIX2
    // CRPIX = (-val[0] / delta) + 1
    // val[0] = 0.0. So CRPIX = 1.0.
    EXPECT_NEAR(std::stod(header_map["CRPIX1"]), 1.0, kTol);
    EXPECT_NEAR(std::stod(header_map["CRPIX2"]), 1.0, kTol);

    // PC Matrix (Identity in mock)
    EXPECT_NEAR(std::stod(header_map["PC1_1"]), 1.0, kTol);
    EXPECT_NEAR(std::stod(header_map["PC1_2"]), 0.0, kTol);
    EXPECT_NEAR(std::stod(header_map["PC2_1"]), 0.0, kTol);
    EXPECT_NEAR(std::stod(header_map["PC2_2"]), 1.0, kTol);

    // 4. Spectral Axis
    EXPECT_EQ(header_map["CTYPE3"], "FREQ");
    // Mock freq: {1.4e9, 1.41e9}
    EXPECT_NEAR(std::stod(header_map["CRVAL3"]), 1.4e9, kTol);
    EXPECT_NEAR(std::stod(header_map["CDELT3"]), 1.0e7, kTol); // 1.41e9 - 1.4e9
    EXPECT_NEAR(std::stod(header_map["CRPIX3"]), 1.0, kTol);

    // 5. Stokes Axis (Defaults because "polarization" array missing in mock)
    // Code defaults to CDELT4=1.0, CRVAL4=1.0
    EXPECT_NEAR(std::stod(header_map["CDELT4"]), 1.0, kTol);
    // If polarization array is missing, code sets CRVAL4=1.0
    EXPECT_NEAR(std::stod(header_map["CRVAL4"]), 1.0, kTol);
    // Check fallback CTYPE4
    // If missing, implementation uses CTYPE4="STOKES" if it tries to parse it?
    // Actually:
    // try { ... ReadStringVector("polarization") ... if (!pol.empty) ... else { ... add CRVAL4/CDELT4 ... "CUNIT4" } ... }
    // It does NOT explicit add CTYPE4 in the `else` block of `if (!pol_strs.empty())`. 
    // Wait, let's re-read line 981: `add_string_header("CTYPE4", "STOKES");` is BEFORE the if check.
    // So CTYPE4 should be STOKES.
    EXPECT_EQ(header_map["CTYPE4"], "STOKES");


    // 6. Object / Observation Metadata
    EXPECT_EQ(header_map["OBJECT"], "MOCK_OBJ");
    EXPECT_EQ(header_map["OBSERVER"], "MOCK_OBS");
    EXPECT_EQ(header_map["BUNIT"], "Jy/beam");

    // 7. END keyword
    // The last element in vector should be END, but in map it's just a key 'END' with empty value (likely)
    EXPECT_TRUE(header_map.find("END") != header_map.end());
}
