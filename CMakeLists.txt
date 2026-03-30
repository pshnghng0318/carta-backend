cmake_minimum_required(VERSION 3.20)
project(carta_backend)

include_directories(src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
set(CMAKE_LINK_DIRECTORIES_BEFORE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/third-party/install)

include(uWebSockets)

if(DEPLOYMENT_TYPE)
    add_definitions(-DDEPLOYMENT_TYPE="${DEPLOYMENT_TYPE}")
endif()

# Enable OpenMP if package is found
find_package(OpenMP)
if (OPENMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    message(WARNING "OpenMP not found!")
endif ()

# Use the -march=native flags when building on the same architecture as deploying to get a slight performance
# increase when running CPU intensive tasks such as compression and down-sampling of data. If targeting AVX-capable
# processes only, set EnableAvx to ON
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
option(EnableAvx "Enable AVX codepaths instead of SSE4" OFF)

# Automatically detect if building on an ARM based system such as the Apple M1.
# It will replace SSE functions with ARM NEON functions using sse2neon.h from 
# https://github.com/DLTcollab/sse2neon, redistributable under the MIT License.

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ARM_ARCH_ -march=armv8-a+fp+simd+crypto+crc")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ARM_ARCH_ -march=armv8-a+fp+simd+crypto+crc")
elseif (EnableAvx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")
endif ()

# Needed by clang-tidy and other clang tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FIND_PACKAGE(GSL QUIET)
if (GSL_FOUND)
    message(STATUS "Found gsl using find_package")
else ()
    if(EXISTS /opt/carta-gsl)
        message(STATUS "Found carta-gsl for el7")
        INCLUDE_DIRECTORIES(/opt/carta-gsl/include)
        LINK_DIRECTORIES(/opt/carta-gsl/lib)
    else ()
        message(FATAL_ERROR "Could not find gsl")
    endif ()
endif()

FIND_PACKAGE(ZFP CONFIG REQUIRED)
FIND_PACKAGE(PkgConfig REQUIRED)

FIND_PACKAGE(CFITSIO QUIET)
if (CFITSIO_FOUND)
    message(STATUS "Found cfitsio using find_package")
else ()
    PKG_SEARCH_MODULE(CFITSIO REQUIRED cfitsio)
    if (CFITSIO_FOUND)
        message(STATUS "Found cfitsio using pkg-config")
        include_directories (${CFITSIO_INCLUDE_DIRS})
    else ()
        message(FATAL_ERROR "Could not find cfitsio")
    endif ()
endif()

FIND_PACKAGE(WCSLIB QUIET)
if (WCSLIB_FOUND)
    message(STATUS "Found wcslib using find_package")
else ()
    PKG_SEARCH_MODULE(WCSLIB REQUIRED wcslib)
    if (WCSLIB_FOUND)
        message(STATUS "Found wcslib using pkg-config")
    else ()
        message(FATAL_ERROR "Could not find wcslib")
    endif ()
endif()

PKG_SEARCH_MODULE(UUID REQUIRED uuid)
if (UUID_FOUND)
    message(STATUS "Found uuid using pkg-config")
else ()
    message(FATAL_ERROR "Could not find uuid")
endif ()

PKG_SEARCH_MODULE(ZSTD REQUIRED libzstd)
if (ZSTD_FOUND)
    message(STATUS "zstd found using pkg-config")
else ()
    message(FATAL_ERROR "Could not find zstd")
endif ()

FIND_PACKAGE(HDF5 REQUIRED COMPONENTS CXX)
FIND_PACKAGE(Protobuf REQUIRED)

if (Protobuf_VERSION VERSION_GREATER_EQUAL "4.25.3")
    # Use TensorStore's Abseil libraries instead of system ones to avoid conflicts
    set(PROTOBUF_LIBRARY
            ${PROTOBUF_LIBRARY})
    message(STATUS "Newer protobuf version (${Protobuf_VERSION}) will use TensorStore's abseil libraries")
endif ()

FIND_PACKAGE(Threads)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})

ADD_SUBDIRECTORY(carta-protobuf)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

if (NOT CARTA_CASACORE_ROOT)
    set(CARTA_CASACORE_ROOT /opt/carta-casacore)
endif()

set(CARTA_CASACORE_INCLUDE_DIRS
    ${CARTA_CASACORE_ROOT}/include
    ${CARTA_CASACORE_ROOT}/include/casacore
    ${CARTA_CASACORE_ROOT}/include/casacode)

option(DevSuppressExternalWarnings "Suppress external warnings (developer use only)" OFF)
if (DevSuppressExternalWarnings)
    message(STATUS "disabling warnings from carta-casacore")
    INCLUDE_DIRECTORIES(SYSTEM ${CARTA_CASACORE_INCLUDE_DIRS})
else()
    INCLUDE_DIRECTORIES(${CARTA_CASACORE_INCLUDE_DIRS})
endif()
LINK_DIRECTORIES(${CARTA_CASACORE_ROOT}/lib)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    SET(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    INCLUDE_DIRECTORIES(/usr/local/include)
    INCLUDE_DIRECTORIES(/usr/local/opt/openssl/include)
    INCLUDE_DIRECTORIES(/usr/local/opt/libomp/include)
    LINK_DIRECTORIES(/usr/local/lib)
    LINK_DIRECTORIES(/usr/local/opt)
    # Use system abseil libraries to match protobuf
    LINK_DIRECTORIES(/opt/homebrew/lib)
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    INCLUDE_DIRECTORIES(/usr/include)
    LINK_DIRECTORIES(/usr/lib/x86_64-linux-gnu)
    LINK_DIRECTORIES(/usr/lib)
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if (DisableContourCompression)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DISABLE_CONTOUR_COMPRESSION_")
endif (DisableContourCompression)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION LESS 11.2.0)
    set(LINK_LIBS ${LINK_LIBS} stdc++fs)
endif ()

# Include uWebSockets headers and build the uSockets lib
include_directories(${CMAKE_SOURCE_DIR}/third-party/include)
install_uWebSockets()

#pugixml
include_directories(${CMAKE_SOURCE_DIR}/third-party/pugixml/src)

#spdlog
add_subdirectory(third-party/spdlog)

# JSON schemas
add_subdirectory(schemas)

# nlohmann/json
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(third-party/nlohmann_json)

# Use TensorStore's nlohmann_json for json-schema-validator to ensure ABI compatibility
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(TENSORSTORE_BUILD_ROOT "${CMAKE_SOURCE_DIR}/tensorstore/build")
    include_directories(BEFORE "${TENSORSTORE_BUILD_ROOT}/_deps/nlohmann_json-src/include")
endif()

# json-schema-validator
set(nlohmann_json_VERSION "v3.11.3")
add_subdirectory(third-party/json-schema-validator)

# xtensor and xtl configuration
include_directories("${CMAKE_SOURCE_DIR}/third-party/xtl/include")
include_directories("${CMAKE_SOURCE_DIR}/third-party/xtensor/include")

# TensorStore configuration
set(TENSORSTORE_ROOT "${CMAKE_SOURCE_DIR}/tensorstore")
set(TENSORSTORE_BUILD_ROOT "${TENSORSTORE_ROOT}/build")

# TensorStore include directories  
include_directories("${TENSORSTORE_ROOT}")
include_directories("${TENSORSTORE_BUILD_ROOT}")
include_directories("${TENSORSTORE_ROOT}/tensorstore")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # macOS: Use old paths for CMake build
    include_directories("${TENSORSTORE_ROOT}/tensorstore")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/re2-src")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/riegeli-src")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-src")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/half-src/include")
    include_directories("/opt/homebrew/opt/c-blosc/include")
    include_directories(BEFORE "${TENSORSTORE_ROOT}/_deps/absl-src")
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Linux: Use TensorStore CMake-built headers from _deps
    # abseil-cpp must come BEFORE system headers to avoid conflicts
    include_directories(BEFORE "${TENSORSTORE_BUILD_ROOT}/_deps/absl-src")
    include_directories(BEFORE "${TENSORSTORE_BUILD_ROOT}/_deps/nlohmann_json-src/include")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/riegeli-src")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/re2-src")
    include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/half-src/include")
    # Include system blosc headers for compression support
    include_directories("/usr/include")
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # macOS: Use system libraries instead of TensorStore static libraries
    find_package(PkgConfig REQUIRED)

    # Use system absl and other dependencies
    set(CMAKE_PREFIX_PATH "/opt/homebrew;${CMAKE_PREFIX_PATH}")
    find_library(ABSL_BASE_LIB absl_base PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_STRINGS_LIB absl_strings PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_STRINGS_INTERNAL_LIB absl_strings_internal PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_STATUS_LIB absl_status PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_STATUSOR_LIB absl_statusor PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_HASH_LIB absl_hash PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_LOG_INTERNAL_MESSAGE_LIB absl_log_internal_message PATHS /opt/homebrew/lib)
    find_library(ABSL_LOG_INTERNAL_CONDITIONS_LIB absl_log_internal_conditions PATHS /opt/homebrew/lib)
    find_library(ABSL_LOG_SEVERITY_LIB absl_log_severity PATHS /opt/homebrew/lib)

    # Additional critical absl libraries for TensorStore linking
    find_library(ABSL_CORD_LIB absl_cord PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_CORD_INTERNAL_LIB absl_cord_internal PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_CORDZ_FUNCTIONS_LIB absl_cordz_functions PATHS /opt/homebrew/lib)
    find_library(ABSL_CORDZ_HANDLE_LIB absl_cordz_handle PATHS /opt/homebrew/lib)
    find_library(ABSL_CORDZ_INFO_LIB absl_cordz_info PATHS /opt/homebrew/lib)

    # Include blosc headers for compression support
    include_directories("/opt/homebrew/opt/c-blosc/include")
    
    # macOS: Use individual find_library for system abseil
    find_library(ABSL_CORDZ_SAMPLE_TOKEN_LIB absl_cordz_sample_token PATHS /opt/homebrew/lib)
    find_library(ABSL_CRC32C_LIB absl_crc32c PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_CRC_INTERNAL_LIB absl_crc_internal PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_CRC_CORD_STATE_LIB absl_crc_cord_state PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_CRC_CPU_DETECT_LIB absl_crc_cpu_detect PATHS /opt/homebrew/lib)
    find_library(ABSL_SYNCHRONIZATION_LIB absl_synchronization PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_TIME_LIB absl_time PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_CIVIL_TIME_LIB absl_civil_time PATHS /opt/homebrew/lib)
    find_library(ABSL_TIME_ZONE_LIB absl_time_zone PATHS /opt/homebrew/lib)
    find_library(ABSL_INT128_LIB absl_int128 PATHS /opt/homebrew/lib)
    find_library(ABSL_RAW_LOGGING_INTERNAL_LIB absl_raw_logging_internal PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_LOG_INTERNAL_CHECK_OP_LIB absl_log_internal_check_op PATHS /opt/homebrew/lib)
    find_library(ABSL_LOG_INTERNAL_GLOBALS_LIB absl_log_internal_globals PATHS /opt/homebrew/lib)
    find_library(ABSL_THROW_DELEGATE_LIB absl_throw_delegate PATHS /opt/homebrew/lib REQUIRED)
    find_library(ABSL_MALLOC_INTERNAL_LIB absl_malloc_internal PATHS /opt/homebrew/lib)
    find_library(ABSL_SPINLOCK_WAIT_LIB absl_spinlock_wait PATHS /opt/homebrew/lib)
    find_library(ABSL_RAW_HASH_SET_LIB absl_raw_hash_set PATHS /opt/homebrew/lib)
    find_library(ABSL_HASHTABLEZ_SAMPLER_LIB absl_hashtablez_sampler PATHS /opt/homebrew/lib)
    find_library(ABSL_CITY_LIB absl_city PATHS /opt/homebrew/lib)
    find_library(ABSL_STR_FORMAT_INTERNAL_LIB absl_str_format_internal PATHS /opt/homebrew/lib)
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# TensorStore link directories - use CMake build path
link_directories("${TENSORSTORE_BUILD_ROOT}")

# Add TensorStore CMake build _deps link directories
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/strings")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/base")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/numeric")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/time")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/types")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/container")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/hash")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/synchronization")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-build/absl/status")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/riegeli-build")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/re2-build")
link_directories("${TENSORSTORE_BUILD_ROOT}/_deps/grpc-build")

# Automatically collect all TensorStore library files from CMake build
file(GLOB_RECURSE TENSORSTORE_ALL_LIBS_FILES "${TENSORSTORE_BUILD_ROOT}/libtensorstore_*.a")
# Filter out the main library (it's in TENSORSTORE_MAIN_LIB) to avoid duplicate linking
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "libtensorstore_tensorstore\\.a$")
# Filter out test libraries
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "test")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "gtest")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "gmock")
# Exclude mock libraries, but keep aws_http_mocking which is required
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "mock_")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "benchmark")

# Collect dependencies from TensorStore CMake build _deps
file(GLOB TENSORSTORE_CMAKE_DEPS_RIEGELI 
    "${TENSORSTORE_BUILD_ROOT}/_deps/riegeli-build/*.a")
# Filter out protobuf-dependent riegeli libraries
list(FILTER TENSORSTORE_CMAKE_DEPS_RIEGELI EXCLUDE REGEX "message")

# Collect only non-system dependencies from TensorStore CMake build
# System already has: re2, protobuf, grpc (but use TensorStore's absl for ABI compatibility)
# We need: absl (from TensorStore), riegeli, image libraries (jpeg, png, webp, etc.), compression (blake3, etc.)
file(GLOB_RECURSE TENSORSTORE_CMAKE_DEPS 
    "${TENSORSTORE_BUILD_ROOT}/_deps/*-build/*.a"
    "${TENSORSTORE_BUILD_ROOT}/_deps/*-build/**/*.a"
)
# Filter out libraries to avoid ABI conflicts, but keep absl from TensorStore
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "protobuf")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "grpc")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "googleapis")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "udpa")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "envoy")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "validate")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "xds")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "cel-spec")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "upb")
# Filter out riegeli (handled separately above)
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "riegeli")
# Filter out problematic libraries with main() conflicts
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "crc_clmul_consts_gen")
# Filter out LibLZMA_lzma.a (contains crc_clmul_consts_gen.c.o with main())
# Keep LibLZMA_src_common.a and use system liblzma
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "libLibLZMA_lzma\\.a$")
# Add filtered riegeli libraries
list(APPEND TENSORSTORE_CMAKE_DEPS ${TENSORSTORE_CMAKE_DEPS_RIEGELI})

# Filter out test libraries
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "_test\\.a$")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "_test_util\\.a$")
# Filter out protobuf-dependent libraries to avoid ABI conflicts with system protobuf
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "proto")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "serialization_protobuf")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "tsgrpc")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "_grpc")
# Filter out cloud storage backends that we don't need (S3, GCS grpc/testbench)
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "kvstore_s3")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "kvstore_gcs")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "gcs_testbench")
# Filter out ZIP kvstore which also depends on serialization
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "kvstore_zip")
# Filter out OCDBT (distributed B-tree) which depends on grpc and protobuf
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "ocdbt")
# Filter out CLI tools
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "tscli")
# Filter out testing/benchmark utilities
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "testing")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "testutil")
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "test_httpserver")
# Filter out OAuth2 test utilities
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "google_auth_test_utils")
# Filter out base metrics lib (keep specific ones like histogram, collect, etc.)
# This avoids multiple definition errors with histogram.cc
list(FILTER TENSORSTORE_ALL_LIBS_FILES EXCLUDE REGEX "libtensorstore_internal_metrics\\.a$")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "_test\\.a$")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "gtest")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "gmock")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "benchmark")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "scoped_mock_log")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "status_matchers")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "test_helpers")
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "test_matchers")
# Filter out protobuf libraries to use system protobuf instead
list(FILTER TENSORSTORE_CMAKE_DEPS EXCLUDE REGEX "protobuf")

# Sort the libraries to ensure consistent linking order
list(SORT TENSORSTORE_ALL_LIBS_FILES)
list(SORT TENSORSTORE_CMAKE_DEPS)

# Display found libraries for debugging
list(LENGTH TENSORSTORE_ALL_LIBS_FILES TENSORSTORE_COUNT)
list(LENGTH TENSORSTORE_CMAKE_DEPS CMAKE_DEPS_COUNT)
message(STATUS "Found ${TENSORSTORE_COUNT} TensorStore libraries")
message(STATUS "Found ${CMAKE_DEPS_COUNT} CMake _deps libraries")

# Ensure critical TensorStore main library is explicitly identified
# Use the main tensorstore library from CMake build
set(TENSORSTORE_MAIN_LIB "${TENSORSTORE_BUILD_ROOT}/libtensorstore_tensorstore.a")

# Create complete TensorStore library list
set(TENSORSTORE_ALL_LIBS
    ${TENSORSTORE_ALL_LIBS_FILES}
    ${TENSORSTORE_CMAKE_DEPS}
)



set(LINK_LIBS
        ${LINK_LIBS}
        carta-protobuf
        ${PROTOBUF_LIBRARY}
        uSockets
        uuid
        z
        zfp
        cfitsio
        gsl
        nlohmann_json::nlohmann_json
        nlohmann_json_schema_validator
        casa_casa
        casa_coordinates
        casa_tables
        casa_images
        casa_lattices
        casa_fits
        casa_measures
        casa_mirlib
        casa_scimath
        casa_imageanalysis
        ${OpenMP_CXX_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        # Add TensorStore libraries and all dependencies
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LINK_LIBS
            ${LINK_LIBS}
            /opt/homebrew/lib/libzstd.dylib
            /opt/homebrew/lib/libwcs.dylib
    )
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(LINK_LIBS
            ${LINK_LIBS}
            zstd
            wcs
            snappy
            lz4
            lzma
            bz2
            blosc
            re2
    )
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

set(SOURCE_FILES
        ${SOURCE_FILES}
        third-party/pugixml/src/pugixml.cpp
        src/Cache/LoaderCache.cc
        src/Cache/TileCache.cc
        src/Cache/TilePool.cc
        src/DataStream/Compression.cc
        src/DataStream/Contouring.cc
        src/DataStream/Smoothing.cc
        src/DataStream/Tile.cc
        src/DataStream/VectorField.cc
        src/FileList/FileExtInfoLoader.cc
        src/FileList/FileInfoLoader.cc
        src/FileList/FileListHandler.cc
        src/FileList/FitsHduList.cc
        src/Frame/Frame.cc
        src/HttpServer/HttpServer.cc
        src/ImageData/CartaFitsImage.cc
        src/ImageData/CartaHdf5Image.cc
        src/ImageData/CartaMiriadImage.cc
        src/ImageData/CompressedFits.cc
        src/ImageData/FileInfo.cc
        src/ImageData/FileLoader.cc
        src/ImageData/FitsLoader.cc
        src/ImageData/FitsUtil.cc
        src/ImageData/Hdf5Attributes.cc
        src/ImageData/Hdf5Loader.cc
        src/ImageData/PolarizationCalculator.cc
        src/ImageData/StokesFilesConnector.cc
        src/ImageData/CartaZarrImage.cc
        src/ImageData/ZarrLoader.cc
        src/ImageData/ZarrDataReader.cc
        # TensorStore zarr driver sources are already compiled in TensorStore libraries
        # tensorstore/tensorstore/driver/zarr/driver.cc
        # tensorstore/tensorstore/driver/zarr/metadata.cc
        # tensorstore/tensorstore/driver/zarr/spec.cc
        # tensorstore/tensorstore/driver/zarr/dtype.cc
        # tensorstore/tensorstore/driver/zarr/compressor.cc
        # tensorstore/tensorstore/driver/zarr/blosc_compressor.cc
        # tensorstore/tensorstore/driver/zarr/zstd_compressor.cc
        # tensorstore/tensorstore/driver/zarr/zlib_compressor.cc
        # tensorstore/tensorstore/driver/zarr/bzip2_compressor.cc
        # TensorStore internal compression implementations are already compiled
        # tensorstore/tensorstore/internal/compression/blosc_compressor.cc
        # tensorstore/tensorstore/internal/compression/zstd_compressor.cc
        # tensorstore/tensorstore/internal/compression/zlib_compressor.cc
        # tensorstore/tensorstore/internal/compression/bzip2_compressor.cc
        # tensorstore/tensorstore/internal/compression/json_specified_compressor.cc
        # tensorstore/tensorstore/context.cc
        # tensorstore/tensorstore/internal/concurrency_resource.cc
        # tensorstore/tensorstore/internal/data_copy_concurrency_resource.cc
        # tensorstore/tensorstore/internal/file_io_concurrency_resource.cc
        # tensorstore/tensorstore/internal/cache/cache_pool_resource.cc
        # TensorStore file KvStore driver is already compiled
        # tensorstore/tensorstore/kvstore/file/file_key_value_store.cc
        # tensorstore/tensorstore/kvstore/file/file_resource.cc
        src/ImageFitting/ImageFitter.cc
        src/ImageFitting/InitialValueCalculator.cc
        src/ImageGenerators/MomentGenerator.cc
        src/ImageGenerators/PvGenerator.cc
        src/ImageGenerators/PvPreviewCube.cc
        src/ImageGenerators/PvPreviewCut.cc
        src/ImageStats/Histogram.cc
        src/ImageStats/StatsCalculator.cc
        src/Logger/CartaLogSink.cc
        src/Logger/Logger.cc
        src/Main/Main.cc
        src/Main/ProgramSettings.cc
        src/Main/WebBrowser.cc
        src/Region/ImageRegion.cc
        src/Region/Region.cc
        src/Region/RegionConverter.cc
        src/Region/RegionHandler.cc
        src/Region/RegionAnalysis/LineBoxRegions.cc
        src/Region/RegionAnalysis/RegionHistogram.cc
        src/Region/RegionAnalysis/RegionSpatialProfile.cc
        src/Region/RegionAnalysis/RegionStatistics.cc
        src/Region/RegionImportExport/CrtfExporter.cc
        src/Region/RegionImportExport/CrtfImporter.cc
        src/Region/RegionImportExport/Ds9Exporter.cc
        src/Region/RegionImportExport/Ds9Importer.cc
        src/Region/RegionImportExport/RegionExporter.cc
        src/Region/RegionImportExport/RegionImporter.cc
        src/Region/RegionImportExport/RegionImportExportUtil.cc
        src/Session/ChannelMapSettings.cc
        src/Session/CursorSettings.cc
        src/Session/OnMessageTask.cc
        src/Session/Session.cc
        src/Session/SessionManager.cc
        src/Table/Column.cc
        src/Table/Table.cc
        src/Table/TableController.cc
        src/Table/TableView.cc
        src/ThreadManager/ThreadManager.cc
        src/Timer/ListProgressReporter.cc
        src/Timer/Timer.cc
        src/Util/App.cc
        src/Util/Casacore.cc
        src/Util/File.cc
        src/Util/Json.cc
        src/Util/Memory.cc
        src/Util/Message.cc
        src/Util/RemoteFiles.cc
        src/Util/Stokes.cc
        src/Util/String.cc
        src/Util/Token.cc)

add_definitions(-DHAVE_HDF5 -DPUGIXML_COMPACT)
add_executable(carta_backend ${SOURCE_FILES})

add_dependencies(carta_backend carta-schemas)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    list(REMOVE_ITEM LINK_LIBS uuid)
    
    # Prepare force-load arguments for compression libraries
    set(FORCE_LOAD_ARGS "")
    foreach(lib ${TENSORSTORE_FORCE_LOAD_LIBS})
        if(EXISTS ${lib})
            list(APPEND FORCE_LOAD_ARGS "-Wl,-force_load,${lib}")
        endif()
    endforeach()
    
    target_link_libraries(carta_backend
        PRIVATE
            protobuf
            /opt/homebrew/lib/libuv.dylib
            ${LINK_LIBS}
            -Wl,-force_load,${TENSORSTORE_MAIN_LIB}
            ${FORCE_LOAD_ARGS})
            
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        target_link_libraries(carta_backend
        PRIVATE
            protobuf
            uv
            # Link TensorStore libraries with whole-archive to ensure all symbols are included
            -Wl,--whole-archive ${TENSORSTORE_MAIN_LIB} ${TENSORSTORE_ALL_LIBS} -Wl,--no-whole-archive
            # Add CMake deps (absl from TensorStore, riegeli, image libs, etc.)
            ${TENSORSTORE_CMAKE_DEPS}
            # Link carta-backend's own libraries after TensorStore
            ${LINK_LIBS})
            
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if (CartaUserFolderPrefix)
    add_definitions(-DCARTA_USER_FOLDER_PREFIX="${CartaUserFolderPrefix}")
endif (CartaUserFolderPrefix)

if (CartaDefaultFrontendFolder)
    add_definitions(-DCARTA_DEFAULT_FRONTEND_FOLDER="${CartaDefaultFrontendFolder}")
endif (CartaDefaultFrontendFolder)

install(TARGETS carta_backend
    RUNTIME DESTINATION bin)

install(PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/carta"
    DESTINATION bin)

install(FILES "${CMAKE_SOURCE_DIR}/static/default_image/default.fits"
    DESTINATION share/carta)

# Tests
option(test "Build tests." OFF)
if (test)
    FIND_PACKAGE(GTest REQUIRED)
    add_subdirectory(test)
endif ()