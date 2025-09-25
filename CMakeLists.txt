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

# json-schema-validator
set(nlohmann_json_VERSION "v3.11.3")
add_subdirectory(third-party/json-schema-validator)

# TensorStore configuration
set(TENSORSTORE_ROOT "${CMAKE_SOURCE_DIR}/tensorstore")
set(TENSORSTORE_BUILD_ROOT "${TENSORSTORE_ROOT}/build")

# TensorStore include directories  
include_directories("${TENSORSTORE_ROOT}")
include_directories("${TENSORSTORE_ROOT}/tensorstore")
include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/re2-src")
include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/riegeli-src")
include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/absl-src")
include_directories("${TENSORSTORE_BUILD_ROOT}/_deps/half-src/include")

# Include blosc headers for compression support
include_directories("/opt/homebrew/opt/c-blosc/include")

# Include TensorStore absl headers for protobuf compatibility
include_directories(BEFORE "${TENSORSTORE_ROOT}/_deps/absl-src")

# Use system libraries instead of TensorStore static libraries
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

# TensorStore link directories - use bazel-bin path
link_directories("${TENSORSTORE_ROOT}/bazel-bin/tensorstore")

# Automatically collect all TensorStore library files
file(GLOB_RECURSE TENSORSTORE_ALL_LIBS_FILES "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/*.a")

# Collect external dependencies from TensorStore Bazel build
file(GLOB_RECURSE TENSORSTORE_EXTERNAL_DEPS "${TENSORSTORE_ROOT}/bazel-bin/external/*.a")

# Sort the libraries to ensure consistent linking order
list(SORT TENSORSTORE_ALL_LIBS_FILES)
list(SORT TENSORSTORE_EXTERNAL_DEPS)

# Display found libraries for debugging
list(LENGTH TENSORSTORE_ALL_LIBS_FILES TENSORSTORE_COUNT)
list(LENGTH TENSORSTORE_EXTERNAL_DEPS EXTERNAL_COUNT)
message(STATUS "Found ${TENSORSTORE_COUNT} TensorStore libraries")
message(STATUS "Found ${EXTERNAL_COUNT} external dependency libraries")
foreach(lib ${TENSORSTORE_ALL_LIBS_FILES})
    get_filename_component(lib_name ${lib} NAME)
    message(STATUS "  TensorStore lib: ${lib_name}")
endforeach()

# Add external dependencies from TensorStore build (re2, riegeli, absl, etc.)
set(TENSORSTORE_EXTERNAL_LIBS
    ${TENSORSTORE_EXTERNAL_DEPS}
)

# Ensure critical TensorStore main library is explicitly identified
set(TENSORSTORE_MAIN_LIB "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/libtensorstore.a")

# Define critical ZARR and compression libraries that must be force-loaded
set(TENSORSTORE_FORCE_LOAD_LIBS
    "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/driver/zarr/libblosc_compressor.a"
    "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/driver/zarr/libzstd_compressor.a"
    "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/internal/compression/libblosc.a"
    "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/internal/compression/libblosc_compressor.a"
    "${TENSORSTORE_ROOT}/bazel-bin/tensorstore/internal/compression/libzstd_compressor.a"
    "/opt/homebrew/opt/c-blosc/lib/libblosc.a"
)

# Additional dependencies that TensorStore might need
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    find_library(SECURITY_LIBRARY Security)
    set(TENSORSTORE_SYSTEM_LIBS ${COREFOUNDATION_LIBRARY} ${SECURITY_LIBRARY})
endif()

# Create complete TensorStore library list with system dependencies
set(TENSORSTORE_ALL_LIBS
    ${TENSORSTORE_ALL_LIBS_FILES}
    ${TENSORSTORE_EXTERNAL_LIBS}
    ${TENSORSTORE_SYSTEM_LIBS}
)


set(LINK_LIBS
        ${LINK_LIBS}
        carta-protobuf
        ${PROTOBUF_LIBRARY}
        uSockets
        uuid
        z
        zfp
        /opt/homebrew/lib/libzstd.dylib
        cfitsio
        /opt/homebrew/lib/libwcs.dylib
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
        ${TENSORSTORE_ALL_LIBS})

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
        # Include TensorStore zarr driver to ensure registration
        tensorstore/tensorstore/driver/zarr/driver.cc
        tensorstore/tensorstore/driver/zarr/metadata.cc
        tensorstore/tensorstore/driver/zarr/spec.cc
        tensorstore/tensorstore/driver/zarr/dtype.cc
        tensorstore/tensorstore/driver/zarr/compressor.cc
        tensorstore/tensorstore/driver/zarr/blosc_compressor.cc
        tensorstore/tensorstore/driver/zarr/zstd_compressor.cc
        tensorstore/tensorstore/driver/zarr/zlib_compressor.cc
        tensorstore/tensorstore/driver/zarr/bzip2_compressor.cc
        # Include internal compression implementations
        tensorstore/tensorstore/internal/compression/blosc_compressor.cc
        tensorstore/tensorstore/internal/compression/zstd_compressor.cc
        tensorstore/tensorstore/internal/compression/zlib_compressor.cc
        tensorstore/tensorstore/internal/compression/bzip2_compressor.cc
        tensorstore/tensorstore/internal/compression/json_specified_compressor.cc
        tensorstore/tensorstore/context.cc
        tensorstore/tensorstore/internal/concurrency_resource.cc
        tensorstore/tensorstore/internal/data_copy_concurrency_resource.cc
        tensorstore/tensorstore/internal/file_io_concurrency_resource.cc
        tensorstore/tensorstore/internal/cache/cache_pool_resource.cc
        # Include file KvStore driver for file:// URLs
        tensorstore/tensorstore/kvstore/file/file_key_value_store.cc
        tensorstore/tensorstore/kvstore/file/file_resource.cc
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
        src/Region/CrtfImportExport.cc
        src/Region/Ds9ImportExport.cc
        src/Region/LineBoxRegions.cc
        src/Region/Region.cc
        src/Region/RegionConverter.cc
        src/Region/RegionHandler.cc
        src/Region/RegionImportExport.cc
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
            ${LINK_LIBS}
            -Wl,--whole-archive
            ${TENSORSTORE_ROOT}/_deps/absl-build/absl/hash/libabsl_hash.a
            -Wl,--no-whole-archive)
            
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