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

# nlohmann/json - use project's version, not system version
# MUST set include directory BEFORE system paths to prevent ABI mismatch
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third-party/nlohmann_json/include")
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(third-party/nlohmann_json)

# json-schema-validator - configure to use project's nlohmann_json
# Set version to match our bundled nlohmann_json (v3.11.3)
set(nlohmann_json_VERSION "v3.11.3" CACHE STRING "nlohmann_json version" FORCE)
# Prevent json-schema-validator from using system nlohmann_json by ensuring target exists
# The validator's CMakeLists.txt checks: if (NOT TARGET nlohmann_json)
# Since we already added nlohmann_json above, it will use our version
add_subdirectory(third-party/json-schema-validator)

# xtensor 0.26.0 (C++17 compatible) for optimized array operations
# Add AFTER json-schema-validator to avoid affecting its compilation
include_directories("${CMAKE_SOURCE_DIR}/third-party/xtensor/include")
if(APPLE)
    include_directories("/opt/homebrew/include")  # for xtl dependency on macOS
else()
    include_directories("/usr/local/include" "/usr/include")
endif()

# TensorStore configuration
set(TENSORSTORE_ROOT "${CMAKE_SOURCE_DIR}/third-party/tensorstore")
set(TENSORSTORE_BUILD_ROOT "${TENSORSTORE_ROOT}/build")
set(TENSORSTORE_DEPS_DIR "${TENSORSTORE_BUILD_ROOT}/_deps")
file(MAKE_DIRECTORY "${TENSORSTORE_BUILD_ROOT}")

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${TENSORSTORE_BUILD_ROOT}")

# Fetch missing TensorStore deps.
function(ts_fetch_if_missing name header_path url url_hash)
    if(NOT EXISTS "${header_path}")
        message(STATUS "Fetching ${name} for TensorStore")
        FetchContent_Declare(
            ${name}
            URL ${url}
            URL_HASH ${url_hash}
        )
        FetchContent_MakeAvailable(${name})
    endif()
endfunction()

ts_fetch_if_missing(
    riegeli
    "${TENSORSTORE_DEPS_DIR}/riegeli-src/riegeli/bytes/reader.h"
    "https://github.com/google/riegeli/archive/4214a733e7aad4654cc789e3b1ff4687f923323d.tar.gz"
    "SHA256=06c618ffa11be504b561a0103bdd469ceae6b3b6b844bba5d581bdffc7301f4f"
)
ts_fetch_if_missing(
    half
    "${TENSORSTORE_DEPS_DIR}/half-src/include/half.hpp"
    "https://sourceforge.net/projects/half/files/half/2.2.1/half-2.2.1.zip"
    "SHA256=76ddbf406e9d9b772ec73af2bf925b38b290b4390cc4064720a08d4b4bca0aa9"
)

FetchContent_Declare(
    re2
    URL https://github.com/google/re2/releases/download/2024-07-02/re2-2024-07-02.tar.gz
    URL_HASH SHA256=eb2df807c781601c14a260a507a5bb4509be1ee626024cb45acbd57cb9d4032b
)
message(STATUS "Fetching re2 for TensorStore")
FetchContent_MakeAvailable(re2)

# TensorStore include directories
set(TENSORSTORE_INCLUDE_DIRS
    "${TENSORSTORE_ROOT}"
    "${TENSORSTORE_ROOT}/tensorstore"
    "${TENSORSTORE_BUILD_ROOT}/_deps/re2-src"
    "${TENSORSTORE_BUILD_ROOT}/_deps/riegeli-src"
    "${TENSORSTORE_BUILD_ROOT}/_deps/absl-src"
    "${TENSORSTORE_BUILD_ROOT}/_deps/half-src/include"
    "${TENSORSTORE_BUILD_ROOT}/riegeli-src"
    "${TENSORSTORE_BUILD_ROOT}/half-src/include"
    "${TENSORSTORE_BUILD_ROOT}/re2-src"
)
list(REMOVE_DUPLICATES TENSORSTORE_INCLUDE_DIRS)
foreach(dir ${TENSORSTORE_INCLUDE_DIRS})
    include_directories("${dir}")
endforeach()

# TensorStore Bazel external dependencies (riegeli, etc.)
include_directories("${TENSORSTORE_ROOT}/bazel-tensorstore/external/riegeli")

# Include blosc headers for compression support
if(APPLE)
    include_directories("/opt/homebrew/opt/c-blosc/include")
else()
    include_directories("/usr/local/include" "/usr/include")
endif()

# Include TensorStore absl headers for protobuf compatibility
include_directories(BEFORE "${TENSORSTORE_ROOT}/_deps/absl-src")

# Build a minimal riegeli static library (TensorStore uses riegeli sources directly).
set(RIEGELI_ROOT "${TENSORSTORE_BUILD_ROOT}/riegeli-src")
if(EXISTS "${RIEGELI_ROOT}/riegeli")
    file(GLOB RIEGELI_BASE_SOURCES "${RIEGELI_ROOT}/riegeli/base/*.cc")
    file(GLOB RIEGELI_BYTES_SOURCES "${RIEGELI_ROOT}/riegeli/bytes/*.cc")
    # reader_cfile/cfile_writer use GNU extensions (off64_t, fopencookie) not on macOS.
    list(FILTER RIEGELI_BYTES_SOURCES EXCLUDE REGEX "cfile")
    file(GLOB RIEGELI_BZIP2_SOURCES "${RIEGELI_ROOT}/riegeli/bzip2/*.cc")
    file(GLOB RIEGELI_ENDIAN_SOURCES "${RIEGELI_ROOT}/riegeli/endian/*.cc")
    file(GLOB RIEGELI_VARINT_SOURCES "${RIEGELI_ROOT}/riegeli/varint/*.cc")
    file(GLOB RIEGELI_ZLIB_SOURCES "${RIEGELI_ROOT}/riegeli/zlib/*.cc")
    file(GLOB RIEGELI_ZSTD_SOURCES "${RIEGELI_ROOT}/riegeli/zstd/*.cc")

    add_library(riegeli STATIC
        ${RIEGELI_BASE_SOURCES}
        ${RIEGELI_BYTES_SOURCES}
        ${RIEGELI_BZIP2_SOURCES}
        ${RIEGELI_ENDIAN_SOURCES}
        ${RIEGELI_VARINT_SOURCES}
        ${RIEGELI_ZLIB_SOURCES}
        ${RIEGELI_ZSTD_SOURCES}
    )
    target_include_directories(riegeli PUBLIC "${RIEGELI_ROOT}")

    find_library(BZIP2_LIB bz2)
    if(BZIP2_LIB)
        target_link_libraries(riegeli PUBLIC ${BZIP2_LIB})
    endif()
    find_library(ZSTD_LIB zstd PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
    target_link_libraries(riegeli PUBLIC z ${ZSTD_LIB})
endif()

# Use system libraries instead of TensorStore static libraries
find_package(PkgConfig REQUIRED)

# Use system absl and other dependencies
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew;${CMAKE_PREFIX_PATH}")
endif()
find_library(ABSL_BASE_LIB absl_base PATHS /opt/homebrew/lib /usr/local/lib /usr/lib REQUIRED)
# Set library search paths based on platform
set(LIB_SEARCH_PATHS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)

find_library(ABSL_STRINGS_LIB absl_strings PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_STRINGS_INTERNAL_LIB absl_strings_internal PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_STATUS_LIB absl_status PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_STATUSOR_LIB absl_statusor PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_HASH_LIB absl_hash PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_LOG_INTERNAL_MESSAGE_LIB absl_log_internal_message PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_LOG_INTERNAL_CONDITIONS_LIB absl_log_internal_conditions PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_LOG_SEVERITY_LIB absl_log_severity PATHS ${LIB_SEARCH_PATHS})

# Additional critical absl libraries for TensorStore linking
find_library(ABSL_CORD_LIB absl_cord PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_CORD_INTERNAL_LIB absl_cord_internal PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_CORDZ_FUNCTIONS_LIB absl_cordz_functions PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_CORDZ_HANDLE_LIB absl_cordz_handle PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_CORDZ_INFO_LIB absl_cordz_info PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_CORDZ_SAMPLE_TOKEN_LIB absl_cordz_sample_token PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_CRC32C_LIB absl_crc32c PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_CRC_INTERNAL_LIB absl_crc_internal PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_CRC_CORD_STATE_LIB absl_crc_cord_state PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_CRC_CPU_DETECT_LIB absl_crc_cpu_detect PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_SYNCHRONIZATION_LIB absl_synchronization PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_TIME_LIB absl_time PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_CIVIL_TIME_LIB absl_civil_time PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_TIME_ZONE_LIB absl_time_zone PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_INT128_LIB absl_int128 PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RAW_LOGGING_INTERNAL_LIB absl_raw_logging_internal PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_LOG_INTERNAL_CHECK_OP_LIB absl_log_internal_check_op PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_LOG_INTERNAL_GLOBALS_LIB absl_log_internal_globals PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_THROW_DELEGATE_LIB absl_throw_delegate PATHS ${LIB_SEARCH_PATHS} REQUIRED)
find_library(ABSL_MALLOC_INTERNAL_LIB absl_malloc_internal PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_SPINLOCK_WAIT_LIB absl_spinlock_wait PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RAW_HASH_SET_LIB absl_raw_hash_set PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_HASHTABLEZ_SAMPLER_LIB absl_hashtablez_sampler PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_CITY_LIB absl_city PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_STR_FORMAT_INTERNAL_LIB absl_str_format_internal PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_DISTRIBUTIONS_LIB absl_random_distributions PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_INTERNAL_ENTROPY_POOL_LIB absl_random_internal_entropy_pool PATHS ${LIB_SEARCH_PATHS})  # Optional
find_library(ABSL_RANDOM_INTERNAL_PLATFORM_LIB absl_random_internal_platform PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_INTERNAL_RANDEN_LIB absl_random_internal_randen PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_INTERNAL_RANDEN_HWAES_LIB absl_random_internal_randen_hwaes PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_INTERNAL_RANDEN_HWAES_IMPL_LIB absl_random_internal_randen_hwaes_impl PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_INTERNAL_RANDEN_SLOW_LIB absl_random_internal_randen_slow PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_INTERNAL_SEED_MATERIAL_LIB absl_random_internal_seed_material PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_SEED_GEN_EXCEPTION_LIB absl_random_seed_gen_exception PATHS ${LIB_SEARCH_PATHS})
find_library(ABSL_RANDOM_SEED_SEQUENCES_LIB absl_random_seed_sequences PATHS ${LIB_SEARCH_PATHS})

set(ABSL_LIBS
    ${ABSL_BASE_LIB}
    ${ABSL_STRINGS_LIB}
    ${ABSL_STRINGS_INTERNAL_LIB}
    ${ABSL_STATUS_LIB}
    ${ABSL_STATUSOR_LIB}
    ${ABSL_HASH_LIB}
    ${ABSL_LOG_INTERNAL_MESSAGE_LIB}
    ${ABSL_LOG_INTERNAL_CONDITIONS_LIB}
    ${ABSL_LOG_SEVERITY_LIB}
    ${ABSL_CORD_LIB}
    ${ABSL_CORD_INTERNAL_LIB}
    ${ABSL_CORDZ_FUNCTIONS_LIB}
    ${ABSL_CORDZ_HANDLE_LIB}
    ${ABSL_CORDZ_INFO_LIB}
    ${ABSL_CORDZ_SAMPLE_TOKEN_LIB}
    ${ABSL_CRC32C_LIB}
    ${ABSL_CRC_INTERNAL_LIB}
    ${ABSL_CRC_CORD_STATE_LIB}
    ${ABSL_CRC_CPU_DETECT_LIB}
    ${ABSL_SYNCHRONIZATION_LIB}
    ${ABSL_TIME_LIB}
    ${ABSL_CIVIL_TIME_LIB}
    ${ABSL_TIME_ZONE_LIB}
    ${ABSL_INT128_LIB}
    ${ABSL_RAW_LOGGING_INTERNAL_LIB}
    ${ABSL_LOG_INTERNAL_CHECK_OP_LIB}
    ${ABSL_LOG_INTERNAL_GLOBALS_LIB}
    ${ABSL_THROW_DELEGATE_LIB}
    ${ABSL_MALLOC_INTERNAL_LIB}
    ${ABSL_SPINLOCK_WAIT_LIB}
    ${ABSL_RAW_HASH_SET_LIB}
    ${ABSL_HASHTABLEZ_SAMPLER_LIB}
    ${ABSL_CITY_LIB}
    ${ABSL_STR_FORMAT_INTERNAL_LIB}
    ${ABSL_RANDOM_DISTRIBUTIONS_LIB}
    $<$<BOOL:${ABSL_RANDOM_INTERNAL_ENTROPY_POOL_LIB}>:${ABSL_RANDOM_INTERNAL_ENTROPY_POOL_LIB}>
    ${ABSL_RANDOM_INTERNAL_PLATFORM_LIB}
    ${ABSL_RANDOM_INTERNAL_RANDEN_LIB}
    ${ABSL_RANDOM_INTERNAL_RANDEN_HWAES_LIB}
    ${ABSL_RANDOM_INTERNAL_RANDEN_HWAES_IMPL_LIB}
    ${ABSL_RANDOM_INTERNAL_RANDEN_SLOW_LIB}
    ${ABSL_RANDOM_INTERNAL_SEED_MATERIAL_LIB}
    ${ABSL_RANDOM_SEED_GEN_EXCEPTION_LIB}
    ${ABSL_RANDOM_SEED_SEQUENCES_LIB}
)

# TensorStore link directories - use bazel-bin path if present
set(TENSORSTORE_BAZEL_BIN_DIR "${TENSORSTORE_ROOT}/bazel-bin/tensorstore")
set(TENSORSTORE_BAZEL_EXTERNAL_DIR "${TENSORSTORE_ROOT}/bazel-bin/external")

if(EXISTS "${TENSORSTORE_BAZEL_BIN_DIR}")
    link_directories("${TENSORSTORE_BAZEL_BIN_DIR}")
    # Automatically collect all TensorStore library files
    file(GLOB_RECURSE TENSORSTORE_ALL_LIBS_FILES "${TENSORSTORE_BAZEL_BIN_DIR}/*.a")
else()
    set(TENSORSTORE_ALL_LIBS_FILES "")
endif()

if(EXISTS "${TENSORSTORE_BAZEL_EXTERNAL_DIR}")
    # Collect external dependencies from TensorStore Bazel build
    file(GLOB_RECURSE TENSORSTORE_EXTERNAL_DEPS "${TENSORSTORE_BAZEL_EXTERNAL_DIR}/*.a")
else()
    set(TENSORSTORE_EXTERNAL_DEPS "")
endif()

# Sort the libraries to ensure consistent linking order
list(SORT TENSORSTORE_ALL_LIBS_FILES)
list(SORT TENSORSTORE_EXTERNAL_DEPS)

# Display found libraries for debugging (only when Bazel outputs are present)
if(EXISTS "${TENSORSTORE_BAZEL_BIN_DIR}" OR EXISTS "${TENSORSTORE_BAZEL_EXTERNAL_DIR}")
    list(LENGTH TENSORSTORE_ALL_LIBS_FILES TENSORSTORE_COUNT)
    list(LENGTH TENSORSTORE_EXTERNAL_DEPS EXTERNAL_COUNT)
    message(STATUS "TensorStore Bazel libraries: ${TENSORSTORE_COUNT}, external deps: ${EXTERNAL_COUNT}")
    foreach(lib ${TENSORSTORE_ALL_LIBS_FILES})
        get_filename_component(lib_name ${lib} NAME)
        message(STATUS "  TensorStore lib: ${lib_name}")
    endforeach()
endif()

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
)
# Find blosc library
find_library(BLOSC_LIB blosc PATHS /opt/homebrew/opt/c-blosc/lib /usr/local/lib /usr/lib)
if(BLOSC_LIB)
    list(APPEND TENSORSTORE_STATIC_LIBS ${BLOSC_LIB})
endif()

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

# TensorStore core sources (exclude tests) for non-Bazel builds
file(GLOB TENSORSTORE_CORE_SOURCES "${TENSORSTORE_ROOT}/tensorstore/*.cc")
list(FILTER TENSORSTORE_CORE_SOURCES EXCLUDE REGEX "test")
list(REMOVE_ITEM TENSORSTORE_CORE_SOURCES "${TENSORSTORE_ROOT}/tensorstore/context.cc")

file(GLOB TENSORSTORE_INDEX_SPACE_SOURCES "${TENSORSTORE_ROOT}/tensorstore/index_space/*.cc")
list(FILTER TENSORSTORE_INDEX_SPACE_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INDEX_SPACE_INTERNAL_SOURCES "${TENSORSTORE_ROOT}/tensorstore/index_space/internal/*.cc")
list(FILTER TENSORSTORE_INDEX_SPACE_INTERNAL_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INTERNAL_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/*.cc")
list(FILTER TENSORSTORE_INTERNAL_SOURCES EXCLUDE REGEX "test")
list(REMOVE_ITEM TENSORSTORE_INTERNAL_SOURCES
    "${TENSORSTORE_ROOT}/tensorstore/internal/concurrency_resource.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/data_copy_concurrency_resource.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/file_io_concurrency_resource.cc"
)

file(GLOB TENSORSTORE_INTERNAL_CONTAINER_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/container/*.cc")
list(FILTER TENSORSTORE_INTERNAL_CONTAINER_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INTERNAL_LOG_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/log/*.cc")
list(FILTER TENSORSTORE_INTERNAL_LOG_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INTERNAL_RIEGELI_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/riegeli/*.cc")
list(FILTER TENSORSTORE_INTERNAL_RIEGELI_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INTERNAL_THREAD_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/thread/*.cc")
list(FILTER TENSORSTORE_INTERNAL_THREAD_SOURCES EXCLUDE REGEX "test|benchmark")

file(GLOB TENSORSTORE_INTERNAL_TRACING_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/tracing/*.cc")
list(FILTER TENSORSTORE_INTERNAL_TRACING_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INTERNAL_CACHE_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/cache/*.cc")
list(FILTER TENSORSTORE_INTERNAL_CACHE_SOURCES EXCLUDE REGEX "test")
list(REMOVE_ITEM TENSORSTORE_INTERNAL_CACHE_SOURCES
    "${TENSORSTORE_ROOT}/tensorstore/internal/cache/cache_pool_resource.cc"
)

file(GLOB TENSORSTORE_INTERNAL_JSON_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/json/*.cc")
list(FILTER TENSORSTORE_INTERNAL_JSON_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_INTERNAL_JSON_BINDING_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/json_binding/*.cc")
list(FILTER TENSORSTORE_INTERNAL_JSON_BINDING_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_KVSTORE_SOURCES "${TENSORSTORE_ROOT}/tensorstore/kvstore/*.cc")
list(FILTER TENSORSTORE_KVSTORE_SOURCES EXCLUDE REGEX "test")

file(GLOB TENSORSTORE_UTIL_SOURCES "${TENSORSTORE_ROOT}/tensorstore/util/*.cc")
list(FILTER TENSORSTORE_UTIL_SOURCES EXCLUDE REGEX "test")

set(TENSORSTORE_INTERNAL_OS_SOURCES
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/aligned_alloc.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/error_code_posix.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/file_descriptor.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/file_info_posix.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/file_lock.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/file_lister_posix.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/file_util.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/file_util_posix.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/filesystem.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/fork_detection.cc"
    "${TENSORSTORE_ROOT}/tensorstore/internal/os/memory_region.cc"
)

file(GLOB TENSORSTORE_INTERNAL_METRICS_SOURCES "${TENSORSTORE_ROOT}/tensorstore/internal/metrics/*.cc")
# Exclude tests/benchmarks, protobuf-backed metrics, and Prometheus push (HTTP deps).
list(FILTER TENSORSTORE_INTERNAL_METRICS_SOURCES EXCLUDE REGEX "test|benchmark|protobuf|prometheus")

file(GLOB TENSORSTORE_SERIALIZATION_SOURCES "${TENSORSTORE_ROOT}/tensorstore/serialization/*.cc")
# Exclude tests and protobuf serialization (pulls in riegeli/protobuf glue).
list(FILTER TENSORSTORE_SERIALIZATION_SOURCES EXCLUDE REGEX "test|protobuf")

set(TENSORSTORE_DRIVER_SOURCES
    "${TENSORSTORE_ROOT}/tensorstore/driver/driver.cc"
    "${TENSORSTORE_ROOT}/tensorstore/driver/read.cc"
    "${TENSORSTORE_ROOT}/tensorstore/driver/write.cc"
    "${TENSORSTORE_ROOT}/tensorstore/driver/copy.cc"
    "${TENSORSTORE_ROOT}/tensorstore/driver/url_registry.cc"
)

set(TENSORSTORE_EXTRA_SOURCES
    ${TENSORSTORE_CORE_SOURCES}
    ${TENSORSTORE_INDEX_SPACE_SOURCES}
    ${TENSORSTORE_INDEX_SPACE_INTERNAL_SOURCES}
    ${TENSORSTORE_INTERNAL_SOURCES}
    ${TENSORSTORE_INTERNAL_CONTAINER_SOURCES}
    ${TENSORSTORE_INTERNAL_LOG_SOURCES}
    ${TENSORSTORE_INTERNAL_CACHE_SOURCES}
    ${TENSORSTORE_INTERNAL_JSON_SOURCES}
    ${TENSORSTORE_INTERNAL_JSON_BINDING_SOURCES}
    ${TENSORSTORE_INTERNAL_RIEGELI_SOURCES}
    ${TENSORSTORE_INTERNAL_OS_SOURCES}
    ${TENSORSTORE_INTERNAL_METRICS_SOURCES}
    ${TENSORSTORE_INTERNAL_THREAD_SOURCES}
    ${TENSORSTORE_INTERNAL_TRACING_SOURCES}
    ${TENSORSTORE_KVSTORE_SOURCES}
    ${TENSORSTORE_SERIALIZATION_SOURCES}
    ${TENSORSTORE_UTIL_SOURCES}
    ${TENSORSTORE_DRIVER_SOURCES}
    "${TENSORSTORE_ROOT}/tensorstore/kvstore/file/util.cc"
)


set(LINK_LIBS
        ${LINK_LIBS}
        carta-protobuf
        ${PROTOBUF_LIBRARY}
        uSockets
        uuid
        zfp
        ${ZSTD_LIB}
        cfitsio
        wcs
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

if(TARGET riegeli)
    set(LINK_LIBS ${LINK_LIBS} riegeli)
endif()

if(TARGET re2)
    set(LINK_LIBS ${LINK_LIBS} re2)
elseif(TARGET re2::re2)
    set(LINK_LIBS ${LINK_LIBS} re2::re2)
endif()

# Ensure absl libraries come after riegeli/static objects for link order.
set(LINK_LIBS ${LINK_LIBS} ${ABSL_LIBS})

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
        # Include TensorStore zarr driver to ensure registration
        third-party/tensorstore/tensorstore/driver/driver_spec.cc
        third-party/tensorstore/tensorstore/driver/kvs_backed_chunk_driver.cc
        third-party/tensorstore/tensorstore/driver/zarr/driver.cc
        third-party/tensorstore/tensorstore/driver/zarr/metadata.cc
        third-party/tensorstore/tensorstore/driver/zarr/spec.cc
        third-party/tensorstore/tensorstore/driver/zarr/dtype.cc
        third-party/tensorstore/tensorstore/driver/zarr/compressor.cc
        third-party/tensorstore/tensorstore/driver/zarr/blosc_compressor.cc
        third-party/tensorstore/tensorstore/driver/zarr/zstd_compressor.cc
        third-party/tensorstore/tensorstore/driver/zarr/zlib_compressor.cc
        third-party/tensorstore/tensorstore/driver/zarr/bzip2_compressor.cc
        # Include internal compression implementations
        third-party/tensorstore/tensorstore/internal/compression/blosc_compressor.cc
        third-party/tensorstore/tensorstore/internal/compression/blosc.cc
        third-party/tensorstore/tensorstore/internal/compression/zstd_compressor.cc
        third-party/tensorstore/tensorstore/internal/compression/zlib_compressor.cc
        third-party/tensorstore/tensorstore/internal/compression/bzip2_compressor.cc
        third-party/tensorstore/tensorstore/internal/compression/json_specified_compressor.cc
        third-party/tensorstore/tensorstore/context.cc
        third-party/tensorstore/tensorstore/internal/concurrency_resource.cc
        third-party/tensorstore/tensorstore/internal/data_copy_concurrency_resource.cc
        third-party/tensorstore/tensorstore/internal/file_io_concurrency_resource.cc
        third-party/tensorstore/tensorstore/internal/cache/cache_pool_resource.cc
        # Include file KvStore driver for file:// URLs
        third-party/tensorstore/tensorstore/kvstore/file/file_key_value_store.cc
        third-party/tensorstore/tensorstore/kvstore/file/file_resource.cc
        ${TENSORSTORE_EXTRA_SOURCES}
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

    # Force-load TensorStore main lib only if it exists (Bazel build optional)
    set(TENSORSTORE_MAIN_FORCE_LOAD "")
    if(EXISTS ${TENSORSTORE_MAIN_LIB})
        set(TENSORSTORE_MAIN_FORCE_LOAD "-Wl,-force_load,${TENSORSTORE_MAIN_LIB}")
    else()
        # No Bazel output: force-load not applicable.
    endif()
    
    target_link_libraries(carta_backend
        PRIVATE
            protobuf
            /opt/homebrew/lib/libuv.dylib
            ${LINK_LIBS}
            ${TENSORSTORE_MAIN_FORCE_LOAD}
            ${FORCE_LOAD_ARGS})
            
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    
    # Prepare force-load arguments for compression libraries
    set(FORCE_LOAD_ARGS "")
    foreach(lib ${TENSORSTORE_FORCE_LOAD_LIBS})
        if(EXISTS ${lib})
            list(APPEND FORCE_LOAD_ARGS "-Wl,--whole-archive" "${lib}" "-Wl,--no-whole-archive")
        endif()
    endforeach()

    # Force-load TensorStore main lib only if it exists (Bazel build optional)
    set(TENSORSTORE_MAIN_FORCE_LOAD "")
    if(EXISTS ${TENSORSTORE_MAIN_LIB})
        list(APPEND TENSORSTORE_MAIN_FORCE_LOAD "-Wl,--whole-archive" "${TENSORSTORE_MAIN_LIB}" "-Wl,--no-whole-archive")
    else()
        # No Bazel output: force-load not applicable.
    endif()

    target_link_libraries(carta_backend
        PRIVATE
            protobuf
            uv
            ${LINK_LIBS}
            ${TENSORSTORE_MAIN_FORCE_LOAD}
            ${FORCE_LOAD_ARGS})
            
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
