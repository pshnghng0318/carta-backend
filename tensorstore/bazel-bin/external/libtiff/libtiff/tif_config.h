#ifndef _TIFF_CONFIG_H_
#define _TIFF_CONFIG_H_
// libtiff/tif_config.h from bazel BUILD file

#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>

#define TIFF_INT8_T int8_t
#define TIFF_INT16_T int16_t
#define TIFF_INT32_T int32_t
#define TIFF_INT64_T int64_t

#define TIFF_UINT8_T uint8_t
#define TIFF_UINT16_T uint16_t
#define TIFF_UINT32_T uint32_t
#define TIFF_UINT64_T uint64_t

#define TIFF_SIZE_T size_t
#define TIFF_SSIZE_T ptrdiff_t
#define TIFF_PTRDIFF_T ptrdiff_t

#define HAVE_IEEEFP 1

/* Set the native cpu bit order (FILLORDER_LSB2MSB or FILLORDER_MSB2LSB) */
#define HOST_FILLORDER FILLORDER_LSB2MSB

#if defined __BIG_ENDIAN__
#  define HOST_BIGENDIAN 1
#else
#  define HOST_BIGENDIAN 0
#endif

#define CCITT_SUPPORT 1
#define JPEG_SUPPORT 1
#define LOGLUV_SUPPORT 1
#define LZW_SUPPORT 1
#define MDI_SUPPORT 1
#define NEXT_SUPPORT 1
#define OJPEG_SUPPORT 1
#define PACKBITS_SUPPORT 1
#define PIXARLOG_SUPPORT 1
#define SUBIFD_SUPPORT 1
#define THUNDER_SUPPORT 1
#define ZIP_SUPPORT 1
#define ZSTD_SUPPORT 1

#define DEFAULT_EXTRASAMPLE_AS_ALPHA 1
#define CHECK_JPEG_YCBCR_SUBSAMPLING 1

#define STRIPCHOP_DEFAULT TIFF_STRIPCHOP
#define STRIP_SIZE_DEFAULT 8192

/* obsolete macros */
#define COLORIMETRY_SUPPORT
#define YCBCR_SUPPORT
#define CMYK_SUPPORT
#define ICC_SUPPORT
#define PHOTOSHOP_SUPPORT
#define IPTC_SUPPORT


#if !defined(__native_client__) && (__WORDSIZE==64 || defined(_WIN64))
#  define SIZEOF_SIZE_T  8
#else
#  define SIZEOF_SIZE_T  4
#endif

#define TIFF_SIZE_FORMAT "z"
#define TIFF_SSIZE_FORMAT "t"

#if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
#endif

#if defined(_MSC_VER)

#define HAVE_IO_H 1
#define USE_WIN32_FILEIO 1

#elif defined(__APPLE__)

#define _DARWIN_USE_64_BIT_INODE 1

#define HAVE_DECL_OPTARG 1
#define HAVE_GETOPT 1
#define HAVE_MMAP 1
#define HAVE_SETMODE 1
#define HAVE_STRINGS_H 1
#define HAVE_UNISTD_H 1

#else

#define HAVE_DECL_OPTARG 1
#define HAVE_GETOPT 1
#define HAVE_MMAP 1
#define HAVE_STRINGS_H 1
#define HAVE_UNISTD_H 1

#endif

#define HAVE_SYS_TYPES_H 1
#define HAVE_FCNTL_H 1
#define CXX_SUPPORT 1
#define HAVE_ASSERT_H 1

#define PACKAGE "tiff"
#define PACKAGE_BUGREPORT "tiff@lists.maptools.org"
#define PACKAGE_NAME "LibTIFF Software"
#define PACKAGE_STRING "LibTIFF Software 4.6.0"
#define PACKAGE_TARNAME "tiff"
#define PACKAGE_URL ""
#define PACKAGE_VERSION "4.6.0"
#define VERSION "4.6.0"

#define X_DISPLAY_MISSING 1

#endif // _TIFF_CONFIG_H_