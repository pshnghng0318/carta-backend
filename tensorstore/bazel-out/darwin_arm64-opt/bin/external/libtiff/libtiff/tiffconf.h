#ifndef _TIFFCONF_
#define _TIFFCONF_
// libtiff/tiffconf.h from bazel BUILD file

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

#endif /* _TIFFCONF_ */