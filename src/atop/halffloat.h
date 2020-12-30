#ifdef __cplusplus
extern "C" {
#endif

/*
 * Half-precision routines
 */
 /* half/float16 isn't a floating-point type in C */

typedef uint16_t npy_half;

///* Conversions */
//float npy_half_to_float(npy_half h);
//double npy_half_to_double(npy_half h);
//npy_half npy_float_to_half(float f);
//npy_half npy_double_to_half(double d);
///* Comparisons */
//int npy_half_eq(npy_half h1, npy_half h2);
//int npy_half_ne(npy_half h1, npy_half h2);
//int npy_half_le(npy_half h1, npy_half h2);
//int npy_half_lt(npy_half h1, npy_half h2);
//int npy_half_ge(npy_half h1, npy_half h2);
//int npy_half_gt(npy_half h1, npy_half h2);
///* faster *_nonan variants for when you know h1 and h2 are not NaN */
//int npy_half_eq_nonan(npy_half h1, npy_half h2);
//int npy_half_lt_nonan(npy_half h1, npy_half h2);
//int npy_half_le_nonan(npy_half h1, npy_half h2);
///* Miscellaneous functions */
//int npy_half_iszero(npy_half h);
//int npy_half_isnan(npy_half h);
//int npy_half_isinf(npy_half h);
//int npy_half_isfinite(npy_half h);
//int npy_half_signbit(npy_half h);
//npy_half npy_half_copysign(npy_half x, npy_half y);
//npy_half npy_half_spacing(npy_half h);
//npy_half npy_half_nextafter(npy_half x, npy_half y);
//npy_half npy_half_divmod(npy_half x, npy_half y, npy_half *modulus);
//
///*
// * Half-precision constants
// */
//
//#define NPY_HALF_ZERO   (0x0000u)
//#define NPY_HALF_PZERO  (0x0000u)
//#define NPY_HALF_NZERO  (0x8000u)
//#define NPY_HALF_ONE    (0x3c00u)
//#define NPY_HALF_NEGONE (0xbc00u)
//#define NPY_HALF_PINF   (0x7c00u)
//#define NPY_HALF_NINF   (0xfc00u)
//#define NPY_HALF_NAN    (0x7e00u)
//
//#define NPY_MAX_HALF    (0x7bffu)
//
///*
// * Bit-level conversions
// */

//uint16_t npy_floatbits_to_halfbits(uint32_t f);
//uint16_t npy_doublebits_to_halfbits(uint64_t d);
//uint32_t npy_halfbits_to_floatbits(uint16_t h);
//uint64_t npy_halfbits_to_doublebits(uint16_t h);

#ifdef __cplusplus
}
#endif
