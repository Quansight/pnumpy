#ifdef __cplusplus
extern "C" {
#endif

/*
 * Half-precision routines
 */
 /* half/float16 isn't a floating-point type in C */

typedef uint16_t atop_half;

///* Conversions */
//float atop_half_to_float(atop_half h);
//double atop_half_to_double(atop_half h);
//atop_half atop_float_to_half(float f);
//atop_half atop_double_to_half(double d);
///* Comparisons */
//int atop_half_eq(atop_half h1, atop_half h2);
//int atop_half_ne(atop_half h1, atop_half h2);
//int atop_half_le(atop_half h1, atop_half h2);
//int atop_half_lt(atop_half h1, atop_half h2);
//int atop_half_ge(atop_half h1, atop_half h2);
//int atop_half_gt(atop_half h1, atop_half h2);
///* faster *_nonan variants for when you know h1 and h2 are not NaN */
//int atop_half_eq_nonan(atop_half h1, atop_half h2);
//int atop_half_lt_nonan(atop_half h1, atop_half h2);
//int atop_half_le_nonan(atop_half h1, atop_half h2);
///* Miscellaneous functions */
//int atop_half_iszero(atop_half h);
//int atop_half_isnan(atop_half h);
//int atop_half_isinf(atop_half h);
//int atop_half_isfinite(atop_half h);
//int atop_half_signbit(atop_half h);
//atop_half atop_half_copysign(atop_half x, atop_half y);
//atop_half atop_half_spacing(atop_half h);
//atop_half atop_half_nextafter(atop_half x, atop_half y);
//atop_half atop_half_divmod(atop_half x, atop_half y, atop_half *modulus);
//
///*
// * Half-precision constants
// */
//
//#define ATOP_HALF_ZERO   (0x0000u)
//#define ATOP_HALF_PZERO  (0x0000u)
//#define ATOP_HALF_NZERO  (0x8000u)
//#define ATOP_HALF_ONE    (0x3c00u)
//#define ATOP_HALF_NEGONE (0xbc00u)
//#define ATOP_HALF_PINF   (0x7c00u)
//#define ATOP_HALF_NINF   (0xfc00u)
//#define ATOP_HALF_NAN    (0x7e00u)
//
//#define ATOP_MAX_HALF    (0x7bffu)
//
///*
// * Bit-level conversions
// */

//uint16_t atop_floatbits_to_halfbits(uint32_t f);
//uint16_t atop_doublebits_to_halfbits(uint64_t d);
//uint32_t atop_halfbits_to_floatbits(uint16_t h);
//uint64_t atop_halfbits_to_doublebits(uint16_t h);

#ifdef __cplusplus
}
#endif
