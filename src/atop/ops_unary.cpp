#include "common_inc.h"
#include <cmath>
#include "invalids.h"

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang attribute push (__attribute__((target("avx2"))), apply_to=function)
#endif

#if defined(__GNUC__)
//#pragma GCC target "arch=core-avx2,tune=core-avx2"
#if __GNUC_PREREQ(4, 4) || (__clang__ > 0 && __clang_major__ >= 3) || !defined(__GNUC__)
/* GCC >= 4.4 or clang or non-GCC compilers */
#include <x86intrin.h>
#elif __GNUC_PREREQ(4, 1)
/* GCC 4.1, 4.2, and 4.3 do not have x86intrin.h, directly include SSE2 header */
#include <emmintrin.h>
#endif
#endif


//#define LOGGING printf
#define LOGGING(...)

static const inline __m256d LOADU(__m256d* x) { return _mm256_loadu_pd((double const*)x); };
static const inline __m256 LOADU(__m256* x) { return _mm256_loadu_ps((float const*)x); };
static const inline __m256i LOADU(__m256i* x) { return _mm256_loadu_si256((__m256i const*)x); };

static const inline void STOREU(__m256d* x, __m256d y) { _mm256_storeu_pd((double*)x, y); }
static const inline void STOREU(__m256* x, __m256 y) { _mm256_storeu_ps((float*)x, y); }
static const inline void STOREU(__m256i* x, __m256i y) { _mm256_storeu_si256((__m256i*)x, y); }

// For aligned loads which must be on 32 byte boundary
static const inline __m256d LOADA(__m256d* x) { return _mm256_load_pd((double const*)x); };
static const inline __m256 LOADA(__m256* x) { return _mm256_load_ps((float const*)x); };
static const inline __m256i LOADA(__m256i* x) { return _mm256_load_si256((__m256i const*)x); };

// Aligned stores
static const inline void STOREA(__m256d* x, __m256d y) { _mm256_store_pd((double*)x, y); }
static const inline void STOREA(__m256* x, __m256 y) { _mm256_store_ps((float*)x, y); }
static const inline void STOREA(__m256i* x, __m256i y) { _mm256_store_si256((__m256i*)x, y); }

// Some missing instrinsics
#define _mm_roundme_ps(val)       _mm256_round_ps((val), _MM_FROUND_NINT)
#define _mm_roundme_pd(val)       _mm256_round_pd((val), _MM_FROUND_NINT)
#define _mm_truncme_ps(val)       _mm256_round_ps((val), _MM_FROUND_TRUNC)
#define _mm_truncme_pd(val)       _mm256_round_pd((val), _MM_FROUND_TRUNC)

// This shuffle is for int32/float32.  It will move byte positions 0, 4, 8, and 12 together into one 32 bit dword
static const __m256i g_shuffle1 = _mm256_set_epi8((char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, 12, 8, 4, 0,
(char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, 12, 8, 4, 0);

// This is the second shuffle for int32/float32.  It will move byte positions 0, 4, 8, and 12 together into one 32 bit dword
static const __m256i g_shuffle2 = _mm256_set_epi8((char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, 12, 8, 4, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
(char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, 12, 8, 4, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80);

static const __m256i g_shuffle3 = _mm256_set_epi8((char)0x80, (char)0x80, (char)0x80, (char)0x80, 12, 8, 4, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
(char)0x80, (char)0x80, (char)0x80, (char)0x80, 12, 8, 4, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80);

static const __m256i g_shuffle4 = _mm256_set_epi8(12, 8, 4, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    12, 8, 4, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80);

// interleave hi lo across 128 bit lanes
static const __m256i g_permute = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
static const __m256i g_ones = _mm256_set1_epi8(1);
static const __m128i g_ones128 = _mm_set1_epi8(1);

//// Examples of how to store a constant in vector math
//MEM_ALIGN(64)
//__m256  __ones_constant32f = _mm256_set1_ps(1.0f);
//__m256d __ones_constant64f = _mm256_set1_pd(1.0);
//__m256i __ones_constant64i = _mm256_set1_epi64x(1);

// This bit mask will remove the sign bit from an IEEE floating point and is how ABS values are done
MEM_ALIGN(64)
const union
{
    int32_t i[8];
    float f[8];
    __m256 m;
} __f32vec8_abs_mask = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };


MEM_ALIGN(64)
const union
{
    int64_t  i[4];
    double d[4];
    __m256d m;
} __f64vec4_abs_mask = { 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff };


MEM_ALIGN(64)
const union
{
    int32_t i[8];
    float f[8];
    __m256 m;
    // all 1 bits in exponent must be 1 (8 bits after sign)
    // and fraction must not be 0
} __f32vec8_finite_compare = { 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000 };


MEM_ALIGN(64)
const union
{
    int32_t i[8];
    float f[8];
    __m256 m;
    // all 1 bits in exponent must be 1 (8 bits after sign)
    // and fraction must not be 0
} __f32vec8_finite_mask = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

MEM_ALIGN(64)
const union
{
    int32_t i[8];
    float f[8];
    __m256 m;
    // all 1 bits in exponent must be 1 (8 bits after sign)
    // and fraction must not be 0
} __f32vec8_inf_mask = { 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff };


MEM_ALIGN(64)
const union
{
    int64_t  i[4];
    double d[4];
    __m256d m;
    // all 1 bits in exponent must be 1 (11 bits after sign)
    // and fraction must not be 0
} __f64vec4_finite_mask = { 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff };

MEM_ALIGN(64)
const union
{
    int64_t  i[4];
    double d[4];
    __m256d m;
    // all 1 bits in exponent must be 1 (11 bits after sign)
    // and fraction must not be 0
} __f64vec4_finite_compare = { 0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000 };

MEM_ALIGN(64)
const union
{
    int64_t  i[4];
    double d[4];
    __m256d m;
    // all 1 bits in exponent must be 1 (11 bits after sign)
    // and fraction must not be 0
} __f64vec4_inf_mask = { 0x000fffffffffffff, 0x000fffffffffffff, 0x000fffffffffffff, 0x000fffffffffffff };

// This is used to multiply the strides
MEM_ALIGN(64)
const union
{
    int32_t i[8];
    __m256i m;
} __vec8_strides = { 0, 1, 2, 3, 4, 5, 6, 7 };

// This is used to multiply the strides
MEM_ALIGN(64)
const union
{
    int64_t i[8];
    __m256i m;
} __vec4_strides = { 0, 1, 2, 3 };

//// IEEE Mask
//// NOTE: Check NAN mask -- if not then return number, else return 0.0 or +INF or -INF
//// For IEEE 754, MSB is the sign bit, then next section is the exponent.  If the exponent is all 1111s, it is some kind of NAN
//#define NAN_TO_NUM_F32(x) ((((*(uint32_t*)&x)  & 0x7f800000) != 0x7f800000) ?  x :  (((*(uint32_t*)&x)  & 0x007fffff) != 0) ?  0.0f : (((*(uint32_t*)&x)  & 0x80000000) == 0) ?  FLT_MAX : -FLT_MAX)
//#define NAN_TO_NUM_F64(x) ((((*(uint64_t*)&x)  & 0x7ff0000000000000) != 0x7ff0000000000000) ?  x :  (((*(uint64_t*)&x)  & 0x000fffffffffffff) != 0) ? 0.0 : (((*(uint64_t*)&x)  & 0x8000000000000000) == 0) ?  DBL_MAX : -DBL_MAX )
//
//#define NAN_TO_ZERO_F32(x) ((((*(uint32_t*)&x)  & 0x7f800000) != 0x7f800000) ?  x :   0.0f )
//#define NAN_TO_ZERO_F64(x) ((((*(uint64_t*)&x)  & 0x7ff0000000000000) != 0x7ff0000000000000) ?  x : 0.0 )
//


template<typename T> static const inline T ABS_OP(T x) { return x < 0 ? -x : x; }
template<typename T> static const inline double FABS_OP(T x) { return x < 0 ? -x : x; }

// Invalid int mode (consider if we should return invalid when we discover invalid)
template<typename T> static const inline T SIGN_OP(T x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

// If we find a nan, we return the same type of nan.  For instance -nan will return -nan instead of nan.
template<typename T> static const inline T FLOATSIGN_OP(T x) { return x > (T)(0.0) ? (T)(1.0) : (x < (T)(0.0) ? (T)(-1.0) : (x == x ? (T)(0.0) : x)); }

template<typename T> static const inline T NEG_OP(T x) { return -x; }
template<typename T> static const inline T BITWISE_NOT_OP(T x) { return ~x; }
template<typename T> static const inline T INVERT_OP(T x) { return ~x; }
template<typename T> static const inline T INVERT_OP_BOOL(int8_t x) { return x ^ 1; }

template<typename T> static const inline bool NOT_OP(T x) { return (bool)(x == (T)0); }

template<typename T> static const inline bool ISNOTNAN_OP(T x) { return !std::isnan(x); }
template<typename T> static const inline bool ISNAN_OP(T x) { return std::isnan(x); }
template<typename T> static const inline bool ISFINITE_OP(T x) { return std::isfinite(x); }
template<typename T> static const inline bool ISNOTFINITE_OP(T x) { return !std::isfinite(x); }
template<typename T> static const inline bool ISINF_OP(T x) { return std::isinf(x); }
template<typename T> static const inline bool ISNOTINF_OP(T x) { return !std::isinf(x); }
template<typename T> static const inline bool ISNORMAL_OP(T x) { return std::isnormal(x); }
template<typename T> static const inline bool ISNOTNORMAL_OP(T x) { return !std::isnormal(x); }
template<typename T> static const inline bool ISNANORZERO_OP(T x) { return x == 0.0 || std::isnan(x); }

template<typename T> static const inline long double ROUND_OP(long double x) { return roundl(x); }
template<typename T> static const inline double ROUND_OP(double x) { return round(x); }
template<typename T> static const inline float ROUND_OP(float x) { return roundf(x); }

template<typename T> static const inline long double FLOOR_OP(long double x) { return floorl(x); }
template<typename T> static const inline double FLOOR_OP(double x) { return floor(x); }
template<typename T> static const inline float FLOOR_OP(float x) { return floorf(x); }

template<typename T> static const inline long double TRUNC_OP(long double x) { return truncl(x); }
template<typename T> static const inline double TRUNC_OP(double x) { return trunc(x); }
template<typename T> static const inline float TRUNC_OP(float x) { return truncf(x); }

template<typename T> static const inline long double CEIL_OP(long double x) { return ceill(x); }
template<typename T> static const inline double CEIL_OP(double x) { return ceil(x); }
template<typename T> static const inline float CEIL_OP(float x) { return ceilf(x); }

template<typename T> static const inline long double SQRT_OP(long double x) { return sqrtl(x); }
template<typename T> static const inline double SQRT_OP(double x) { return sqrt(x); }
template<typename T> static const inline float SQRT_OP(float x) { return sqrtf(x); }

template<typename T> static const inline long double LOG_OP(long double x) { return logl(x); }
template<typename T> static const inline double LOG_OP(double x) { return log(x); }
template<typename T> static const inline float LOG_OP(float x) { return logf(x); }

template<typename T> static const inline long double LOG2_OP(long double x) { return log2l(x); }
template<typename T> static const inline double LOG2_OP(double x) { return log2(x); }
template<typename T> static const inline float LOG2_OP(float x) { return log2f(x); }

template<typename T> static const inline long double LOG10_OP(long double x) { return log10l(x); }
template<typename T> static const inline double LOG10_OP(double x) { return log10(x); }
template<typename T> static const inline float LOG10_OP(float x) { return log10f(x); }

template<typename T> static const inline long double EXP_OP(long double x) { return expl(x); }
template<typename T> static const inline double EXP_OP(double x) { return exp(x); }
template<typename T> static const inline float EXP_OP(float x) { return expf(x); }

template<typename T> static const inline long double EXP2_OP(long double x) { return exp2l(x); }
template<typename T> static const inline double EXP2_OP(double x) { return exp2(x); }
template<typename T> static const inline float EXP2_OP(float x) { return exp2f(x); }

template<typename T> static const inline long double CBRT_OP(long double x) { return cbrtl(x); }
template<typename T> static const inline double CBRT_OP(double x) { return cbrt(x); }
template<typename T> static const inline float CBRT_OP(float x) { return cbrtf(x); }

template<typename T> static const inline long double TAN_OP(long double x) { return tanl(x); }
template<typename T> static const inline double TAN_OP(double x) { return tan(x); }
template<typename T> static const inline float TAN_OP(float x) { return tanf(x); }

template<typename T> static const inline long double COS_OP(long double x) { return cosl(x); }
template<typename T> static const inline double COS_OP(double x) { return cos(x); }
template<typename T> static const inline float COS_OP(float x) { return cosf(x); }

template<typename T> static const inline long double SIN_OP(long double x) { return sinl(x); }
template<typename T> static const inline double SIN_OP(double x) { return sin(x); }
template<typename T> static const inline float SIN_OP(float x) { return sinf(x); }

// NOTE: These routines can be vectorized
template<typename T> static const inline bool SIGNBIT_OP(long double x) { return std::signbit(x); }
template<typename T> static const inline bool SIGNBIT_OP(double x) { return std::signbit(x); }
template<typename T> static const inline bool SIGNBIT_OP(float x) { return std::signbit(x); }

//static const inline __m256  ABS_OP_256(const float z, __m256 x) { return _mm256_and_ps(x, __f32vec8_abs_mask.m); }
//static const inline __m256d ABS_OP_256(const double z, __m256d x) { return _mm256_and_pd(x, __f64vec4_abs_mask.m); }
static const inline __m256i ABS_OP_256i32(__m256i x) { return _mm256_abs_epi32(x); }
static const inline __m256i ABS_OP_256i16(__m256i x) { return _mm256_abs_epi16(x); }
static const inline __m256i ABS_OP_256i8(__m256i x) { return _mm256_abs_epi8(x); }

template<typename T> static const inline __m256  ABS_OP_256(__m256 x) {
    const __m256 m8 = __f32vec8_abs_mask.m;
    // The second operand does not need to be unaligned loaded
    return _mm256_and_ps(m8, x);
}

template<typename T> static const inline __m256d  ABS_OP_256(__m256d x) {
    const __m256d m8 = __f64vec4_abs_mask.m;
    // The second operand does not need to be unaligned loaded
    return _mm256_and_pd(m8, x);
}

static const inline __m256i NEG_OP_256i64(__m256i x) { const __m256i m8 = _mm256_setzero_si256(); return _mm256_sub_epi64(m8, x); }
static const inline __m256i NEG_OP_256i32(__m256i x) { const __m256i m8 = _mm256_setzero_si256(); return _mm256_sub_epi32(m8, x); }
static const inline __m256i NEG_OP_256i16(__m256i x) { const __m256i m8 = _mm256_setzero_si256(); return _mm256_sub_epi16(m8, x); }
static const inline __m256i NEG_OP_256i8(__m256i x) { const __m256i m8 = _mm256_setzero_si256(); return _mm256_sub_epi8(m8, x); }

template<typename T> static const inline __m256  NEG_OP_256(__m256 x) {
    const __m256 m8 = _mm256_setzero_ps();
    // The second operand does not need to be unaligned loaded
    return _mm256_sub_ps(m8, x);
}

template<typename T> static const inline __m256d  NEG_OP_256(__m256d x) {
    const __m256d m8 = _mm256_setzero_pd();
    // The second operand does not need to be unaligned loaded
    return _mm256_sub_pd(m8, x);
}

template<typename T> static const inline __m256  ROUND_OP_256(__m256 x) { return _mm_roundme_ps(x); }
template<typename T> static const inline __m256d ROUND_OP_256(__m256d x) { return _mm_roundme_pd(x); }

template<typename T> static const inline __m256  TRUNC_OP_256(__m256 x) { return _mm_truncme_ps(x); }
template<typename T> static const inline __m256d TRUNC_OP_256(__m256d x) { return _mm_truncme_pd(x); }

template<typename T> static const inline __m256  FLOOR_OP_256(__m256 x) { return _mm256_floor_ps(x); }
template<typename T> static const inline __m256d FLOOR_OP_256(__m256d x) { return _mm256_floor_pd(x); }

template<typename T> static const inline __m256  CEIL_OP_256(__m256 x) { return _mm256_ceil_ps(x); }
template<typename T> static const inline __m256d CEIL_OP_256(__m256d x) { return _mm256_ceil_pd(x); }

template<typename T> static const inline __m256  SQRT_OP_256(__m256 x) { return _mm256_sqrt_ps(x); }
template<typename T> static const inline __m256d SQRT_OP_256(__m256d x) { return _mm256_sqrt_pd(x); }

//_mm256_xor_si256

//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, const T MATH_OP(T), const U256 MATH_OP256(U256)>
static inline void UnaryOpFast(void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    T* pOut = (T*)pDataOut;
    T* pLastOut = (T*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);
    LOGGING("unary op fast strides %lld %lld   sizeof: %lld\n", strideIn, strideOut, sizeof(T));
    if (sizeof(T) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {

        // possible to align 32 bit boundary?
        //if (((int64_t)pDataOut & 31) != 0) {
        //   int64_t babylen = (32 - ((int64_t)pDataOut & 31)) / sizeof(T);
        //   if (babylen > len) babylen = len;
        //   for (int64_t i = 0; i < babylen; i++) {
        //      *pOut++ = MATH_OP(*pIn++);
        //   }
        //   len -= babylen;
        //}

        T* pEnd = &pOut[chunkSize * (len / chunkSize)];
        U256* pEnd_256 = (U256*)pEnd;

        U256* pIn1_256 = (U256*)pIn;
        U256* pOut_256 = (U256*)pOut;

        // possible to align?
        do {
            // Use 256 bit registers which hold 8 floats or 4 doubles
            // The first operand should allow unaligned loads
            STOREU(pOut_256, MATH_OP256(*pIn1_256));
            pIn1_256 += 1;
            pOut_256 += 1;

        } while (pOut_256 < pEnd_256);

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (T*)pOut_256;

    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(T, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, const T MATH_OP(T), const U256 MATH_OP256(U256)>
static inline void UnaryOpFastStrided(void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    T* pOut = (T*)pDataOut;
    T* pLastOut = (T*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = (sizeof(U256)) / sizeof(T);

    // TOOD: ensure stride*len < 2billion
    // assumes output not strided
    if (sizeof(T) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        const int64_t innerloopLen = chunkSize * (len / chunkSize);
        T* pEnd = &pOut[innerloopLen];
        U256* pEnd_256 = (U256*)pEnd;

        U256* pIn1_256 = (U256*)pIn;
        U256* pOut_256 = (U256*)pOut;

        int32_t babyStride = (int32_t)strideIn;
        // add 8 strides everytime we process 8
        __m256i mindex = _mm256_mullo_epi32(_mm256_set1_epi32(babyStride), __vec8_strides.m);

        // possible to align?
        do {
            // Use 256 bit registers which hold 8 floats or 4 doubles
            // The first operand should allow unaligned loads         
            const __m256 gather = _mm256_i32gather_ps((float const*)pIn, mindex, 1);
            STOREU(pOut_256, MATH_OP256(gather));

            // Advance in and out pointers
            pIn = (T*)((char*)pIn + (strideIn * 8));
            pOut_256 += 1;

        } while (pOut_256 < pEnd_256);

        // update thin pointers to last location of wide pointers
        pOut = (T*)pOut_256;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(T, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}



//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryNanFastFloat(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    // Loops unrolled 4 times, 4 * 8 = 32
    int64_t chunkSize = 32; // sizeof(U256) / sizeof(T);

    if (sizeof(bool) == strideOut && sizeof(T) == strideIn) {
        const __m256* pSrc1Fast = (const __m256*)pDataIn;
        __m256i* pDestFast = (__m256i*)pDataOut;
        int8_t* pEnd = (int8_t*)pDataOut + len;

        __m256i* pDestFastEnd = &pDestFast[len / 32];
        while (pDestFast != pDestFastEnd) {
            // the shuffle will move all 8 comparisons together
            __m256 m0 = _mm256_loadu_ps((const float*)(pSrc1Fast + 0));
            __m256i m10 = _mm256_castps_si256(_mm256_cmp_ps(m0, m0, _CMP_NEQ_UQ));
            __m256 m1 = _mm256_loadu_ps((const float*)(pSrc1Fast + 1));
            __m256i m11 = _mm256_castps_si256(_mm256_cmp_ps(m1, m1, _CMP_NEQ_UQ));
            __m256 m2 = _mm256_loadu_ps((const float*)(pSrc1Fast + 2));
            __m256i m12 = _mm256_castps_si256(_mm256_cmp_ps(m2, m2, _CMP_NEQ_UQ));
            __m256 m3 = _mm256_loadu_ps((const float*)(pSrc1Fast + 3));
            __m256i m13 = _mm256_castps_si256(_mm256_cmp_ps(m3, m3, _CMP_NEQ_UQ));
            m10 = _mm256_packs_epi32(m10, m11);
            m12 = _mm256_packs_epi32(m12, m13);
            m10 = _mm256_packs_epi16(m10, m12);
            STOREU(pDestFast, _mm256_and_si256(_mm256_permutevar8x32_epi32(m10, g_permute), g_ones));
            pSrc1Fast += 4;
            pDestFast++;
        }

        float* pDataInX = (float*)pSrc1Fast;
        int8_t* pDataOutX = (int8_t*)pDestFast;
        while (pDataOutX < pEnd) {
            *pDataOutX++ = *pDataInX != *pDataInX;
            pDataInX++;
        }
        return;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryNanFastDouble(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);
    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {

        const __m128d* pSrc1Fast = (const __m128d*)pDataIn;
        INT64* pDestFast = (INT64*)pDataOut;
        int8_t* pEnd = (int8_t*)pDataOut + len;

        INT64* pDestFastEnd = &pDestFast[len / 8];
        while (pDestFast != pDestFastEnd) {
            // the shuffle will move all 8 comparisons together
            __m128d m0 = _mm_loadu_pd((const double*)(pSrc1Fast + 0));
            __m128i m10 = _mm_castpd_si128(_mm_cmp_pd(m0, m0, _CMP_NEQ_UQ));
            __m128d m1 = _mm_loadu_pd((const double*)(pSrc1Fast + 1));
            __m128i m11 = _mm_castpd_si128(_mm_cmp_pd(m1, m1, _CMP_NEQ_UQ));
            m10 = _mm_packs_epi32(m10, m11);

            __m128d m2 = _mm_loadu_pd((const double*)(pSrc1Fast + 2));
            __m128i m12 = _mm_castpd_si128(_mm_cmp_pd(m2, m2, _CMP_NEQ_UQ));
            __m128d m3 = _mm_loadu_pd((const double*)(pSrc1Fast + 3));
            __m128i m13 = _mm_castpd_si128(_mm_cmp_pd(m3, m3, _CMP_NEQ_UQ));
            m12 = _mm_packs_epi32(m12, m13);

            m10 = _mm_packs_epi16(m10, m12);
            m10 = _mm_packs_epi16(m10, m10);

            // Write 8 booleans
            _mm_storel_epi64((__m128i*)pDestFast, _mm_and_si128(m10, g_ones128));

            pSrc1Fast += 4;
            pDestFast++;
        }

        double* pDataInX = (double*)pSrc1Fast;
        int8_t* pDataOutX = (int8_t*)pDestFast;
        while (pDataOutX < pEnd) {
            *pDataOutX++ = *pDataInX != *pDataInX;
            pDataInX++;
        }

        return;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryNotNanFastFloat(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);

    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        bool* pEnd = &pOut[chunkSize * (len / chunkSize)];
        int64_t* pEnd_i64 = (int64_t*)pEnd;

        U256* pIn1_256 = (U256*)pDataIn;
        int64_t* pOut_i64 = (int64_t*)pDataOut;

        while (pOut_i64 < pEnd_i64) {
            U256 m0 = LOADU(pIn1_256);
            pIn1_256++;

            // nans will NOT equal eachother
            // +/-inf will equal eachother
            *pOut_i64++ = gBooleanLUT64[_mm256_movemask_ps(_mm256_cmp_ps(m0, m0, _CMP_EQ_OQ)) & 255];
        }

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (bool*)pOut_i64;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryNotNanFastDouble(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);

    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        bool* pEnd = &pOut[chunkSize * (len / chunkSize)];
        int32_t* pEnd_i32 = (int32_t*)pEnd;

        U256* pIn1_256 = (U256*)pDataIn;
        int32_t* pOut_i32 = (int32_t*)pDataOut;

        while (pOut_i32 < pEnd_i32) {
            U256 m0 = LOADU(pIn1_256);
            pIn1_256++;

            *pOut_i32++ = gBooleanLUT32[_mm256_movemask_pd(_mm256_cmp_pd(m0, m0, _CMP_EQ_OQ)) & 15];
        }

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (bool*)pOut_i32;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}



//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryFiniteFastFloat(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);

    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        bool* pEnd = &pOut[chunkSize * (len / chunkSize)];
        int64_t* pEnd_i64 = (int64_t*)pEnd;

        U256* pIn1_256 = (U256*)pDataIn;
        int64_t* pOut_i64 = (int64_t*)pDataOut;

        // finite_comp 0x7f800000
        const __m256 m_finitecomp = _mm256_load_ps(__f32vec8_finite_compare.f);

        while (pOut_i64 < pEnd_i64) {
            U256 m0 = LOADU(pIn1_256);
            pIn1_256++;

            int32_t bitmask = _mm256_movemask_ps(
                _mm256_cmp_ps(m_finitecomp,
                    _mm256_and_ps(m0, m_finitecomp), _CMP_NEQ_OQ));

            *pOut_i64++ = gBooleanLUT64[bitmask & 255];
        }

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (bool*)pOut_i64;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryFiniteFastDouble(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);
    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        bool* pEnd = &pOut[chunkSize * (len / chunkSize)];
        int32_t* pEnd_i32 = (int32_t*)pEnd;

        U256* pIn1_256 = (U256*)pDataIn;
        int32_t* pOut_i32 = (int32_t*)pDataOut;

        const __m256d m_finitecomp = _mm256_load_pd(__f64vec4_finite_compare.d);

        while (pOut_i32 < pEnd_i32) {
            U256 m0 = LOADU(pIn1_256);
            pIn1_256++;

            int32_t bitmask = _mm256_movemask_pd(
                _mm256_cmp_pd(m_finitecomp,
                    _mm256_and_pd(m0, m_finitecomp), _CMP_NEQ_OQ));

            *pOut_i32++ = gBooleanLUT32[bitmask & 15];
        }

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (bool*)pOut_i32;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryNotFiniteFastFloat(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);

    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        bool* pEnd = &pOut[chunkSize * (len / chunkSize)];
        int64_t* pEnd_i64 = (int64_t*)pEnd;

        U256* pIn1_256 = (U256*)pDataIn;
        int64_t* pOut_i64 = (int64_t*)pDataOut;

        // finite_comp 0x7f800000
        const __m256 m_finitecomp = _mm256_load_ps(__f32vec8_finite_compare.f);

        while (pOut_i64 < pEnd_i64) {
            U256 m0 = LOADU(pIn1_256);
            pIn1_256++;

            int32_t bitmask = _mm256_movemask_ps(
                _mm256_cmp_ps(m_finitecomp,
                    _mm256_and_ps(m0, m_finitecomp), _CMP_EQ_OQ));

            *pOut_i64++ = gBooleanLUT64[bitmask & 255];
        }

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (bool*)pOut_i64;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}


//-------------------------------------------------------------------
// T = data type as input
// MathOp operation to perform
template<typename T, typename U256, typename MathFunctionPtr>
static inline void UnaryNotFiniteFastDouble(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    int64_t chunkSize = sizeof(U256) / sizeof(T);

    if (sizeof(bool) == strideOut && sizeof(T) == strideIn && len >= chunkSize) {
        bool* pEnd = &pOut[chunkSize * (len / chunkSize)];
        int32_t* pEnd_i32 = (int32_t*)pEnd;

        U256* pIn1_256 = (U256*)pDataIn;
        int32_t* pOut_i32 = (int32_t*)pDataOut;

        const __m256d m_finitecomp = _mm256_load_pd(__f64vec4_finite_compare.d);

        while (pOut_i32 < pEnd_i32) {
            U256 m0 = LOADU(pIn1_256);
            pIn1_256++;

            // After masking for all exponents == 1 check to see if same value
            int32_t bitmask = _mm256_movemask_pd(
                _mm256_cmp_pd(m_finitecomp,
                    _mm256_and_pd(m0, m_finitecomp), _CMP_EQ_OQ));

            *pOut_i32++ = gBooleanLUT32[bitmask & 15];
        }

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (bool*)pOut_i32;
    }

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

//-------------------------------------------------------------------
// T = data type as input
// Output same as input type
// MathOp operation to perform
template<typename T, typename MathFunctionPtr>
static void UnaryOpSlow(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    T* pOut = (T*)pDataOut;
    T* pLastOut = (T*)((char*)pOut + (strideOut * len));

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(T, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

//-------------------------------------------------------------------
// T = data type as input
// Output always returns a double
// MathOp operation to perform
template<typename T, typename MathFunctionPtr>
static void UnaryOpSlowDouble(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    double* pOut = (double*)pDataOut;
    double* pLastOut = (double*)((char*)pOut + (strideOut * len));

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP((double)*pIn);
        pOut = STRIDE_NEXT(double, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

//-------------------------------------------------------------------
// T = data type as input
// Output always returns a double
// MathOp operation to perform
template<typename T, typename MathFunctionPtr>
static void UnaryOpSlowBool(MathFunctionPtr MATH_OP, void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {

    T* pIn = (T*)pDataIn;
    bool* pOut = (bool*)pDataOut;
    bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        *pOut = MATH_OP(*pIn);
        pOut = STRIDE_NEXT(bool, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

static void UnaryOpSlow_FillTrue(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    if (strideOut == sizeof(bool)) {
        memset(pDataOut, 1, len);
    }
    else {
        bool* pOut = (bool*)pDataOut;
        bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));
        while (pOut != pLastOut) {
            *pOut = 1;
            pOut = STRIDE_NEXT(bool, pOut, strideOut);
        }
    }
}

static void UnaryOpSlow_FillFalse(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    if (strideOut == sizeof(bool)) {
        memset(pDataOut, 0, len);
    }
    else {
        bool* pOut = (bool*)pDataOut;
        bool* pLastOut = (bool*)((char*)pOut + (strideOut * len));
        while (pOut != pLastOut) {
            *pOut = 0;
            pOut = STRIDE_NEXT(bool, pOut, strideOut);
        }
    }
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_ABS(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(ABS_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_FABS(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const double(*)(T)>(FABS_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_SIGN(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(SIGN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_FLOATSIGN(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(FLOATSIGN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_NEG(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(NEG_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}


//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_ISNOTNAN(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNOTNAN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
template<typename T>
static void UnaryOpSlow_ISNAN(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNAN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISNANORZERO(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNANORZERO_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISINVALID(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    T* pIn = (T*)pDataIn1;
    T invalid = GetInvalid((T)0);
    int8_t* pOut = (int8_t*)pDataOut;
    int8_t* pLastOut = (int8_t*)((char*)pOut + (strideOut * len));

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        if (*pIn == invalid) {
            *pOut = TRUE;
        }
        else {
            *pOut = FALSE;
        }
        pOut = STRIDE_NEXT(int8_t, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

template<typename T>
static void UnaryOpSlow_ISINVALIDORZERO(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    T* pIn = (T*)pDataIn1;
    T invalid = GetInvalid((T)0);
    int8_t* pOut = (int8_t*)pDataOut;
    int8_t* pLastOut = (int8_t*)((char*)pOut + (strideOut * len));

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        if (*pIn == invalid || *pIn == 0) {
            *pOut = TRUE;
        }
        else {
            *pOut = FALSE;
        }
        pOut = STRIDE_NEXT(int8_t, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

template<typename T>
static void UnaryOpSlow_ISNOTINVALID(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    T* pIn = (T*)pDataIn1;
    T invalid = GetInvalid((T)0);
    int8_t* pOut = (int8_t*)pDataOut;
    int8_t* pLastOut = (int8_t*)((char*)pOut + (strideOut * len));

    // Slow loop, handle 1 at a time
    while (pOut != pLastOut) {
        if (*pIn == invalid) {
            *pOut = FALSE;
        }
        else {
            *pOut = TRUE;
        }
        pOut = STRIDE_NEXT(int8_t, pOut, strideOut);
        pIn = STRIDE_NEXT(T, pIn, strideIn);
    }
}

template<typename T>
static void UnaryOpSlow_ISFINITE(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISFINITE_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISNOTFINITE(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNOTFINITE_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISINF(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISINF_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISNOTINF(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNOTINF_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISNORMAL(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNORMAL_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_ISNOTNORMAL(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(ISNOTNORMAL_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T>
static void UnaryOpSlow_SIGNBIT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(SIGNBIT_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}


//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_NOT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowBool<T, const bool(*)(T)>(NOT_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_INVERT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(INVERT_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_INVERT_BOOL(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(INVERT_OP_BOOL<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_FLOOR(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(FLOOR_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_CEIL(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(CEIL_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_TRUNC(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(TRUNC_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_ROUND(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(ROUND_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_SQRT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(SQRT_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_SQRT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(SQRT_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}


// NEG
//__m128 val = /* some value */;
//__m128 minusval = _mm_xor_ps(val, SIGNMASK); // minusval = -val


//------------------------------------------------------------------------------------
// This routines will call the proper templated ABS function based on the type
// It provides the standard signature
template<typename T, typename U256> static inline void UnaryOpFast_NANF32(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryNanFastFloat<T, U256, const bool(*)(T)>(ISNAN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
template<typename T, typename U256> static inline void UnaryOpFast_NANF64(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryNanFastDouble<T, U256, const bool(*)(T)>(ISNAN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T, typename U256> static inline void UnaryOpFast_NOTNANF32(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryNotNanFastFloat<T, U256, const bool(*)(T)>(ISNOTNAN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
template<typename T, typename U256> static inline void UnaryOpFast_NOTNANF64(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryNotNanFastDouble<T, U256, const bool(*)(T)>(ISNOTNAN_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T, typename U256> static inline void UnaryOpFast_FINITEF32(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryFiniteFastFloat<T, U256, const bool(*)(T)>(ISFINITE_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
template<typename T, typename U256> static inline void UnaryOpFast_FINITEF64(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryFiniteFastDouble<T, U256, const bool(*)(T)>(ISFINITE_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

template<typename T, typename U256> static inline void UnaryOpFast_NOTFINITEF32(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryNotFiniteFastFloat<T, U256, const bool(*)(T)>(ISNOTFINITE_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
template<typename T, typename U256> static inline void UnaryOpFast_NOTFINITEF64(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryNotFiniteFastDouble<T, U256, const bool(*)(T)>(ISNOTFINITE_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

UNARY_FUNC GetUnaryOpSlow(int func, int numpyInType1, int numpyOutType, int* wantedOutType) {
    LOGGING("Looking for func slow %d  type %d  outtype: %d\n", func, numpyInType1, numpyOutType);

    switch (func) {
    case UNARY_OPERATION::FABS:
        //*wantedOutType = ATOP_DOUBLE;
        //switch (numpyInType1) {
        //case ATOP_INT32:  return UnaryOpSlow_FABS<int32_t>;
        //case ATOP_INT64  return UnaryOpSlow_FABS<int64_t>;
        //case ATOP_INT8   return UnaryOpSlow_FABS<int8_t>;
        //case ATOP_INT16  return UnaryOpSlow_FABS<int16_t>;

        //}
        break;

    case UNARY_OPERATION::ABS:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_FLOAT:  return UnaryOpSlow_ABS<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_ABS<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_ABS<long double>;
        case ATOP_INT32:  return UnaryOpSlow_ABS<int32_t>;
        case ATOP_INT64:  return UnaryOpSlow_ABS<int64_t>;
        case ATOP_INT8:   return UnaryOpSlow_ABS<int8_t>;
        case ATOP_INT16:  return UnaryOpSlow_ABS<int16_t>;

        }
        break;

    case UNARY_OPERATION::SIGN:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_FLOAT:  return UnaryOpSlow_FLOATSIGN<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_FLOATSIGN<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_FLOATSIGN<long double>;
        case ATOP_INT32:  return UnaryOpSlow_SIGN<int32_t>;
        case ATOP_INT64:  return UnaryOpSlow_SIGN<int64_t>;
        case ATOP_INT8:   return UnaryOpSlow_SIGN<int8_t>;
        case ATOP_INT16:  return UnaryOpSlow_SIGN<int16_t>;

        }
        break;

    case UNARY_OPERATION::NEGATIVE:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_FLOAT:  return UnaryOpSlow_NEG<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_NEG<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_NEG<long double>;
        case ATOP_INT32:  return UnaryOpSlow_NEG<int32_t>;
        case ATOP_INT64:  return UnaryOpSlow_NEG<int64_t>;
        case ATOP_INT8:   return UnaryOpSlow_NEG<int8_t>;
        case ATOP_INT16:  return UnaryOpSlow_NEG<int16_t>;

        }
        break;

    case UNARY_OPERATION::LOGICAL_NOT:
        *wantedOutType = ATOP_BOOL;
        // TJD: Not sure why bool works and others do not
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_UINT8:
            case ATOP_INT8:
            case ATOP_BOOL:   return UnaryOpSlow_NOT<int8_t>;
            case ATOP_INT16:
            case ATOP_UINT16:   return UnaryOpSlow_NOT<int16_t>;
            case ATOP_INT32:
            case ATOP_UINT32:   return UnaryOpSlow_NOT<int32_t>;
            case ATOP_INT64:
            case ATOP_UINT64:   return UnaryOpSlow_NOT<int64_t>;
            case ATOP_FLOAT:  return UnaryOpSlow_NOT<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_NOT<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_NOT<long double>;
            }
        }
        break;

    case UNARY_OPERATION::BITWISE_NOT:
    case UNARY_OPERATION::INVERT:
        // bitwise on floats not allowed
        if (numpyInType1 <= ATOP_UINT64) {
            *wantedOutType = numpyInType1;
            switch (numpyInType1) {
            case ATOP_INT32:   return UnaryOpSlow_INVERT<int32_t>;
            case ATOP_UINT32:  return UnaryOpSlow_INVERT<uint32_t>;
            case ATOP_INT64:   return UnaryOpSlow_INVERT<int64_t>;
            case ATOP_UINT64:  return UnaryOpSlow_INVERT<uint64_t>;
            case ATOP_BOOL:    return UnaryOpSlow_INVERT_BOOL<int8_t>;
            case ATOP_INT8:    return UnaryOpSlow_INVERT<int8_t>;
            case ATOP_UINT8:   return UnaryOpSlow_INVERT<uint8_t>;
            case ATOP_INT16:   return UnaryOpSlow_INVERT<int16_t>;
            case ATOP_UINT16:  return UnaryOpSlow_INVERT<uint16_t>;
            }
        }
        break;

    case UNARY_OPERATION::ISFINITE:
        *wantedOutType = ATOP_BOOL;
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISFINITE<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISFINITE<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISFINITE<long double>;
            case ATOP_INT32:      return UnaryOpSlow_ISNOTINVALID<int32_t>;
            case ATOP_UINT32:     return UnaryOpSlow_ISNOTINVALID<uint32_t>;
            case ATOP_INT64:      return UnaryOpSlow_ISNOTINVALID<int64_t>;
            case ATOP_UINT64:     return UnaryOpSlow_ISNOTINVALID<uint64_t>;
            case ATOP_BOOL:
            case ATOP_INT8:       return UnaryOpSlow_ISNOTINVALID<int8_t>;
            case ATOP_UINT8:      return UnaryOpSlow_ISNOTINVALID<uint8_t>;
            case ATOP_INT16:      return UnaryOpSlow_ISNOTINVALID<int16_t>;
            case ATOP_UINT16:     return UnaryOpSlow_ISNOTINVALID<uint16_t>;
            default: return UnaryOpSlow_FillTrue;
            }
        }
        break;

    case UNARY_OPERATION::ISNOTFINITE:
        *wantedOutType = ATOP_BOOL;

        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNOTFINITE<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNOTFINITE<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNOTFINITE<long double>;
            case ATOP_INT32:      return UnaryOpSlow_ISINVALID<int32_t>;
            case ATOP_UINT32:     return UnaryOpSlow_ISINVALID<uint32_t>;
            case ATOP_INT64:      return UnaryOpSlow_ISINVALID<int64_t>;
            case ATOP_UINT64:     return UnaryOpSlow_ISINVALID<uint64_t>;
            case ATOP_BOOL:
            case ATOP_INT8:       return UnaryOpSlow_ISINVALID<int8_t>;
            case ATOP_UINT8:      return UnaryOpSlow_ISINVALID<uint8_t>;
            case ATOP_INT16:      return UnaryOpSlow_ISINVALID<int16_t>;
            case ATOP_UINT16:     return UnaryOpSlow_ISINVALID<uint16_t>;
            default: return UnaryOpSlow_FillTrue;
            }
        }
        break;

    case UNARY_OPERATION::ISNAN:
        *wantedOutType = ATOP_BOOL;

        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNAN<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNAN<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNAN<long double>;
            case ATOP_INT32:      return UnaryOpSlow_ISINVALID<int32_t>;
            case ATOP_UINT32:     return UnaryOpSlow_ISINVALID<uint32_t>;
            case ATOP_INT64:      return UnaryOpSlow_ISINVALID<int64_t>;
            case ATOP_UINT64:     return UnaryOpSlow_ISINVALID<uint64_t>;
            case ATOP_BOOL:
            case ATOP_INT8:       return UnaryOpSlow_ISINVALID<int8_t>;
            case ATOP_UINT8:      return UnaryOpSlow_ISINVALID<uint8_t>;
            case ATOP_INT16:      return UnaryOpSlow_ISINVALID<int16_t>;
            case ATOP_UINT16:     return UnaryOpSlow_ISINVALID<uint16_t>;

            default: return UnaryOpSlow_FillFalse;
            }
        }
        break;

    case UNARY_OPERATION::ISNANORZERO:
        *wantedOutType = ATOP_BOOL;

        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNANORZERO<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNANORZERO<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNANORZERO<long double>;
            case ATOP_INT32:      return UnaryOpSlow_ISINVALIDORZERO<int32_t>;
            case ATOP_UINT32:     return UnaryOpSlow_ISINVALIDORZERO<uint32_t>;
            case ATOP_INT64:      return UnaryOpSlow_ISINVALIDORZERO<int64_t>;
            case ATOP_UINT64:     return UnaryOpSlow_ISINVALIDORZERO<uint64_t>;
            case ATOP_BOOL:
            case ATOP_INT8:       return UnaryOpSlow_ISINVALIDORZERO<int8_t>;
            case ATOP_UINT8:      return UnaryOpSlow_ISINVALIDORZERO<uint8_t>;
            case ATOP_INT16:      return UnaryOpSlow_ISINVALIDORZERO<int16_t>;
            case ATOP_UINT16:     return UnaryOpSlow_ISINVALIDORZERO<uint16_t>;

            default: return UnaryOpSlow_FillFalse;
            }
        }
        break;

    case UNARY_OPERATION::ISNOTNAN:
        *wantedOutType = ATOP_BOOL;
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNOTNAN<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNOTNAN<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNOTNAN<long double>;
            case ATOP_INT32:      return UnaryOpSlow_ISNOTINVALID<int32_t>;
            case ATOP_UINT32:     return UnaryOpSlow_ISNOTINVALID<uint32_t>;
            case ATOP_INT64:      return UnaryOpSlow_ISNOTINVALID<int64_t>;
            case ATOP_UINT64:     return UnaryOpSlow_ISNOTINVALID<uint64_t>;
            case ATOP_BOOL:
            case ATOP_INT8:       return UnaryOpSlow_ISNOTINVALID<int8_t>;
            case ATOP_UINT8:      return UnaryOpSlow_ISNOTINVALID<uint8_t>;
            case ATOP_INT16:      return UnaryOpSlow_ISNOTINVALID<int16_t>;
            case ATOP_UINT16:     return UnaryOpSlow_ISNOTINVALID<uint16_t>;
            default: return UnaryOpSlow_FillTrue;
            }
        }
        break;
    case UNARY_OPERATION::ISINF:
        *wantedOutType = ATOP_BOOL;
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISINF<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISINF<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISINF<long double>;
            default: return UnaryOpSlow_FillFalse;
            }
        }
        break;
    case UNARY_OPERATION::ISNOTINF:
        *wantedOutType = ATOP_BOOL;
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNOTINF<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNOTINF<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNOTINF<long double>;
            default: return UnaryOpSlow_FillTrue;
            }
        }
        break;
    case UNARY_OPERATION::ISNORMAL:
        *wantedOutType = ATOP_BOOL;
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNORMAL<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNORMAL<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNORMAL<long double>;
            default: return UnaryOpSlow_FillTrue;
            }
        }
        break;
    case UNARY_OPERATION::ISNOTNORMAL:
        *wantedOutType = ATOP_BOOL;
        // Can only handle when output type is bool or not defined
        if (numpyOutType == 0 || numpyOutType == -1) {
            switch (numpyInType1) {
            case ATOP_FLOAT:  return UnaryOpSlow_ISNOTNORMAL<float>;
            case ATOP_DOUBLE: return UnaryOpSlow_ISNOTNORMAL<double>;
            case ATOP_LONGDOUBLE: return UnaryOpSlow_ISNOTNORMAL<long double>;
            default: return UnaryOpSlow_FillFalse;
            }
        }
        break;

    case UNARY_OPERATION::FLOOR:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_DOUBLE: return UnaryOpSlow_FLOOR<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_FLOOR<long double>;
        }
        break;
    case UNARY_OPERATION::CEIL:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_DOUBLE: return UnaryOpSlow_CEIL<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_CEIL<long double>;
        }
        break;
    case UNARY_OPERATION::TRUNC:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_DOUBLE: return UnaryOpSlow_TRUNC<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_TRUNC<long double>;
        }
        break;
    case UNARY_OPERATION::ROUND:
        *wantedOutType = numpyInType1;
        switch (numpyInType1) {
        case ATOP_DOUBLE: return UnaryOpSlow_ROUND<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_ROUND<long double>;
        }
        break;
    case UNARY_OPERATION::SQRT:
        *wantedOutType = ATOP_DOUBLE;
        if (numpyInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        if (numpyInType1 == ATOP_LONGDOUBLE) {
            *wantedOutType = ATOP_LONGDOUBLE;
        }
        switch (numpyInType1) {
        case ATOP_INT32:   return UnaryOpSlowDouble_SQRT<int32_t>;
        case ATOP_UINT32:  return UnaryOpSlowDouble_SQRT<uint32_t>;
        case ATOP_INT64:   return UnaryOpSlowDouble_SQRT<int64_t>;
        case ATOP_UINT64:  return UnaryOpSlowDouble_SQRT<int64_t>;
        case ATOP_DOUBLE:  return UnaryOpSlow_SQRT<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_SQRT<long double>;
        }
        break;

    case UNARY_OPERATION::SIGNBIT:
        *wantedOutType = ATOP_BOOL;
        switch (numpyInType1) {
        case ATOP_FLOAT: return UnaryOpSlow_SIGNBIT<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_SIGNBIT<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_SIGNBIT<long double>;
            //case ATOP_INT32:  return UnaryOpSlowDouble_SIGNBIT<int32_t>;
        }
        break;

    }

    return NULL;
}

extern "C"
UNARY_FUNC GetUnaryOpFast(int func, int atopInType1, int* wantedOutType) {

    LOGGING("Looking for func %d  type:%d \n", func, atopInType1);

    switch (func) {
    case UNARY_OPERATION::FABS:
        break;

    case UNARY_OPERATION::ABS:
        *wantedOutType = atopInType1;
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ABS_OP<float>, ABS_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ABS_OP<double>, ABS_OP_256<__m256d>>;
        case ATOP_INT32:  return UnaryOpFast<int32_t, __m256i, ABS_OP<int32_t>, ABS_OP_256i32>;
        case ATOP_INT16:  return UnaryOpFast<int16_t, __m256i, ABS_OP<int16_t>, ABS_OP_256i16>;
        case ATOP_INT8:   return UnaryOpFast<int8_t, __m256i, ABS_OP<int8_t>, ABS_OP_256i8>;
        }
        break;
    case UNARY_OPERATION::ISNAN:
        *wantedOutType = ATOP_BOOL;
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast_NANF32<float, __m256>;
        case ATOP_DOUBLE: return UnaryOpFast_NANF64<double, __m256d>;
        }
        break;

    case UNARY_OPERATION::ISNOTNAN:
        *wantedOutType = ATOP_BOOL;
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast_NOTNANF32<float, __m256>;
        case ATOP_DOUBLE: return UnaryOpFast_NOTNANF64<double, __m256d>;
        }
        break;

    case UNARY_OPERATION::ISFINITE:
        *wantedOutType = ATOP_BOOL;
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast_FINITEF32<float, __m256>;
        case ATOP_DOUBLE: return UnaryOpFast_FINITEF64<double, __m256d>;
        }
        break;

    case UNARY_OPERATION::ISNOTFINITE:
        *wantedOutType = ATOP_BOOL;
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast_NOTFINITEF32<float, __m256>;
        case ATOP_DOUBLE: return UnaryOpFast_NOTFINITEF64<double, __m256d>;
        }
        break;

    case UNARY_OPERATION::NEGATIVE:
        // neg on bool not allowed
        if (atopInType1 > ATOP_BOOL) {
            *wantedOutType = atopInType1;
            switch (atopInType1) {
            case ATOP_FLOAT:  return UnaryOpFast<float, __m256, NEG_OP<float>, NEG_OP_256<__m256>>;
            case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, NEG_OP<double>, NEG_OP_256<__m256d>>;
            case ATOP_INT64:  return UnaryOpFast<int64_t, __m256i, NEG_OP<int64_t>, NEG_OP_256i64>;
            case ATOP_INT32:  return UnaryOpFast<int32_t, __m256i, NEG_OP<int32_t>, NEG_OP_256i32>;
            case ATOP_INT16:  return UnaryOpFast<int16_t, __m256i, NEG_OP<int16_t>, NEG_OP_256i16>;
            case ATOP_INT8:   return UnaryOpFast<int8_t, __m256i, NEG_OP<int8_t>, NEG_OP_256i8>;
            }
        }
        break;
    case UNARY_OPERATION::INVERT:
    case UNARY_OPERATION::BITWISE_NOT:
        // bitwise on floats not allowed
        if (atopInType1 <= ATOP_UINT64) {
            *wantedOutType = atopInType1;
        }
        break;
    case UNARY_OPERATION::FLOOR:
        // Floor on bool not allowed
        if (atopInType1 >= ATOP_FLOAT) {
            *wantedOutType = atopInType1;
            switch (atopInType1) {
            case ATOP_FLOAT:  return UnaryOpFast<float, __m256, FLOOR_OP<float>, FLOOR_OP_256<__m256>>;
            case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, FLOOR_OP<double>, FLOOR_OP_256<__m256d>>;
            }
        }
        break;
    case UNARY_OPERATION::CEIL:
        // ceil on bool not allowed
        if (atopInType1 >= ATOP_FLOAT) {
            *wantedOutType = atopInType1;
            switch (atopInType1) {
            case ATOP_FLOAT:  return UnaryOpFast<float, __m256, CEIL_OP<float>, CEIL_OP_256<__m256>>;
            case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, CEIL_OP<double>, CEIL_OP_256<__m256d>>;
            }
        }
        break;
    case UNARY_OPERATION::TRUNC:
        // trunc on bool not allowed
        if (atopInType1 >= ATOP_FLOAT) {
            *wantedOutType = atopInType1;
            switch (atopInType1) {
            case ATOP_FLOAT:  return UnaryOpFast<float, __m256, TRUNC_OP<float>, TRUNC_OP_256<__m256>>;
            case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, TRUNC_OP<double>, TRUNC_OP_256<__m256d>>;
            }
        }
        break;
    case UNARY_OPERATION::ROUND:
        // round on bool not allowed
        if (atopInType1 >= ATOP_FLOAT) {
            *wantedOutType = atopInType1;
            // TODO: Needs review as fails test
            //switch (atopInType1) {
            //case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ROUND_OP<float>, ROUND_OP_256<__m256>>;
            //case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ROUND_OP<double>, ROUND_OP_256<__m256d>>;
            //}
        }
        break;
    case UNARY_OPERATION::SQRT:
        // Numpy odd square root rules
        // sometimes outputs float16 or float32
        if (atopInType1 >= ATOP_FLOAT) {
            *wantedOutType = ATOP_DOUBLE;
            if (atopInType1 == ATOP_FLOAT) {
                *wantedOutType = ATOP_FLOAT;
            }
            switch (atopInType1) {
            case ATOP_FLOAT:  return UnaryOpFast<float, __m256, SQRT_OP<float>, SQRT_OP_256<__m256>>;
            case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, SQRT_OP<double>, SQRT_OP_256<__m256d>>;
            }
        }
        break;

    }
    return NULL;
}

//--------------------------------------------------------------------
//
UNARY_FUNC GetUnaryOpFastStrided(int func, int atopInType1, int* wantedOutType) {

    LOGGING("Looking for func %d  type:%d\n", func, atopInType1);

    switch (func) {
    case UNARY_OPERATION::FABS:
        *wantedOutType = atopInType1;
        break;

    case UNARY_OPERATION::ABS:
        *wantedOutType = atopInType1;
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFastStrided<float, __m256, ABS_OP<float>, ABS_OP_256<__m256>>;
            //case ATOP_DOUBLE: return UnaryOpFastStrided<double, __m256d, ABS_OP<double>, ABS_OP_256<__m256d>>;
            //case ATOP_INT32:  return UnaryOpFastStrided<int32_t, __m256i, ABS_OP<int32_t>, ABS_OP_256i32>;
            //case ATOP_INT16  return UnaryOpFastStrided<int16_t, __m256i, ABS_OP<int16_t>, ABS_OP_256i16>;
            //case ATOP_INT8   return UnaryOpFastStrided<int8_t, __m256i, ABS_OP<int8_t>, ABS_OP_256i8>;

        }
    }
    return NULL;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
