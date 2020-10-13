#include "common_inc.h"
#include <cmath>

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
#pragma GCC diagnostic ignored "-Wunused-function"
#endif


//#define LOGGING printf
#define LOGGING(...)

static FORCE_INLINE const __m256i MM_SET(bool* pData) { return _mm256_set1_epi8(*(int8_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(int8_t* pData) { return _mm256_set1_epi8(*(int8_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(uint8_t* pData) { return _mm256_set1_epi8(*(int8_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(int16_t* pData) { return _mm256_set1_epi16(*(int16_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(uint16_t* pData) { return _mm256_set1_epi16(*(int16_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(int32_t* pData) { return _mm256_set1_epi32(*(int32_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(uint32_t* pData) { return _mm256_set1_epi32(*(int32_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(int64_t* pData) { return _mm256_set1_epi64x(*(int64_t*)pData); }
static FORCE_INLINE const __m256i MM_SET(uint64_t* pData) { return _mm256_set1_epi64x(*(int64_t*)pData); }
static FORCE_INLINE const __m256  MM_SET(float* pData) { return _mm256_set1_ps(*(float*)pData); }
static FORCE_INLINE const __m256d MM_SET(double* pData) { return _mm256_set1_pd(*(double*)pData); }

#if !RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED
// MSVC compiler by default assumed unaligned loads
#define LOADU(X) *(X)
#define STOREU(X,Y) *(X)=Y
#define STOREU128(X,Y) *(X)=Y

//inline __m256d LOADU(const __m256d* x) { return _mm256_stream_pd((double const *)x); };
//inline __m256 LOADU(const __m256* x) { return _mm256_stream_ps((float const *)x); };
//inline __m256i LOADU(const __m256i* x) { return _mm256_stream_si256((__m256i const *)x); };

#else
//#define LOADU(X) *(X)
#define STOREU(X,Y) _mm256_storeu_si256(X, Y)
#define STOREU128(X,Y) _mm_storeu_si128(X, Y)

inline __m256d LOADU(const __m256d* x) { return _mm256_loadu_pd((double const*)x); };
inline __m256 LOADU(const __m256* x) { return _mm256_loadu_ps((float const*)x); };
inline __m256i LOADU(const __m256i* x) { return _mm256_loadu_si256((__m256i const*)x); };

#endif


// For unsigned... (have not done yet) ---------------------------------------------------------
#define _mm_cmpge_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)
#define _mm_cmple_epu8(a, b) _mm_cmpge_epu8(b, a)
#define _mm_cmpgt_epu8(a, b) _mm_xor_si128(_mm_cmple_epu8(a, b), _mm_set1_epi8(-1))
#define _mm_cmplt_epu8(a, b) _mm_cmpgt_epu8(b, a)

//// For signed 32 ------------------------------------------------------------------------------
//#define _mm256_cmpge_epi32(a, b) _mm256_cmpeq_epi32(_mm256_max_epi32(a, b), a)
//#define _mm256_cmplt_epi32(a, b) _mm256_cmpgt_epi32(b, a)
//#define _mm256_cmple_epi32(a, b) _mm256_cmpge_epi32(b, a)
//
//// For signed 64 ------------------------------------------------------------------------------
//#define _mm256_cmpge_epi64(a, b) _mm256_cmpeq_epi64(_mm256_max_epi64(a, b), a)
//#define _mm256_cmplt_epi64(a, b) _mm256_cmpgt_epi64(b, a)
//#define _mm256_cmple_epi64(a, b) _mm256_cmpge_epi64(b, a)

// Debug routine to dump 8 int32 values
//void printm256(__m256i m0) {
//   int32_t* pData = (int32_t*)&m0;
//   printf("Value is 0x%.8x 0x%.8x 0x%.8x 0x%.8x 0x%.8x 0x%.8x 0x%.8x 0x%.8x\n", pData[0], pData[1], pData[2], pData[3], pData[4], pData[5], pData[6], pData[7]);
//}

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

// This will compute 32 x int32 comparison at a time, returning 32 bools
template<typename T> FORCE_INLINE const __m256i COMP32i_EQS(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4) {

    // the shuffle will move all 8 comparisons together
    __m256i m0 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x1, y1), g_shuffle1);
    __m256i m1 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x2, y2), g_shuffle2);
    __m256i m2 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x3, y3), g_shuffle3);
    __m256i m3 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x4, y4), g_shuffle4);
    __m256i m4 = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));

    return _mm256_and_si256(_mm256_permutevar8x32_epi32(m4, g_permute), g_ones);
}

// This will compute 32 x int32 comparison at a time
template<typename T> FORCE_INLINE const __m256i COMP32i_NES(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4) {

    // the shuffle will move all 8 comparisons together
    __m256i m0 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x1, y1), g_shuffle1);
    __m256i m1 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x2, y2), g_shuffle2);
    __m256i m2 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x3, y3), g_shuffle3);
    __m256i m3 = _mm256_shuffle_epi8(_mm256_cmpeq_epi32(x4, y4), g_shuffle4);
    __m256i m4 = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));

    // the and will flip all 0xff to 0 and all 0 to 1 -- an invert
    return _mm256_and_si256(_mm256_permutevar8x32_epi32(m4, g_permute), g_ones);
}


// This will compute 32 x int32 comparison at a time
template<typename T> FORCE_INLINE const __m256i COMP32i_GTS(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4) {

    // the shuffle will move all 8 comparisons together
    __m256i m0 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x1, y1), g_shuffle1);
    __m256i m1 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x2, y2), g_shuffle2);
    __m256i m2 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x3, y3), g_shuffle3);
    __m256i m3 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x4, y4), g_shuffle4);
    __m256i m4 = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));

    return _mm256_and_si256(_mm256_permutevar8x32_epi32(m4, g_permute), g_ones);
}

// This will compute 32 x int32 comparison at a time
template<typename T> FORCE_INLINE const __m256i COMP32i_LTS(T y1, T x1, T y2, T x2, T y3, T x3, T y4, T x4) {

    // the shuffle will move all 8 comparisons together
    __m256i m0 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x1, y1), g_shuffle1);
    __m256i m1 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x2, y2), g_shuffle2);
    __m256i m2 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x3, y3), g_shuffle3);
    __m256i m3 = _mm256_shuffle_epi8(_mm256_cmpgt_epi32(x4, y4), g_shuffle4);
    __m256i m4 = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));

    return _mm256_and_si256(_mm256_permutevar8x32_epi32(m4, g_permute), g_ones);
}

// This series of functions processes 8 int32 and returns 8 bools
template<typename T> FORCE_INLINE const int64_t COMP32i_EQ(T x, T y) { return gBooleanLUT64[_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(x, y))) & 255]; }
template<typename T> FORCE_INLINE const int64_t COMP32i_NE(T x, T y) { return gBooleanLUT64Inverse[_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(x, y))) & 255]; }
template<typename T> FORCE_INLINE const int64_t COMP32i_GT(T x, T y) { return gBooleanLUT64[_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, y))) & 255]; }
template<typename T> FORCE_INLINE const int64_t COMP32i_LT(T x, T y) { return gBooleanLUT64[_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(y, x))) & 255]; }
template<typename T> FORCE_INLINE const int64_t COMP32i_GE(T x, T y) { return gBooleanLUT64Inverse[_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(y, x))) & 255]; }
template<typename T> FORCE_INLINE const int64_t COMP32i_LE(T x, T y) { return gBooleanLUT64Inverse[_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, y))) & 255]; }

// This will compute 4 x int64 comparison at a time
template<typename T> FORCE_INLINE const int32_t COMP64i_EQ(T x, T y) {
    return gBooleanLUT32[_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, y))) & 15];
}

// This series of functions processes 4 int64 and returns 4 bools
template<typename T> FORCE_INLINE const int32_t COMP64i_NE(T x, T y) { return gBooleanLUT32Inverse[_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, y))) & 15]; }
template<typename T> FORCE_INLINE const int32_t COMP64i_GT(T x, T y) { return gBooleanLUT32[_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(x, y))) & 15]; }
template<typename T> FORCE_INLINE const int32_t COMP64i_LT(T x, T y) { return gBooleanLUT32[_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(y, x))) & 15]; }
template<typename T> FORCE_INLINE const int32_t COMP64i_GE(T x, T y) { return gBooleanLUT32Inverse[_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(y, x))) & 15]; }
template<typename T> FORCE_INLINE const int32_t COMP64i_LE(T x, T y) { return gBooleanLUT32Inverse[_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(x, y))) & 15]; }

// This will compute 32 x int8 comparison at a time
template<typename T> FORCE_INLINE const __m256i COMP8i_EQ(T x, T y, T mask1) { return _mm256_and_si256(_mm256_cmpeq_epi8(x, y), mask1); }
template<typename T> FORCE_INLINE const __m256i COMP8i_NE(T x, T y, T mask1) { return _mm256_xor_si256(_mm256_and_si256(_mm256_cmpeq_epi8(x, y), mask1), mask1); }
template<typename T> FORCE_INLINE const __m256i COMP8i_GT(T x, T y, T mask1) { return _mm256_and_si256(_mm256_cmpgt_epi8(x, y), mask1); }
template<typename T> FORCE_INLINE const __m256i COMP8i_LT(T x, T y, T mask1) { return _mm256_and_si256(_mm256_cmpgt_epi8(y, x), mask1); }
template<typename T> FORCE_INLINE const __m256i COMP8i_GE(T x, T y, T mask1) { return _mm256_xor_si256(_mm256_and_si256(_mm256_cmpgt_epi8(y, x), mask1), mask1); }
template<typename T> FORCE_INLINE const __m256i COMP8i_LE(T x, T y, T mask1) { return _mm256_xor_si256(_mm256_and_si256(_mm256_cmpgt_epi8(x, y), mask1), mask1); }

// This will compute 16 x int16 comparison at a time
template<typename T> FORCE_INLINE const __m128i COMP16i_EQ(T x1, T y1, T mask1) {
    __m256i m0 = _mm256_and_si256(_mm256_cmpeq_epi16(x1, y1), mask1);
    // move upper 128 in m0 to lower 128 in m1
    __m256i m1 = _mm256_inserti128_si256(m0, _mm256_extracti128_si256(m0, 1), 0);
    return _mm256_extracti128_si256(_mm256_packs_epi16(m0, m1), 0);
}
template<typename T> FORCE_INLINE const __m128i COMP16i_NE(T x1, T y1, T mask1) {
    __m256i m0 = _mm256_xor_si256(_mm256_and_si256(_mm256_cmpeq_epi16(x1, y1), mask1), mask1);
    // move upper 128 in m0 to lower 128 in m1
    __m256i m1 = _mm256_inserti128_si256(m0, _mm256_extracti128_si256(m0, 1), 0);
    return _mm256_extracti128_si256(_mm256_packs_epi16(m0, m1), 0);
}
template<typename T> FORCE_INLINE const __m128i COMP16i_GT(T x1, T y1, T mask1) {
    __m256i m0 = _mm256_and_si256(_mm256_cmpgt_epi16(x1, y1), mask1);
    // move upper 128 in m0 to lower 128 in m1
    __m256i m1 = _mm256_inserti128_si256(m0, _mm256_extracti128_si256(m0, 1), 0);
    return _mm256_extracti128_si256(_mm256_packs_epi16(m0, m1), 0);
}
template<typename T> FORCE_INLINE const __m128i COMP16i_LT(T x1, T y1, T mask1) {
    __m256i m0 = _mm256_and_si256(_mm256_cmpgt_epi16(y1, x1), mask1);
    // move upper 128 in m0 to lower 128 in m1
    __m256i m1 = _mm256_inserti128_si256(m0, _mm256_extracti128_si256(m0, 1), 0);
    return _mm256_extracti128_si256(_mm256_packs_epi16(m0, m1), 0);
}
template<typename T> FORCE_INLINE const __m128i COMP16i_GE(T x1, T y1, T mask1) {
    __m256i m0 = _mm256_xor_si256(_mm256_and_si256(_mm256_cmpgt_epi16(y1, x1), mask1), mask1);
    // move upper 128 in m0 to lower 128 in m1
    __m256i m1 = _mm256_inserti128_si256(m0, _mm256_extracti128_si256(m0, 1), 0);
    return _mm256_extracti128_si256(_mm256_packs_epi16(m0, m1), 0);
}
template<typename T> FORCE_INLINE const __m128i COMP16i_LE(T x1, T y1, T mask1) {
    __m256i m0 = _mm256_xor_si256(_mm256_and_si256(_mm256_cmpgt_epi16(x1, y1), mask1), mask1);
    // move upper 128 in m0 to lower 128 in m1
    __m256i m1 = _mm256_inserti128_si256(m0, _mm256_extracti128_si256(m0, 1), 0);
    return _mm256_extracti128_si256(_mm256_packs_epi16(m0, m1), 0);
}


// Build template of comparison functions
template<typename T> FORCE_INLINE const bool COMP_EQ(T X, T Y) { return (X == Y); }
template<typename T> FORCE_INLINE const bool COMP_GT(T X, T Y) { return (X > Y); }
template<typename T> FORCE_INLINE const bool COMP_GE(T X, T Y) { return (X >= Y); }
template<typename T> FORCE_INLINE const bool COMP_LT(T X, T Y) { return (X < Y); }
template<typename T> FORCE_INLINE const bool COMP_LE(T X, T Y) { return (X <= Y); }
template<typename T> FORCE_INLINE const bool COMP_NE(T X, T Y) { return (X != Y); }

// Comparing int64_t to uint64_t
template<typename T> FORCE_INLINE const bool COMP_GT_INT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return FALSE; return (X > Y); }
template<typename T> FORCE_INLINE const bool COMP_GE_INT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return FALSE; return (X >= Y); }
template<typename T> FORCE_INLINE const bool COMP_LT_INT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return TRUE; return (X < Y); }
template<typename T> FORCE_INLINE const bool COMP_LE_INT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return TRUE; return (X <= Y); }

// Comparing uint64_t to int64_t
template<typename T> FORCE_INLINE const bool COMP_EQ_UINT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return FALSE; return (X == Y); }
template<typename T> FORCE_INLINE const bool COMP_NE_UINT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return TRUE; return (X != Y); }
template<typename T> FORCE_INLINE const bool COMP_GT_UINT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return TRUE; return (X > Y); }
template<typename T> FORCE_INLINE const bool COMP_GE_UINT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return TRUE; return (X >= Y); }
template<typename T> FORCE_INLINE const bool COMP_LT_UINT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return FALSE; return (X < Y); }
template<typename T> FORCE_INLINE const bool COMP_LE_UINT64(T X, T Y) { if ((X | Y) & 0x8000000000000000) return FALSE; return (X <= Y); }



//------------------------------------------------------------------------------------------------------
// This template takes ANY type such as 32 bit floats and uses C++ functions to apply the operation
// It can handle scalars
// Used by comparison of integers
template<typename T, const bool COMPARE(T, T)>
static void CompareAny(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut) {
    int8_t* pDataOutX = (int8_t*)pDataOut;
    T* pDataInX = (T*)pDataIn;
    T* pDataIn2X = (T*)pDataIn2;

    LOGGING("compare any sizeof(T) %lld  len: %lld  %lld  %lld  out: %lld\n", sizeof(T), len, strideIn1, strideIn2, strideOut);

    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        
        if (strideIn1 == 0) {
            if (strideIn2 == sizeof(T)) {
                T arg1 = *pDataInX;
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(arg1, pDataIn2X[i]);
                }
                return;
            }
        }
        else
        if (strideIn2 == 0) {
            if (strideIn1 == sizeof(T)) {
                T arg2 = *pDataIn2X;
                LOGGING("arg2 is %lld or %llu\n", (int64_t)arg2, (uint64_t)arg2);
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[i], arg2);
                }
                return;
            }
        }
        else {
            if (strideIn2 == sizeof(T) && strideIn1 == sizeof(T)) {
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                }
                return;
            }
        }
    }
    // punt to generic loop if fall to here
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(T, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(T, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }

}


//----------------------------------------------------------------------------------
// Lookup to go from 1 byte to 8 byte boolean values
template<const int COMP_OPCODE, const bool COMPARE(float, float)>
static void CompareFloat(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut) {

    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        if (strideIn1 == 0) {
            if (strideIn2 == sizeof(float)) {
                const __m256* pSrc2Fast = (const __m256*)pDataIn2;
                int64_t* pDestFast = (int64_t*)pDataOut;
                __m256 m0 = MM_SET((float*)pDataIn);
                int64_t fastCount = len / 8;
                for (int64_t i = 0; i < fastCount; i++) {
                    int32_t bitmask = _mm256_movemask_ps(_mm256_cmp_ps(m0, LOADU(pSrc2Fast + i), COMP_OPCODE));
                    pDestFast[i] = gBooleanLUT64[bitmask & 255];
                }
                len = len - (fastCount * 8);
                const float* pDataInX = (float*)pDataIn;
                const float* pDataIn2X = &((float*)pDataIn2)[fastCount * 8];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 8];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[0], pDataIn2X[i]);
                }
                return;
            }
        }
        else
        if (strideIn2 == 0) {
            if (strideIn1 == sizeof(float)) {
                __m256* pSrc1Fast = (__m256*)pDataIn;
                __m256i* pDestFast = (__m256i*)pDataOut;
                __m256 m5 = MM_SET((float*)pDataIn2);
                int8_t* pEnd = (int8_t*)pDataOut + len;

                __m256i* pDestFastEnd = &pDestFast[len / 32];
                while (pDestFast != pDestFastEnd) {
                    // the shuffle will move all 8 comparisons together
                    __m256i m0 = _mm256_shuffle_epi8(_mm256_castps_si256(_mm256_cmp_ps(pSrc1Fast[0], m5, COMP_OPCODE)), g_shuffle1);
                    __m256i m1 = _mm256_shuffle_epi8(_mm256_castps_si256(_mm256_cmp_ps(pSrc1Fast[1], m5, COMP_OPCODE)), g_shuffle2);
                    __m256i m2 = _mm256_shuffle_epi8(_mm256_castps_si256(_mm256_cmp_ps(pSrc1Fast[2], m5, COMP_OPCODE)), g_shuffle3);
                    __m256i m3 = _mm256_shuffle_epi8(_mm256_castps_si256(_mm256_cmp_ps(pSrc1Fast[3], m5, COMP_OPCODE)), g_shuffle4);
                    m0 = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));

                    STOREU(pDestFast, _mm256_and_si256(_mm256_permutevar8x32_epi32(m0, g_permute), g_ones));
                    pSrc1Fast += 4;
                    pDestFast ++;
                }

                float* pDataInX = (float*)pSrc1Fast;
                float arg2 = *(float*)pDataIn2;
                int8_t* pDataOutX = (int8_t*)pDestFast;
                while (pDataOutX < pEnd) {
                    *pDataOutX++ = COMPARE(*pDataInX++, arg2);
                }

                //int64_t fastCount = len / 8;
                //for (int64_t i = 0; i < fastCount; i++) {
                //    int32_t bitmask = _mm256_movemask_ps(_mm256_cmp_ps(LOADU(pSrc1Fast + i), m0, COMP_OPCODE));
                //    pDestFast[i] = gBooleanLUT64[bitmask & 255];
                //}
                //len = len - (fastCount * 8);
                //const float* pDataInX = &((float*)pDataIn)[fastCount * 8];
                //const float* pDataIn2X = (float*)pDataIn2;
                //int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 8];
                //for (int64_t i = 0; i < len; i++) {
                //    pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[0]);
                //}
                return;
            }
        }
        else {
            if (strideIn1 == sizeof(float) && strideIn2 == sizeof(float)) {
                const __m256* pSrc1Fast = (const __m256*)pDataIn;
                const __m256* pSrc2Fast = (const __m256*)pDataIn2;
                int64_t* pDestFast = (int64_t*)pDataOut;
                int64_t fastCount = len / 8;
                for (int64_t i = 0; i < fastCount; i++) {
                    // Alternate way
                    int32_t bitmask = _mm256_movemask_ps(_mm256_cmp_ps(LOADU(pSrc1Fast + i), LOADU(pSrc2Fast + i), COMP_OPCODE));
                    pDestFast[i] = gBooleanLUT64[bitmask & 255];
                }
                len = len - (fastCount * 8);
                const float* pDataInX = &((float*)pDataIn)[fastCount * 8];
                const float* pDataIn2X = &((float*)pDataIn2)[fastCount * 8];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 8];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                }
                return;
            }
        }
    }
    // generic rare case
    float* pDataInX = (float*)pDataIn;
    float* pDataIn2X = (float*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(float, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(float, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }

}

//=======================================================================================================
//
template<const int COMP_OPCODE, const bool COMPARE(double, double)>
static void CompareDouble(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut)
{
    LOGGING("compare double   len: %lld  %lld  %lld  out: %lld\n",  len, strideIn1, strideIn2, strideOut);

    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        int32_t* pDestFast = (int32_t*)pDataOut;

        if (strideIn1 == 0)
        {
            if (strideIn2 == sizeof(double)) {

                const __m256d* pSrc2Fast = (const __m256d*)pDataIn2;
                int64_t fastCount = len / 4;

                __m256d m0 = MM_SET((double*)pDataIn);
                for (int64_t i = 0; i < fastCount; i++)
                {
                    int32_t bitmask = _mm256_movemask_pd(_mm256_cmp_pd(m0, LOADU(pSrc2Fast + i), COMP_OPCODE));
                    pDestFast[i] = gBooleanLUT32[bitmask & 15];
                }
                len = len - (fastCount * 4);
                const double arg1 = *(double*)pDataIn;
                const double* pDataIn2X = &((double*)pDataIn2)[fastCount * 4];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 4];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(arg1, pDataIn2X[i]);
                }
                return;
            }
        }
        else
        if (strideIn2 == 0)
        {
            if (strideIn1 == sizeof(double)) {
                const __m256d* pSrc1Fast = (const __m256d*)pDataIn;
                int64_t fastCount = len / 4;

                __m256d m0 = MM_SET((double*)pDataIn2);
                for (int64_t i = 0; i < fastCount; i++)
                {
                    int32_t bitmask = _mm256_movemask_pd(_mm256_cmp_pd(LOADU(pSrc1Fast + i), m0, COMP_OPCODE));
                    pDestFast[i] = gBooleanLUT32[bitmask & 15];
                }
                len = len - (fastCount * 4);
                const double* pDataInX = &((double*)pDataIn)[fastCount * 4];
                const double arg2 = *(double*)pDataIn2;
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 4];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[i], arg2);
                }
                return;
            }
        }
        else {
            if (strideIn2 == sizeof(double) && strideIn1 == sizeof(double)) {
                const __m256d* pSrc1Fast = (const __m256d*)pDataIn;
                const __m256d* pSrc2Fast = (const __m256d*)pDataIn2;
                int64_t fastCount = len / 4;

                for (int64_t i = 0; i < fastCount; i++)
                {
                    int32_t bitmask = _mm256_movemask_pd(_mm256_cmp_pd(LOADU(pSrc1Fast + i), LOADU(pSrc2Fast + i), COMP_OPCODE));
                    //printf("bitmask is %d\n", bitmask);
                    //printf("bitmask & 15 is %d\n", bitmask & 15);
                    pDestFast[i] = gBooleanLUT32[bitmask & 15];
                }
                len = len - (fastCount * 4);
                const double* pDataInX = &((double*)pDataIn)[fastCount * 4];
                const double* pDataIn2X = &((double*)pDataIn2)[fastCount * 4];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 4];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                }
                return;
            }
        }
    }
    // generic rare case
    double* pDataInX = (double*)pDataIn;
    double* pDataIn2X = (double*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(double, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(double, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }
}


//=======================================================================================================
//
template<const int32_t COMP_256(__m256i, __m256i), const bool COMPARE(int64_t, int64_t)>
static void CompareInt64(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut)
{

    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        const __m256i* pSrc1Fast = (const __m256i*)pDataIn;
        const __m256i* pSrc2Fast = (const __m256i*)pDataIn2;
        int64_t fastCount = len / 4;
        int32_t* pDestFast = (int32_t*)pDataOut;

        if (strideIn1 == 0)
        {
            if (strideIn2 == sizeof(int64_t)) {
                __m256i m0 = MM_SET((int64_t*)pDataIn);
                for (int64_t i = 0; i < fastCount; i++)
                {
                    pDestFast[i] = COMP_256(m0, LOADU(pSrc2Fast + i));
                }
                len = len - (fastCount * 4);
                const int64_t* pDataInX = (int64_t*)pDataIn;
                const int64_t* pDataIn2X = &((int64_t*)pDataIn2)[fastCount * 4];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 4];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[0], pDataIn2X[i]);
                }
                return;
            }
        }
        else
            if (strideIn2 == 0)
            {
                if (strideIn2 == sizeof(int64_t)) {
                    __m256i m0 = MM_SET((int64_t*)pDataIn2);
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        pDestFast[i] = COMP_256(LOADU(pSrc1Fast + i), m0);
                    }
                    len = len - (fastCount * 4);
                    const int64_t* pDataInX = &((int64_t*)pDataIn)[fastCount * 4];
                    const int64_t* pDataIn2X = (int64_t*)pDataIn2;
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 4];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[0]);
                    }
                    return;
                }
            }
            else {
                if (strideIn1 == sizeof(int64_t) && strideIn2 == sizeof(int64_t)) {
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        pDestFast[i] = COMP_256(LOADU(pSrc1Fast + i), LOADU(pSrc2Fast + i));
                    }
                    len = len - (fastCount * 4);
                    const int64_t* pDataInX = &((int64_t*)pDataIn)[fastCount * 4];
                    const int64_t* pDataIn2X = &((int64_t*)pDataIn2)[fastCount * 4];
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 4];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                    }
                    return;
                }
            }
    }
    // generic rare case
    int64_t* pDataInX = (int64_t*)pDataIn;
    int64_t* pDataIn2X = (int64_t*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(int64_t, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(int64_t, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }
}


//=======================================================================================================
// Compare 4x8xint32 using 256bit vector intrinsics
// This routine is currently disabled and runs at slightly faster speed as CompareInt32
// Leave the code because as is a good example on how to compute 32 bools
template<const __m256i COMP_256(__m256i, __m256i, __m256i, __m256i, __m256i, __m256i, __m256i, __m256i), const bool COMPARE(int32_t, int32_t)>
static void CompareInt32S(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut)
{
    LOGGING("int32 %lld %lld %lld\n", strideIn1, strideIn2, strideOut);
    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        const __m256i* pSrc1Fast = (const __m256i*)pDataIn;
        const __m256i* pSrc2Fast = (const __m256i*)pDataIn2;
        // compute 32 bools at once
        int64_t fastCount = len / 32;
        __m256i* pDestFast = (__m256i*)pDataOut;
        __m256i* pDestFastEnd = pDestFast + fastCount;

        if (strideIn1 == 0)
        {
            if (strideIn2 == sizeof(int32_t)) {
                __m256i m0 = MM_SET((int32_t*)pDataIn);
                while (pDestFast < pDestFastEnd)
                {
                    STOREU(pDestFast, COMP_256(
                        m0, LOADU(pSrc2Fast),
                        m0, LOADU(pSrc2Fast + 1),
                        m0, LOADU(pSrc2Fast + 2),
                        m0, LOADU(pSrc2Fast + 3)
                    ));
                    pSrc2Fast += 4;
                    pDestFast++;
                }
                len = len - (fastCount * 32);
                const int32_t* pDataInX = (int32_t*)pDataIn;
                const int32_t* pDataIn2X = (int32_t*)pSrc2Fast;
                int8_t* pDataOutX = (int8_t*)pDestFast;
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[0], pDataIn2X[i]);
                }
                return;
            }
        }
        else
            if (strideIn2 == 0)
            {
                if (strideIn1 == sizeof(int32_t)) {
                    __m256i m0 = MM_SET((int32_t*)pDataIn2);
                    while (pDestFast < pDestFastEnd)
                    {
                        STOREU(pDestFast, COMP_256(
                            LOADU(pSrc1Fast), m0,
                            LOADU(pSrc1Fast + 1), m0,
                            LOADU(pSrc1Fast + 2), m0,
                            LOADU(pSrc1Fast + 3), m0
                        ));

                        pSrc1Fast += 4;
                        pDestFast++;
                    }
                    len = len - (fastCount * 32);
                    const int32_t* pDataInX = (int32_t*)pSrc1Fast;
                    const int32_t* pDataIn2X = (int32_t*)pDataIn2;
                    int8_t* pDataOutX = (int8_t*)pDestFast;
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[0]);
                    }
                    return;
                }
            }
            else {
                if (strideIn1 == sizeof(int32_t) && strideIn2 == sizeof(int32_t)) {
                    while (pDestFast < pDestFastEnd)
                    {
                        // the result is 32 bools __m256i
                        STOREU(pDestFast, COMP_256(
                            LOADU(pSrc1Fast), LOADU(pSrc2Fast),
                            LOADU(pSrc1Fast + 1), LOADU(pSrc2Fast + 1),
                            LOADU(pSrc1Fast + 2), LOADU(pSrc2Fast + 2),
                            LOADU(pSrc1Fast + 3), LOADU(pSrc2Fast + 3)
                        ));
                        pSrc1Fast += 4;
                        pSrc2Fast += 4;
                        pDestFast++;
                    }
                    len = len - (fastCount * 32);
                    const int32_t* pDataInX = (int32_t*)pSrc1Fast;
                    const int32_t* pDataIn2X = (int32_t*)pSrc2Fast;
                    int8_t* pDataOutX = (int8_t*)pDestFast;
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                    }
                    return;
                }
            }
    }
    // generic rare case
    int32_t* pDataInX = (int32_t*)pDataIn;
    int32_t* pDataIn2X = (int32_t*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(int32_t, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(int32_t, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }
}

//=======================================================================================================
// Compare 8xint32 using 256bit vector intrinsics
template<const int64_t COMP_256(__m256i, __m256i), const bool COMPARE(int32_t, int32_t)>
static void CompareInt32(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut)
{
    LOGGING("int32 %lld %lld %lld\n", strideIn1, strideIn2, strideOut);
    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        const __m256i* pSrc1Fast = (const __m256i*)pDataIn;
        const __m256i* pSrc2Fast = (const __m256i*)pDataIn2;
        // compute 8 bools at once
        int64_t fastCount = len / 8;
        int64_t* pDestFast = (int64_t*)pDataOut;

        if (strideIn1 == 0)
        {
            if (strideIn2 == sizeof(int32_t)) {
                __m256i m0 = MM_SET((int32_t*)pDataIn);
                for (int64_t i = 0; i < fastCount; i++)
                {
                    pDestFast[i] = COMP_256(m0, LOADU(pSrc2Fast + i));
                }
                len = len - (fastCount * 8);
                const int32_t* pDataInX = (int32_t*)pDataIn;
                const int32_t* pDataIn2X = &((int32_t*)pDataIn2)[fastCount * 8];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 8];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[0], pDataIn2X[i]);
                }
                return;
            }
        }
        else
            if (strideIn2 == 0)
            {
                if (strideIn1 == sizeof(int32_t)) {
                    __m256i m0 = MM_SET((int32_t*)pDataIn2);
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        pDestFast[i] = COMP_256(LOADU(pSrc1Fast + i), m0);
                    }
                    len = len - (fastCount * 8);
                    const int32_t* pDataInX = &((int32_t*)pDataIn)[fastCount * 8];
                    const int32_t* pDataIn2X = (int32_t*)pDataIn2;
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 8];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[0]);
                    }
                    return;
                }
            }
            else {
                if (strideIn1 == sizeof(int32_t) && strideIn2 == sizeof(int32_t)) {
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        pDestFast[i] = COMP_256(LOADU(pSrc1Fast + i), LOADU(pSrc2Fast + i));
                    }
                    len = len - (fastCount * 8);
                    const int32_t* pDataInX = &((int32_t*)pDataIn)[fastCount * 8];
                    const int32_t* pDataIn2X = &((int32_t*)pDataIn2)[fastCount * 8];
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 8];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                    }
                    return;
                }
            }
    }
    // generic rare case
    int32_t* pDataInX = (int32_t*)pDataIn;
    int32_t* pDataIn2X = (int32_t*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(int32_t, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(int32_t, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }

}


//=======================================================================================================
// Compare 8xint16 using 256bit vector intrinsics
template<const __m128i COMP_256(__m256i, __m256i, __m256i), const bool COMPARE(int16_t, int16_t)>
static void CompareInt16(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut)
{

    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        const __m256i* pSrc1Fast = (const __m256i*)pDataIn;
        const __m256i* pSrc2Fast = (const __m256i*)pDataIn2;
        // compute 16 bools at once
        int64_t fastCount = len / 16;
        __m128i* pDestFast = (__m128i*)pDataOut;
        __m256i mask1 = _mm256_set1_epi16(1);

        if (strideIn1 == 0)
        {
            if (strideIn2 == sizeof(int16_t)) {
                __m256i m0 = MM_SET((int16_t*)pDataIn);
                for (int64_t i = 0; i < fastCount; i++)
                {
                    STOREU128(pDestFast + i, COMP_256(m0, LOADU(pSrc2Fast + i), mask1));
                }
                len = len - (fastCount * 16);
                const int16_t* pDataInX = (int16_t*)pDataIn;
                const int16_t* pDataIn2X = &((int16_t*)pDataIn2)[fastCount * 16];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 16];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[0], pDataIn2X[i]);
                }
                return;
            }
        }
        else
            if (strideIn2 == 0)
            {
                if (strideIn1 == sizeof(int16_t)) {
                    __m256i m0 = MM_SET((int16_t*)pDataIn2);
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        STOREU128(pDestFast + i, COMP_256(LOADU(pSrc1Fast + i), m0, mask1));
                    }
                    len = len - (fastCount * 16);
                    const int16_t* pDataInX = &((int16_t*)pDataIn)[fastCount * 16];
                    const int16_t* pDataIn2X = (int16_t*)pDataIn2;
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 16];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[0]);
                    }
                    return;
                }
            }
            else {
                if (strideIn1 == sizeof(int16_t) && strideIn2 == sizeof(int16_t)) {
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        STOREU128(pDestFast + i, COMP_256(LOADU(pSrc1Fast + i), LOADU(pSrc2Fast + i), mask1));
                    }
                    len = len - (fastCount * 16);
                    const int16_t* pDataInX = &((int16_t*)pDataIn)[fastCount * 16];
                    const int16_t* pDataIn2X = &((int16_t*)pDataIn2)[fastCount * 16];
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 16];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                    }
                    return;
                }
            }
    }
    // generic rare case
    int16_t* pDataInX = (int16_t*)pDataIn;
    int16_t* pDataIn2X = (int16_t*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(int16_t, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(int16_t, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }
}


//=======================================================================================================
// Compare 8xint8 using 256bit vector intrinsics
template<const __m256i COMP_256(__m256i, __m256i, __m256i), const bool COMPARE(int8_t, int8_t)>
static void CompareInt8(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut)
{
    // check for strided output first
    if (strideOut == sizeof(int8_t)) {
        const __m256i* pSrc1Fast = (const __m256i*)pDataIn;
        const __m256i* pSrc2Fast = (const __m256i*)pDataIn2;
        // compute 32 bools at once
        int64_t fastCount = len / 32;
        __m256i* pDestFast = (__m256i*)pDataOut;
        __m256i mask1 = _mm256_set1_epi8(1);

        if (strideIn1 == 0)
        {
            if (strideIn2 == sizeof(int8_t)) {
                __m256i m0 = MM_SET((int8_t*)pDataIn);
                for (int64_t i = 0; i < fastCount; i++)
                {
                    STOREU(pDestFast + i, COMP_256(m0, LOADU(pSrc2Fast + i), mask1));
                }
                len = len - (fastCount * 32);
                const int8_t* pDataInX = (int8_t*)pDataIn;
                const int8_t* pDataIn2X = &((int8_t*)pDataIn2)[fastCount * 32];
                int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 32];
                for (int64_t i = 0; i < len; i++) {
                    pDataOutX[i] = COMPARE(pDataInX[0], pDataIn2X[i]);
                }
                return;
            }
        }
        else
            if (strideIn2 == 0)
            {
                if (strideIn1 == sizeof(int8_t)) {
                    __m256i m0 = MM_SET((int8_t*)pDataIn2);
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        STOREU(pDestFast + i, COMP_256(LOADU(pSrc1Fast + i), m0, mask1));
                    }
                    len = len - (fastCount * 32);
                    const int8_t* pDataInX = &((int8_t*)pDataIn)[fastCount * 32];
                    const int8_t* pDataIn2X = (int8_t*)pDataIn2;
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 32];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[0]);
                    }
                    return;
                }
            }
            else {
                if (strideIn1 == sizeof(int8_t) && strideIn2 == sizeof(int8_t)) {
                    for (int64_t i = 0; i < fastCount; i++)
                    {
                        STOREU(pDestFast + i, COMP_256(LOADU(pSrc1Fast + i), LOADU(pSrc2Fast + i), mask1));
                    }
                    len = len - (fastCount * 32);
                    const int8_t* pDataInX = &((int8_t*)pDataIn)[fastCount * 32];
                    const int8_t* pDataIn2X = &((int8_t*)pDataIn2)[fastCount * 32];
                    int8_t* pDataOutX = &((int8_t*)pDataOut)[fastCount * 32];
                    for (int64_t i = 0; i < len; i++) {
                        pDataOutX[i] = COMPARE(pDataInX[i], pDataIn2X[i]);
                    }
                }
            }
    }
    // generic rare case
    int8_t* pDataInX = (int8_t*)pDataIn;
    int8_t* pDataIn2X = (int8_t*)pDataIn2;
    int8_t* pDataOutX = (int8_t*)pDataOut;
    for (int64_t i = 0; i < len; i++) {
        *pDataOutX = COMPARE(*pDataInX, *pDataIn2X);
        pDataInX = STRIDE_NEXT(int8_t, pDataInX, strideIn1);
        pDataIn2X = STRIDE_NEXT(int8_t, pDataIn2X, strideIn2);
        pDataOutX = STRIDE_NEXT(int8_t, pDataOutX, strideOut);
    }

}




// example of stub
//static void Compare32(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int32_t scalarMode) { return CompareFloat<_CMP_EQ_OS>(pDataIn, pDataIn2, pDataOut, len, scalarMode); }
//const int CMP_LUT[6] = { _CMP_EQ_OS, _CMP_NEQ_OS, _CMP_LT_OS, _CMP_GT_OS, _CMP_LE_OS, _CMP_GE_OS };

//==========================================================
// May return NULL if it cannot handle type or function
extern "C"
ANY_TWO_FUNC GetComparisonOpFast(int func, int atopInType1, int atopInType2, int* wantedOutType) {

    BOOL bSpecialComparison = FALSE;

    if (atopInType1 != atopInType2) {
        // Because upcasting an int64_t to a float64 results in precision loss, we try comparisons
        if (atopInType1 >= ATOP_INT64 && atopInType1 <= ATOP_UINT64 && atopInType2 >= ATOP_INT64 && atopInType2 <= ATOP_UINT64) {
            bSpecialComparison = TRUE;        
        }

        if (!bSpecialComparison)
            return NULL;
    }

    *wantedOutType = ATOP_BOOL;
    int mainType = atopInType1;

    LOGGING("Comparison maintype %d for func %d  inputs: %d %d\n", mainType, func, atopInType1, atopInType2);

    // NOTE: Intel on Nans
    // Use _CMP_NEQ_US instead of OS because it works with != nan comparisons
    //The unordered relationship is true when at least one of the two source operands being compared is a NaN; the ordered relationship is true when neither source operand is a NaN.
    //A subsequent computational instruction that uses the mask result in the destination operand as an input operand will not generate an exception, 
    //because a mask of all 0s corresponds to a floating - point value of + 0.0 and a mask of all 1s corresponds to a QNaN.
    /*Ordered comparison of NaN and 1.0 gives false.
       Unordered comparison of NaN and 1.0 gives true.
       Ordered comparison of 1.0 and 1.0 gives true.
       Unordered comparison of 1.0 and 1.0 gives false.
       Ordered comparison of NaN and Nan gives false.
       Unordered comparison of NaN and NaN gives true.
   */

    switch (mainType) {
    case ATOP_FLOAT:
        switch (func) {
        case COMP_OPERATION::CMP_EQ:      return CompareFloat<_CMP_EQ_OS, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareFloat<_CMP_NEQ_US, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareFloat<_CMP_GT_OS, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareFloat<_CMP_GE_OS, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareFloat<_CMP_LT_OS, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareFloat<_CMP_LE_OS, COMP_LE>;
        }
        break;
    case ATOP_DOUBLE:
        switch (func) {
        case COMP_OPERATION::CMP_EQ:      return CompareDouble<_CMP_EQ_OS, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareDouble<_CMP_NEQ_US, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareDouble<_CMP_GT_OS, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareDouble<_CMP_GE_OS, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareDouble<_CMP_LT_OS, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareDouble<_CMP_LE_OS, COMP_LE>;
        }
        break;
    case ATOP_INT32:
        switch (func) {
        case COMP_OPERATION::CMP_EQ:      return CompareInt32S<COMP32i_EQS<__m256i>, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareInt32<COMP32i_NE<__m256i>, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareInt32<COMP32i_GT<__m256i>, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareInt32<COMP32i_GE<__m256i>, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareInt32<COMP32i_LT<__m256i>, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareInt32<COMP32i_LE<__m256i>, COMP_LE>;
        }
        break;
    case ATOP_UINT32:
        switch (func) {
            // NOTE: if this needs to get sped up, upcast from uint32_t to int64_t  using _mm256_cvtepu32_epi64 and cmpint64
            // For equal, not equal the sign does not matter
        case COMP_OPERATION::CMP_EQ:      return CompareInt32<COMP32i_EQ<__m256i>, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareInt32<COMP32i_NE<__m256i>, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareAny<uint32_t, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareAny<uint32_t, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareAny<uint32_t, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareAny<uint32_t, COMP_LE>;
        }
        break;
    case ATOP_INT64:
        // signed ints in atop will have last bit set
        if (atopInType1 != atopInType2 && !(atopInType2 & 1)) {
            switch (func) {
            case COMP_OPERATION::CMP_EQ:      return CompareAny<int64_t, COMP_EQ_UINT64>;
            case COMP_OPERATION::CMP_NE:      return CompareAny<int64_t, COMP_NE_UINT64>;
            case COMP_OPERATION::CMP_GT:      return CompareAny<int64_t, COMP_GT_INT64>;
            case COMP_OPERATION::CMP_GTE:     return CompareAny<int64_t, COMP_GE_INT64>;
            case COMP_OPERATION::CMP_LT:      return CompareAny<int64_t, COMP_LT_INT64>;
            case COMP_OPERATION::CMP_LTE:     return CompareAny<int64_t, COMP_LE_INT64>;
            }
        }
        else {
            switch (func) {
            case COMP_OPERATION::CMP_EQ:      return CompareInt64<COMP64i_EQ<__m256i>, COMP_EQ>;
            case COMP_OPERATION::CMP_NE:      return CompareInt64<COMP64i_NE<__m256i>, COMP_NE>;
            case COMP_OPERATION::CMP_GT:      return CompareInt64<COMP64i_GT<__m256i>, COMP_GT>;
            case COMP_OPERATION::CMP_GTE:     return CompareInt64<COMP64i_GE<__m256i>, COMP_GE>;
            case COMP_OPERATION::CMP_LT:      return CompareInt64<COMP64i_LT<__m256i>, COMP_LT>;
            case COMP_OPERATION::CMP_LTE:     return CompareInt64<COMP64i_LE<__m256i>, COMP_LE>;
            }
        }
        break;
    case ATOP_UINT64:
        // signed ints in atop will have last bit set
        if (atopInType1 != atopInType2 && (atopInType2 & 1)) {
            switch (func) {
                // For equal, not equal the sign does not matter
            case COMP_OPERATION::CMP_EQ:      return CompareAny<int64_t, COMP_EQ>;
            case COMP_OPERATION::CMP_NE:      return CompareAny<int64_t, COMP_NE>;
            case COMP_OPERATION::CMP_GT:      return CompareAny<uint64_t, COMP_GT_UINT64>;
            case COMP_OPERATION::CMP_GTE:     return CompareAny<uint64_t, COMP_GE_UINT64>;
            case COMP_OPERATION::CMP_LT:      return CompareAny<uint64_t, COMP_LT_UINT64>;
            case COMP_OPERATION::CMP_LTE:     return CompareAny<uint64_t, COMP_LE_UINT64>;
            }
        }
        else {
            switch (func) {
                // For equal, not equal the sign does not matter
            case COMP_OPERATION::CMP_EQ:      return CompareInt64<COMP64i_EQ<__m256i>, COMP_EQ>;
            case COMP_OPERATION::CMP_NE:      return CompareInt64<COMP64i_NE<__m256i>, COMP_NE>;
            case COMP_OPERATION::CMP_GT:      return CompareAny<uint64_t, COMP_GT>;
            case COMP_OPERATION::CMP_GTE:     return CompareAny<uint64_t, COMP_GE>;
            case COMP_OPERATION::CMP_LT:      return CompareAny<uint64_t, COMP_LT>;
            case COMP_OPERATION::CMP_LTE:     return CompareAny<uint64_t, COMP_LE>;
            }

        }
        break;
    case ATOP_BOOL:
        // TJD: TODO - special compare
        break;
    case ATOP_INT8:
        switch (func) {
        case COMP_OPERATION::CMP_EQ:      return CompareInt8<COMP8i_EQ<__m256i>, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareInt8<COMP8i_NE<__m256i>, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareInt8<COMP8i_GT<__m256i>, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareInt8<COMP8i_GE<__m256i>, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareInt8<COMP8i_LT<__m256i>, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareInt8<COMP8i_LE<__m256i>, COMP_LE>;
        }
        break;
    case ATOP_UINT8:
        switch (func) {
        case COMP_OPERATION::CMP_EQ:      return CompareInt8<COMP8i_EQ<__m256i>, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareInt8<COMP8i_NE<__m256i>, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareAny<uint8_t, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareAny<uint8_t, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareAny<uint8_t, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareAny<uint8_t, COMP_LE>;
        }
    case ATOP_INT16:
        switch (func) {
        case COMP_OPERATION::CMP_EQ:      return CompareInt16<COMP16i_EQ<__m256i>, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareInt16<COMP16i_NE<__m256i>, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareInt16<COMP16i_GT<__m256i>, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareInt16<COMP16i_GE<__m256i>, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareInt16<COMP16i_LT<__m256i>, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareInt16<COMP16i_LE<__m256i>, COMP_LE>;
        }
        break;
    case ATOP_UINT16:
        switch (func) {
            // NOTE: if this needs to get sped up, upcast from uint16_t to int32_t  using _mm256_cvtepu16_epi32 and cmpint32
        case COMP_OPERATION::CMP_EQ:      return CompareInt16<COMP16i_EQ<__m256i>, COMP_EQ>;
        case COMP_OPERATION::CMP_NE:      return CompareInt16<COMP16i_NE<__m256i>, COMP_NE>;
        case COMP_OPERATION::CMP_GT:      return CompareAny<uint16_t, COMP_GT>;
        case COMP_OPERATION::CMP_GTE:     return CompareAny<uint16_t, COMP_GE>;
        case COMP_OPERATION::CMP_LT:      return CompareAny<uint16_t, COMP_LT>;
        case COMP_OPERATION::CMP_LTE:     return CompareAny<uint16_t, COMP_LE>;
        }
        break;
    }

    return NULL;
}


#if defined(__clang__)
#pragma clang attribute pop
#endif
