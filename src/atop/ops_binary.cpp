#include "threads.h"
#include <cmath>

#if defined(__GNUC__)
#pragma GCC target "arch=core-avx2,tune=core-avx2"
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

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang attribute push (__attribute__((target("avx""))), apply_to=function)
#endif



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

//=========================================================================================
template<typename T> static const inline T AddOp(T x, T y) { return x + y; }
template<typename T> static const inline T SubOp(T x, T y) { return x - y; }
template<typename T> static const inline T MulOp(T x, T y) { return x * y; }

// bitwise operations
template<typename T> static const inline T AndOp(T x, T y) { return x & y; }
template<typename T> static const inline T XorOp(T x, T y) { return x ^ y; }
template<typename T> static const inline T OrOp(T x, T y) { return x | y; }
// NOTE: mimics intel intrinsic
template<typename T> static const inline T AndNotOp(T x, T y) { return ~x & y; }

//=========================================================================================
static const inline __m256  ADD_OP_256f32(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
static const inline __m256d ADD_OP_256f64(__m256d x, __m256d y) { return _mm256_add_pd(x, y); }
static const inline __m256i ADD_OP_256i8(__m256i x, __m256i y) { return _mm256_add_epi8(x, y); }
static const inline __m256i ADD_OP_256i16(__m256i x, __m256i y) { return _mm256_add_epi16(x, y); }
static const inline __m256i ADD_OP_256i32(__m256i x, __m256i y) { return _mm256_add_epi32(x, y); }
static const inline __m256i ADD_OP_256i64(__m256i x, __m256i y) { return _mm256_add_epi64(x, y); }

static const inline __m256  SUB_OP_256f32(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
static const inline __m256d SUB_OP_256f64(__m256d x, __m256d y) { return _mm256_sub_pd(x, y); }
static const inline __m256i SUB_OP_256i8(__m256i x, __m256i y) { return _mm256_sub_epi8(x, y); }
static const inline __m256i SUB_OP_256i16(__m256i x, __m256i y) { return _mm256_sub_epi16(x, y); }
static const inline __m256i SUB_OP_256i32(__m256i x, __m256i y) { return _mm256_sub_epi32(x, y); }
static const inline __m256i SUB_OP_256i64(__m256i x, __m256i y) { return _mm256_sub_epi64(x, y); }

static const inline __m256  MUL_OP_256f32(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
static const inline __m256d MUL_OP_256f64(__m256d x, __m256d y) { return _mm256_mul_pd(x, y); }
static const inline __m256i MUL_OP_256i16(__m256i x, __m256i y) { return _mm256_mullo_epi16(x, y); }
static const inline __m256i MUL_OP_256i32(__m256i x, __m256i y) { return _mm256_mullo_epi32(x, y); }

// mask off low 32bits
static const __m256i masklo = _mm256_set1_epi64x(0xFFFFFFFFLL);
static const __m128i shifthigh = _mm_set1_epi64x(32);

// This routine only works for positive integers
static const inline __m256i MUL_OP_256u64(__m256i x, __m256i y) {
    // Algo is lo1*lo2 + (lo1*hi2) << 32 + (lo2*hi1) << 32
    // To get to 128 bit int would have to add (hi1*hi2) << 64
    __m256i lo1 = _mm256_and_si256(x, masklo);
    __m256i lo2 = _mm256_and_si256(y, masklo);
    __m256i hi1 = _mm256_srl_epi64(x, shifthigh); // need to sign extend
    __m256i hi2 = _mm256_srl_epi64(y, shifthigh);
    __m256i add1 = _mm256_mul_epu32(lo1, lo2);
    __m256i add2 = _mm256_sll_epi64(_mm256_mul_epu32(lo1, hi2), shifthigh);
    __m256i add3 = _mm256_sll_epi64(_mm256_mul_epu32(lo2, hi1), shifthigh);
    // add all the results together
    return _mm256_add_epi64(add1, _mm256_add_epi64(add2, add3));
}


static const inline __m256i AND_OP_256(__m256i x, __m256i y) { return _mm256_and_si256(x, y); }
static const inline __m256i OR_OP_256(__m256i x, __m256i y) { return _mm256_or_si256(x, y); }
static const inline __m256i XOR_OP_256(__m256i x, __m256i y) { return _mm256_xor_si256(x, y); }
static const inline __m256i ANDNOT_OP_256(__m256i x, __m256i y) { return _mm256_andnot_si256(x, y); }



//=====================================================================================================
// Not symmetric -- arg1 must be first, arg2 must be second
template<typename T, typename U256, const T MATH_OP(T, T), const U256 MATH_OP256(U256, U256)>
inline void SimpleMathOpFast(void* pDataIn1X, void* pDataIn2X, void* pDataOutX, int64_t datalen, int32_t scalarMode) {
    T* pDataOut = (T*)pDataOutX;
    T* pDataIn1 = (T*)pDataIn1X;
    T* pDataIn2 = (T*)pDataIn2X;

    const int64_t NUM_LOOPS_UNROLLED = 1;
    const int64_t chunkSize = NUM_LOOPS_UNROLLED * (sizeof(U256) / sizeof(T));
    int64_t perReg = sizeof(U256) / sizeof(T);

    LOGGING("mathopfast datalen %llu  chunkSize %llu  perReg %llu\n", datalen, chunkSize, perReg);

    switch (scalarMode) {
    case SCALAR_MODE::NO_SCALARS:
    {
        if (datalen >= chunkSize) {
            T* pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
            U256* pEnd_256 = (U256*)pEnd;
            U256* pIn1_256 = (U256*)pDataIn1;
            U256* pIn2_256 = (U256*)pDataIn2;
            U256* pOut_256 = (U256*)pDataOut;

            do {
                // clang requires LOADU on last operand
#ifdef RT_COMPILER_MSVC
            // Microsoft will create the opcode where the second argument is an address
                STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), *pIn2_256));
#else
                STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), LOADU(pIn2_256)));
#endif
                pOut_256 += NUM_LOOPS_UNROLLED;
                pIn1_256 += NUM_LOOPS_UNROLLED;
                pIn2_256 += NUM_LOOPS_UNROLLED;
            } while (pOut_256 < pEnd_256);

            // update pointers to last location of wide pointers
            pDataIn1 = (T*)pIn1_256;
            pDataIn2 = (T*)pIn2_256;
            pDataOut = (T*)pOut_256;
        }

        datalen = datalen & (chunkSize - 1);
        for (int64_t i = 0; i < datalen; i++) {
            pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
        }

        break;
    }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
    {
        // NOTE: the unrolled loop is faster
        T arg1 = *pDataIn1;

        if (datalen >= chunkSize) {
            T* pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
            U256* pEnd_256 = (U256*)pEnd;
            U256* pIn1_256 = (U256*)pDataIn1;
            U256* pIn2_256 = (U256*)pDataIn2;
            U256* pOut_256 = (U256*)pDataOut;

            const U256 m0 = LOADU(pIn1_256);

            do {
#ifdef RT_COMPILER_MSVC
                STOREU(pOut_256, MATH_OP256(m0, *pIn2_256));
#else
                STOREU(pOut_256, MATH_OP256(m0, LOADU(pIn2_256)));
#endif

                pOut_256 += NUM_LOOPS_UNROLLED;
                pIn2_256 += NUM_LOOPS_UNROLLED;

            } while (pOut_256 < pEnd_256);

            // update pointers to last location of wide pointers
            pDataIn2 = (T*)pIn2_256;
            pDataOut = (T*)pOut_256;
        }
        datalen = datalen & (chunkSize - 1);
        for (int64_t i = 0; i < datalen; i++) {
            pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
        }
        break;
    }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
    {
        T arg2 = *pDataIn2;

        // Check if the output is the same as the input
        if (pDataOut == pDataIn1) {

            // align the load to 32 byte boundary
            int64_t babylen = (int64_t)pDataIn1 & 31;
            if (babylen != 0) {
                // calc how much to align data
                babylen = (32 - babylen) / sizeof(T);
                if (babylen <= datalen) {
                    for (int64_t i = 0; i < babylen; i++) {
                        pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
                    }
                    pDataIn1 += babylen;
                    datalen -= babylen;
                }
            }

            // inplace operation
            if (datalen >= chunkSize) {
                T* pEnd = &pDataIn1[chunkSize * (datalen / chunkSize)];
                U256* pEnd_256 = (U256*)pEnd;
                U256* pIn1_256 = (U256*)pDataIn1;
                U256* pIn2_256 = (U256*)pDataIn2;

                const U256 m1 = LOADU(pIn2_256);

                // apply 256bit aligned operations
                while (pIn1_256 < pEnd_256) {
                    STOREA(pIn1_256, MATH_OP256(LOADA(pIn1_256), m1));
                    pIn1_256++;
                }

                // update pointers to last location of wide pointers
                pDataIn1 = (T*)pIn1_256;
            }
            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++) {
                pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
            }

        }
        else {

            if (datalen >= chunkSize) {
                T* pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                U256* pEnd_256 = (U256*)pEnd;
                U256* pIn1_256 = (U256*)pDataIn1;
                U256* pIn2_256 = (U256*)pDataIn2;
                U256* pOut_256 = (U256*)pDataOut;

                const U256 m1 = LOADU((U256*)pIn2_256);

                // apply 256bit unaligned operations
                do {
                    STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), m1));

                    pOut_256 += NUM_LOOPS_UNROLLED;
                    pIn1_256 += NUM_LOOPS_UNROLLED;

                } while (pOut_256 < pEnd_256);

                // update pointers to last location of wide pointers
                pDataIn1 = (T*)pIn1_256;
                pDataOut = (T*)pOut_256;
            }
            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++) {
                pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
            }
        }
        break;
    }
    default:
        printf("**error - impossible scalar mode\n");
    }
}

//=====================================================================================================
// symmetric -- arg1 and arg2 can be swapped and the operation will return the same result (like addition or multiplication)
template<typename T, typename U256, const T MATH_OP(T, T), const U256 MATH_OP256(U256, U256)>
inline void SimpleMathOpFastSymmetric(void* pDataIn1X, void* pDataIn2X, void* pDataOutX, int64_t datalen, int32_t scalarMode) {
    T* pDataOut = (T*)pDataOutX;
    T* pDataIn1 = (T*)pDataIn1X;
    T* pDataIn2 = (T*)pDataIn2X;

    const int64_t NUM_LOOPS_UNROLLED = 1;
    const int64_t chunkSize = NUM_LOOPS_UNROLLED * (sizeof(U256) / sizeof(T));
    int64_t perReg = sizeof(U256) / sizeof(T);

    LOGGING("mathopfast datalen %llu  chunkSize %llu  perReg %llu\n", datalen, chunkSize, perReg);

    switch (scalarMode) {
    case SCALAR_MODE::NO_SCALARS:
    {
        if (datalen >= chunkSize) {
            T* pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
            U256* pEnd_256 = (U256*)pEnd;
            U256* pIn1_256 = (U256*)pDataIn1;
            U256* pIn2_256 = (U256*)pDataIn2;
            U256* pOut_256 = (U256*)pDataOut;

            do {
                // clang requires LOADU on last operand
#ifdef RT_COMPILER_MSVC
                STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), *pIn2_256));
#else
                STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256), LOADU(pIn2_256)));
#endif
                pOut_256 += NUM_LOOPS_UNROLLED;
                pIn1_256 += NUM_LOOPS_UNROLLED;
                pIn2_256 += NUM_LOOPS_UNROLLED;
            } while (pOut_256 < pEnd_256);

            // update pointers to last location of wide pointers
            pDataIn1 = (T*)pIn1_256;
            pDataIn2 = (T*)pIn2_256;
            pDataOut = (T*)pOut_256;
        }

        datalen = datalen & (chunkSize - 1);
        for (int64_t i = 0; i < datalen; i++) {
            pDataOut[i] = MATH_OP(pDataIn1[i], pDataIn2[i]);
        }

        break;
    }
    case SCALAR_MODE::FIRST_ARG_SCALAR:
    {
        // NOTE: the unrolled loop is faster
        T arg1 = *pDataIn1;

        if (datalen >= chunkSize) {
            T* pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
            U256* pEnd_256 = (U256*)pEnd;
            U256* pIn1_256 = (U256*)pDataIn1;
            U256* pIn2_256 = (U256*)pDataIn2;
            U256* pOut_256 = (U256*)pDataOut;

            const U256 m0 = LOADU(pIn1_256);

            do {
#ifdef RT_COMPILER_MSVC
                STOREU(pOut_256, MATH_OP256(m0, *pIn2_256));
#else
                STOREU(pOut_256, MATH_OP256(m0, LOADU(pIn2_256)));
#endif

                pOut_256 += NUM_LOOPS_UNROLLED;
                pIn2_256 += NUM_LOOPS_UNROLLED;

            } while (pOut_256 < pEnd_256);

            // update pointers to last location of wide pointers
            pDataIn2 = (T*)pIn2_256;
            pDataOut = (T*)pOut_256;
        }
        datalen = datalen & (chunkSize - 1);
        for (int64_t i = 0; i < datalen; i++) {
            pDataOut[i] = MATH_OP(arg1, pDataIn2[i]);
        }
        break;
    }
    case SCALAR_MODE::SECOND_ARG_SCALAR:
    {
        T arg2 = *pDataIn2;

        // Check if the output is the same as the input
        if (pDataOut == pDataIn1) {

            // align the load to 32 byte boundary
            int64_t babylen = (int64_t)pDataIn1 & 31;
            if (babylen != 0) {
                // calc how much to align data
                babylen = (32 - babylen) / sizeof(T);
                if (babylen <= datalen) {
                    for (int64_t i = 0; i < babylen; i++) {
                        pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
                    }
                    pDataIn1 += babylen;
                    datalen -= babylen;
                }
            }

            // inplace operation
            if (datalen >= chunkSize) {
                T* pEnd = &pDataIn1[chunkSize * (datalen / chunkSize)];
                U256* pEnd_256 = (U256*)pEnd;
                U256* pIn1_256 = (U256*)pDataIn1;
                U256* pIn2_256 = (U256*)pDataIn2;

                const U256 m1 = LOADU(pIn2_256);

                // apply 256bit aligned operations
                while (pIn1_256 < pEnd_256) {
                    //pin1_256 is aligned
                    STOREA(pIn1_256, MATH_OP256(m1, *pIn1_256));
                    pIn1_256++;
                }

                // update pointers to last location of wide pointers
                pDataIn1 = (T*)pIn1_256;
            }
            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++) {
                pDataIn1[i] = MATH_OP(pDataIn1[i], arg2);
            }

        }
        else {

            if (datalen >= chunkSize) {
                T* pEnd = &pDataOut[chunkSize * (datalen / chunkSize)];
                U256* pEnd_256 = (U256*)pEnd;
                U256* pIn1_256 = (U256*)pDataIn1;
                U256* pIn2_256 = (U256*)pDataIn2;
                U256* pOut_256 = (U256*)pDataOut;

                const U256 m1 = LOADU((U256*)pIn2_256);

                // apply 256bit unaligned operations
                do {
#ifdef RT_COMPILER_MSVC
                    STOREU(pOut_256, MATH_OP256(m1, *pIn1_256));
#else
                    STOREU(pOut_256, MATH_OP256(m1, LOADU(pIn1_256)));
#endif
                    pOut_256 += NUM_LOOPS_UNROLLED;
                    pIn1_256 += NUM_LOOPS_UNROLLED;

                } while (pOut_256 < pEnd_256);

                // update pointers to last location of wide pointers
                pDataIn1 = (T*)pIn1_256;
                pDataOut = (T*)pOut_256;
            }
            datalen = datalen & (chunkSize - 1);
            for (int64_t i = 0; i < datalen; i++) {
                pDataOut[i] = MATH_OP(pDataIn1[i], arg2);
            }
        }
        break;
    }
    default:
        printf("**error - impossible scalar mode\n");
    }
}


extern "C"
ANY_TWO_FUNC GetSimpleMathOpFast(int func, int atopInType1, int atopInType2, int* wantedOutType) {
    LOGGING("GetSimpleMathOpFastFunc %d %d\n", atopInType1, func);

    switch (func) {
    case MATH_OPERATION::ADD:
        *wantedOutType = atopInType1;
        switch (*wantedOutType) {
        case ATOP_BOOL:   return SimpleMathOpFastSymmetric<int8_t, __m256i, OrOp<int8_t>, OR_OP_256>;
        case ATOP_FLOAT:  return SimpleMathOpFastSymmetric<float, __m256, AddOp<float>, ADD_OP_256f32>;
        case ATOP_DOUBLE: return SimpleMathOpFastSymmetric<double, __m256d, AddOp<double>, ADD_OP_256f64>;
            // proof of concept for i32 addition loop
        case ATOP_INT32:  return SimpleMathOpFastSymmetric<int32_t, __m256i, AddOp<int32_t>, ADD_OP_256i32>;
        case ATOP_INT64:  return SimpleMathOpFastSymmetric<int64_t, __m256i, AddOp<int64_t>, ADD_OP_256i64>;
        case ATOP_INT16:  return SimpleMathOpFastSymmetric<int16_t, __m256i, AddOp<int16_t>, ADD_OP_256i16>;
        case ATOP_INT8:   return SimpleMathOpFastSymmetric<int8_t, __m256i, AddOp<int8_t>, ADD_OP_256i8>;
        }
        return NULL;

    case MATH_OPERATION::MUL:
        *wantedOutType = atopInType1;
        switch (*wantedOutType) {
        case ATOP_BOOL:   return SimpleMathOpFastSymmetric<int8_t, __m256i, AndOp<int8_t>, AND_OP_256>;
        case ATOP_FLOAT:  return SimpleMathOpFastSymmetric<float, __m256, MulOp<float>, MUL_OP_256f32>;
        case ATOP_DOUBLE: return SimpleMathOpFastSymmetric<double, __m256d, MulOp<double>, MUL_OP_256f64>;
        case ATOP_INT32:  return SimpleMathOpFastSymmetric<int32_t, __m256i, MulOp<int32_t>, MUL_OP_256i32>;

            //CASE_ATOP_INT64:  return SimpleMathOpFast<int64_t, __m256i, MulOp<int64_t>, MUL_OP_256i64>;
        case ATOP_INT16:  return SimpleMathOpFastSymmetric<int16_t, __m256i, MulOp<int16_t>, MUL_OP_256i16>;

            // Below the intrinsic to multiply is slower so we disabled it (really wants 32bit -> 64bit)
            //CASE_ATOP_UINT32:  return SimpleMathOpFastMul<UINT32, __m256i>;
            // TODO: 64bit multiply can be done with algo..
            // lo1 * lo2 + (lo1 * hi2) << 32 + (hi1 *lo2) << 32)
        case ATOP_UINT64: return SimpleMathOpFastSymmetric<uint64_t, __m256i, MulOp<uint64_t>, MUL_OP_256u64>;
        }
        return NULL;

    case MATH_OPERATION::SUB:
        *wantedOutType = atopInType1;
        switch (*wantedOutType) {
        case ATOP_FLOAT:  return SimpleMathOpFast<float, __m256, SubOp<float>, SUB_OP_256f32>;
        case ATOP_DOUBLE: return SimpleMathOpFast<double, __m256d, SubOp<double>, SUB_OP_256f64>;
        case ATOP_INT32:  return SimpleMathOpFast<int32_t, __m256i, SubOp<int32_t>, SUB_OP_256i32>;
        case ATOP_INT64:  return SimpleMathOpFast<int64_t, __m256i, SubOp<int64_t>, SUB_OP_256i64>;
        case ATOP_INT16:  return SimpleMathOpFast<int16_t, __m256i, SubOp<int16_t>, SUB_OP_256i16>;
        case ATOP_INT8:   return SimpleMathOpFast<int8_t, __m256i, SubOp<int8_t>, SUB_OP_256i8>;
        }
        return NULL;
    }
    return NULL;
}


#if defined(__clang__)
#pragma clang attribute pop
#endif
