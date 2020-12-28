#include "common_inc.h"
#include <cmath>
#include "invalids.h"
#include "thread"
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

#if !RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED
// MSVC compiler by default assumed unaligned loads
#define LOADU(X) *(X)

#else
static const inline __m256d LOADU(__m256d* x) { return _mm256_loadu_pd((double const*)x); };
static const inline __m256 LOADU(__m256* x) { return _mm256_loadu_ps((float const*)x); };
static const inline __m256i LOADU(__m256i* x) { return _mm256_loadu_si256((__m256i const*)x); };
#endif


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


template<typename T> static const inline long double LOG_OP(long double x) { return logl(x); }
template<typename T> static const inline double LOG_OP(double x) { return log(x); }
template<typename T> static const inline float LOG_OP(float x) { return logf(x); }

template<typename T> static const inline long double LOG2_OP(long double x) { return log2l(x); }
template<typename T> static const inline double LOG2_OP(double x) { return log2(x); }
template<typename T> static const inline float LOG2_OP(float x) { return log2f(x); }

template<typename T> static const inline long double LOG10_OP(long double x) { return log10l(x); }
template<typename T> static const inline double LOG10_OP(double x) { return log10(x); }
template<typename T> static const inline float LOG10_OP(float x) { return log10f(x); }

template<typename T> static const inline long double LOG1P_OP(long double x) { return log1pl(x); }
template<typename T> static const inline double LOG1P_OP(double x) { return log1p(x); }
template<typename T> static const inline float LOG1P_OP(float x) { return log1pf(x); }

template<typename T> static const inline long double EXP_OP(long double x) { return expl(x); }
template<typename T> static const inline double EXP_OP(double x) { return exp(x); }
template<typename T> static const inline float EXP_OP(float x) { return expf(x); }

template<typename T> static const inline long double EXP2_OP(long double x) { return exp2l(x); }
template<typename T> static const inline double EXP2_OP(double x) { return exp2(x); }
template<typename T> static const inline float EXP2_OP(float x) { return exp2f(x); }

template<typename T> static const inline long double EXPM1_OP(long double x) { return expm1l(x); }
template<typename T> static const inline double EXPM1_OP(double x) { return expm1(x); }
template<typename T> static const inline float EXPM1_OP(float x) { return expm1f(x); }

template<typename T> static const inline long double CBRT_OP(long double x) { return cbrtl(x); }
template<typename T> static const inline double CBRT_OP(double x) { return cbrt(x); }
template<typename T> static const inline float CBRT_OP(float x) { return cbrtf(x); }

template<typename T> static const inline long double SIN_OP(long double x) { return sinl(x); }
template<typename T> static const inline double SIN_OP(double x) { return sin(x); }
template<typename T> static const inline float SIN_OP(float x) { return sinf(x); }
template<typename T> static const inline long double COS_OP(long double x) { return cosl(x); }
template<typename T> static const inline double COS_OP(double x) { return cos(x); }
template<typename T> static const inline float COS_OP(float x) { return cosf(x); }
template<typename T> static const inline long double TAN_OP(long double x) { return tanl(x); }
template<typename T> static const inline double TAN_OP(double x) { return tan(x); }
template<typename T> static const inline float TAN_OP(float x) { return tanf(x); }

template<typename T> static const inline long double ASIN_OP(long double x) { return asinl(x); }
template<typename T> static const inline double ASIN_OP(double x) { return asin(x); }
template<typename T> static const inline float ASIN_OP(float x) { return asinf(x); }
template<typename T> static const inline long double ACOS_OP(long double x) { return acosl(x); }
template<typename T> static const inline double ACOS_OP(double x) { return acos(x); }
template<typename T> static const inline float ACOS_OP(float x) { return acosf(x); }
template<typename T> static const inline long double ATAN_OP(long double x) { return atanl(x); }
template<typename T> static const inline double ATAN_OP(double x) { return atan(x); }
template<typename T> static const inline float ATAN_OP(float x) { return atanf(x); }

template<typename T> static const inline long double SINH_OP(long double x) { return sinhl(x); }
template<typename T> static const inline double SINH_OP(double x) { return sinh(x); }
template<typename T> static const inline float SINH_OP(float x) { return sinhf(x); }
template<typename T> static const inline long double COSH_OP(long double x) { return coshl(x); }
template<typename T> static const inline double COSH_OP(double x) { return cosh(x); }
template<typename T> static const inline float COSH_OP(float x) { return coshf(x); }
template<typename T> static const inline long double TANH_OP(long double x) { return tanhl(x); }
template<typename T> static const inline double TANH_OP(double x) { return tanh(x); }
template<typename T> static const inline float TANH_OP(float x) { return tanhf(x); }

template<typename T> static const inline long double ASINH_OP(long double x) { return asinhl(x); }
template<typename T> static const inline double ASINH_OP(double x) { return asinh(x); }
template<typename T> static const inline float ASINH_OP(float x) { return asinhf(x); }
template<typename T> static const inline long double ACOSH_OP(long double x) { return acoshl(x); }
template<typename T> static const inline double ACOSH_OP(double x) { return acosh(x); }
template<typename T> static const inline float ACOSH_OP(float x) { return acoshf(x); }
template<typename T> static const inline long double ATANH_OP(long double x) { return atanhl(x); }
template<typename T> static const inline double ATANH_OP(double x) { return atanh(x); }
template<typename T> static const inline float ATANH_OP(float x) { return atanhf(x); }


#if defined(RT_COMPILER_MSVC)

template<typename T> static const inline __m256  SIN_OP_256(__m256 x) { return _mm256_sin_ps(x); }
template<typename T> static const inline __m256d SIN_OP_256(__m256d x) { return _mm256_sin_pd(x); }
template<typename T> static const inline __m256  COS_OP_256(__m256 x) { return _mm256_cos_ps(x); }
template<typename T> static const inline __m256d COS_OP_256(__m256d x) { return _mm256_cos_pd(x); }
template<typename T> static const inline __m256  TAN_OP_256(__m256 x) { return _mm256_tan_ps(x); }
template<typename T> static const inline __m256d TAN_OP_256(__m256d x) { return _mm256_tan_pd(x); }

template<typename T> static const inline __m256  ASIN_OP_256(__m256 x) { return _mm256_asin_ps(x); }
template<typename T> static const inline __m256d ASIN_OP_256(__m256d x) { return _mm256_asin_pd(x); }
template<typename T> static const inline __m256  ACOS_OP_256(__m256 x) { return _mm256_acos_ps(x); }
template<typename T> static const inline __m256d ACOS_OP_256(__m256d x) { return _mm256_acos_pd(x); }
template<typename T> static const inline __m256  ATAN_OP_256(__m256 x) { return _mm256_atan_ps(x); }
template<typename T> static const inline __m256d ATAN_OP_256(__m256d x) { return _mm256_atan_pd(x); }

template<typename T> static const inline __m256  SINH_OP_256(__m256 x) { return _mm256_sinh_ps(x); }
template<typename T> static const inline __m256d SINH_OP_256(__m256d x) { return _mm256_sinh_pd(x); }
template<typename T> static const inline __m256  COSH_OP_256(__m256 x) { return _mm256_cosh_ps(x); }
template<typename T> static const inline __m256d COSH_OP_256(__m256d x) { return _mm256_cosh_pd(x); }
template<typename T> static const inline __m256  TANH_OP_256(__m256 x) { return _mm256_tanh_ps(x); }
template<typename T> static const inline __m256d TANH_OP_256(__m256d x) { return _mm256_tanh_pd(x); }

template<typename T> static const inline __m256  ASINH_OP_256(__m256 x) { return _mm256_asinh_ps(x); }
template<typename T> static const inline __m256d ASINH_OP_256(__m256d x) { return _mm256_asinh_pd(x); }
template<typename T> static const inline __m256  ACOSH_OP_256(__m256 x) { return _mm256_acosh_ps(x); }
template<typename T> static const inline __m256d ACOSH_OP_256(__m256d x) { return _mm256_acosh_pd(x); }
template<typename T> static const inline __m256  ATANH_OP_256(__m256 x) { return _mm256_atanh_ps(x); }
template<typename T> static const inline __m256d ATANH_OP_256(__m256d x) { return _mm256_atanh_pd(x); }

template<typename T> static const inline __m256  LOG_OP_256(__m256 x) { return _mm256_log_ps(x); }
template<typename T> static const inline __m256d LOG_OP_256(__m256d x) { return _mm256_log_pd(x); }
template<typename T> static const inline __m256  LOG1P_OP_256(__m256 x) { return _mm256_log1p_ps(x); }
template<typename T> static const inline __m256d LOG1P_OP_256(__m256d x) { return _mm256_log1p_pd(x); }
template<typename T> static const inline __m256  LOG10_OP_256(__m256 x) { return _mm256_log10_ps(x); }
template<typename T> static const inline __m256d LOG10_OP_256(__m256d x) { return _mm256_log10_pd(x); }
template<typename T> static const inline __m256  LOG2_OP_256(__m256 x) { return _mm256_log2_ps(x); }
template<typename T> static const inline __m256d LOG2_OP_256(__m256d x) { return _mm256_log2_pd(x); }

template<typename T> static const inline __m256  EXP_OP_256(__m256 x) { return _mm256_exp_ps(x); }
template<typename T> static const inline __m256d EXP_OP_256(__m256d x) { return _mm256_exp_pd(x); }
template<typename T> static const inline __m256  EXP2_OP_256(__m256 x) { return _mm256_exp2_ps(x); }
template<typename T> static const inline __m256d EXP2_OP_256(__m256d x) { return _mm256_exp2_pd(x); }
template<typename T> static const inline __m256  EXPM1_OP_256(__m256 x) { return _mm256_expm1_ps(x); }
template<typename T> static const inline __m256d EXPM1_OP_256(__m256d x) { return _mm256_expm1_pd(x); }

template<typename T> static const inline __m256  CBRT_OP_256(__m256 x) { return _mm256_cbrt_ps(x); }
template<typename T> static const inline __m256d CBRT_OP_256(__m256d x) { return _mm256_cbrt_pd(x); }

#endif

#if defined(__GNUC__)
// May require -lm for linker

extern "C" {
    __m256d _ZGVdN4v_cos(__m256d x);
    __m256d _ZGVdN4v_exp(__m256d x);
    __m256d _ZGVdN4v_log(__m256d x);
    __m256d _ZGVdN4v_sin(__m256d x);
    __m256d _ZGVdN4vv_pow(__m256d x, __m256d y);
    void    _ZGVdN4vvv_sincos(__m256d x, __m256i ptrs, __m256i ptrc);

    __m256  _ZGVdN8v_cosf(__m256 x);
    __m256  _ZGVdN8v_expf(__m256 x);
    __m256  _ZGVdN8v_logf(__m256 x);
    __m256  _ZGVdN8v_sinf(__m256 x);
    __m256  _ZGVdN8vv_powf(__m256 x, __m256 y);
    void    _ZGVdN8vvv_sincosf(__m256 x, __m256i ptrs_lo, __m256i ptrs_hi, __m256i ptrc_lo, __m256i ptrc_hi);
}

template<typename T> static const inline __m256  SIN_OP_256(__m256 x) { return _ZGVdN8v_sinf(x); }
template<typename T> static const inline __m256d SIN_OP_256(__m256d x) { return _ZGVdN4v_sin(x); }

template<typename T> static const inline __m256  COS_OP_256(__m256 x) { return _ZGVdN8v_cosf(x); }
template<typename T> static const inline __m256d COS_OP_256(__m256d x) { return _ZGVdN4v_cos(x); }

#endif


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
            STOREU(pOut_256, MATH_OP256(LOADU(pIn1_256)));
            pIn1_256 += 1;
            pOut_256 += 1;

        } while (pOut_256 < pEnd_256);

        // update thin pointers to last location of wide pointers
        pIn = (T*)pIn1_256;
        pOut = (T*)pOut_256;

        // Slow loop, handle 1 at a time
        while (pOut != pLastOut) {
            *pOut++ = MATH_OP(*pIn++);
        }
        return;
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
template<typename T>
static void UnaryOpSlow_CBRT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(CBRT_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_LOG(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(LOG_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_LOG2(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(LOG2_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_LOG10(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(LOG10_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_EXP(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(EXP_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlow_EXP2(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlow<T, const T(*)(T)>(EXP2_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_CBRT(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(CBRT_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_LOG(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(LOG_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_LOG2(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(LOG2_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_LOG10(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(LOG10_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_EXP(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(EXP_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}
//-------------------------------------------------------------------
template<typename T>
static void UnaryOpSlowDouble_EXP2(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
    return UnaryOpSlowDouble<T, const double(*)(double)>(EXP2_OP<double>, pDataIn1, pDataOut, len, strideIn, strideOut);
}

/*
 * Constants used in vector implementation of exp(x)
 */
#define NPY_RINT_CVT_MAGICf 0x1.800000p+23f
#define NPY_CODY_WAITE_LOGE_2_HIGHf -6.93145752e-1f
#define NPY_CODY_WAITE_LOGE_2_LOWf -1.42860677e-6f
#define NPY_COEFF_P0_EXPf 9.999999999980870924916e-01f
#define NPY_COEFF_P1_EXPf 7.257664613233124478488e-01f
#define NPY_COEFF_P2_EXPf 2.473615434895520810817e-01f
#define NPY_COEFF_P3_EXPf 5.114512081637298353406e-02f
#define NPY_COEFF_P4_EXPf 6.757896990527504603057e-03f
#define NPY_COEFF_P5_EXPf 5.082762527590693718096e-04f
#define NPY_COEFF_Q0_EXPf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_EXPf -2.742335390411667452936e-01f
#define NPY_COEFF_Q2_EXPf 2.159509375685829852307e-02f

 /*
  * Constants used in vector implementation of log(x)
  */
#define NPY_COEFF_P0_LOGf 0.000000000000000000000e+00f
#define NPY_COEFF_P1_LOGf 9.999999999999998702752e-01f
#define NPY_COEFF_P2_LOGf 2.112677543073053063722e+00f
#define NPY_COEFF_P3_LOGf 1.480000633576506585156e+00f
#define NPY_COEFF_P4_LOGf 3.808837741388407920751e-01f
#define NPY_COEFF_P5_LOGf 2.589979117907922693523e-02f
#define NPY_COEFF_Q0_LOGf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_LOGf 2.612677543073109236779e+00f
#define NPY_COEFF_Q2_LOGf 2.453006071784736363091e+00f
#define NPY_COEFF_Q3_LOGf 9.864942958519418960339e-01f
#define NPY_COEFF_Q4_LOGf 1.546476374983906719538e-01f
#define NPY_COEFF_Q5_LOGf 5.875095403124574342950e-03f
  /*
   * Constants used in vector implementation of sinf/cosf(x)
   */
#define NPY_TWO_O_PIf 0x1.45f306p-1f
#define NPY_CODY_WAITE_PI_O_2_HIGHf -0x1.921fb0p+00f
#define NPY_CODY_WAITE_PI_O_2_MEDf -0x1.5110b4p-22f
#define NPY_CODY_WAITE_PI_O_2_LOWf -0x1.846988p-48f
#define NPY_COEFF_INVF0_COSINEf 0x1.000000p+00f
#define NPY_COEFF_INVF2_COSINEf -0x1.000000p-01f
#define NPY_COEFF_INVF4_COSINEf 0x1.55553cp-05f
#define NPY_COEFF_INVF6_COSINEf -0x1.6c06dcp-10f
#define NPY_COEFF_INVF8_COSINEf 0x1.98e616p-16f
#define NPY_COEFF_INVF3_SINEf -0x1.555556p-03f
#define NPY_COEFF_INVF5_SINEf 0x1.11119ap-07f
#define NPY_COEFF_INVF7_SINEf -0x1.a06bbap-13f
#define NPY_COEFF_INVF9_SINEf 0x1.7d3bbcp-19f



/*
 * Vectorized implementation of log using AVX2 and AVX512
 * 1) if x < 0.0f; return -NAN (invalid input)
 * 2) Range reduction: y = x/2^k;
 *      a) y = normalized mantissa, k is the exponent (0.5 <= y < 1)
 * 3) Compute log(y) = P/Q, ratio of 2 polynomials P and Q
 *      b) P = 5th order and Q = 5th order polynomials obtained from Remez's
 *      algorithm (mini-max polynomial approximation)
 * 5) Compute log(x) = log(y) + k*ln(2)
 * 6) Max ULP error measured across all 32-bit FP's = 3.83 (x = 0x3f486945)
 * 7) Max relative error measured across all 32-bit FP's = 2.359E-07 (for same
 * x = 0x3f486945)
 */

//static NPY_GCC_OPT_3 NPY_GCC_TARGET_@ISA@ void
//@ISA@_log_FLOAT(npy_float* op,
//    npy_float* ip,
//    const npy_intp array_size,
//    const npy_intp steps)
//{
//    const npy_intp stride = steps / (npy_intp)sizeof(npy_float);
//    const npy_int num_lanes = @BYTES@ / (npy_intp)sizeof(npy_float);
//
//    /*
//     * Note: while generally indices are npy_intp, we ensure that our maximum index
//     * will fit in an int32 as a precondition for this function via
//     * IS_OUTPUT_BLOCKABLE_UNARY
//     */
//    npy_int32 indexarr[16];
//    for (npy_int32 ii = 0; ii < 16; ii++) {
//        indexarr[ii] = ii * stride;
//    }
//
//    /* Load up frequently used constants */
//    @vtype@ log_p0 = _mm@vsize@_set1_ps(NPY_COEFF_P0_LOGf);
//    @vtype@ log_p1 = _mm@vsize@_set1_ps(NPY_COEFF_P1_LOGf);
//    @vtype@ log_p2 = _mm@vsize@_set1_ps(NPY_COEFF_P2_LOGf);
//    @vtype@ log_p3 = _mm@vsize@_set1_ps(NPY_COEFF_P3_LOGf);
//    @vtype@ log_p4 = _mm@vsize@_set1_ps(NPY_COEFF_P4_LOGf);
//    @vtype@ log_p5 = _mm@vsize@_set1_ps(NPY_COEFF_P5_LOGf);
//    @vtype@ log_q0 = _mm@vsize@_set1_ps(NPY_COEFF_Q0_LOGf);
//    @vtype@ log_q1 = _mm@vsize@_set1_ps(NPY_COEFF_Q1_LOGf);
//    @vtype@ log_q2 = _mm@vsize@_set1_ps(NPY_COEFF_Q2_LOGf);
//    @vtype@ log_q3 = _mm@vsize@_set1_ps(NPY_COEFF_Q3_LOGf);
//    @vtype@ log_q4 = _mm@vsize@_set1_ps(NPY_COEFF_Q4_LOGf);
//    @vtype@ log_q5 = _mm@vsize@_set1_ps(NPY_COEFF_Q5_LOGf);
//    @vtype@ loge2 = _mm@vsize@_set1_ps(NPY_LOGE2f);
//    @vtype@ nan = _mm@vsize@_set1_ps(NPY_NANF);
//    @vtype@ neg_nan = _mm@vsize@_set1_ps(-NPY_NANF);
//    @vtype@ neg_inf = _mm@vsize@_set1_ps(-NPY_INFINITYF);
//    @vtype@ inf = _mm@vsize@_set1_ps(NPY_INFINITYF);
//    @vtype@ zeros_f = _mm@vsize@_set1_ps(0.0f);
//    @vtype@ ones_f = _mm@vsize@_set1_ps(1.0f);
//    @vtype@i vindex = _mm@vsize@_loadu_si@vsize@((@vtype@i*)indexarr);
//    @vtype@ poly, num_poly, denom_poly, exponent;
//
//    @mask@ inf_mask, nan_mask, sqrt2_mask, zero_mask, negx_mask;
//    @mask@ invalid_mask = @isa@_get_partial_load_mask_ps(0, num_lanes);
//    @mask@ divide_by_zero_mask = invalid_mask;
//    @mask@ load_mask = @isa@_get_full_load_mask_ps();
//    npy_intp num_remaining_elements = array_size;
//
//    while (num_remaining_elements > 0) {
//
//        if (num_remaining_elements < num_lanes) {
//            load_mask = @isa@_get_partial_load_mask_ps(num_remaining_elements,
//                num_lanes);
//        }
//
//        @vtype@ x_in;
//        if (stride == 1) {
//            x_in = @isa@_masked_load_ps(load_mask, ip);
//        }
//        else {
//            x_in = @isa@_masked_gather_ps(zeros_f, ip, vindex, load_mask);
//        }
//
//        negx_mask = _mm@vsize@_cmp_ps@vsub@(x_in, zeros_f, _CMP_LT_OQ);
//        zero_mask = _mm@vsize@_cmp_ps@vsub@(x_in, zeros_f, _CMP_EQ_OQ);
//        inf_mask = _mm@vsize@_cmp_ps@vsub@(x_in, inf, _CMP_EQ_OQ);
//        nan_mask = _mm@vsize@_cmp_ps@vsub@(x_in, x_in, _CMP_NEQ_UQ);
//        divide_by_zero_mask = @or_masks@(divide_by_zero_mask,
//            @and_masks@(zero_mask, load_mask));
//        invalid_mask = @or_masks@(invalid_mask, negx_mask);
//
//        @vtype@ x = @isa@_set_masked_lanes_ps(x_in, zeros_f, negx_mask);
//
//        /* set x = normalized mantissa */
//        exponent = @isa@_get_exponent(x);
//        x = @isa@_get_mantissa(x);
//
//        /* if x < sqrt(2) {exp = exp-1; x = 2*x} */
//        sqrt2_mask = _mm@vsize@_cmp_ps@vsub@(x, _mm@vsize@_set1_ps(NPY_SQRT1_2f), _CMP_LE_OQ);
//        x = @isa@_blend(x, _mm@vsize@_add_ps(x, x), sqrt2_mask);
//        exponent = @isa@_blend(exponent,
//            _mm@vsize@_sub_ps(exponent, ones_f), sqrt2_mask);
//
//        /* x = x - 1 */
//        x = _mm@vsize@_sub_ps(x, ones_f);
//
//        /* Polynomial approximation for log(1+x) */
//        num_poly = @fmadd@(log_p5, x, log_p4);
//        num_poly = @fmadd@(num_poly, x, log_p3);
//        num_poly = @fmadd@(num_poly, x, log_p2);
//        num_poly = @fmadd@(num_poly, x, log_p1);
//        num_poly = @fmadd@(num_poly, x, log_p0);
//        denom_poly = @fmadd@(log_q5, x, log_q4);
//        denom_poly = @fmadd@(denom_poly, x, log_q3);
//        denom_poly = @fmadd@(denom_poly, x, log_q2);
//        denom_poly = @fmadd@(denom_poly, x, log_q1);
//        denom_poly = @fmadd@(denom_poly, x, log_q0);
//        poly = _mm@vsize@_div_ps(num_poly, denom_poly);
//        poly = @fmadd@(exponent, loge2, poly);
//
//        /*
//         * x < 0.0f; return -NAN
//         * x = +/- NAN; return NAN
//         * x = 0.0f; return -INF
//         */
//        poly = @isa@_set_masked_lanes_ps(poly, nan, nan_mask);
//        poly = @isa@_set_masked_lanes_ps(poly, neg_nan, negx_mask);
//        poly = @isa@_set_masked_lanes_ps(poly, neg_inf, zero_mask);
//        poly = @isa@_set_masked_lanes_ps(poly, inf, inf_mask);
//
//        @masked_store@(op, @cvtps_epi32@(load_mask), poly);
//
//        ip += num_lanes * stride;
//        op += num_lanes;
//        num_remaining_elements -= num_lanes;
//    }
//
//    if (@mask_to_int@(invalid_mask)) {
//        npy_set_floatstatus_invalid();
//    }
//    if (@mask_to_int@(divide_by_zero_mask)) {
//        npy_set_floatstatus_divbyzero();
//    }
//}
//

extern "C"
UNARY_FUNC GetTrigOpFast(int func, int atopInType1, int* wantedOutType) {

    LOGGING("Looking for func %d  type:%d \n", func, atopInType1);

    switch (func) {
    case TRIG_OPERATION::SIN:
        *wantedOutType = atopInType1;
#if !defined(__clang__)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, SIN_OP<float>, SIN_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, SIN_OP<double>, SIN_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::COS:
        *wantedOutType = atopInType1;
#if !defined(__clang__)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, COS_OP<float>, COS_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, COS_OP<double>, COS_OP_256<__m256d>>;
        }
#endif
        break;
    case TRIG_OPERATION::TAN:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, TAN_OP<float>, TAN_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, TAN_OP<double>, TAN_OP_256<__m256d>>;
        }
#endif
        break;

    //=========================================
    case TRIG_OPERATION::ASIN:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ASIN_OP<float>, ASIN_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ASIN_OP<double>, ASIN_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::ACOS:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ACOS_OP<float>, ACOS_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ACOS_OP<double>, ACOS_OP_256<__m256d>>;
        }
#endif
        break;
    case TRIG_OPERATION::ATAN:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ATAN_OP<float>, ATAN_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ATAN_OP<double>, ATAN_OP_256<__m256d>>;
        }
#endif
        break;


    //=========================================
    case TRIG_OPERATION::SINH:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, SINH_OP<float>, SINH_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, SINH_OP<double>, SINH_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::COSH:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, COSH_OP<float>, COSH_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, COSH_OP<double>, COSH_OP_256<__m256d>>;
        }
#endif
        break;
    case TRIG_OPERATION::TANH:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, TANH_OP<float>, TANH_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, TANH_OP<double>, TANH_OP_256<__m256d>>;
        }
#endif
        break;


    //=========================================
    case TRIG_OPERATION::ASINH:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ASINH_OP<float>, ASINH_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ASINH_OP<double>, ASINH_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::ACOSH:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ACOSH_OP<float>, ACOSH_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ACOSH_OP<double>, ACOSH_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::ATANH:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
// TJD NOTE: crashes on 0, MSVC compiler believed to be buggy on AVX2 version of ATANH
//        switch (atopInType1) {
//        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, ATANH_OP<float>, ATANH_OP_256<__m256>>;
//        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, ATANH_OP<double>, ATANH_OP_256<__m256d>>;
//        }
#endif
        break;

    case TRIG_OPERATION::EXP:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, EXP_OP<float>, EXP_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, EXP_OP<double>, EXP_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::EXP2:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, EXP2_OP<float>, EXP2_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, EXP2_OP<double>, EXP2_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::EXPM1:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, EXPM1_OP<float>, EXPM1_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, EXPM1_OP<double>, EXPM1_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::LOG:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, LOG_OP<float>, LOG_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, LOG_OP<double>, LOG_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::LOG2:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, LOG2_OP<float>, LOG2_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, LOG2_OP<double>, LOG2_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::LOG10:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, LOG10_OP<float>, LOG10_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, LOG10_OP<double>, LOG10_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::LOG1P:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, LOG1P_OP<float>, LOG1P_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, LOG1P_OP<double>, LOG1P_OP_256<__m256d>>;
        }
#endif
        break;

    case TRIG_OPERATION::CBRT:
        *wantedOutType = atopInType1;
#if defined(RT_COMPILER_MSVC)
        switch (atopInType1) {
        case ATOP_FLOAT:  return UnaryOpFast<float, __m256, CBRT_OP<float>, CBRT_OP_256<__m256>>;
        case ATOP_DOUBLE: return UnaryOpFast<double, __m256d, CBRT_OP<double>, CBRT_OP_256<__m256d>>;
        }
#endif
        break;

    }
    return NULL;
}

extern "C"
UNARY_FUNC GetTrigOpSlow(int func, int numpyInType1, int* wantedOutType) {
    LOGGING("Looking for func slow %d  type %d  \n", func, numpyInType1);

    switch (func) {
    case TRIG_OPERATION::CBRT:
        *wantedOutType = ATOP_DOUBLE;
        if (numpyInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        if (numpyInType1 == ATOP_LONGDOUBLE) {
            *wantedOutType = ATOP_LONGDOUBLE;
        }
        switch (numpyInType1) {
        case ATOP_INT32:   return UnaryOpSlowDouble_CBRT<int32_t>;
        case ATOP_UINT32:  return UnaryOpSlowDouble_CBRT<uint32_t>;
        case ATOP_INT64:   return UnaryOpSlowDouble_CBRT<int64_t>;
        case ATOP_UINT64:  return UnaryOpSlowDouble_CBRT<int64_t>;
        }
        break;

    case TRIG_OPERATION::LOG:
        *wantedOutType = ATOP_DOUBLE;
        if (numpyInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        if (numpyInType1 == ATOP_LONGDOUBLE) {
            *wantedOutType = ATOP_LONGDOUBLE;
        }
        switch (numpyInType1) {
        case ATOP_FLOAT: return UnaryOpSlow_LOG<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_LOG<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_LOG<long double>;
        case ATOP_INT32:   return UnaryOpSlowDouble_LOG<int32_t>;
        case ATOP_UINT32:  return UnaryOpSlowDouble_LOG<uint32_t>;
        case ATOP_INT64:   return UnaryOpSlowDouble_LOG<int64_t>;
        case ATOP_UINT64:  return UnaryOpSlowDouble_LOG<int64_t>;
        }
        break;

    case TRIG_OPERATION::LOG10:
        *wantedOutType = ATOP_DOUBLE;
        if (numpyInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        if (numpyInType1 == ATOP_LONGDOUBLE) {
            *wantedOutType = ATOP_LONGDOUBLE;
        }
        switch (numpyInType1) {
        case ATOP_FLOAT: return UnaryOpSlow_LOG10<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_LOG10<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_LOG10<long double>;
        case ATOP_INT32:  return UnaryOpSlowDouble_LOG10<int32_t>;
        case ATOP_UINT32:  return UnaryOpSlowDouble_LOG10<uint32_t>;
        case ATOP_INT64:  return UnaryOpSlowDouble_LOG10<int64_t>;
        case ATOP_UINT64:  return UnaryOpSlowDouble_LOG10<int64_t>;
        }
        break;

    case TRIG_OPERATION::EXP:
        *wantedOutType = ATOP_DOUBLE;
        if (numpyInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        if (numpyInType1 == ATOP_LONGDOUBLE) {
            *wantedOutType = ATOP_LONGDOUBLE;
        }
        switch (numpyInType1) {
        case ATOP_FLOAT: return UnaryOpSlow_EXP<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_EXP<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_EXP<long double>;
        case ATOP_INT32:  return UnaryOpSlowDouble_EXP<int32_t>;
        case ATOP_UINT32:  return UnaryOpSlowDouble_EXP<uint32_t>;
        case ATOP_INT64:  return UnaryOpSlowDouble_EXP<int64_t>;
        case ATOP_UINT64:  return UnaryOpSlowDouble_EXP<int64_t>;
        }
        break;

    case TRIG_OPERATION::EXP2:
        *wantedOutType = ATOP_DOUBLE;
        if (numpyInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        if (numpyInType1 == ATOP_LONGDOUBLE) {
            *wantedOutType = ATOP_LONGDOUBLE;
        }
        switch (numpyInType1) {
        case ATOP_FLOAT: return UnaryOpSlow_EXP2<float>;
        case ATOP_DOUBLE: return UnaryOpSlow_EXP2<double>;
        case ATOP_LONGDOUBLE: return UnaryOpSlow_EXP2<long double>;
        case ATOP_INT32:  return UnaryOpSlowDouble_EXP2<int32_t>;
        case ATOP_UINT32:  return UnaryOpSlowDouble_EXP2<uint32_t>;
        case ATOP_INT64:  return UnaryOpSlowDouble_EXP2<int64_t>;
        case ATOP_UINT64:  return UnaryOpSlowDouble_EXP2<int64_t>;
        }
        break;

    }
    return NULL;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
