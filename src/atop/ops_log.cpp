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

#if !RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED
// MSVC compiler by default assumed unaligned loads
#define LOADU(X) *(X)

#else
static const inline __m256d LOADU(__m256d* x) { return _mm256_loadu_pd((double const*)x); };
static const inline __m256 LOADU(__m256* x) { return _mm256_loadu_ps((float const*)x); };
static const inline __m256i LOADU(__m256i* x) { return _mm256_loadu_si256((__m256i const*)x); };
#endif

static const inline __m256i LOADUI(__m256i* x) { return _mm256_loadu_si256((__m256i const*)x); };

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


#if defined(RT_COMPILER_MSVC)

template<typename T> static const inline __m256  LOG_OP_256(__m256 x) { return _mm256_log_ps(x); }
template<typename T> static const inline __m256d LOG_OP_256(__m256d x) { return _mm256_log_pd(x); }
#include <float.h>
#endif

#if defined(__GNUC__)
// May require -lm for linker

extern "C" {
    __m256d _ZGVdN4v_log(__m256d x);
    __m256  _ZGVdN8v_logf(__m256 x);
}

template<typename T> static const inline __m256  LOG_OP_256(__m256 x) { return _ZGVdN8v_logf(x); }
template<typename T> static const inline __m256d LOG_OP_256(__m256d x) { return _ZGVdN4v_log(x); }

#endif


#  include <fenv.h>

void npy_set_floatstatus_divbyzero(void)
{
    feraiseexcept(FE_DIVBYZERO);
}

void npy_set_floatstatus_overflow(void)
{
    feraiseexcept(FE_OVERFLOW);
}

void npy_set_floatstatus_underflow(void)
{
    feraiseexcept(FE_UNDERFLOW);
}

void npy_set_floatstatus_invalid(void)
{
    feraiseexcept(FE_INVALID);
}

//-------------------------------------------------------------------
//template<typename T>
//static void UnaryOpSlow_LOG(void* pDataIn1, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut) {
//    return UnaryOpSlow<T, const T(*)(T)>(LOG_OP<T>, pDataIn1, pDataOut, len, strideIn, strideOut);
//}


/*
 * NAN and INFINITY like macros (same behavior as glibc for NAN, same as C99
 * for INFINITY)
 *
 * XXX: I should test whether INFINITY and NAN are available on the platform
 */
static const inline float __npy_inff(void)
{
    const union { uint32_t __i; float __f; } __bint = { 0x7f800000UL };
    return __bint.__f;
}

static const inline  float __npy_nanf(void)
{
    const union { uint32_t __i; float __f; } __bint = { 0x7fc00000UL };
    return __bint.__f;
}

static const inline float __npy_pzerof(void)
{
    const union { uint32_t __i; float __f; } __bint = { 0x00000000UL };
    return __bint.__f;
}

static const inline float __npy_nzerof(void)
{
    const union { uint32_t __i; float __f; } __bint = { 0x80000000UL };
    return __bint.__f;
}

#define NPY_INFINITYF __npy_inff()
#define NPY_NANF __npy_nanf()
#define NPY_PZEROF __npy_pzerof()
#define NPY_NZEROF __npy_nzerof()

#define NPY_INFINITY ((npy_double)NPY_INFINITYF)
#define NPY_NAN ((npy_double)NPY_NANF)
#define NPY_PZERO ((npy_double)NPY_PZEROF)
#define NPY_NZERO ((npy_double)NPY_NZEROF)

#define NPY_INFINITYL ((npy_longdouble)NPY_INFINITYF)
#define NPY_NANL ((npy_longdouble)NPY_NANF)
#define NPY_PZEROL ((npy_longdouble)NPY_PZEROF)
#define NPY_NZEROL ((npy_longdouble)NPY_NZEROF)

/*
 * Useful constants
 */
#define NPY_E         2.718281828459045235360287471352662498  /* e */
#define NPY_LOG2E     1.442695040888963407359924681001892137  /* log_2 e */
#define NPY_LOG10E    0.434294481903251827651128918916605082  /* log_10 e */
#define NPY_LOGE2     0.693147180559945309417232121458176568  /* log_e 2 */
#define NPY_LOGE10    2.302585092994045684017991454684364208  /* log_e 10 */
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */
#define NPY_PI_2      1.570796326794896619231321691639751442  /* pi/2 */
#define NPY_PI_4      0.785398163397448309615660845819875721  /* pi/4 */
#define NPY_1_PI      0.318309886183790671537767526745028724  /* 1/pi */
#define NPY_2_PI      0.636619772367581343075535053490057448  /* 2/pi */
#define NPY_EULER     0.577215664901532860606512090082402431  /* Euler constant */
#define NPY_SQRT2     1.414213562373095048801688724209698079  /* sqrt(2) */
#define NPY_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

#define NPY_Ef        2.718281828459045235360287471352662498F /* e */
#define NPY_LOG2Ef    1.442695040888963407359924681001892137F /* log_2 e */
#define NPY_LOG10Ef   0.434294481903251827651128918916605082F /* log_10 e */
#define NPY_LOGE2f    0.693147180559945309417232121458176568F /* log_e 2 */
#define NPY_LOGE10f   2.302585092994045684017991454684364208F /* log_e 10 */
#define NPY_PIf       3.141592653589793238462643383279502884F /* pi */
#define NPY_PI_2f     1.570796326794896619231321691639751442F /* pi/2 */
#define NPY_PI_4f     0.785398163397448309615660845819875721F /* pi/4 */
#define NPY_1_PIf     0.318309886183790671537767526745028724F /* 1/pi */
#define NPY_2_PIf     0.636619772367581343075535053490057448F /* 2/pi */
#define NPY_EULERf    0.577215664901532860606512090082402431F /* Euler constant */
#define NPY_SQRT2f    1.414213562373095048801688724209698079F /* sqrt(2) */
#define NPY_SQRT1_2f  0.707106781186547524400844362104849039F /* 1/sqrt(2) */

#define NPY_El        2.718281828459045235360287471352662498L /* e */
#define NPY_LOG2El    1.442695040888963407359924681001892137L /* log_2 e */
#define NPY_LOG10El   0.434294481903251827651128918916605082L /* log_10 e */
#define NPY_LOGE2l    0.693147180559945309417232121458176568L /* log_e 2 */
#define NPY_LOGE10l   2.302585092994045684017991454684364208L /* log_e 10 */
#define NPY_PIl       3.141592653589793238462643383279502884L /* pi */
#define NPY_PI_2l     1.570796326794896619231321691639751442L /* pi/2 */
#define NPY_PI_4l     0.785398163397448309615660845819875721L /* pi/4 */
#define NPY_1_PIl     0.318309886183790671537767526745028724L /* 1/pi */
#define NPY_2_PIl     0.636619772367581343075535053490057448L /* 2/pi */
#define NPY_EULERl    0.577215664901532860606512090082402431L /* Euler constant */
#define NPY_SQRT2l    1.414213562373095048801688724209698079L /* sqrt(2) */
#define NPY_SQRT1_2l  0.707106781186547524400844362104849039L /* 1/sqrt(2) */

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

#define FLT_MIN          1.175494351e-38F        // min normalized positive value

static const inline int mask_to_int(__m256 _x_) { return _mm256_movemask_ps(_x_); };

static const inline __m256 set1_ps(float _x_) { return _mm256_set1_ps(_x_); };

//static const inline __mmask16
//get_full_load_mask_ps(void)
//{
//    return 0xFFFF;
//}
//
//static const inline __mmask8
//get_full_load_mask_pd(void)
//{
//    return 0xFF;
//}
//
//static const inline __mmask16
//get_partial_load_mask_ps(const int num_elem, const int total_elem)
//{
//    return (0x0001 << num_elem) - 0x0001;
//}
//
//static const inline __mmask8
//get_partial_load_mask_pd(const int num_elem, const int total_elem)
//{
//    return (0x01 << num_elem) - 0x01;
//}


static const inline __m256
get_full_load_mask_ps(void)
{
    return _mm256_set1_ps(-1.0);
}

static const inline __m256i
get_full_load_mask_pd(void)
{
    return _mm256_castpd_si256(_mm256_set1_pd(-1.0));
}

static const inline __m256
get_partial_load_mask_ps(const int num_elem, const int num_lanes)
{
    float maskint[16] = { -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
    float* addr = maskint + num_lanes - num_elem;
    return _mm256_loadu_ps(addr);
}

static const inline __m256i
get_partial_load_mask_pd(const int num_elem, const int num_lanes)
{
    int maskint[16] = { -1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1 };
    int* addr = maskint + 2 * num_lanes - 2 * num_elem;
    return _mm256_loadu_si256((__m256i*) addr);
}

static const inline __m256
masked_gather_ps(__m256 src,
    float* addr,
    __m256i vindex,
    __m256 mask)
{
    return _mm256_mask_i32gather_ps(src, addr, vindex, mask, 4);
}

static const inline __m256d
masked_gather_pd(__m256d src,
    double* addr,
    __m128i vindex,
    __m256d mask)
{
    return _mm256_mask_i32gather_pd(src, addr, vindex, mask, 8);
}

static const inline __m256
masked_load_ps(__m256 mask, float* addr)
{
    return _mm256_maskload_ps(addr, _mm256_cvtps_epi32(mask));
}

static const inline __m256d
masked_load_pd(__m256i mask, double* addr)
{
    return _mm256_maskload_pd(addr, mask);
}

static const inline __m256
fma_set_masked_lanes_ps(__m256 x, __m256 val, __m256 mask)
{
    return _mm256_blendv_ps(x, val, mask);
}

static const inline __m256d
fma_set_masked_lanes_pd(__m256d x, __m256d val, __m256d mask)
{
    return _mm256_blendv_pd(x, val, mask);
}

static const inline __m256
or_masks(__m256 x, __m256 y) { return _mm256_or_ps(x, y); }

static const inline __m256
and_masks(__m256 x, __m256 y) { return _mm256_and_ps(x, y); }

static const inline __m256
xor_masks(__m256 x, __m256 y) { return _mm256_xor_ps(x, y); }

static const inline __m256
set_masked_lanes_ps(__m256 x, __m256 val, __m256 mask)
{
    return _mm256_blendv_ps(x, val, mask);
}

static const inline __m256d
set_masked_lanes_pd(__m256d x, __m256d val, __m256d mask)
{
    return _mm256_blendv_pd(x, val, mask);
}

static const inline __m256
blend(__m256 x, __m256 y, __m256 ymask)
{
    return _mm256_blendv_ps(x, y, ymask);
}

template<const int COMP_OP>
static inline __m256
cmp_ps(__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, COMP_OP); };

static const inline __m256
add_ps(__m256 x, __m256 y) { return _mm256_add_ps(x, y); };

static const inline __m256
sub_ps(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); };

static const inline __m256
div_ps(__m256 x, __m256 y) { return _mm256_div_ps(x, y); };

static const inline void
maskstore_ps(float * x, __m256i y, __m256 z) {
    return _mm256_maskstore_ps(x, y, z);
};

static const inline __m256i
cvtps_epi32(__m256 x) { return _mm256_cvtps_epi32(x); };

//static const inline __m256
//fma_add_ps(__m256 x, __m256 y, __m256 z) { return _mm256_fmadd_ps(x, y, z); };

static const inline __m256
fma_add_ps(__m256 x, __m256 y, __m256 z) { return _mm256_add_ps(z, _mm256_mul_ps(x, y)); };

static const inline __m256
get_exponent(__m256 x)
{
    /*
     * Special handling of denormals:
     * 1) Multiply denormal elements with 2**100 (0x71800000)
     * 2) Get the 8 bits of unbiased exponent
     * 3) Subtract 100 from exponent of denormals
     */

    __m256 two_power_100 = _mm256_castsi256_ps(_mm256_set1_epi32(0x71800000));
    __m256 denormal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_LT_OQ);
    __m256 normal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_GE_OQ);

    __m256 temp1 = _mm256_blendv_ps(x, _mm256_set1_ps(0.0f), normal_mask);
    __m256 temp = _mm256_mul_ps(temp1, two_power_100);
    x = _mm256_blendv_ps(x, temp, denormal_mask);

    __m256 exp = _mm256_cvtepi32_ps(
        _mm256_sub_epi32(
            _mm256_srli_epi32(
                _mm256_castps_si256(x), 23), _mm256_set1_epi32(0x7E)));

    __m256 denorm_exp = _mm256_sub_ps(exp, _mm256_set1_ps(100.0f));
    return _mm256_blendv_ps(exp, denorm_exp, denormal_mask);
}

static const inline __m256
get_mantissa(__m256 x)
{
    /*
     * Special handling of denormals:
     * 1) Multiply denormal elements with 2**100 (0x71800000)
     * 2) Get the 23 bits of mantissa
     * 3) Mantissa for denormals is not affected by the multiplication
     */

    __m256 two_power_100 = _mm256_castsi256_ps(_mm256_set1_epi32(0x71800000));
    __m256 denormal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_LT_OQ);
    __m256 normal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_GE_OQ);

    __m256 temp1 = _mm256_blendv_ps(x, _mm256_set1_ps(0.0f), normal_mask);
    __m256 temp = _mm256_mul_ps(temp1, two_power_100);
    x = _mm256_blendv_ps(x, temp, denormal_mask);

    __m256i mantissa_bits = _mm256_set1_epi32(0x7fffff);
    __m256i exp_126_bits = _mm256_set1_epi32(126 << 23);
    return _mm256_castsi256_ps(
        _mm256_or_si256(
            _mm256_and_si256(
                _mm256_castps_si256(x), mantissa_bits), exp_126_bits));
}




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
template<typename VTYPE, typename VTYPEi>
static void
log_FLOAT(void* pDataIn,
    void* pDataOut,
    const int64_t array_size,
    const int64_t steps,
    const int64_t stepsOut)
{
    const int64_t stride = steps / (int64_t)sizeof(float);
    const int32_t num_lanes = sizeof(VTYPE) / (int64_t)sizeof(float);

    float* op = (float*)pDataOut;
    float* ip = (float*)pDataIn;

    /*
     * Note: while generally indices are int64_t, we ensure that our maximum index
     * will fit in an int32 as a precondition for this function via
     * IS_OUTPUT_BLOCKABLE_UNARY
     */
    int32_t indexarr[16];
    for (int32_t ii = 0; ii < 16; ii++) {
        indexarr[ii] = ii * (int32_t)stride;
    }

    /* Load up frequently used constants */
    VTYPE log_p0 = set1_ps(NPY_COEFF_P0_LOGf);
    VTYPE log_p1 = set1_ps(NPY_COEFF_P1_LOGf);
    VTYPE log_p2 = set1_ps(NPY_COEFF_P2_LOGf);
    VTYPE log_p3 = set1_ps(NPY_COEFF_P3_LOGf);
    VTYPE log_p4 = set1_ps(NPY_COEFF_P4_LOGf);
    VTYPE log_p5 = set1_ps(NPY_COEFF_P5_LOGf);
    VTYPE log_q0 = set1_ps(NPY_COEFF_Q0_LOGf);
    VTYPE log_q1 = set1_ps(NPY_COEFF_Q1_LOGf);
    VTYPE log_q2 = set1_ps(NPY_COEFF_Q2_LOGf);
    VTYPE log_q3 = set1_ps(NPY_COEFF_Q3_LOGf);
    VTYPE log_q4 = set1_ps(NPY_COEFF_Q4_LOGf);
    VTYPE log_q5 = set1_ps(NPY_COEFF_Q5_LOGf);
    VTYPE loge2 =  set1_ps(NPY_LOGE2f);
    VTYPE nan = set1_ps(NPY_NANF);
    VTYPE neg_nan = set1_ps(-NPY_NANF);
    VTYPE neg_inf = set1_ps(-NPY_INFINITYF);
    VTYPE inf = set1_ps(NPY_INFINITYF);
    VTYPE zeros_f = set1_ps(0.0f);
    VTYPE ones_f = set1_ps(1.0f);
    VTYPEi vindex = LOADUI((VTYPEi*)indexarr);
    VTYPE poly, num_poly, denom_poly, exponent;

    VTYPE inf_mask, nan_mask, sqrt2_mask, zero_mask, negx_mask;
    VTYPE invalid_mask = get_partial_load_mask_ps(0, num_lanes);
    VTYPE divide_by_zero_mask = invalid_mask;
    VTYPE load_mask = get_full_load_mask_ps();

    int64_t num_remaining_elements = array_size;

    while (num_remaining_elements > 0) {

        if (num_remaining_elements < num_lanes) {
            load_mask = get_partial_load_mask_ps((int)num_remaining_elements,
                num_lanes);
        }

        VTYPE x_in;
        if (stride == 1) {
            x_in = masked_load_ps(load_mask, ip);
        }
        else {
            x_in = masked_gather_ps(zeros_f, ip, vindex, load_mask);
        }

        negx_mask = cmp_ps< _CMP_LT_OQ>(x_in, zeros_f);
        zero_mask = cmp_ps< _CMP_EQ_OQ>(x_in, zeros_f);
        inf_mask = cmp_ps< _CMP_EQ_OQ>(x_in, inf);
        nan_mask = cmp_ps< _CMP_NEQ_UQ>(x_in, x_in);

        divide_by_zero_mask = or_masks(divide_by_zero_mask,
            and_masks(zero_mask, load_mask));
        invalid_mask = or_masks(invalid_mask, negx_mask);

        VTYPE x = set_masked_lanes_ps(x_in, zeros_f, negx_mask);

        /* set x = normalized mantissa */
        exponent = get_exponent(x);
        x = get_mantissa(x);

        /* if x < sqrt(2) {exp = exp-1; x = 2*x} */
        sqrt2_mask = cmp_ps< _CMP_LE_OQ>(x, set1_ps(NPY_SQRT1_2f));

        x = blend(x, add_ps(x, x), sqrt2_mask);
        exponent = blend(exponent, sub_ps(exponent, ones_f), sqrt2_mask);

        /* x = x - 1 */
        x = sub_ps(x, ones_f);

        /* Polynomial approximation for log(1+x) */
        num_poly = fma_add_ps(log_p5, x, log_p4);
        num_poly = fma_add_ps(num_poly, x, log_p3);
        num_poly = fma_add_ps(num_poly, x, log_p2);
        num_poly = fma_add_ps(num_poly, x, log_p1);
        num_poly = fma_add_ps(num_poly, x, log_p0);
        denom_poly = fma_add_ps(log_q5, x, log_q4);
        denom_poly = fma_add_ps(denom_poly, x, log_q3);
        denom_poly = fma_add_ps(denom_poly, x, log_q2);
        denom_poly = fma_add_ps(denom_poly, x, log_q1);
        denom_poly = fma_add_ps(denom_poly, x, log_q0);
        poly = div_ps(num_poly, denom_poly);
        poly = fma_add_ps(exponent, loge2, poly);

        /*
         * x < 0.0f; return -NAN
         * x = +/- NAN; return NAN
         * x = 0.0f; return -INF
         */
        poly = set_masked_lanes_ps(poly, nan, nan_mask);
        poly = set_masked_lanes_ps(poly, neg_nan, negx_mask);
        poly = set_masked_lanes_ps(poly, neg_inf, zero_mask);
        poly = set_masked_lanes_ps(poly, inf, inf_mask);

        maskstore_ps(op, cvtps_epi32(load_mask), poly);

        ip += num_lanes * stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (mask_to_int(invalid_mask)) {
        npy_set_floatstatus_invalid();
    }
    if (mask_to_int(divide_by_zero_mask)) {
        npy_set_floatstatus_divbyzero();
    }
}


extern "C"
UNARY_FUNC GetLogOpFast(int func, int atopInType1, int* wantedOutType) {

    switch (func) {
    case TRIG_OPERATION::LOG:
        *wantedOutType = ATOP_DOUBLE;
        if (atopInType1 == ATOP_FLOAT) {
            *wantedOutType = ATOP_FLOAT;
        }
        switch (atopInType1) {
        case ATOP_FLOAT: return log_FLOAT<__m256, __m256i>;
        }
    }
    return NULL;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
