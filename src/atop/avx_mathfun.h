/* 
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#include <immintrin.h>

#if defined(RT_COMPILER_MSVC)
# define ALIGN32_BEG __declspec(align(32))
# define ALIGN32_END
#else
# define ALIGN32_BEG
# define ALIGN32_END __attribute__((aligned(32)))
#endif

#define v8sf __m256 
#define v8si __m256i
#define v4df __m256d
#define v4di __m256i

// Two ways to define the constants, as variables, or inline with mm256_set1_xxx
// If in aligned memory (32 bytes of AVX2 register)- Intel seems to run faster, probably in L1 cache
// AMD appears to run faster the other way, load the registers ahead of time with broadcast.  Perhaps due to cache ownership per thread.
//
//#define _PS256_CONST(Name, Val) static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { (float)Val, (float)Val, (float)Val, (float)Val, (float)Val, (float)Val, (float)Val, (float)Val }
//#define _PI32_CONST256(Name, Val) static const ALIGN32_BEG int32_t _pi32_256_##Name[8] ALIGN32_END = { (int32_t)Val, (int32_t)Val, (int32_t)Val, (int32_t)Val, (int32_t)Val, (int32_t)Val, (int32_t)Val, (int32_t)Val }
//#define _PS256_CONST_TYPE(Name, Val) static const ALIGN32_BEG uint32_t _ps256_##Name[8] ALIGN32_END = { (uint32_t)Val, (uint32_t)Val, (uint32_t)Val, (uint32_t)Val, (uint32_t)Val, (uint32_t)Val, (uint32_t)Val, (uint32_t)Val }

//#define _PS256d_CONST(Name, Val) static const ALIGN32_BEG double _pd256d_##Name[4] ALIGN32_END = { (double)Val, (double)Val, (double)Val, (double)Val }
//#define _PI32_CONST256d(Name, Val) static const ALIGN32_BEG int64_t _pi32_256d_##Name[4] ALIGN32_END = { (int64_t)Val, (int64_t)Val, (int64_t)Val, (int64_t)Val }
//#define _PS256d_CONST_TYPE(Name, Val) static const ALIGN32_BEG uint64_t _pd256d_##Name[4] ALIGN32_END = { (uint64_t)Val, (uint64_t)Val, (uint64_t)Val, (uint64_t)Val }

//#define _PS256_CONST(Name, Val) static const __m256 _ps256_##Name = _mm256_set1_ps(Val);
//#define _PI32_CONST256(Name, Val) static const  __m256i _pi32_256_##Name = _mm256_set1_epi32(Val);
//#define _PS256_CONST_TYPE(Name, Val) static const  __m256 _ps256_##Name = _mm256_castsi256_ps( _mm256_set1_epi32(Val)); 

#define _PS256d_CONST(Name, Val) static const __m256d _pd256d_##Name = _mm256_set1_pd(Val); 
#define _PI32_CONST256d(Name, Val) static const __m256i _pi32_256d_##Name = _mm256_set1_epi64x((int64_t)Val); 
#define _PS256d_CONST_TYPE(Name, Val) static const __m256d _pd256d_##Name = _mm256_castsi256_pd(_mm256_set1_epi64x((int64_t)Val));

#define _PS256_CONST(Name, Val) static const float _ps256t_##Name = Val; static const __m256 _ps256_##Name = _mm256_broadcast_ss(&_ps256t_##Name);
#define _PI32_CONST256(Name, Val) static const  __m256i _pi32_256_##Name = _mm256_set1_epi32(Val);
#define _PS256_CONST_TYPE(Name, Val) static const uint32_t _ps256t_##Name = Val; static const  __m256 _ps256_##Name = _mm256_broadcast_ss((const float*)&_ps256t_##Name); 


_PS256_CONST(1  , 1.0f);
_PS256d_CONST(1, 1.0f);

_PS256_CONST(0p5, 0.5f);
_PS256d_CONST(0p5, 0.5f);

/* the smallest non denormalized float number */
// First 8 bits all zero
_PS256_CONST_TYPE(min_norm_pos,  0x00800000);
_PS256_CONST_TYPE(mant_mask,     0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, 0x807fffff);
_PS256_CONST_TYPE(sign_mask,     0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, 0x7fffffff);

// First 11 bits all zero
_PS256d_CONST_TYPE(min_norm_pos,  0x0010000000000000ULL);
_PS256d_CONST_TYPE(mant_mask,     0x7ff0000000000000ULL); 
_PS256d_CONST_TYPE(inv_mant_mask, 0x800fffffffffffffULL);
_PS256d_CONST_TYPE(sign_mask,     0x8000000000000000LL);
_PS256d_CONST_TYPE(inv_sign_mask, 0x7fffffffffffffffULL);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
// srli 23 then mask off exponent (8 bits), note 0xff = infinity
_PI32_CONST256(0x7f, 0x7f);

_PI32_CONST256d(0, 0);
_PI32_CONST256d(1, 1);
_PI32_CONST256d(inv1, ~(int64_t)1);   // or ~1LL  or -2?
_PI32_CONST256d(2, 2);
_PI32_CONST256d(4, 4);
// srli 52 then mask off exponent (11 bits), note 0xffe = infinity
_PI32_CONST256d(0x7f, 0x7fe);


//  Only works for inputs in the range: [-2^51, 2^51]
FORCE_INLINE static __m256i double_to_int64(__m256d x) {
    x = _mm256_add_pd(x, _mm256_set1_pd(0x0018000000000000));
    return _mm256_sub_epi64(
        _mm256_castpd_si256(x),
        _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
    );
}

//  Only works for inputs in the range: [-2^51, 2^51]
FORCE_INLINE static __m256d int64_to_double(__m256i x) {
    x = _mm256_add_epi64(x, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0018000000000000));
}

// TJD: Note not used yet
FORCE_INLINE static __m256d int64_to_double_fast_precise(const __m256i v)
/* Optimized full range int64_t to double conversion           */
/* Emulate _mm256_cvtepi64_pd()                                */
{
    const __m256i magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);              /* 2^52               encoded as floating-point  */
    const __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000);            /* 2^84 + 2^63        encoded as floating-point  */
    const __m256i magic_i_all = _mm256_set1_epi64x(0x4530000080100000);             /* 2^84 + 2^63 + 2^52 encoded as floating-point  */
    const __m256d magic_d_all = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         /* Blend the 32 lowest significant bits of v with magic_int_lo                                                   */
    __m256i v_hi = _mm256_srli_epi64(v, 32);                              /* Extract the 32 most significant bits of v                                                                     */
    v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);                  /* Flip the msb of v_hi and blend with 0x45300000                                                                */
    __m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision:                                                                                  */
    __m256d result = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));    /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!                        */
    return result;                                                        /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                                                                          /* With icc use -fp-model precise                                                                                */
}

//==========================================================================
// Natural Log
//
_PS256d_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS256d_CONST(cephes_log_p0, 7.0376836292E-2);
_PS256d_CONST(cephes_log_p1, -1.1514610310E-1);
_PS256d_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256d_CONST(cephes_log_p3, -1.2420140846E-1);
_PS256d_CONST(cephes_log_p4, +1.4249322787E-1);
_PS256d_CONST(cephes_log_p5, -1.6668057665E-1);
_PS256d_CONST(cephes_log_p6, +2.0000714765E-1);
_PS256d_CONST(cephes_log_p7, -2.4999993993E-1);
_PS256d_CONST(cephes_log_p8, +3.3333331174E-1);
_PS256d_CONST(cephes_log_q1, -2.12194440e-4);
_PS256d_CONST(cephes_log_q2, 0.693359375);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS256_CONST(cephes_log_q1, -2.12194440e-4f);
_PS256_CONST(cephes_log_q2, 0.693359375f);


/* natural logarithm computed for 8 simultaneous float 
   return NaN for x <= 0
*/
FORCE_INLINE static v8sf log256_ps(v8sf x) {

    v8si imm0;
    v8sf one = _ps256_1;

    //v8sf invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, _ps256_min_norm_pos);  /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, _ps256_inv_mant_mask);
    x = _mm256_or_ps(x, _ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, _pi32_256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    /* part2: 
        if( x < SQRTHF ) {
        e -= 1;
        x = x + x - 1.0;
        } else { x = x - 1.0; }
    */
    //v8sf mask = _mm256_cmplt_ps(x, _ps256_cephes_SQRTHF);
    v8sf mask = _mm256_cmp_ps(x, _ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x,x);

    v8sf y = _ps256_cephes_log_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);
  
    tmp = _mm256_mul_ps(e, _ps256_cephes_log_q1);
    y = _mm256_add_ps(y, tmp);


    tmp = _mm256_mul_ps(z, _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, _ps256_cephes_log_q2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
    return x;
}


FORCE_INLINE static v4df log256_pd(v4df x) {
    v4di imm0;
    v4df one = _pd256d_1;

    //v4df invalid_mask = _mm256_cmple_pd(x, _mm256_setzero_pd());
    v4df invalid_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LE_OS);

    x = _mm256_max_pd(x, _pd256d_min_norm_pos);  /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi64(_mm256_castpd_si256(x), 52); // 23);

    /* keep only the fractional part */
    x = _mm256_and_pd(x, _pd256d_inv_mant_mask);
    x = _mm256_or_pd(x, _pd256d_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi64(imm0, _pi32_256d_0x7f);
    v4df e =int64_to_double(imm0);

    e = _mm256_add_pd(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    //v4df mask = _mm256_cmplt_pd(x, _pd256d_cephes_SQRTHF);
    v4df mask = _mm256_cmp_pd(x, _pd256d_cephes_SQRTHF, _CMP_LT_OS);
    v4df tmp = _mm256_and_pd(x, mask);
    x = _mm256_sub_pd(x, one);
    e = _mm256_sub_pd(e, _mm256_and_pd(one, mask));
    x = _mm256_add_pd(x, tmp);

    v4df z = _mm256_mul_pd(x, x);

    v4df y = _pd256d_cephes_log_p0;
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p1);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p2);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p3);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p4);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p5);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p6);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p7);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_log_p8);
    y = _mm256_mul_pd(y, x);

    y = _mm256_mul_pd(y, z);

    tmp = _mm256_mul_pd(e, _pd256d_cephes_log_q1);
    y = _mm256_add_pd(y, tmp);


    tmp = _mm256_mul_pd(z, _pd256d_0p5);
    y = _mm256_sub_pd(y, tmp);

    tmp = _mm256_mul_pd(e, _pd256d_cephes_log_q2);
    x = _mm256_add_pd(x, y);
    x = _mm256_add_pd(x, tmp);
    x = _mm256_or_pd(x, invalid_mask); // negative arg will be NAN
    return x;
}

//==========================================================================
// Exponent
//
_PS256d_CONST(exp_hi, 88.3762626647949);
_PS256d_CONST(exp_lo, -88.3762626647949);

_PS256d_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256d_CONST(cephes_exp_C1, 0.693359375);
_PS256d_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256d_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256d_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256d_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256d_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256d_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256d_CONST(cephes_exp_p5, 5.0000001201E-1);

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

FORCE_INLINE static v8sf exp256_ps(v8sf x) {

    v8sf tmp = _mm256_setzero_ps(), fx;
    v8si imm0;
    v8sf one = _ps256_1;

    x = _mm256_min_ps(x, _ps256_exp_hi);
    x = _mm256_max_ps(x, _ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_ps(x, _ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, _ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    //imm0 = _mm256_cvttps_epi32(fx);
    //tmp  = _mm256_cvtepi32_ps(imm0);
  
    tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    //v8sf mask = _mm256_cmpgt_ps(tmp, fx);    
    v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, _ps256_cephes_exp_C1);
    v8sf z = _mm256_mul_ps(fx, _ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x,x);
  
    v8sf y = _ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, _pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    v8sf pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

FORCE_INLINE static v4df exp256_pd(v4df x) {
    v4df tmp = _mm256_setzero_pd(), fx;
    v4di imm0;
    v4df one = _pd256d_1;

    x = _mm256_min_pd(x, _pd256d_exp_hi);
    x = _mm256_max_pd(x, _pd256d_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_pd(x, _pd256d_cephes_LOG2EF);
    fx = _mm256_add_pd(fx, _pd256d_0p5);

    /* how to perform a floorf with SSE: just below */
    //imm0 = _mm256d_cvttps_epi32(fx);
    //tmp  = _mm256d_cvtepi32_pd(imm0);

    tmp = _mm256_floor_pd(fx);

    /* if greater, substract 1 */
    //v4df mask = _mm256d_cmpgt_pd(tmp, fx);    
    v4df mask = _mm256_cmp_pd(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_pd(mask, one);
    fx = _mm256_sub_pd(tmp, mask);

    tmp = _mm256_mul_pd(fx, _pd256d_cephes_exp_C1);
    v4df z = _mm256_mul_pd(fx, _pd256d_cephes_exp_C2);
    x = _mm256_sub_pd(x, tmp);
    x = _mm256_sub_pd(x, z);

    z = _mm256_mul_pd(x, x);

    v4df y = _pd256d_cephes_exp_p0;
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_exp_p1);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_exp_p2);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_exp_p3);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_exp_p4);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _pd256d_cephes_exp_p5);
    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, x);
    y = _mm256_add_pd(y, one);

    /* build 2^n */
    imm0 = double_to_int64(fx);
    // another two AVX2 instructions
    // TJD: 127 << 23
    imm0 = _mm256_add_epi64(imm0, _pi32_256d_0x7f);
    imm0 = _mm256_slli_epi64(imm0, 52); // 23);
    v4df pow2n = _mm256_castsi256_pd(imm0);
    y = _mm256_mul_pd(y, pow2n);
    return y;
}

//==========================================================================
// Sin function
//

_PS256d_CONST(minus_cephes_DP1, -0.78515625);
_PS256d_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS256d_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS256d_CONST(sincof_p0, -1.9515295891E-4);
_PS256d_CONST(sincof_p1, 8.3321608736E-3);
_PS256d_CONST(sincof_p2, -1.6666654611E-1);
_PS256d_CONST(coscof_p0, 2.443315711809948E-005);
_PS256d_CONST(coscof_p1, -1.388731625493765E-003);
_PS256d_CONST(coscof_p2, 4.166664568298827E-002);
_PS256d_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI

_PS256_CONST(minus_cephes_DP1, -0.78515625f);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS256_CONST(sincof_p0, -1.9515295891E-4f);
_PS256_CONST(sincof_p1, 8.3321608736E-3f);
_PS256_CONST(sincof_p2, -1.6666654611E-1f);
_PS256_CONST(coscof_p0, 2.443315711809948E-005f);
_PS256_CONST(coscof_p1, -1.388731625493765E-003f);
_PS256_CONST(coscof_p2, 4.166664568298827E-002f);
_PS256_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
FORCE_INLINE static v8sf sin256_ps(v8sf x) {

    v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
    v8si imm0, imm2;

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, _ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, _ps256_sign_mask);
  
    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, _ps256_cephes_FOPI);

    /*
    Here we start a series of integer operations, which are in the
    realm of AVX2.
    If we don't have AVX, let's perform them using SSE2 directives
    */

    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_add_epi32(imm2, _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, _pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    // IEEE 754 32bit/64bit
    // 31 = sign    63 = sign  (1 bit)
    // 30-23 (8 bits)= exponent  62-52 (11 bits)
    // 22-0  = fraction (23 bits)  51-0 = fraction (52 bits)

    imm0 = _mm256_and_si256(imm2, _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask 
        there is one polynom for 0 <= x <= Pi/4
        and another one for Pi/4<x<=Pi/2

        Both branches will be computed.
    */
    imm2 = _mm256_and_si256(imm2, _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, _pi32_256_0);
 
    v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic" 
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _ps256_minus_cephes_DP1;
    xmm2 = _ps256_minus_cephes_DP2;
    xmm3 = _ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = _ps256_coscof_p0;
    v8sf z = _mm256_mul_ps(x,x);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, _ps256_1);
  
    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = _ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */  
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y,y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
FORCE_INLINE static v8sf cos256_ps(v8sf x) { // any x
    v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
    v8si imm0, imm2;

    /* take the absolute value */
    x = _mm256_and_ps(x, _ps256_inv_sign_mask);
  
    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, _ps256_cephes_FOPI);
  
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, _pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_sub_epi32(imm2, _pi32_256_2);
  
    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, _pi32_256_0);

    v8sf sign_bit = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic" 
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _ps256_minus_cephes_DP1;
    xmm2 = _ps256_minus_cephes_DP2;
    xmm3 = _ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);
  
    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = _ps256_coscof_p0;
    v8sf z = _mm256_mul_ps(x,x);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, _ps256_1);
  
    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = _ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */  
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y,y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
FORCE_INLINE static void sincos256_ps(v8sf x, v8sf *s, v8sf *c) {

    v8sf xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
    v8si imm0, imm2, imm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, _ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, _ps256_sign_mask);
  
    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, _ps256_cephes_FOPI);

    /* store the integer part of y in imm2 */
    imm2 = _mm256_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, _pi32_256_1);
    imm2 = _mm256_and_si256(imm2, _pi32_256_inv1);

    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, _pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    //v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, _pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, _pi32_256_0);

    //v8sf poly_mask = _mm256_castsi256_ps(imm2);
    v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    v8sf poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic" 
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _ps256_minus_cephes_DP1;
    xmm2 = _ps256_minus_cephes_DP2;
    xmm3 = _ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    imm4 = _mm256_sub_epi32(imm4, _pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, _pi32_256_4);
    imm4 = _mm256_slli_epi32(imm4, 29);

    v8sf sign_bit_cos = _mm256_castsi256_ps(imm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);
  
    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v8sf z = _mm256_mul_ps(x,x);
    y = _ps256_coscof_p0;

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, _ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, _ps256_1);
  
    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = _ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */  
    xmm3 = poly_mask;
    v8sf ysin2 = _mm256_and_ps(xmm3, y2);
    v8sf ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2,ysin2);
    y = _mm256_sub_ps(y, ysin1);

    xmm1 = _mm256_add_ps(ysin1,ysin2);
    xmm2 = _mm256_add_ps(y,y2);
 
    /* update the sign */
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

// ===============================================================================
// double versions sin, cos, sincos
FORCE_INLINE static v4df sin256_pd(v4df x) { // any x
    v4df xmm1, xmm2 = _mm256_setzero_pd(), xmm3, sign_bit, y;
    v4di imm0, imm2;

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_pd(x, _pd256d_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_pd(sign_bit, _pd256d_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_pd(x, _pd256d_cephes_FOPI);

    /*
      Here we start a series of integer operations, which are in the
      realm of AVX2.
      If we don't have AVX, let's perform them using SSE2 directives
    */

    /* store the integer part of y in mm0 */
    imm2 = double_to_int64(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_add_epi64(imm2, _pi32_256d_1);
    imm2 = _mm256_and_si256(imm2, _pi32_256d_inv1);
    y = int64_to_double(imm2);

    /* get the swap sign flag */
    // IEEE 754 32bit/64bit
    // 31 = sign    63 = sign  (1 bit)
    // 30-23 (8 bits)= exponent  62-52 (11 bits)
    // 22-0  = fraction (23 bits)  51-0 = fraction (52 bits)
    imm0 = _mm256_and_si256(imm2, _pi32_256d_4);
    imm0 = _mm256_slli_epi64(imm0, 32+29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    imm2 = _mm256_and_si256(imm2, _pi32_256d_2);
    imm2 = _mm256_cmpeq_epi64(imm2, _pi32_256d_0);

    v4df swap_sign_bit = _mm256_castsi256_pd(imm0);
    v4df poly_mask = _mm256_castsi256_pd(imm2);
    sign_bit = _mm256_xor_pd(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _pd256d_minus_cephes_DP1;
    xmm2 = _pd256d_minus_cephes_DP2;
    xmm3 = _pd256d_minus_cephes_DP3;
    xmm1 = _mm256_mul_pd(y, xmm1);
    xmm2 = _mm256_mul_pd(y, xmm2);
    xmm3 = _mm256_mul_pd(y, xmm3);
    x = _mm256_add_pd(x, xmm1);
    x = _mm256_add_pd(x, xmm2);
    x = _mm256_add_pd(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = _pd256d_coscof_p0;
    v4df z = _mm256_mul_pd(x, x);

    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, _pd256d_coscof_p1);
    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, _pd256d_coscof_p2);
    y = _mm256_mul_pd(y, z);
    y = _mm256_mul_pd(y, z);
    v4df tmp = _mm256_mul_pd(z, _pd256d_0p5);
    y = _mm256_sub_pd(y, tmp);
    y = _mm256_add_pd(y, _pd256d_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v4df y2 = _pd256d_sincof_p0;
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_add_pd(y2, _pd256d_sincof_p1);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_add_pd(y2, _pd256d_sincof_p2);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_mul_pd(y2, x);
    y2 = _mm256_add_pd(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_pd(xmm3, y2); //, xmm3);
    y = _mm256_andnot_pd(xmm3, y);
    y = _mm256_add_pd(y, y2);
    /* update the sign */
    y = _mm256_xor_pd(y, sign_bit);

    return y;
}

/* almost the same as sin_pd */
FORCE_INLINE static v4df cos256_pd(v4df x) { // any x
    v4df xmm1, xmm2 = _mm256_setzero_pd(), xmm3, y;
    v4di imm0, imm2;

    /* take the absolute value */
    x = _mm256_and_pd(x, _pd256d_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_pd(x, _pd256d_cephes_FOPI);

    /* store the integer part of y in mm0 */
    imm2 = double_to_int64(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi64(imm2, _pi32_256d_1);
    imm2 = _mm256_and_si256(imm2, _pi32_256d_inv1);
    y = int64_to_double(imm2);
    imm2 = _mm256_sub_epi64(imm2, _pi32_256d_2);

    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, _pi32_256d_4);
    imm0 = _mm256_slli_epi64(imm0, 32+29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, _pi32_256d_2);
    imm2 = _mm256_cmpeq_epi64(imm2, _pi32_256d_0);

    v4df sign_bit = _mm256_castsi256_pd(imm0);
    v4df poly_mask = _mm256_castsi256_pd(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _pd256d_minus_cephes_DP1;
    xmm2 = _pd256d_minus_cephes_DP2;
    xmm3 = _pd256d_minus_cephes_DP3;
    xmm1 = _mm256_mul_pd(y, xmm1);
    xmm2 = _mm256_mul_pd(y, xmm2);
    xmm3 = _mm256_mul_pd(y, xmm3);
    x = _mm256_add_pd(x, xmm1);
    x = _mm256_add_pd(x, xmm2);
    x = _mm256_add_pd(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = _pd256d_coscof_p0;
    v4df z = _mm256_mul_pd(x, x);

    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, _pd256d_coscof_p1);
    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, _pd256d_coscof_p2);
    y = _mm256_mul_pd(y, z);
    y = _mm256_mul_pd(y, z);
    v4df tmp = _mm256_mul_pd(z, _pd256d_0p5);
    y = _mm256_sub_pd(y, tmp);
    y = _mm256_add_pd(y, _pd256d_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v4df y2 = _pd256d_sincof_p0;
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_add_pd(y2, _pd256d_sincof_p1);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_add_pd(y2, _pd256d_sincof_p2);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_mul_pd(y2, x);
    y2 = _mm256_add_pd(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_pd(xmm3, y2); //, xmm3);
    y = _mm256_andnot_pd(xmm3, y);
    y = _mm256_add_pd(y, y2);
    /* update the sign */
    y = _mm256_xor_pd(y, sign_bit);

    return y;
}

/* since sin256_pd and cos256_pd are almost identical, sincos256_pd could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
FORCE_INLINE static void sincos256_pd(v4df x, v4df* s, v4df* c) {

    v4df xmm1, xmm2, xmm3 = _mm256_setzero_pd(), sign_bit_sin, y;
    v4di imm0, imm2, imm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_pd(x, _pd256d_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_pd(sign_bit_sin, _pd256d_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_pd(x, _pd256d_cephes_FOPI);

    /* store the integer part of y in imm2 */
    imm2 = double_to_int64(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi64(imm2, _pi32_256d_1);
    imm2 = _mm256_and_si256(imm2, _pi32_256d_inv1);

    y = int64_to_double(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, _pi32_256d_4);
    imm0 = _mm256_slli_epi64(imm0, 32+29);
    //v4df swap_sign_bit_sin = _mm256_castsi256_pd(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, _pi32_256d_2);
    imm2 = _mm256_cmpeq_epi64(imm2, _pi32_256d_0);

    //v4df poly_mask = _mm256_castsi256_pd(imm2);
    v4df swap_sign_bit_sin = _mm256_castsi256_pd(imm0);
    v4df poly_mask = _mm256_castsi256_pd(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _pd256d_minus_cephes_DP1;
    xmm2 = _pd256d_minus_cephes_DP2;
    xmm3 = _pd256d_minus_cephes_DP3;
    xmm1 = _mm256_mul_pd(y, xmm1);
    xmm2 = _mm256_mul_pd(y, xmm2);
    xmm3 = _mm256_mul_pd(y, xmm3);
    x = _mm256_add_pd(x, xmm1);
    x = _mm256_add_pd(x, xmm2);
    x = _mm256_add_pd(x, xmm3);

    imm4 = _mm256_sub_epi64(imm4, _pi32_256d_2);
    imm4 = _mm256_andnot_si256(imm4, _pi32_256d_4);
    imm4 = _mm256_slli_epi64(imm4, 32+29);

    v4df sign_bit_cos = _mm256_castsi256_pd(imm4);

    sign_bit_sin = _mm256_xor_pd(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v4df z = _mm256_mul_pd(x, x);
    y = _pd256d_coscof_p0;

    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, _pd256d_coscof_p1);
    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, _pd256d_coscof_p2);
    y = _mm256_mul_pd(y, z);
    y = _mm256_mul_pd(y, z);
    v4df tmp = _mm256_mul_pd(z, _pd256d_0p5);
    y = _mm256_sub_pd(y, tmp);
    y = _mm256_add_pd(y, _pd256d_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v4df y2 = _pd256d_sincof_p0;
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_add_pd(y2, _pd256d_sincof_p1);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_add_pd(y2, _pd256d_sincof_p2);
    y2 = _mm256_mul_pd(y2, z);
    y2 = _mm256_mul_pd(y2, x);
    y2 = _mm256_add_pd(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    v4df ysin2 = _mm256_and_pd(xmm3, y2);
    v4df ysin1 = _mm256_andnot_pd(xmm3, y);
    y2 = _mm256_sub_pd(y2, ysin2);
    y = _mm256_sub_pd(y, ysin1);

    xmm1 = _mm256_add_pd(ysin1, ysin2);
    xmm2 = _mm256_add_pd(y, y2);

    /* update the sign */
    *s = _mm256_xor_pd(xmm1, sign_bit_sin);
    *c = _mm256_xor_pd(xmm2, sign_bit_cos);
}
