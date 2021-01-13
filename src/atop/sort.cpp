#include "common_inc.h"
#include "threads.h"
#include "halffloat.h"
#include <cmath>
#include <algorithm>

//#define LOGGING printf
#define LOGGING(...)

#define PLOGGING(...)
//#define PLOGGING printf

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

FORCE_INLINE static int npy_get_msb(uint64_t unum)
{
    int depth_limit = 0;
    while (unum >>= 1) {
        depth_limit++;
    }
    return depth_limit;
}

// Reverse mem copy to help with cache
FORCE_INLINE static void MEMCPYR(void* pDestV, void *pSrcV, int64_t length) {
    memcpy(pDestV, pSrcV, length);

    // reverse copy below, but it's slower
    //char* pDest = (char*)pDestV;
    //char* pSrc = (char*)pSrcV;
    //if (length >= 32) {
    //    char* pDestEnd = pDest + length;
    //    char* pSrcEnd = pSrc + length;
    //    do  {
    //        pDestEnd -= 32;
    //        pSrcEnd -= 32;
    //        __m256i m0 = _mm256_loadu_si256((const __m256i*)pSrcEnd);
    //        _mm256_storeu_si256((__m256i*)pDestEnd, m0);
    //        length -= 32;
    //    } while (length > 32);
    //    // write first two bytes
    //    __m256i m0 = _mm256_loadu_si256((const __m256i*)pSrc);
    //    _mm256_storeu_si256((__m256i*)pDest, m0);
    //}
    //else {
    //    while (length--) {
    //        *pDest = *pSrc;
    //        pDest++; pSrc++;
    //    }
    //}
}

#define SMALL_MERGESORT 16
#define PYA_QS_STACK 128
#define SMALL_QUICKSORT 15
#define INTP_SWAP(_X_,_Y_) { auto temp=_X_; _X_=_Y_; _Y_=temp;}
#define T_SWAP(_X_, _Y_) { auto temp= _X_; _X_ = _Y_; _Y_ = temp; }
//#define T_SWAP(_X_, _Y_) std::swap(_X_,_Y_); 

FORCE_INLINE static void STRING_SWAP(void* _X_, void* _Y_, int64_t len) {
    char* pSrc = (char*)_X_;
    char* pDest = (char*)_Y_;
    while (len >= 8) {
        int64_t temp = *(int64_t*)pSrc;
        *(int64_t*)pSrc = *(int64_t*)pDest;
        *(int64_t*)pDest = temp;
        pSrc += 8;
        pDest += 8;
        len -= 8;
    }
    while (len) {
        char temp = *pSrc;
        *pSrc++ = *pDest;
        *pDest++ = temp;
        len--;
    }
}

FORCE_INLINE static void STRING_COPY(void* _X_, void* _Y_, int64_t len) {
    char* pDest = (char*)_X_;
    char* pSrc = (char*)_Y_;
    while (len >= 8) {
        *(int64_t*)pDest = *(int64_t*)pSrc;
        pSrc += 8;
        pDest += 8;
        len -= 8;
    }
    char* pEnd = pDest + len;
    while (pDest < pEnd) {
        *pDest++ = *pSrc++;
    }
}

// NOTE: For MSVS compiler DO NOT use /fp:fast (floating point fast as X==X will not work on Nans)
//A signalling NaN (NANS) is represented by any bit pattern
//between 7F800001 and 7FBFFFFF or between FF800001 and FFBFFFFF
//A quiet NaN(NANQ) is represented by any bit pattern
//between 7FC00000 and 7FFFFFFF or between FFC00000 and FFFFFFFF
// For floats anything compared to a nan will return 0/false
// TODO: Add compare for COMPLEX
//#define COMPARE_LT(X,Y) ((X) < (Y) || ((Y) != (Y) && (X) == (X)))
FORCE_INLINE static bool COMPARE_LT(float X, float Y) { return (X < Y || (Y != Y && X == X)); }
FORCE_INLINE static bool COMPARE_LT(double X, double Y) { return (X < Y || (Y != Y && X == X)); }
FORCE_INLINE static bool COMPARE_LT(long double X, long double Y) { return (X < Y || (Y != Y && X == X)); }
FORCE_INLINE static bool COMPARE_LT(int32_t X, int32_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(int64_t X, int64_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(uint32_t X, uint32_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(uint64_t X, uint64_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(int8_t X, int8_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(int16_t X, int16_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(uint8_t X, uint8_t Y) { return (X < Y); }
FORCE_INLINE static bool COMPARE_LT(uint16_t X, uint16_t Y) { return (X < Y); }

// Routines for HALF_FLOAT
FORCE_INLINE static bool
atop_half_isnan(atop_half h)
{
    return ((h & 0x7c00u) == 0x7c00u) && ((h & 0x03ffu) != 0x0000u);
}

FORCE_INLINE static bool
atop_half_lt_nonan(atop_half h1, atop_half h2)
{
    if (h1 & 0x8000u) {
        if (h2 & 0x8000u) {
            return (h1 & 0x7fffu) > (h2 & 0x7fffu);
        }
        else {
            /* Signed zeros are equal, have to check for it */
            return (h1 != 0x8000u) || (h2 != 0x0000u);
        }
    }
    else {
        if (h2 & 0x8000u) {
            return false;
        }
        else {
            return (h1 & 0x7fffu) < (h2 & 0x7fffu);
        }
    }
}

FORCE_INLINE static bool
HALF_LT(atop_half a, atop_half b)
{
    int ret;

    if (atop_half_isnan(b)) {
        ret = !atop_half_isnan(a);
    }
    else {
        ret = !atop_half_isnan(a) && atop_half_lt_nonan(a, b);
    }

    return ret;
}

//======================================
// For one byte strings
FORCE_INLINE static bool
STRING_LT(const unsigned char* c1, const unsigned char* c2, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
        }
    }
    return false;
}

//---------------------------------
// Assumes Py_UCS4
// Assumes int is 32bits
FORCE_INLINE static bool
STRING_LT(const uint32_t* c1, const uint32_t* c2, size_t len)
{
    size_t lenunicode = len / 4;

    for (size_t i = 0; i < lenunicode; ++i) {
        if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
        }
    }
    return false;
}

// NOTE: This routine may not be useful
// It will sort an opaque amount of data
FORCE_INLINE static bool
STRING_LT(const char* s1, const char* s2, size_t len)
{
    const unsigned char* c1 = (unsigned char*)s1;
    const unsigned char* c2 = (unsigned char*)s2;

    // compare 8 bytes at a time
    while (len > 8) {
        if (*(uint64_t*)c1 != *(uint64_t*)c2) {
            return *(uint64_t*)c1 < *(uint64_t*)c2;
        }
        c1 += 8;
        c2 += 8;
        len -= 8;
    }
    switch (len) {
    case 1:
        if (*c1 != *c2) {
            return *c1 < *c2;
        }
        return 0;
    case 2:
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return *(uint16_t*)c1 < *(uint16_t*)c2;
        }
        return 0;
    case 3:
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return *(uint16_t*)c1 < *(uint16_t*)c2;
        }
        c1 += 2;
        c2 += 2;
        if (*c1 != *c2) {
            return *c1 < *c2;
        }
        return 0;
    case 4:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return *(uint32_t*)c1 < *(uint32_t*)c2;
        }
        return 0;
    case 5:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return *(uint32_t*)c1 < *(uint32_t*)c2;
        }
        c1 += 4;
        c2 += 4;
        if (*c1 != *c2) {
            return *c1 < *c2;
        }
        return 0;
    case 6:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return *(uint32_t*)c1 < *(uint32_t*)c2;
        }
        c1 += 4;
        c2 += 4;
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return *(uint16_t*)c1 < *(uint16_t*)c2;
        }
        return 0;
    case 7:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return *(uint32_t*)c1 < *(uint32_t*)c2;
        }
        c1 += 4;
        c2 += 4;
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return *(uint16_t*)c1 < *(uint16_t*)c2;
        }
        c1 += 2;
        c2 += 2;
        if (*c1 != *c2) {
            return *c1 < *c2;
        }
        return 0;
    case 8:
        if (*(uint64_t*)c1 != *(uint64_t*)c2) {
            return *(uint64_t*)c1 < *(uint64_t*)c2;
        }
        return 0;
    default:
        return 0;
    }
    return 0;
}


// Returns 0 if equal, else 1
FORCE_INLINE static int
BINARY_LT(const char* s1, const char* s2, size_t len)
{
    const unsigned char* c1 = (unsigned char*)s1;
    const unsigned char* c2 = (unsigned char*)s2;

    while (len > 8) {
        if (*(uint64_t*)c1 != *(uint64_t*)c2) {
            return 1;
        }
        c1 += 8;
        c2 += 8;
        len -= 8;
    }
    switch (len) {
    case 1:
        if (*c1 != *c2) {
            return 1;
        }
        return 0;
    case 2:
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return 1;
        }
        return 0;
    case 3:
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return 1;
        }
        c1 += 2;
        c2 += 2;
        if (*c1 != *c2) {
            return 1;
        }
        return 0;
    case 4:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return 1;
        }
        return 0;
    case 5:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return 1;
        }
        c1 += 4;
        c2 += 4;
        if (*c1 != *c2) {
            return 1;
        }
        return 0;
    case 6:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return 1;
        }
        c1 += 4;
        c2 += 4;
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return 1;
        }
        return 0;
    case 7:
        if (*(uint32_t*)c1 != *(uint32_t*)c2) {
            return 1;
        }
        c1 += 4;
        c2 += 4;
        if (*(uint16_t*)c1 != *(uint16_t*)c2) {
            return 1;
        }
        c1 += 2;
        c2 += 2;
        if (*c1 != *c2) {
            return 1;
        }
        return 0;
    case 8:
        if (*(uint64_t*)c1 != *(uint64_t*)c2) {
            return 1;
        }
        return 0;
    default:
        return 0;
    }
    return 0;
}



//-----------------------------------------------------------------------------------------------
// inplace heapsort
template <typename T>
static int
heapsort_(void* pVoidStart, int64_t n)
{
    T* start = (T*)pVoidStart;
    T     tmp, * a;
    int64_t i, j, l;

    /* The array needs to be offset by one for heapsort indexing */
    a = (T*)start - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n&& COMPARE_LT(a[j], a[j + 1])) {
                j += 1;
            }
            if (COMPARE_LT(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && COMPARE_LT(a[j], a[j + 1])) {
                j++;
            }
            if (COMPARE_LT(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}



//-----------------------------------------------------------------------------------------------
// indirect heapsort
template <typename T, typename UINDEX>
static int
aheapsort_(void* vv1, void* tosort1, int64_t n)
{
    T* vv = (T*)vv1;
    UINDEX* tosort = (UINDEX*)tosort1;

    T* v = vv;
    UINDEX* a, tmp;
    int64_t i, j, l;
    /* The arrays need to be offset by one for heapsort indexing */
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n&& COMPARE_LT(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (COMPARE_LT(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && COMPARE_LT(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (COMPARE_LT(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}



//---------------------------
// For intergers and floats
template <typename T>
static int
quicksort_(void* pVoidStart, int64_t length)
{
    T* start = (T*)pVoidStart;
    T vp;
    T* pl = start;
    T* pr = pl + length - 1;
    T* stack[PYA_QS_STACK];
    T** sptr = stack;
    T* pm, * pi, * pj, * pk;

    int depth[PYA_QS_STACK];
    int* psdepth = depth;
    int cdepth = npy_get_msb(length) * 2;

    for (;;) {
        if (cdepth >= 0) {
            while ((pr - pl) > SMALL_QUICKSORT) {
                /* quicksort partition */
                pm = pl + ((pr - pl) >> 1);
                if (COMPARE_LT(*pm, *pl)) T_SWAP(*pm, *pl);
                if (COMPARE_LT(*pr, *pm)) T_SWAP(*pr, *pm);
                if (COMPARE_LT(*pm, *pl)) T_SWAP(*pm, *pl);
                vp = *pm;
                pi = pl;
                pj = pr - 1;
                T_SWAP(*pm, *pj);
                for (;;) {
                    do ++pi; while (COMPARE_LT(*pi, vp));
                    do --pj; while (COMPARE_LT(vp, *pj));
                    if (pi >= pj) {
                        break;
                    }
                    T_SWAP(*pi, *pj);
                }
                pk = pr - 1;
                T_SWAP(*pi, *pk);
                /* push largest partition on stack */
                if (pi - pl < pr - pi) {
                    *sptr++ = pi + 1;
                    *sptr++ = pr;
                    pr = pi - 1;
                }
                else {
                    *sptr++ = pl;
                    *sptr++ = pi - 1;
                    pl = pi + 1;
                }
                *psdepth++ = --cdepth;
            }

            /* insertion sort */
            for (pi = pl + 1; pi <= pr; ++pi) {
                vp = *pi;
                pj = pi;
                pk = pi - 1;
                while (pj > pl&& COMPARE_LT(vp, *pk)) {
                    *pj-- = *pk--;
                }
                *pj = vp;
            }
            if (sptr == stack) {
                break;
            }
            pr = *(--sptr);
            pl = *(--sptr);
            cdepth = *(--psdepth);
            continue;
        }
        else {
            heapsort_<T>(pl, pr - pl + 1);
            if (sptr == stack) {
                break;
            }
            pr = *(--sptr);
            pl = *(--sptr);
            cdepth = *(--psdepth);
        }
    }

    return 0;
}


//-----------------------------------------------------------------------------------------------
// argsort (indirect quicksort)
template <typename T, typename UINDEX>
static int
aquicksort_(void* vv1, void* tosort1, int64_t num)
{
    T* vv = (T*)vv1;
    UINDEX* tosort = (UINDEX *) tosort1;
    T* v = vv;
    T vp;
    UINDEX* pl = tosort;
    UINDEX* pr = tosort + num - 1;
    UINDEX* stack[PYA_QS_STACK];
    UINDEX** sptr = stack;
    UINDEX* pm, * pi, * pj, * pk, vi;
    int depth[PYA_QS_STACK];
    int* psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (cdepth >= 0) {

            while ((pr - pl) > SMALL_QUICKSORT) {
                /* quicksort partition */
                pm = pl + ((pr - pl) >> 1);
                if (COMPARE_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
                if (COMPARE_LT(v[*pr], v[*pm])) INTP_SWAP(*pr, *pm);
                if (COMPARE_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
                vp = v[*pm];
                pi = pl;
                pj = pr - 1;
                INTP_SWAP(*pm, *pj);
                for (;;) {
                    do ++pi; while (COMPARE_LT(v[*pi], vp));
                    do --pj; while (COMPARE_LT(vp, v[*pj]));
                    if (pi >= pj) {
                        break;
                    }
                    INTP_SWAP(*pi, *pj);
                }
                pk = pr - 1;
                INTP_SWAP(*pi, *pk);
                /* push largest partition on stack */
                if (pi - pl < pr - pi) {
                    *sptr++ = pi + 1;
                    *sptr++ = pr;
                    pr = pi - 1;
                }
                else {
                    *sptr++ = pl;
                    *sptr++ = pi - 1;
                    pl = pi + 1;
                }
                *psdepth++ = --cdepth;
            }

            /* insertion sort */
            for (pi = pl + 1; pi <= pr; ++pi) {
                vi = *pi;
                vp = v[vi];
                pj = pi;
                pk = pi - 1;
                while (pj > pl&& COMPARE_LT(vp, v[*pk])) {
                    *pj-- = *pk--;
                }
                *pj = vi;
            }
        }
        else {
            aheapsort_<T, UINDEX>(vv, pl, (UINDEX)(pr - pl + 1));
        }

        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}



//--------------------------------------------------------------------------------------
template <typename T>
static void
mergesort0_(T* pl, T* pr, T* pw)
{
    T vp, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        mergesort0_(pl, pm, pw);
        mergesort0_(pm, pr, pw);

        if (COMPARE_LT(*pm, *(pm - 1))) {
            MEMCPYR(pw, pl, (pm - pl) * sizeof(T));

                pi = pw + (pm - pl);
                pj = pw;
                pk = pl;
                while (pj < pi && pm < pr) {
                    if (COMPARE_LT(*pm, *pj)) {
                        *pk++ = *pm++;
                    }
                    else {
                        *pk++ = *pj++;
                    }
                }
            while (pj < pi) {
                *pk++ = *pj++;
            }
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& COMPARE_LT(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    }
}


//--------------------------------------------------------------------------------------
// variable length merge sort
template <typename T>
static void
mergesort0string_(char* pl, char* pr, char* pw, int64_t strLen)
{
    char * pi, * pj, * pk, * pm;

    if (((pr - pl) / strLen) > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl +  strLen * (((pr - pl)/strLen) >> 1);
        mergesort0string_<T>(pl, pm, pw, strLen);
        mergesort0string_<T>(pm, pr, pw, strLen);

        if (STRING_LT((T*)pm, (T*)(pm - strLen), strLen)) {
            // copy front (pl) to worksapce
            MEMCPYR(pw, pl, pm - pl);

            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;
            while (pj < pi && pm < pr) {
                if (STRING_LT((T*)pm, (T*)pj, strLen)) {
                    STRING_COPY(pk, pm, strLen);
                    pk += strLen;
                    pm += strLen;
                }
                else {
                    STRING_COPY(pk, pj, strLen);
                    pk += strLen;
                    pj += strLen;
                }
            }
            while (pj < pi) {
                STRING_COPY(pk, pj, strLen);
                pk += strLen;
                pj += strLen;
            }
        }
    }
    else {

        /* insertion sort */
        for (pi = pl + strLen; pi < pr; pi += strLen) {
            // Copy into vp, a temp buffer
            STRING_COPY(pw, pi, strLen);
            pj = pi;
            pk = pi - strLen;
            while (pj > pl&& STRING_LT((T*)pw, (T*)pk, strLen)) {
                STRING_COPY(pj, pk, strLen);
                pj -= strLen;
                pk -= strLen;
            }
            STRING_COPY(pj, pw, strLen);
        }

    }
}

//-----------------------------------------------------------------------------------------------
template <typename T>
static int
mergesort_(void* start, int64_t length)
{
    T* pl, * pr, * pw;

    pl = (T*)start;
    pr = pl + length;

    // Consider alloc on stack
    const int64_t allocSize = (length / 2) * sizeof(T);
    pw = POSSIBLY_STACK_ALLOC_TYPE(T*, allocSize);
    if (pw == NULL) {
        return -1;
    }
    mergesort0_(pl, pr, pw);

    POSSIBLY_STACK_FREE(allocSize, pw);
    return 0;
}


//-----------------------------------------------------------------------------------------------
template <typename T>
static int
mergesortstring_(void* start, int64_t length, int64_t strLen)
{
    char* pl, * pr, * pw;

    pl = (char*)start;
    pr = pl + (length * strLen);

    // Consider alloc on stack
    const int64_t allocSize = (length / 2) * strLen;
    pw = (char*)WORKSPACE_ALLOC(allocSize);
    if (pw == NULL) {
        return -1;
    }
    mergesort0string_<T>(pl, pr, pw, strLen);

    WORKSPACE_FREE(pw);
    return 0;
}


//-----------------------------------------------------------------------------------------------
template <typename DATATYPE, typename UINDEX>
static void
amergesort0_string(UINDEX* pl, UINDEX* pr, const char* strItem, UINDEX* pw, int64_t strLen)
{
    const char* vp;
    UINDEX vi, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        //printf("merge sort %p %p %p diff:%lld\n", pl, pm, pr, pr-pl);
        amergesort0_string<DATATYPE>(pl, pm, strItem, pw, strLen);
        amergesort0_string<DATATYPE>(pm, pr, strItem, pw, strLen);
        pm = pl + ((pr - pl) >> 1);

        if (STRING_LT((DATATYPE)(strItem + (*pm) * strLen), (DATATYPE)(strItem + (*(pm - 1)) * strLen), strLen)) {

            MEMCPYR(pw, pl, (pm - pl) * sizeof(UINDEX));

            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;
            while (pj < pi && pm < pr) {
                if (STRING_LT((DATATYPE)(strItem + (*pm) * strLen), (DATATYPE)(strItem + (*pj) * strLen), strLen)) {
                    *pk++ = *pm++;
                }
                else {
                    *pk++ = *pj++;
                }
            }
            while (pj < pi) {
                *pk++ = *pj++;
            }
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = strItem + (vi * strLen);
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& STRING_LT((DATATYPE)vp, (DATATYPE)(strItem + (*pk) * strLen), strLen)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}



//-----------------------------------------------------------------------------------------------
// T= data type == int16,int32,uint32,int64.uint64
// UINDEX = int32_t or int64_t
template <typename T, typename UINDEX>
static void
amergesort0_(UINDEX* pl, UINDEX* pr, T* v, UINDEX* pw)
{
    T vp;
    UINDEX vi, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_(pl, pm, v, pw);
        amergesort0_(pm, pr, v, pw);

        // check if already sorted
        // if the first element on the right is less than the last element on the left
        //printf("comparing %d to %d ", (int)pm[0], (int)pm[-1]);
        if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {
            if ((pm - pl) >= 32) {
                MEMCPYR(pw, pl, (pm - pl) * sizeof(UINDEX));
            }
            else {
                // Copy left side into workspace
                pi = pw;
                pj = pl;
                while (pj < pm) {
                    *pi++ = *pj++;
                }
            }

            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;
            while (pj < pi && pm < pr) {
                if (COMPARE_LT(v[*pm], v[*pj])) {
                    *pk++ = *pm++;
                }
                else {
                    *pk++ = *pj++;
                }
            }
            while (pj < pi) {
                *pk++ = *pj++;
            }
        }

    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& COMPARE_LT(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

//-----------------------------------------------------------------------------------------------
// allocates workspace
template <typename T, typename UINDEX>
static int
amergesort_(void* v1, void* tosort1, int64_t length)
{
    T* v = (T*)v1;
    UINDEX* tosort = (UINDEX*)tosort1;

    UINDEX* pl, * pr, * pworkspace;

    pl = tosort;
    pr = pl + length;

    pworkspace = (UINDEX*)WORKSPACE_ALLOC((length / 2) * sizeof(UINDEX));
    if (pworkspace == NULL) {
        return -1;
    }
    amergesort0_<T, UINDEX>(pl, pr, v, pworkspace);
    WORKSPACE_FREE(pworkspace);

    return 0;
}

//-----------------------------------------------------------------------------------------------
// does not allocate workspace
template <typename T, typename UINDEX>
static void
amergesortworkspace_(void* v1, void* tosort1, int64_t length, int64_t notused, void* pworkspace)
{
    T* v = (T*)v1;
    UINDEX* tosort = (UINDEX*)tosort1;
    UINDEX* pl, * pr;

    pl = tosort;
    pr = pl + length;

    amergesort0_<T, UINDEX>(pl, pr, v, (UINDEX*)pworkspace);
}


//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename DATATYPE, typename UINDEX>
static void
ParMergeString(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    //PLOGGING("string calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_string<DATATYPE>(pl, pr, pValue, pWorkSpace, strlen);
}

//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename DATATYPE, typename UINDEX>
static void
ParMergeNormal(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    DATATYPE* pValue = (DATATYPE*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    PLOGGING("normal calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_(pl, pr, pValue, pWorkSpace);
}


//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename DATATYPE, typename UINDEX>
static void
ParMergeMergeString(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    UINDEX* pi, * pj, * pk, * pm;
    pm = pl + ((pr - pl) >> 1);

    PLOGGING("Comparing %s to %s  ALSO %s\n", pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, pValue + (*pm + 1) * strlen);

    if ( STRING_LT((DATATYPE)(pValue + (*pm) * strlen), (DATATYPE)(pValue + (*pm - 1) * strlen), strlen)) {

        // copy the left to workspace
        MEMCPYR(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

        // pi is end of workspace
        pi = pWorkSpace + (pm - pl);
        pj = pWorkSpace;
        pk = pl;

        while (pj < pi && pm < pr) {
            if (STRING_LT((DATATYPE)(pValue + (*pm) * strlen), (DATATYPE)(pValue + (*pj) * strlen), strlen)) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }

        //printf("last items %lld %lld %lld\n", pk[-3], pk[-2], pk[-1]);
    }
    else {
        PLOGGING("refusing to merge\n");
    }

}


//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
// T is type to sort -- int32_t, Float64, etc.
// UINDEX is the argsort index -- int32_t or int64_t often
//
template <typename T, typename UINDEX>
static void
ParMergeMerge(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    T* pValue = (T*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    UINDEX* pi, * pj, * pk, * pm;
    pm = pl + ((pr - pl) >> 1);


    PLOGGING("merging len %lld\n", totalLen);

    // quickcheck to see if we have to copy
    if (COMPARE_LT(pValue[*pm], pValue[*(pm - 1)])) {

        MEMCPYR(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

        // pi is end of workspace
        pi = pWorkSpace + (pm - pl);
        pj = pWorkSpace;
        pk = pl;

        while (pj < pi && pm < pr) {
            if (COMPARE_LT(pValue[*pm], pValue[*pj])) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }

        //printf("last items %lld %lld %lld\n", pk[-3], pk[-2], pk[-1]);
    }
    else {
        //printf("**Already sorted %lld\n", (int64_t)(*pm), (int64_t)*(pm - 1), (int64_t)pValue[*pm], (int64_t)pValue[*(pm - 1)]);
    }

}



//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
// T is type to sort -- int32_t, Float64, etc.
// UINDEX is the argsort index -- int32_t or int64_t often
//
template <typename T>
static void
ParInPlaceMerge(void* pValue1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    T* pl, * pr;

    T* pw = (T*)pWorkSpace1;

    pl = (T*)pValue1;
    pr = pl + totalLen;

    T* pi, * pj, * pk, * pm;
    pm = pl + ((pr - pl) >> 1);

    PLOGGING("merging len %lld\n", totalLen);

    // quickcheck to see if we have to copy
    if (COMPARE_LT(*pm, *(pm - 1))) {

        MEMCPYR(pw, pl, (pm - pl) * strlen);

        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (COMPARE_LT(*pm, *pj)) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }

        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        //printf("**Already sorted %lld\n", (int64_t)(*pm), (int64_t)*(pm - 1), (int64_t)pValue[*pm], (int64_t)pValue[*(pm - 1)]);
    }

}

typedef int(*SORT_FUNCTION)(void* pValue, int64_t length);
typedef int(*SORT_FUNCTION_STRING)(void* pValue, int64_t length, int64_t strlen);
struct SORT_FUNCTION_ANY {
    SORT_FUNCTION sortfunc;
    SORT_FUNCTION_STRING sortstringfunc;
};

typedef void(*SORT_STEP_TWO)(void* pValue1, int64_t totalLen, int64_t strlen, void* pWorkSpace1);
typedef void(*MERGE_STEP_ONE)(void* pValue, void* pToSort, int64_t num, int64_t strlen, void* pWorkSpace);
typedef void(*MERGE_STEP_TWO)(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1);

typedef int(*ARGSORT_FUNCTION)(void* pValue, void* pToSort, int64_t length);
typedef void(*ARGSORT_FUNCTION_STRING)(void* pValue, void* pToSort, int64_t length, int64_t strlen, void* pWorkSpace);

// One of them will be NULL
struct ARGSORT_FUNCTION_ANY {
    ARGSORT_FUNCTION argsortfunc;
    ARGSORT_FUNCTION_STRING argsortstringfunc;
    void init() {
        argsortfunc = NULL;
        argsortstringfunc = NULL;
    }
};


//========================================================================
//
enum PAR_SORT_TYPE {
    Normal = 0,
    Float = 1,
    String = 2,
    Unicode = 3,
    Void = 4
};

struct MERGE_SPLITTER {
    // used to synchronize parallel merges
    int     NumLevels;
    int     NumCores;
    int64_t EndPositions[17];
    int64_t Level[4];

    // How many threads to use
    // Caller can set numCores ==0 to auto pick
    void SetLevels(int64_t arrayLength, int numCores) {

        NumCores = numCores;
        if (NumCores == 0) {
            //NOTE set this value to 2,4 or 8 (or 16) ?
            NumCores = THREADER->GetNumCores();
            int maxCores = THREADER->GetFutexWakeup() + 1;
            if (maxCores < NumCores) {
                NumCores = maxCores;
            }
        }

        if (NumCores <= 2) {
            NumCores = 2;
            NumLevels = 1;
        }
        else if (NumCores <= 4) {
            NumCores = 4;
            NumLevels = 2;
        }
        else if (NumCores <= 8) {
            NumCores = 8;
            NumLevels = 3;
        }
        else {
            NumCores = 16;
            NumLevels = 4;
        }

        //stParMergeCallback.MergeBlocks = 8;

        for (int i = 0; i < 4; i++) {
            Level[i] = 0;
        }

        EndPositions[0] = 0;
        switch (NumLevels) {
        case 1:
            EndPositions[1] = arrayLength;
            break;
        case 2:
            // We use an 8 way merge, we need the size breakdown
            EndPositions[4] = arrayLength;
            EndPositions[2] = arrayLength / 2;
            EndPositions[3] = EndPositions[4] + (arrayLength - EndPositions[4]) / 2;
            break;
        case 3:
            // We use an 8 way merge, we need the size breakdown
            EndPositions[8] = arrayLength;
            EndPositions[4] = arrayLength / 2;
            EndPositions[6] = EndPositions[4] + (arrayLength - EndPositions[4]) / 2;
            EndPositions[2] = 0 + (EndPositions[4] - 0) / 2;
            EndPositions[7] = EndPositions[6] + (arrayLength - EndPositions[6]) / 2;
            EndPositions[5] = EndPositions[4] + (EndPositions[6] - EndPositions[4]) / 2;
            EndPositions[3] = EndPositions[2] + (EndPositions[4] - EndPositions[2]) / 2;
            EndPositions[1] = 0 + (EndPositions[2] - 0) / 2;
            break;
        case 4:
            // We use an 8 way merge, we need the size breakdown
            EndPositions[16] = arrayLength;
            EndPositions[8] = arrayLength / 2;
            EndPositions[12] = EndPositions[8] + (arrayLength - EndPositions[8]) / 2;
            EndPositions[4] = 0 + (EndPositions[8] - 0) / 2;
            EndPositions[14] = EndPositions[12] + (arrayLength - EndPositions[12]) / 2;
            EndPositions[10] = EndPositions[8] + (EndPositions[12] - EndPositions[8]) / 2;
            EndPositions[6] = EndPositions[4] + (EndPositions[8] - EndPositions[4]) / 2;
            EndPositions[2] = 0 + (EndPositions[4] - 0) / 2;

            EndPositions[15] = EndPositions[14] + (EndPositions[16] - EndPositions[14]) / 2;
            EndPositions[13] = EndPositions[12] + (EndPositions[14] - EndPositions[12]) / 2;
            EndPositions[11] = EndPositions[10] + (EndPositions[12] - EndPositions[10]) / 2;
            EndPositions[9] = EndPositions[0] + (EndPositions[10] - EndPositions[8]) / 2;
            EndPositions[7] = EndPositions[6] + (EndPositions[8] - EndPositions[6]) / 2;
            EndPositions[5] = EndPositions[4] + (EndPositions[6] - EndPositions[4]) / 2;
            EndPositions[3] = EndPositions[2] + (EndPositions[4] - EndPositions[2]) / 2;
            EndPositions[1] = EndPositions[0] + (EndPositions[2] - EndPositions[0]) / 2;

            break;
        }
    }
};

//--------------------------------------------------------------------
struct MERGE_STEP_ONE_CALLBACK {
    union {
        ARGSORT_FUNCTION_ANY ArgSortCallbackOne;
        SORT_FUNCTION_ANY  SortCallbackOne;
    };
    union {
        MERGE_STEP_TWO MergeCallbackTwo;
        SORT_STEP_TWO SortCallbackTwo;
    };
    void* pValues;
    void* pToSort;
    int64_t ArrayLength;

    // set to 0 if not a string, otherwise the string length
    int64_t StrLen;

    // pointer to the merge workspace (usually half array length in size)
    char* pWorkSpace;

    PAR_SORT_TYPE  SortType;

    // how much was used per 1/8 chunk when allocating the workspace
    int64_t AllocChunk;
    int64_t TypeSizeInput;

    // not valid for inplace sorting, otherwise is sizeof(int32) or sizeof(int64) depending on index size
    int64_t TypeSizeOutput;

    // used to synchronize parallel merges
    MERGE_SPLITTER  MergeSplitter;
    //int64_t EndPositions[9];
    //int64_t Level[3];

};


//------------------------------------------------------------------------------
// Checks to see if adjacent bit is set
// Used when merging sorts
static bool IsBuddyBitSet(int64_t index, int64_t* pBitMask) {
    int64_t bitshift = 1LL << index;

    // Now find the buddy bit (adjacent bit)
    int64_t buddy = 0;
    if (index & 1) {
        buddy = 1LL << (index - 1);
    }
    else {
        buddy = 1LL << (index + 1);
    }

    // Get back which bits were set before the OR operation
    int64_t result = FMInterlockedOr(pBitMask, bitshift);

    // Check if our buddy was already set
    PLOGGING("index -- LEVEL 1: %lld  %lld %lld -- %s\n", index, buddy, (result & buddy), buddy == (result & buddy) ? "GOOD" : "WAIT");

    return (buddy == (result & buddy));
}


//=========================================
// stridesOut is contiguous and should be itemSize
// stridesIn may be non-contiguous
//
static void
CopyData(
    void* pValues,
    int64_t         arrayLength,
    int64_t         stridesIn,
    int64_t         itemSize,
    void* pOut,
    int64_t         stridesOut) {

    if (stridesIn == stridesOut) {
        MEMCPYR(pOut, pValues, arrayLength * itemSize);
    }
    else {
        // TODO: Check for string?
        if (stridesOut == itemSize) {
            // strided copy
            char* pDataIn = (char*)pValues;
            char* pDataOut = (char*)pOut;
            char* pDataLast = pDataOut + (stridesOut * arrayLength);
            switch (stridesOut) {
            case 1:
                while (pDataOut != pDataLast) {
                    *pDataOut = *pDataIn;
                    pDataOut = STRIDE_NEXT(char, pDataOut, stridesOut);
                    pDataIn = STRIDE_NEXT(char, pDataIn, stridesIn);
                }
                break;
            case 2:
                while (pDataOut != pDataLast) {
                    *(int16_t*)pDataOut = *(int16_t*)pDataIn;
                    pDataOut = STRIDE_NEXT(char, pDataOut, stridesOut);
                    pDataIn = STRIDE_NEXT(char, pDataIn, stridesIn);
                }
                break;
            case 4:
                while (pDataOut != pDataLast) {
                    *(int32_t*)pDataOut = *(int32_t*)pDataIn;
                    pDataOut = STRIDE_NEXT(char, pDataOut, stridesOut);
                    pDataIn = STRIDE_NEXT(char, pDataIn, stridesIn);
                }
                break;
            case 8:
                while (pDataOut != pDataLast) {
                    *(int64_t*)pDataOut = *(int64_t*)pDataIn;
                    pDataOut = STRIDE_NEXT(char, pDataOut, stridesOut);
                    pDataIn = STRIDE_NEXT(char, pDataIn, stridesIn);
                }
                break;
            default:
                while (pDataOut != pDataLast) {
                    MEMCPYR(pDataOut, pDataIn, itemSize);
                    pDataOut = STRIDE_NEXT(char, pDataOut, stridesOut);
                    pDataIn = STRIDE_NEXT(char, pDataIn, stridesIn);
                }
                break;
            }
        }
        else {
            // strided copy
            char* pDataIn = (char*)pValues;
            char* pDataOut = (char*)pOut;
            char* pDataLast = STRIDE_NEXT(char, pDataOut, stridesOut * arrayLength);
            while (pDataOut != pDataLast) {
                MEMCPYR(pDataOut, pDataIn, itemSize);
                pDataOut = STRIDE_NEXT(char, pDataOut, stridesOut);
                pDataIn = STRIDE_NEXT(char, pDataIn, stridesIn);
            }

        }
    }
}

//------------------------------------------------------------------------------
// Concurrent callback from multiple threads
// this routine is for indirect sorting
static int64_t ParArgSortCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    MERGE_STEP_ONE_CALLBACK* Callback = (MERGE_STEP_ONE_CALLBACK*)pstWorkerItem->WorkCallbackArg;
    int64_t didSomeWork = FALSE;

    int64_t index;
    int64_t workBlock;

    // As long as there is work to do
    while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0) {
        // First index is 1 so we subtract
        index--;

        // the very first index starts at 0
        int64_t pFirst =  Callback->MergeSplitter.EndPositions[index];
        int64_t pSecond = Callback->MergeSplitter.EndPositions[index+1];

        PLOGGING("[%d] DoWork start loop -- %lld  index: %lld   pFirst: %lld   pSecond: %lld\n", core, workIndex, index, pFirst, pSecond);

        char* pToSort1 = (char*)(Callback->pToSort);

        int64_t MergeSize = (pSecond - pFirst);
        PLOGGING("%d : MergeOne index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);

        // Workspace uses half the size
        char* pWorkSpace1 = Callback->pWorkSpace + (index * Callback->AllocChunk * Callback->TypeSizeOutput);

        if (Callback->ArgSortCallbackOne.argsortfunc) {
            Callback->ArgSortCallbackOne.argsortfunc(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize);
        }
        else {
            Callback->ArgSortCallbackOne.argsortstringfunc(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);
        }

        if (IsBuddyBitSet(index, &Callback->MergeSplitter.Level[0])) {

            // Move to next level -- 4 things to sort
            index = index / 2;
            pWorkSpace1 = Callback->pWorkSpace + (index * 2 * Callback->AllocChunk * Callback->TypeSizeOutput);

            pFirst = Callback->MergeSplitter.EndPositions[index*2];
            pSecond = Callback->MergeSplitter.EndPositions[index*2 + 2];
            MergeSize = (pSecond - pFirst);

            PLOGGING("size:%lld  first: %lld  second: %lld   expected: %lld\n", MergeSize, pFirst, pSecond, pFirst + (MergeSize >> 1));

            Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

            if (Callback->MergeSplitter.NumLevels > 1 && IsBuddyBitSet(index, &Callback->MergeSplitter.Level[1])) {
                index /= 2;

                // Move to next level -- 2 things to sort
                pWorkSpace1 = Callback->pWorkSpace + (index * 4 * Callback->AllocChunk * Callback->TypeSizeOutput);

                pFirst = Callback->MergeSplitter.EndPositions[index*4];
                pSecond = Callback->MergeSplitter.EndPositions[index*4 + 4];
                MergeSize = (pSecond - pFirst);

                PLOGGING("%d : MergeThree index: %llu  %lld  %lld\n", core, index, pFirst, MergeSize);
                //pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
                PLOGGING("Level 2 %p %p,  size: %lld,  pworkspace: %p\n", Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, pWorkSpace1);
                Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

                if (Callback->MergeSplitter.NumLevels > 2 && IsBuddyBitSet(index, &Callback->MergeSplitter.Level[2])) {
                    // Final merge
                    PLOGGING("%d : MergeFinal index: %llu  %lld  %lld  %lld\n", core, index, 0LL, Callback->ArrayLength, 0LL);
                    Callback->MergeCallbackTwo(Callback->pValues, Callback->pToSort, Callback->ArrayLength, Callback->StrLen, Callback->pWorkSpace);
                }
            }
        }

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock(core);
    }

    return didSomeWork;
}



//------------------------------------------------------------------------------
// Concurrent callback from multiple threads
// this routine is for inplace sorting
static int64_t ParMergeInPlaceThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    MERGE_STEP_ONE_CALLBACK* Callback = (MERGE_STEP_ONE_CALLBACK*)pstWorkerItem->WorkCallbackArg;
    int64_t didSomeWork = FALSE;

    int64_t index;
    int64_t workBlock;

    // As long as there is work to do
    while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0) {
        // First index is 1 so we subtract
        index--;

        int64_t itemSizeInput = Callback->TypeSizeInput;
        int64_t strLen = Callback->StrLen;
        char* pSrcData = (char*)(Callback->pValues);
        char* pData = (char*)(Callback->pToSort);

        // the very first index starts at 0
        int64_t pFirst = Callback->MergeSplitter.EndPositions[index];
        int64_t pSecond = Callback->MergeSplitter.EndPositions[index + 1];

        PLOGGING("[%d] DoWork start loop -- %lld  index: %lld   pFirst: %lld   pSecond: %lld\n", core, workIndex, index, pFirst, pSecond);

        int64_t MergeSize = (pSecond - pFirst);
        PLOGGING("%d : MergeOne index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);

        // Workspace uses half the size
        char* pWorkSpace1 = Callback->pWorkSpace + (index * Callback->AllocChunk * itemSizeInput);

        // Copy data over
        // TODO: Fix for strided data
        // For a pure inplace sort, pSrcData will be the same as pData
        if (pSrcData != pData)
            CopyData(pSrcData + (pFirst * itemSizeInput), MergeSize, Callback->TypeSizeInput, Callback->StrLen, pData + (pFirst * itemSizeInput), Callback->TypeSizeOutput);

        if (Callback->SortCallbackOne.sortfunc) {
            Callback->SortCallbackOne.sortfunc(pData + (pFirst * itemSizeInput), MergeSize);
        }
        else {
            Callback->SortCallbackOne.sortstringfunc(pData + (pFirst * itemSizeInput), MergeSize, strLen);
        }

        if (IsBuddyBitSet(index, &Callback->MergeSplitter.Level[0])) {
            // Move to next level -- 4 things to sort
            index = index / 2;
            pWorkSpace1 = Callback->pWorkSpace + (index * 2 * Callback->AllocChunk * itemSizeInput);

            pFirst = Callback->MergeSplitter.EndPositions[index * 2];
            pSecond = Callback->MergeSplitter.EndPositions[index * 2 + 2];
            MergeSize = (pSecond - pFirst);

            PLOGGING("size:%lld  first: %lld  second: %lld   expected: %lld\n", MergeSize, pFirst, pSecond, pFirst + (MergeSize >> 1));

            Callback->SortCallbackTwo(pData + (pFirst * itemSizeInput), MergeSize, strLen, pWorkSpace1);

            if (Callback->MergeSplitter.NumLevels > 1 && IsBuddyBitSet(index, &Callback->MergeSplitter.Level[1])) {
                index /= 2;

                // Move to next level -- 2 things to sort
                pWorkSpace1 = Callback->pWorkSpace + (index * 4 * Callback->AllocChunk * itemSizeInput);

                pFirst = Callback->MergeSplitter.EndPositions[index * 4];
                pSecond = Callback->MergeSplitter.EndPositions[index * 4 + 4];
                MergeSize = (pSecond - pFirst);

                PLOGGING("%d : MergeThree index: %llu  %lld  %lld\n", core, index, pFirst, MergeSize);
                Callback->SortCallbackTwo(pData + (pFirst * itemSizeInput), MergeSize, strLen, pWorkSpace1);

                if (Callback->MergeSplitter.NumLevels > 2 && IsBuddyBitSet(index, &Callback->MergeSplitter.Level[2])) {
                    // Final merge
                    PLOGGING("%d : MergeFinal index: %llu  %lld  %lld  %lld\n", core, index, 0LL, Callback->ArrayLength, 0LL);
                    Callback->SortCallbackTwo(pData, Callback->ArrayLength, strLen, Callback->pWorkSpace);
                }
            }
        }

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock(core);
    }

    return didSomeWork;
}


typedef int(*SINGLE_MERGESORT)(
    void* pValuesT,
    void* pToSortU,
    int64_t          arrayLength,
    int64_t          strlen,
    PAR_SORT_TYPE  sortType);

//------------------------------------------------------------------------
// single threaded version
// T is the dtype int32/float32/float64/etc.
// UINDEX is either int32_t or int64_t
// Returns -1 on failure
template <typename T, typename UINDEX>
static int
single_amergesort(
    void* pValuesT,
    void* pToSortU,
    int64_t          arrayLength,
    int64_t          strlen,
    PAR_SORT_TYPE  sortType)
{
    T* pValues = (T*)pValuesT;
    UINDEX* pToSort = (UINDEX*)pToSortU;

    // single threaded sort
    UINDEX* pWorkSpace;

    pWorkSpace = (UINDEX*)WORKSPACE_ALLOC((arrayLength / 2) * sizeof(UINDEX));
    if (pWorkSpace == NULL) {
        return -1;
    }

    switch (sortType) {
    case PAR_SORT_TYPE::String:
        amergesort0_string<const unsigned char*>(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
        break;
    case PAR_SORT_TYPE::Unicode:
        amergesort0_string< const uint32_t* > (pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
        break;
    case PAR_SORT_TYPE::Void:
        amergesort0_string<const char *>(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
        break;
    default:
        amergesort0_(pToSort, pToSort + arrayLength, pValues, pWorkSpace);
    }

    WORKSPACE_FREE(pWorkSpace);
    return 0;
}


//------------------------------------------------------------------------
// parallel version
// if strlen==0, then not string (int or float)
// If pCutOffs is not null, will go parallel per partition
// If pCutOffs is null, the entire array is sorted
// If the array is large enough, a parallel merge sort is invoked
// Returns -1 on failure
template <typename T, typename UINDEX>
static int
par_amergesort(
    int64_t* pCutOffs,      // May be NULL (if so no partitions)
    int64_t          cutOffLength,

    T* pValues,
    UINDEX* pToSort,

    int64_t          arrayLength,
    int64_t          strlen,
    PAR_SORT_TYPE  sortType,
    ARGSORT_FUNCTION_ANY argSortStepOne)
{
    if (pCutOffs) {
        PLOGGING("partition version col: %lld  %p  %p  %p\n", cutOffLength, pToSort, pToSort + arrayLength, pValues);

        struct stPSORT {
            SINGLE_MERGESORT  funcSingleMerge;
            int64_t* pCutOffs;       // May be NULL (if so no partitions)
            int64_t             cutOffLength;

            char* pValues;
            char* pToSort;
            int64_t             strlen;
            PAR_SORT_TYPE     sortType;

            int64_t             sizeofT;
            int64_t             sizeofUINDEX;

        } psort;

        psort.funcSingleMerge = single_amergesort<T, UINDEX>;
        psort.pCutOffs = pCutOffs;
        psort.cutOffLength = cutOffLength;
        psort.pValues = (char*)pValues;
        psort.pToSort = (char*)pToSort;
        psort.strlen = strlen;
        psort.sortType = sortType;

        psort.sizeofUINDEX = sizeof(UINDEX);
        if (strlen > 0) {
            psort.sizeofT = strlen;
        }
        else {
            psort.sizeofT = sizeof(T);
        }

        // Use threads per partition
        auto lambdaPSCallback = [](void* callbackArgT, int core, int64_t workIndex) -> int64_t {
            stPSORT* callbackArg = (stPSORT*)callbackArgT;
            int64_t t = workIndex;
            int64_t partLength;
            int64_t partStart;

            if (t == 0) {
                partStart = 0;
            }
            else {
                partStart = callbackArg->pCutOffs[t - 1];
            }

            partLength = callbackArg->pCutOffs[t] - partStart;

            PLOGGING("[%lld] start: %lld  length: %lld\n", t, partStart, partLength);

            // shift the data pointers to match the partition
            // call a single threaded merge
            callbackArg->funcSingleMerge(
                callbackArg->pValues + (partStart * callbackArg->sizeofT),
                callbackArg->pToSort + (partStart * callbackArg->sizeofUINDEX),
                partLength,
                callbackArg->strlen,
                callbackArg->sortType);

            return 1;
        };

        THREADER->DoMultiThreadedWork((int)cutOffLength, lambdaPSCallback, &psort);

    }
    else {

        // If size is large, go parallel
        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(arrayLength);
        if (pWorkItem != NULL) {
            MERGE_STEP_ONE_CALLBACK stParMergeCallback;
            stParMergeCallback.MergeSplitter.SetLevels(arrayLength, 8);

            PLOGGING("Parallel version  %p  %p  %p\n", pToSort, pToSort + arrayLength, pValues);
            // Divide into 8 jobs
            // Allocate all memory up front
            // Allocate enough for 8 
            int64_t allocChunk = (arrayLength / (stParMergeCallback.MergeSplitter.NumCores * 2)) + 1;
            void* pWorkSpace = NULL;

            // Allocate half the size since the workspace is only needed for left
            uint64_t allocSize = allocChunk * stParMergeCallback.MergeSplitter.NumCores * sizeof(UINDEX);
            pWorkSpace = WORKSPACE_ALLOC(allocSize);

            if (pWorkSpace == NULL) {
                return -1;
            }

            pWorkItem->DoWorkCallback = ParArgSortCallback;
            pWorkItem->WorkCallbackArg = &stParMergeCallback;

            stParMergeCallback.SortType = sortType;
            stParMergeCallback.ArgSortCallbackOne = argSortStepOne;
            switch (sortType) {

            case PAR_SORT_TYPE::String:
                stParMergeCallback.MergeCallbackTwo = ParMergeMergeString< const unsigned char*, UINDEX>;
                break;
            case PAR_SORT_TYPE::Unicode:
                stParMergeCallback.MergeCallbackTwo = ParMergeMergeString< const uint32_t*, UINDEX>;
                break;
            case PAR_SORT_TYPE::Void:
                stParMergeCallback.MergeCallbackTwo = ParMergeMergeString< const char*, UINDEX>;
                break;
            default:
                // Last Merge
                stParMergeCallback.MergeCallbackTwo = ParMergeMerge<T, UINDEX>;
            };


            stParMergeCallback.pValues = pValues;
            stParMergeCallback.pToSort = pToSort;
            stParMergeCallback.ArrayLength = arrayLength;
            stParMergeCallback.StrLen = strlen;
            stParMergeCallback.AllocChunk = allocChunk;
            stParMergeCallback.pWorkSpace = (char*)pWorkSpace;
            stParMergeCallback.TypeSizeInput = sizeof(T);
            if (strlen) {
                stParMergeCallback.TypeSizeInput = strlen;
            }
            stParMergeCallback.TypeSizeOutput = sizeof(UINDEX);

            // This will notify the worker threads of a new work item
            // Default thead wakeup to 7
            THREADER->WorkMain(pWorkItem, stParMergeCallback.MergeSplitter.NumCores, (int32_t)(stParMergeCallback.MergeSplitter.NumCores - 1), 1, FALSE);

            // Free temp memory used
            WORKSPACE_FREE(pWorkSpace);

        }
        else {
            // TODO
            // Call merge step one
            // single threaded sort

            if (argSortStepOne.argsortfunc) {
                argSortStepOne.argsortfunc(
                    pValues,
                    pToSort,
                    arrayLength);
            }
            else {
                // NOTE: currently all strings use a mergesort since the SWAP function for quicksort is so slow
                // Allocate half the size since the workspace is only needed for left
                uint64_t allocSize = (arrayLength / 2) * sizeof(UINDEX);
                void* pWorkSpace = WORKSPACE_ALLOC(allocSize);

                if (pWorkSpace == NULL) {
                    return -1;
                }

                argSortStepOne.argsortstringfunc(
                    pValues,
                    pToSort,
                    arrayLength,
                    strlen,
                    pWorkSpace);

                WORKSPACE_FREE(pWorkSpace);

            }
            //return
            //    single_amergesort<T, UINDEX>(
            //        pValues,
            //        pToSort,
            //        arrayLength,
            //        strlen,
            //        sortType);
        }
    }
    return 0;
}



//------------------------------------------------------------------------
// parallel version
// if strlen==0, then not string (int or float)
// If the array is large enough, a parallel quick sort is invoked
// if pValues == pOut, then no copy is performed
// Returns -1 on failure
template <typename DATATYPE>
static int
par_sort(
    void*           pValues,
    int64_t         arrayLength,
    int64_t         stridesIn,
    int64_t         itemSize,
    void*           pOut,
    int64_t         stridesOut,
    SORT_FUNCTION_ANY   pSortFunction)
{
    // If size is large, go parallel
    LOGGING("Parallel version  %p  %p  %p\n", pSortFunction.sortfunc, pOut , pValues);

    stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(arrayLength);

    if (pWorkItem) {
        MERGE_STEP_ONE_CALLBACK stParMergeCallback;
        //NOTE set this value to 2,4 or 8
        stParMergeCallback.MergeSplitter.SetLevels(arrayLength, 8);
        LOGGING("!! sort with %d cores\n", stParMergeCallback.MergeSplitter.NumCores);

        // Divide into 8 jobs
        // Allocate all memory up front
        // Allocate enough for 8 
        int64_t allocChunk = (arrayLength / (stParMergeCallback.MergeSplitter.NumCores * 2)) + 1;
        void* pWorkSpace = NULL;

        // Allocate half the size since the workspace is only needed for left
        uint64_t allocSize = allocChunk * stParMergeCallback.MergeSplitter.NumCores * itemSize;
        pWorkSpace = WORKSPACE_ALLOC(allocSize);

        if (pWorkSpace == NULL) {
            return -1;
        }

        pWorkItem->DoWorkCallback = ParMergeInPlaceThreadCallback;
        pWorkItem->WorkCallbackArg = &stParMergeCallback;

        // First pass is a quicksort in place
        stParMergeCallback.SortCallbackOne = pSortFunction;

        // second pass is a mergesort in place
        stParMergeCallback.SortCallbackTwo = ParInPlaceMerge<DATATYPE>;

        stParMergeCallback.pValues = pValues;
        stParMergeCallback.pToSort = pOut;
        stParMergeCallback.ArrayLength = arrayLength;
        stParMergeCallback.StrLen = itemSize;
        stParMergeCallback.AllocChunk = allocChunk;
        stParMergeCallback.pWorkSpace = (char*)pWorkSpace;
        stParMergeCallback.TypeSizeInput = itemSize;
        stParMergeCallback.TypeSizeOutput = stridesOut;

        // This will notify the worker threads of a new work item
        // Default thead wakeup to 7
        THREADER->WorkMain(pWorkItem, stParMergeCallback.MergeSplitter.NumCores, (int32_t)(stParMergeCallback.MergeSplitter.NumCores-1), 1, FALSE);

        // Free temp memory used
        WORKSPACE_FREE(pWorkSpace);
        return 0;
    }

    // No threading path

    // Check if in place or a copy needs to be done
    if (pValues != pOut) {
        CopyData(
            pValues,
            arrayLength,
            stridesIn,
            itemSize,
            pOut,
            stridesOut);
    }

    // single threaded sort
    if (pSortFunction.sortfunc)
        return  pSortFunction.sortfunc(pOut, arrayLength);
    if (pSortFunction.sortstringfunc)
        return  pSortFunction.sortstringfunc(pOut, arrayLength, itemSize);
    printf("!!sort failed %p\n", pSortFunction.sortfunc);
    return -1;
}


//-----------------------------------------------------------------------------------------------
// Sorts in place
// returns a thread safe function to sort based on mode and <DATATYPE>
template <typename DATATYPE>
static SORT_FUNCTION
SortInPlace( SORT_MODE mode) {

    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        return quicksort_<DATATYPE>;

    case SORT_MODE::SORT_MODE_MERGE:
        return mergesort_<DATATYPE>;
        break;

    case SORT_MODE::SORT_MODE_HEAP:
        return  heapsort_<DATATYPE>;
    }

    return NULL;
}

//-----------------------------------------------------------------------------------------------
// Sorts in place
// returns a thread safe function to sort based on mode and <DATATYPE>
template <typename DATATYPE>
static SORT_FUNCTION_STRING
SortInPlaceString(SORT_MODE mode) {
    return mergesortstring_<DATATYPE>;
    //switch (mode) {
    //case SORT_MODE::SORT_MODE_QSORT:
    //    return quicksortstring_<DATATYPE>;
    // Only merge works well
    //case SORT_MODE::SORT_MODE_MERGE:
    //    return mergesortstring_<DATATYPE>;
    //    break;

    //case SORT_MODE::SORT_MODE_HEAP:
    //    return  heapsortstring_<DATATYPE>;
    //}

    //return NULL;
}

//-----------------------------------------------------------------------------------------------
template <typename DATATYPE, typename UINDEX>
static int
SortIndex(
    int64_t* pCutOffs, int64_t cutOffLength, void* pDataIn1, UINDEX* toSort, int64_t arraySize1, SORT_MODE mode) {

    ARGSORT_FUNCTION_ANY    argsort ;
    argsort.init();


    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        argsort.argsortfunc = aquicksort_<DATATYPE, UINDEX>;
        break;

    case SORT_MODE::SORT_MODE_HEAP:
        argsort.argsortfunc = aheapsort_<DATATYPE, UINDEX>;
        break;
    
    default:
        //argsort.argsortfunc = amergesort_<DATATYPE, UINDEX>;
        argsort.argsortstringfunc = amergesortworkspace_<DATATYPE, UINDEX>;
        break;

    }

    int result = 0;
    result = par_amergesort<DATATYPE, UINDEX>(pCutOffs, cutOffLength, (DATATYPE*)pDataIn1, (UINDEX*)toSort, arraySize1, 0, PAR_SORT_TYPE::Normal, argsort);

    if (result != 0) {
        LOGGING("**Error sorting.  size %llu   mode %d\n", (int64_t)arraySize1, mode);
    }

    return result;
}

//------------------------------------------------------------------------------------------
// Internal and can be called from groupby
// caller must allocate the pDataOut1 as int64_t with size arraySize1
// UINDEX = int32_t or int64_t
template <typename UINDEX>
static int SortIndex(
    int64_t*   pCutOffs,
    int64_t    cutOffLength,
    void*      pDataIn1,
    UINDEX     arraySize1,
    UINDEX*    pDataOut1,
    SORT_MODE  mode,
    int        arrayType1,
    int64_t    strlen) {


    ARGSORT_FUNCTION_ANY mergeStepOne;
    mergeStepOne.init();


    switch (arrayType1) {
    case ATOP_UNICODE:
        mergeStepOne.argsortstringfunc = ParMergeString<const uint32_t*, UINDEX>;
        return par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, pDataOut1, arraySize1, strlen, PAR_SORT_TYPE::Unicode, mergeStepOne);
    case ATOP_STRING:
        mergeStepOne.argsortstringfunc = ParMergeString<const unsigned char*, UINDEX>;
        return par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, pDataOut1, arraySize1, strlen, PAR_SORT_TYPE::String, mergeStepOne);
    case ATOP_VOID:
        mergeStepOne.argsortstringfunc = ParMergeString<const char*, UINDEX>;
        return par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, pDataOut1, arraySize1, strlen, PAR_SORT_TYPE::Void, mergeStepOne);
    case ATOP_BOOL:
    case ATOP_INT8:
        return SortIndex<int8_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_INT16:
        return SortIndex<int16_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_INT32:
        return SortIndex<int32_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_INT64:
        return SortIndex<int64_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_UINT8:
        return SortIndex<uint8_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_UINT16:
        return SortIndex<uint16_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_UINT32:
        return SortIndex<uint32_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_UINT64:
        return SortIndex<uint64_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_FLOAT:
        return SortIndex<float, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_DOUBLE:
        return SortIndex<double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    case ATOP_LONGDOUBLE:
        return SortIndex<long double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
    default:
        LOGGING("SortIndex does not understand type %d\n", arrayType1);
    }
    return -1;
}

// Stub for lexsort to return int32
extern "C" int SortIndex32(
    int64_t * pCutOffs,
    int64_t     cutOffLength,
    void* pDataIn1,
    int64_t     arraySize1,
    int32_t * pDataOut1,
    SORT_MODE   mode,
    int         arrayType1,
    int64_t     strlen) {

    return SortIndex<int32_t>(pCutOffs, cutOffLength, pDataIn1, (int32_t)arraySize1, pDataOut1, mode, arrayType1, strlen);
}

// Stub for lexsort to return int64
extern "C" int SortIndex64(
    int64_t * pCutOffs,
    int64_t     cutOffLength,
    void* pDataIn1,
    int64_t     arraySize1,
    int64_t * pDataOut1,
    SORT_MODE   mode,
    int         arrayType1,
    int64_t     strlen) {

    return SortIndex<int64_t>(pCutOffs, cutOffLength, pDataIn1, arraySize1, pDataOut1, mode, arrayType1, strlen);
}



//================================================================================================
//===============================================================================
// TODO: Build table ahead of time
//
static SORT_FUNCTION GetSortFunction(void* pDataIn1, int64_t arraySize1, int32_t atype, SORT_MODE mode) {
    switch (atype) {
    case ATOP_BOOL:
        return SortInPlace<bool>(mode);
    case ATOP_INT8:
        return SortInPlace<int8_t>(mode);
    case ATOP_INT16:
        return SortInPlace<int16_t>(mode);
    case ATOP_INT32:
        return SortInPlace<int32_t>(mode);
    case ATOP_INT64:
        return SortInPlace<int64_t>(mode);
    case ATOP_UINT8:
        return SortInPlace<uint8_t>(mode);
    case ATOP_UINT16:
        return SortInPlace<uint16_t>(mode);
    case ATOP_UINT32:
        return SortInPlace<uint32_t>(mode);
    case ATOP_UINT64:
        return SortInPlace<uint64_t>(mode);
    case ATOP_FLOAT:
        return SortInPlace<float>(mode);
    case ATOP_DOUBLE:
        return SortInPlace<double>(mode);
    case ATOP_LONGDOUBLE:
        return SortInPlace<long double>(mode);
    case ATOP_HALF_FLOAT:
        return SortInPlace<atop_half>(mode);
    default:
        LOGGING("SortArray does not understand type %d\n", atype);
        return NULL;
        break;
    }
    return NULL;
}

static SORT_FUNCTION_STRING GetSortFunctionString(void* pDataIn1, int64_t arraySize1, int32_t atype, SORT_MODE mode) {
    switch (atype) {
    case ATOP_STRING:
        return SortInPlaceString<unsigned char>(mode);
    case ATOP_UNICODE:
        return SortInPlaceString<uint32_t>(mode);
    default:
        LOGGING("SortArray does not understand type %d\n", atype);
        return NULL;
        break;
    }
    return NULL;
}

//===============================================================================
// Returns < 0 on error
extern "C" int Sort(
    SORT_MODE sortmode,
    int atype,
    void* pDataIn1,
    int64_t arraySize1,
    int64_t stridesIn,
    int64_t itemSize,
    void* pDataOut1,
    int64_t stridesOut) {

    SORT_FUNCTION_ANY pSortFunction = { NULL, NULL };
    LOGGING("sort called for %d %d\n", atype, sortmode);

    pSortFunction.sortfunc= GetSortFunction(pDataIn1, arraySize1, atype, sortmode);
    if (!pSortFunction.sortfunc) {
        pSortFunction.sortstringfunc = GetSortFunctionString(pDataIn1, arraySize1, atype, sortmode);
        if (!pSortFunction.sortstringfunc) {
            return -1;
        }
    }

    switch (atype) {
    case ATOP_BOOL:
        return par_sort<bool>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_INT8:
        return par_sort<int8_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_INT16:
        return par_sort<int16_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_INT32:
        return par_sort<int32_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_INT64:
        return par_sort<int64_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_UINT8:
        return par_sort<uint8_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_UINT16:
        return par_sort<uint16_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_UINT32:
        return par_sort<uint32_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_UINT64:
        return par_sort<uint64_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_FLOAT:
        return par_sort<float>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_DOUBLE:
        return par_sort<double>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_LONGDOUBLE:
        return par_sort<long double>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_HALF_FLOAT:
        return par_sort<atop_half>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_STRING:
        return par_sort<unsigned char>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    case ATOP_UNICODE:
        return par_sort<uint32_t>(pDataIn1, arraySize1, stridesIn, itemSize, pDataOut1, stridesOut, pSortFunction);
    default:
        LOGGING("SortArray does not understand type %d\n", atype);
        return -1;
    }

}


//===============================================================================

template<typename UINDEX>
static BOOL ARangeCallback(void* callbackArgT, int core, int64_t start, int64_t length) {

    UINDEX* pDataOut = (UINDEX*)callbackArgT;
    UINDEX istart = (UINDEX)start;
    UINDEX iend = istart + (UINDEX)length;

    for (UINDEX i = istart; i < iend; i++) {
        pDataOut[i] = i;
    }

    return TRUE;
}


//================================================================================
// Group via lexsort
// CountOut
//================================================================================
//typedef int64_t *(GROUP_INDEX_FUNC)()

template <typename T, typename UINDEX>
static int64_t GroupIndexStep2(
    void* pDataIn1,
    UINDEX      arraySize1,
    UINDEX* pDataIndexIn,
    UINDEX* pGroupOut,
    UINDEX* pFirstOut,
    UINDEX* pCountOut,
    bool* pFilter,
    int64_t       base_index,
    UINDEX      strlen = 0) {

    T* pDataIn = (T*)pDataIn1;
    UINDEX      curIndex = pDataIndexIn[0];
    UINDEX      curCount = 1;
    T           val1 = pDataIn[curIndex];
    UINDEX      baseIndex = (UINDEX)base_index;
    UINDEX      curGroup = 0;

    // NOTE: filtering does not work!!
    //if (pFilter) {
    //   // base index must be 1
    //   // Invalid bin init
    //   // currently nothing filtered out
    //   pCountOut[0] = 0;

    //   UINDEX invalid = *(UINDEX*)GetInvalid<UINDEX>();

    //   pFirstOut[0] = invalid;
    //   val1 = -1;

    //   // TJD NOTE...
    //   // Think have to see how many filtered out values up front
    //   curCount = 1;

    //   UINDEX zeroCount = 0;

    //   for (UINDEX i = 0; i < arraySize1; i++) {
    //      curIndex = pDataIndexIn[i];

    //      if (pFilter[i]) {
    //         T val2 = pDataIn[curIndex];

    //         if (val1 == val2) {
    //            curCount++;
    //            pGroupOut[curIndex] = curGroup + 1;
    //         }
    //         else {
    //            curGroup++;
    //            pCountOut[curGroup] = curCount;
    //            pFirstOut[curGroup] = curIndex;
    //            pGroupOut[curIndex] = curGroup + 1;
    //            val1 = val2;
    //            curCount = 1;
    //         }
    //      }
    //      else {
    //         zeroCount++;
    //         pGroupOut[curIndex] = 0;
    //         if (pFirstOut[0] == invalid) {
    //            pFirstOut[0] = curIndex;
    //         }
    //      }
    //   }
    //   curGroup++;
    //   pCountOut[curGroup] = curCount;

    //   // the zero count tally
    //   pCountOut[0] = zeroCount;

    //   return curGroup;
    //}


    {
        if (base_index == 0) {
            // SHIFT countout

            // Invalid bin init
            // currently nothing filtered out
            pFirstOut[0] = curIndex;
            pGroupOut[curIndex] = 0;

            for (UINDEX i = 1; i < arraySize1; i++) {
                curIndex = pDataIndexIn[i];
                T val2 = pDataIn[curIndex];

                if (val1 == val2) {
                    curCount++;
                    pGroupOut[curIndex] = curGroup;
                }
                else {
                    pCountOut[curGroup] = curCount;
                    curGroup++;
                    pFirstOut[curGroup] = curIndex;
                    pGroupOut[curIndex] = curGroup;
                    val1 = val2;
                    curCount = 1;
                }
            }
            pCountOut[curGroup] = curCount;
            curGroup++;
        }
        else {
            // Invalid bin init
            // currently nothing filtered out
            pCountOut[0] = 0;

            pFirstOut[0] = curIndex;
            pGroupOut[curIndex] = 1;

            for (UINDEX i = 1; i < arraySize1; i++) {
                curIndex = pDataIndexIn[i];
                T val2 = pDataIn[curIndex];

                if (val1 == val2) {
                    curCount++;
                    pGroupOut[curIndex] = curGroup + 1;
                }
                else {
                    curGroup++;
                    pCountOut[curGroup] = curCount;
                    pFirstOut[curGroup] = curIndex;
                    pGroupOut[curIndex] = curGroup + 1;
                    val1 = val2;
                    curCount = 1;
                }
            }
            curGroup++;
            pCountOut[curGroup] = curCount;
        }
        return curGroup;
    }
}


template <typename T, typename UINDEX>
static int64_t GroupIndexStep2String(
    void* pDataIn1,
    UINDEX      arraySize1,
    UINDEX* pDataIndexIn,
    UINDEX* pGroupOut,
    UINDEX* pFirstOut,
    UINDEX* pCountOut,
    bool* pFilter,
    int64_t       base_index,
    int64_t       strlen) {

    T* pDataIn = (T*)pDataIn1;
    UINDEX      curIndex = pDataIndexIn[0];
    UINDEX      curCount = 1;
    T* val1 = &pDataIn[curIndex * strlen];
    UINDEX      baseIndex = (UINDEX)base_index;
    UINDEX      curGroup = 0;

    // Invalid bin init when base_index is 1
    pCountOut[0] = 0;

    pFirstOut[0] = curIndex;
    pGroupOut[curIndex] = baseIndex;

    for (UINDEX i = 1; i < arraySize1; i++) {
        curIndex = pDataIndexIn[i];
        T* val2 = &pDataIn[curIndex * strlen];

        if (BINARY_LT(val1, val2, strlen) == 0) {
            curCount++;
            pGroupOut[curIndex] = curGroup + baseIndex;
        }
        else {
            curGroup++;
            pCountOut[curGroup] = curCount;
            pFirstOut[curGroup] = curIndex;
            pGroupOut[curIndex] = curGroup + baseIndex;
            val1 = val2;
            curCount = 1;
        }
    }
    curGroup++;
    pCountOut[curGroup] = curCount;
    return curGroup;
}


//------------------------------------------------------------------------------------------
// Internal and can be called from groupby
// caller must allocate the pGroupOut as int32_t or int64_t with size arraySize1
// UINDEX = int32_t or int64_t
template <typename UINDEX>
static int64_t GroupIndex(
    void* pDataIn1,
    int64_t       arraySize1V,
    void* pDataIndexInV,
    void* pGroupOutV,
    void* pFirstOutV,
    void* pCountOutV,
    bool* pFilter,       // optional
    int64_t       base_index,
    int64_t       strlen) {

    int64_t       uniqueCount = 0;

    UINDEX* pDataIndexIn = (UINDEX*)pDataIndexInV;
    UINDEX* pGroupOut = (UINDEX*)pGroupOutV;
    UINDEX* pFirstOut = (UINDEX*)pFirstOutV;
    UINDEX* pCountOut = (UINDEX*)pCountOutV;
    UINDEX      arraySize1 = (UINDEX)arraySize1V;

    switch (strlen) {
    case 1:
        uniqueCount =
            GroupIndexStep2<int8_t, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
        break;
    case 2:
        uniqueCount =
            GroupIndexStep2<int16_t, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
        break;
    case 4:
        uniqueCount =
            GroupIndexStep2<int32_t, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
        break;
    case 8:
        uniqueCount =
            GroupIndexStep2<int64_t, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
        break;
    default:
        uniqueCount =
            GroupIndexStep2String<const char, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, strlen);
        break;
    }

    return uniqueCount;
}

extern "C" int64_t GroupIndex32(
    void* pDataIn1,
    int64_t    arraySize1V,
    void* pDataIndexInV,
    void* pGroupOutV,
    void* pFirstOutV,
    void* pCountOutV,
    bool* pFilter,       // optional
    int64_t       base_index,
    int64_t       strlen
) {
    return GroupIndex<int32_t>(pDataIn1, arraySize1V, pDataIndexInV, pGroupOutV, pFirstOutV, pCountOutV, pFilter, base_index, strlen);
}

extern "C" int64_t GroupIndex64(
    void* pDataIn1,
    int64_t    arraySize1V,
    void* pDataIndexInV,
    void* pGroupOutV,
    void* pFirstOutV,
    void* pCountOutV,
    bool* pFilter,       // optional
    int64_t       base_index,
    int64_t       strlen
) {
    return GroupIndex<int64_t>(pDataIn1, arraySize1V, pDataIndexInV, pGroupOutV, pFirstOutV, pCountOutV, pFilter, base_index, strlen);

}

//================================================================================
typedef int(*IS_SORTED_FUNC)(const char* pDataIn1, int64_t arraySize1, int64_t strlennotused);
//-----------------------------------------------------------------------------------------------
template <typename T>
static int
IsSortedFloat(const char* pDataIn1, int64_t arraySize1, int64_t strlennotused) {

    int result = 0;
    T* pData = (T*)pDataIn1;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && (pData[i] != pData[i])) {
        i--;
    }

    while ((i > 0) && pData[i] >= pData[i - 1]) {
        i--;
    }

    return i <= 0;
}


//-----------------------------------------------------------------------------------------------
template <typename T>
static int
IsSorted(const char* pDataIn1, int64_t arraySize1, int64_t strlennotused) {

    int result = 0;
    T* pData = (T*)pDataIn1;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && pData[i] >= pData[i - 1]) {
        i--;
    }

    return i <= 0;
}


//-----------------------------------------------------------------------------------------------
static int
IsSortedString(const char* pData, int64_t arraySize1, int64_t strlen) {

    int result = 0;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && !(STRING_LT((const unsigned char*)&pData[i * strlen], (const unsigned char*)&pData[(i - 1) * strlen], strlen))) {
        i--;
    }

    return i <= 0;
}


//-----------------------------------------------------------------------------------------------
static int
IsSortedUnicode(const char* pData, int64_t arraySize1, int64_t strlen) {

    int result = 0;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && !(STRING_LT((const uint32_t*)&pData[i * strlen], (const uint32_t*)&pData[(i - 1) * strlen], strlen))) {
        i--;
    }

    return i <= 0;
}

//-----------------------------------------------------------------------------------------------
static int
IsSortedVoid(const char* pData, int64_t arraySize1, int64_t strlen) {

    int result = 0;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && !(STRING_LT(&pData[i * strlen], &pData[(i - 1) * strlen], strlen))) {
        i--;
    }

    return i <= 0;
}

//====================================
// Returns -1 on failure
// 1 for sorted
// 0 for not sorted
extern "C" int64_t IsSorted(
    void* pDataIn1,
    int64_t arraySize1,
    int32_t arrayType1,
    int64_t itemSize
) {

    LOGGING("issorted size %llu  type %d\n", arraySize1, arrayType1);

    int64_t result = 0;
    IS_SORTED_FUNC pSortedFunc = NULL;

    switch (arrayType1) {
    case ATOP_BOOL:
    case ATOP_INT8:
        pSortedFunc = IsSorted<int8_t>;
        break;
    case ATOP_INT16:
        pSortedFunc = IsSorted<int16_t>;
        break;
    case ATOP_INT32:
        pSortedFunc = IsSorted<int32_t>;
        break;
    case ATOP_INT64:
        pSortedFunc = IsSorted<int64_t>;
        break;
    case ATOP_UINT8:
        pSortedFunc = IsSorted<uint8_t>;
        break;
    case ATOP_UINT16:
        pSortedFunc = IsSorted<uint16_t>;
        break;
    case ATOP_UINT32:
        pSortedFunc = IsSorted<uint32_t>;
        break;
    case ATOP_UINT64:
        pSortedFunc = IsSorted<uint64_t>;
        break;
    case ATOP_FLOAT:
        pSortedFunc = IsSortedFloat<float>;
        break;
    case ATOP_DOUBLE:
        pSortedFunc = IsSortedFloat<double>;
        break;
    case ATOP_LONGDOUBLE:
        pSortedFunc = IsSortedFloat<long double>;
        break;
    case ATOP_VOID:
        pSortedFunc = IsSortedVoid;
        break;
    case ATOP_STRING:
        pSortedFunc = IsSortedString;
        break;
    case ATOP_UNICODE:
        pSortedFunc = IsSortedUnicode;
        break;
    case ATOP_HALF_FLOAT:
        pSortedFunc = IsSorted<atop_half>;
        break;

    default:
        // do not understand how to sort
        return -1;
    }


    // MT callback
    struct IsSortedCallbackStruct {
        int64_t             IsSorted;
        IS_SORTED_FUNC    pSortedFunc;
        const char* pDataIn1;
        int64_t             ArraySize;
        int64_t             ItemSize;
    } stISCallback{ 1, pSortedFunc, (const char*)pDataIn1, arraySize1, itemSize };

    // This is the routine that will be called back from multiple threads
    auto lambdaISCallback = [](void* callbackArgT, int core, int64_t start, int64_t length) -> int64_t {
        IsSortedCallbackStruct* cb = (IsSortedCallbackStruct*)callbackArgT;

        // check if short circuited (any segment not sorted)
        if (cb->IsSorted) {
            // If not the first segment, then overlap by going back
            if (start != 0) {
                start--;
                length++;
            }
            int result = cb->pSortedFunc(cb->pDataIn1 + (start * cb->ItemSize), length, cb->ItemSize);

            // on success, return TRUE 
            if (result) return TRUE;

            // on failure, set the failure flag and return FALSE
            cb->IsSorted = 0;
        }

        return FALSE;
    };

    // A zero length array is considered sorted
    THREADER->DoMultiThreadedChunkWork(arraySize1, lambdaISCallback, &stISCallback);

    result = stISCallback.IsSorted;

    return result;

}



#if defined(__clang__)
#pragma clang attribute pop
#endif
