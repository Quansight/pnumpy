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

#define SMALL_MERGESORT 16

#define PYA_QS_STACK 128
#define SMALL_QUICKSORT 15


#define INTP_SWAP(_X_,_Y_) { auto temp=_X_; _X_=_Y_; _Y_=temp;}

//#define T_SWAP(_X_, _Y_) { auto temp;  temp = _X_; _X_ = _Y_; _Y_ = temp; }
#define T_SWAP(_X_, _Y_)  std::swap(_X_,_Y_); 

// For floats anything compared to a nan will return 0
// TODO: Add compare for HALF_FLOAT and COMPLEX
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

FORCE_INLINE static int
npy_half_isnan(npy_half h)
{
    return ((h & 0x7c00u) == 0x7c00u) && ((h & 0x03ffu) != 0x0000u);
}


FORCE_INLINE static int
npy_half_lt_nonan(npy_half h1, npy_half h2)
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
            return 0;
        }
        else {
            return (h1 & 0x7fffu) < (h2 & 0x7fffu);
        }
    }
}


FORCE_INLINE static int
HALF_LT(npy_half a, npy_half b)
{
    int ret;

    if (npy_half_isnan(b)) {
        ret = !npy_half_isnan(a);
    }
    else {
        ret = !npy_half_isnan(a) && npy_half_lt_nonan(a, b);
    }

    return ret;
}

FORCE_INLINE static int
STRING_LT(const char* s1, const char* s2, size_t len)
{
    const unsigned char* c1 = (unsigned char*)s1;
    const unsigned char* c2 = (unsigned char*)s2;
    size_t i;

    for (i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
        }
    }
    return 0;
}

//---------------------------------
// Assumes Py_UCS4
// Assumes int is 32bits
FORCE_INLINE static int
UNICODE_LT(const char* s1, const char* s2, size_t len)
{
    const unsigned int* c1 = (unsigned int*)s1;
    const unsigned int* c2 = (unsigned int*)s2;
    size_t i;

    size_t lenunicode = len / 4;

    for (i = 0; i < lenunicode; ++i) {
        if (c1[i] != c2[i]) {
            return c1[i] < c2[i];
        }
    }
    return 0;
}


FORCE_INLINE static int
VOID_LT(const char* s1, const char* s2, size_t len)
{
    const unsigned char* c1 = (unsigned char*)s1;
    const unsigned char* c2 = (unsigned char*)s2;

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
    {
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
    }
    }
    return 0;
}



FORCE_INLINE static int
BINARY_LT(const char* s1, const char* s2, size_t len)
{
    const unsigned char* c1 = (unsigned char*)s1;
    const unsigned char* c2 = (unsigned char*)s2;

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
            return *c1 < *c2;
        }
        return 0;
    case 8:
        if (*(uint64_t*)c1 != *(uint64_t*)c2) {
            return 1;
        }
        return 0;
    default:
    {
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
    }
    }
    return 0;
}



//-----------------------------------------------------------------------------------------------
template <typename T>
/*static*/ int
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
template <typename T, typename UINDEX>
static int
aheapsort_(T* vv, UINDEX* tosort, UINDEX n)
{
    T* v = vv;
    UINDEX* a, i, j, l, tmp;
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
quicksort_(void* pVoidStart, int64_t num)
{
    T* start = (T*)pVoidStart;
    T vp;
    T* pl = start;
    T* pr = pl + num - 1;
    T* stack[PYA_QS_STACK];
    T** sptr = stack;
    T* pm, * pi, * pj, * pk;

    int depth[PYA_QS_STACK];
    int* psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

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
        }
        else {
            heapsort_<T>(pl, pr - pl + 1);
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


//-----------------------------------------------------------------------------------------------
// argsort (indirect quicksort)
template <typename T, typename UINDEX>
static int
aquicksort_(T* vv, UINDEX* tosort, int64_t num)
{
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
/*static*/ void
mergesort0_(T* pl, T* pr, T* pw)
{
    T vp, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        mergesort0_(pl, pm, pw);
        mergesort0_(pm, pr, pw);

#ifndef USE_MEMCPY
        memcpy(pw, pl, (pm - pl) * sizeof(T));
#else
        pi = pw;
        pj = pl;
        while (pj < pm) {
            *pi++ = *pj++;
        }
#endif

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
#ifdef USE_MEMCPY
        diff = pi - pj;
        if (diff > 0) {
            memcpy(pk, pj, sizeof(T) * diff);
            pk += diff;
            pj += diff;
        }
#else
        while (pj < pi) {
            *pk++ = *pj++;
        }
#endif
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


//-----------------------------------------------------------------------------------------------
template <typename T>
/*static*/ int
mergesort_(T* start, int64_t num)
{
    T* pl, * pr, * pw;

    pl = start;
    pr = pl + num;

    // TODO: Consider alloc on stack
    pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
    if (pw == NULL) {
        return -1;
    }
    mergesort0_(pl, pr, pw);

    WORKSPACE_FREE(pw);
    return 0;
}



//-----------------------------------------------------------------------------------------------
template <typename UINDEX>
static void
amergesort0_string(UINDEX* pl, UINDEX* pr, const char* strItem, UINDEX* pw, int64_t strlen)
{
    const char* vp;
    UINDEX vi, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        //printf("merge sort %p %p %p diff:%lld\n", pl, pm, pr, pr-pl);
        amergesort0_string(pl, pm, strItem, pw, strlen);
        amergesort0_string(pm, pr, strItem, pw, strlen);
        pm = pl + ((pr - pl) >> 1);

        if (STRING_LT(strItem + (*pm) * strlen, strItem + (*(pm - 1)) * strlen, strlen)) {

            if ((pm - pl) >= 32) {
                memcpy(pw, pl, (pm - pl) * sizeof(UINDEX));
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
                if (STRING_LT(strItem + (*pm) * strlen, strItem + (*pj) * strlen, strlen)) {
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
            vp = strItem + (vi * strlen);
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& STRING_LT(vp, strItem + (*pk) * strlen, strlen)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}


//-----------------------------------------------------------------------------------------------
template <typename UINDEX>
static void
amergesort0_unicode(UINDEX* pl, UINDEX* pr, const char* strItem, UINDEX* pw, int64_t strlen)
{
    const char* vp;
    UINDEX vi, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_unicode(pl, pm, strItem, pw, strlen);
        amergesort0_unicode(pm, pr, strItem, pw, strlen);

        if (UNICODE_LT(strItem + (*pm) * strlen, strItem + (*(pm - 1)) * strlen, strlen)) {
            if ((pm - pl) >= 32) {
                memcpy(pw, pl, (pm - pl) * sizeof(UINDEX));
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
                if (UNICODE_LT(strItem + (*pm) * strlen, strItem + (*pj) * strlen, strlen)) {
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
            vp = strItem + (vi * strlen);
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& UNICODE_LT(vp, strItem + (*pk) * strlen, strlen)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}



//-----------------------------------------------------------------------------------------------
template <typename UINDEX>
static void
amergesort0_void(UINDEX* pl, UINDEX* pr, const char* strItem, UINDEX* pw, int64_t strlen)
{
    const char* vp;
    UINDEX vi, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_void(pl, pm, strItem, pw, strlen);
        amergesort0_void(pm, pr, strItem, pw, strlen);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (VOID_LT(strItem + (*pm) * strlen, strItem + (*pj) * strlen, strlen)) {
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
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = strItem + (vi * strlen);
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& VOID_LT(vp, strItem + (*pk) * strlen, strlen)) {
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
                memcpy(pw, pl, (pm - pl) * sizeof(UINDEX));
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
amergesort_(T* v, UINDEX* tosort, UINDEX num)
{
    UINDEX* pl, * pr, * pworkspace;

    pl = tosort;
    pr = pl + num;

    pworkspace = (UINDEX*)WORKSPACE_ALLOC((num / 2) * sizeof(UINDEX));
    if (pworkspace == NULL) {
        return -1;
    }
    amergesort0_(pl, pr, v, pworkspace);
    WORKSPACE_FREE(pworkspace);

    return 0;
}



//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename UINDEX>
static void
ParMergeString(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    //PLOGGING("string calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_string(pl, pr, pValue, pWorkSpace, strlen);
}


//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename UINDEX>
static void
ParMergeUnicode(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    //PLOGGING("unicode calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_unicode(pl, pr, pValue, pWorkSpace, strlen);
}

//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename UINDEX>
static void
ParMergeVoid(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    //PLOGGING("void calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_void(pl, pr, pValue, pWorkSpace, strlen);
}

//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename T, typename UINDEX>
static void
ParMergeNormal(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    T* pValue = (T*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    PLOGGING("normal calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_(pl, pr, pValue, pWorkSpace);
}


//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename UINDEX>
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

    if ( STRING_LT(pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, strlen)) {

        // copy the left to workspace
        memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

        // pi is end of workspace
        pi = pWorkSpace + (pm - pl);
        pj = pWorkSpace;
        pk = pl;

        while (pj < pi && pm < pr) {
            if (STRING_LT(pValue + (*pm) * strlen, pValue + (*pj) * strlen, strlen)) {
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
template <typename UINDEX>
static void
ParMergeMergeUnicode(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    UINDEX* pi, * pj, * pk, * pm;
    pm = pl + ((pr - pl) >> 1);

    if (UNICODE_LT(pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, strlen)) {

        memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

        // pi is end of workspace
        pi = pWorkSpace + (pm - pl);
        pj = pWorkSpace;
        pk = pl;

        while (pj < pi && pm < pr) {
            if (UNICODE_LT(pValue + (*pm) * strlen, pValue + (*pj) * strlen, strlen)) {
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
        PLOGGING("**Already sorted unicode %lld\n", (int64_t)(*pm));
    }

}

//---------------------------------------------------------------------------
// Called to combine the result of the left and right merge
template <typename UINDEX>
static void
ParMergeMergeVoid(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    const char* pValue = (char*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    UINDEX* pi, * pj, * pk, * pm;
    pm = pl + ((pr - pl) >> 1);

    if (VOID_LT(pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, strlen)) {

        memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

        // pi is end of workspace
        pi = pWorkSpace + (pm - pl);
        pj = pWorkSpace;
        pk = pl;

        while (pj < pi && pm < pr) {
            if (VOID_LT(pValue + (*pm) * strlen, pValue + (*pj) * strlen, strlen)) {
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
        PLOGGING("**Already sorted void %lld\n", (int64_t)(*pm));
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

        memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

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

        memcpy(pw, pl, (pm - pl) * strlen);

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
typedef void(*SORT_STEP_TWO)(void* pValue1, int64_t totalLen, int64_t strlen, void* pWorkSpace1);
typedef void(*MERGE_STEP_ONE)(void* pValue, void* pToSort, int64_t num, int64_t strlen, void* pWorkSpace);
typedef void(*MERGE_STEP_TWO)(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1);
//--------------------------------------------------------------------
struct MERGE_STEP_ONE_CALLBACK {
    union {
        MERGE_STEP_ONE MergeCallbackOne;
        SORT_FUNCTION  SortCallbackOne;
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
    void* pWorkSpace;

    // how much was used per 1/8 chunk when allocating the workspace
    int64_t AllocChunk;
    int64_t MergeBlocks;
    int64_t TypeSizeInput;

    // not valid for inplace sorting, otherwise is sizeof(int32) or sizeof(int64) depending on index size
    int64_t TypeSizeOutput;

    // used to synchronize parallel merges
    int64_t EndPositions[9];
    int64_t Level[3];

} stParMergeCallback;


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

//------------------------------------------------------------------------------
// Concurrent callback from multiple threads
// this routine is for indirect sorting
static int64_t ParMergeThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    MERGE_STEP_ONE_CALLBACK* Callback = (MERGE_STEP_ONE_CALLBACK*)pstWorkerItem->WorkCallbackArg;
    int64_t didSomeWork = FALSE;

    int64_t index;
    int64_t workBlock;

    // As long as there is work to do
    while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0) {
        // First index is 1 so we subtract
        index--;

        // the very first index starts at 0
        int64_t pFirst =  Callback->EndPositions[index];
        int64_t pSecond = Callback->EndPositions[index+1];

        PLOGGING("[%d] DoWork start loop -- %lld  index: %lld   pFirst: %lld   pSecond: %lld\n", core, workIndex, index, pFirst, pSecond);

        char* pToSort1 = (char*)(Callback->pToSort);

        int64_t MergeSize = (pSecond - pFirst);
        PLOGGING("%d : MergeOne index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);

        // Workspace uses half the size
        //char* pWorkSpace1 = (char*)pWorkSpace + (offsetAdjToSort / 2);
        char* pWorkSpace1 = (char*)Callback->pWorkSpace + (index * Callback->AllocChunk * Callback->TypeSizeOutput);

        Callback->MergeCallbackOne(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

        if (IsBuddyBitSet(index, &Callback->Level[0])) {

            // Move to next level -- 4 things to sort
            index = index / 2;
            pWorkSpace1 = (char*)Callback->pWorkSpace + (index * 2 * Callback->AllocChunk * Callback->TypeSizeOutput);

            pFirst = Callback->EndPositions[index*2];
            pSecond = Callback->EndPositions[index*2 + 2];
            MergeSize = (pSecond - pFirst);

            PLOGGING("size:%lld  first: %lld  second: %lld   expected: %lld\n", MergeSize, pFirst, pSecond, pFirst + (MergeSize >> 1));

            //pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
            Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

            if (IsBuddyBitSet(index, &Callback->Level[1])) {
                index /= 2;

                // Move to next level -- 2 things to sort
                pWorkSpace1 = (char*)Callback->pWorkSpace + (index * 4 * Callback->AllocChunk * Callback->TypeSizeOutput);

                pFirst = Callback->EndPositions[index*4];
                pSecond = Callback->EndPositions[index*4 + 4];
                MergeSize = (pSecond - pFirst);

                PLOGGING("%d : MergeThree index: %llu  %lld  %lld\n", core, index, pFirst, MergeSize);
                //pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
                PLOGGING("Level 2 %p %p,  size: %lld,  pworkspace: %p\n", Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, pWorkSpace1);
                Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (pFirst * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

                if (IsBuddyBitSet(index, &Callback->Level[2])) {
                    // Final merge
                    PLOGGING("%d : MergeFinal index: %llu  %lld  %lld  %lld\n", core, index, 0LL, Callback->ArrayLength, 0LL);
                    stParMergeCallback.MergeCallbackTwo(Callback->pValues, Callback->pToSort, Callback->ArrayLength, Callback->StrLen, Callback->pWorkSpace);
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

        // the very first index starts at 0
        int64_t pFirst = Callback->EndPositions[index];
        int64_t pSecond = Callback->EndPositions[index + 1];

        PLOGGING("[%d] DoWork start loop -- %lld  index: %lld   pFirst: %lld   pSecond: %lld\n", core, workIndex, index, pFirst, pSecond);

        int64_t MergeSize = (pSecond - pFirst);
        PLOGGING("%d : MergeOne index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);

        // Workspace uses half the size
        //char* pWorkSpace1 = (char*)pWorkSpace + (offsetAdjToSort / 2);
        char* pWorkSpace1 = (char*)Callback->pWorkSpace + (index * Callback->AllocChunk * Callback->TypeSizeInput);

        Callback->SortCallbackOne((char*)(Callback->pValues) + (pFirst * Callback->TypeSizeInput), MergeSize);

        if (IsBuddyBitSet(index, &Callback->Level[0])) {
            // Move to next level -- 4 things to sort
            index = index / 2;
            pWorkSpace1 = (char*)Callback->pWorkSpace + (index * 2 * Callback->AllocChunk * Callback->TypeSizeInput);

            pFirst = Callback->EndPositions[index * 2];
            pSecond = Callback->EndPositions[index * 2 + 2];
            MergeSize = (pSecond - pFirst);

            PLOGGING("size:%lld  first: %lld  second: %lld   expected: %lld\n", MergeSize, pFirst, pSecond, pFirst + (MergeSize >> 1));

            //pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeInput);
            // int64_t totalLen, int64_t strlen, void* pWorkSpace1);
            Callback->SortCallbackTwo((char*)(Callback->pValues) + (pFirst * Callback->TypeSizeInput), MergeSize, Callback->StrLen, pWorkSpace1);

            if (IsBuddyBitSet(index, &Callback->Level[1])) {
                index /= 2;

                // Move to next level -- 2 things to sort
                pWorkSpace1 = (char*)Callback->pWorkSpace + (index * 4 * Callback->AllocChunk * Callback->TypeSizeInput);

                pFirst = Callback->EndPositions[index * 4];
                pSecond = Callback->EndPositions[index * 4 + 4];
                MergeSize = (pSecond - pFirst);

                PLOGGING("%d : MergeThree index: %llu  %lld  %lld\n", core, index, pFirst, MergeSize);
                Callback->SortCallbackTwo((char*)(Callback->pValues) + (pFirst * Callback->TypeSizeInput), MergeSize, Callback->StrLen, pWorkSpace1);

                if (IsBuddyBitSet(index, &Callback->Level[2])) {
                    // Final merge
                    PLOGGING("%d : MergeFinal index: %llu  %lld  %lld  %lld\n", core, index, 0LL, Callback->ArrayLength, 0LL);
                    stParMergeCallback.SortCallbackTwo(Callback->pValues, Callback->ArrayLength, Callback->StrLen, Callback->pWorkSpace);
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

//========================================================================
//
enum PAR_SORT_TYPE {
    Normal = 0,
    Float = 1,
    String = 2,
    Unicode = 3,
    Void = 4
};

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
        amergesort0_string(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
        break;
    case PAR_SORT_TYPE::Unicode:
        amergesort0_unicode(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
        break;
    case PAR_SORT_TYPE::Void:
        amergesort0_void(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
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
    PAR_SORT_TYPE  sortType)
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
    else

        // If size is large, go parallel
        if (arrayLength >= CMathWorker::WORK_ITEM_BIG) {

            PLOGGING("Parallel version  %p  %p  %p\n", pToSort, pToSort + arrayLength, pValues);
            // Divide into 8 jobs
            // Allocate all memory up front
            // Allocate enough for 8 
            int64_t allocChunk = (arrayLength /16) + 1;
            void* pWorkSpace = NULL;

            // Allocate half the size since the workspace is only needed for left
            uint64_t allocSize = allocChunk * 8 * sizeof(UINDEX);
            pWorkSpace = WORKSPACE_ALLOC(allocSize);

            if (pWorkSpace == NULL) {
                return -1;
            }

            MERGE_STEP_ONE mergeStepOne = NULL;

            switch (sortType) {
            case PAR_SORT_TYPE::String:
                mergeStepOne = ParMergeString<UINDEX>;
                break;
            case PAR_SORT_TYPE::Unicode:
                mergeStepOne = ParMergeUnicode<UINDEX>;
                break;
            case PAR_SORT_TYPE::Void:
                mergeStepOne = ParMergeVoid<UINDEX>;
                break;
            default:
                mergeStepOne = ParMergeNormal<T, UINDEX>;
            }

            stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(arrayLength);

            if (pWorkItem == NULL) {

                // Threading not allowed for this work item, call it directly from main thread
                mergeStepOne(pValues, pToSort, arrayLength, strlen, pWorkSpace);
            }
            else {

                pWorkItem->DoWorkCallback = ParMergeThreadCallback;
                pWorkItem->WorkCallbackArg = &stParMergeCallback;

                stParMergeCallback.MergeCallbackOne = mergeStepOne;
                switch (sortType) {

                case PAR_SORT_TYPE::String:
                    stParMergeCallback.MergeCallbackTwo = ParMergeMergeString< UINDEX>;
                    break;
                case PAR_SORT_TYPE::Unicode:
                    stParMergeCallback.MergeCallbackTwo = ParMergeMergeUnicode< UINDEX>;
                    break;
                case PAR_SORT_TYPE::Void:
                    stParMergeCallback.MergeCallbackTwo = ParMergeMergeVoid< UINDEX>;
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
                stParMergeCallback.pWorkSpace = pWorkSpace;
                stParMergeCallback.TypeSizeInput = sizeof(T);
                if (strlen) {
                    stParMergeCallback.TypeSizeInput = strlen;
                }
                stParMergeCallback.TypeSizeOutput = sizeof(UINDEX);

                //NOTE set this value to 2,4 or 8
                stParMergeCallback.MergeBlocks = 8;

                for (int i = 0; i < 3; i++) {
                    stParMergeCallback.Level[i] = 0;
                }

                // We use an 8 way merge, we need the size breakdown
                stParMergeCallback.EndPositions[8] = arrayLength;
                stParMergeCallback.EndPositions[4] = arrayLength / 2;
                stParMergeCallback.EndPositions[6] = stParMergeCallback.EndPositions[4] + (arrayLength - stParMergeCallback.EndPositions[4]) / 2;
                stParMergeCallback.EndPositions[2] = 0 + (stParMergeCallback.EndPositions[4] - 0) / 2;
                stParMergeCallback.EndPositions[7] = stParMergeCallback.EndPositions[6] + (arrayLength - stParMergeCallback.EndPositions[6]) / 2;
                stParMergeCallback.EndPositions[5] = stParMergeCallback.EndPositions[4] + (stParMergeCallback.EndPositions[6] - stParMergeCallback.EndPositions[4]) / 2;
                stParMergeCallback.EndPositions[3] = stParMergeCallback.EndPositions[2] + (stParMergeCallback.EndPositions[4] - stParMergeCallback.EndPositions[2]) / 2;
                stParMergeCallback.EndPositions[1] = 0 + (stParMergeCallback.EndPositions[2] - 0) / 2;
                stParMergeCallback.EndPositions[0] = 0;

                // This will notify the worker threads of a new work item
                // Default thead wakeup to 7
                THREADER->WorkMain(pWorkItem, stParMergeCallback.MergeBlocks, 7, 1, FALSE);

            }

            // Free temp memory used
            WORKSPACE_FREE(pWorkSpace);

        }
        else {

            // single threaded sort
            return
                single_amergesort<T, UINDEX>(
                    pValues,
                    pToSort,
                    arrayLength,
                    strlen,
                    sortType);
        }

    return 0;
}



//------------------------------------------------------------------------
// parallel version
// if strlen==0, then not string (int or float)
// If the array is large enough, a parallel quick sort is invoked
// Returns -1 on failure
template <typename T>
static int
par_quicksort(
    void*           pValues,
    int64_t         arrayLength,
    int64_t         strides,
    int64_t         itemSize,
    SORT_FUNCTION   pSortFunction)
{
    // If size is large, go parallel
    if (arrayLength >= CMathWorker::WORK_ITEM_BIG) {

        PLOGGING("Parallel version  %p  %p  %p\n", pToSort, pToSort + arrayLength, pValues);
        // Divide into 8 jobs
        // Allocate all memory up front
        // Allocate enough for 8 
        int64_t allocChunk = (arrayLength / 16) + 1;
        void* pWorkSpace = NULL;

        // Allocate half the size since the workspace is only needed for left
        uint64_t allocSize = allocChunk * 8 * itemSize;
        pWorkSpace = WORKSPACE_ALLOC(allocSize);

        if (pWorkSpace == NULL) {
            return -1;
        }


        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(arrayLength);

        if (pWorkItem == NULL) {

            // Threading not allowed for this work item, call it directly from main thread
            pSortFunction(pValues, arrayLength);
        }
        else {

            pWorkItem->DoWorkCallback = ParMergeInPlaceThreadCallback;
            pWorkItem->WorkCallbackArg = &stParMergeCallback;

            // First pass is a quicksort in place
            stParMergeCallback.SortCallbackOne = pSortFunction;

            // second pass is a mergesort in place
            stParMergeCallback.SortCallbackTwo = ParInPlaceMerge<T>;

            stParMergeCallback.pValues = pValues;
            stParMergeCallback.pToSort = NULL;
            stParMergeCallback.ArrayLength = arrayLength;
            stParMergeCallback.StrLen = itemSize;
            stParMergeCallback.AllocChunk = allocChunk;
            stParMergeCallback.pWorkSpace = pWorkSpace;
            //stParMergeCallback.TypeSizeInput = sizeof(T);
            stParMergeCallback.TypeSizeInput = itemSize;

            stParMergeCallback.TypeSizeOutput = 0;

            //NOTE set this value to 2,4 or 8
            stParMergeCallback.MergeBlocks = 8;

            for (int i = 0; i < 3; i++) {
                stParMergeCallback.Level[i] = 0;
            }

            // We use an 8 way merge, we need the size breakdown
            stParMergeCallback.EndPositions[8] = arrayLength;
            stParMergeCallback.EndPositions[4] = arrayLength / 2;
            stParMergeCallback.EndPositions[6] = stParMergeCallback.EndPositions[4] + (arrayLength - stParMergeCallback.EndPositions[4]) / 2;
            stParMergeCallback.EndPositions[2] = 0 + (stParMergeCallback.EndPositions[4] - 0) / 2;
            stParMergeCallback.EndPositions[7] = stParMergeCallback.EndPositions[6] + (arrayLength - stParMergeCallback.EndPositions[6]) / 2;
            stParMergeCallback.EndPositions[5] = stParMergeCallback.EndPositions[4] + (stParMergeCallback.EndPositions[6] - stParMergeCallback.EndPositions[4]) / 2;
            stParMergeCallback.EndPositions[3] = stParMergeCallback.EndPositions[2] + (stParMergeCallback.EndPositions[4] - stParMergeCallback.EndPositions[2]) / 2;
            stParMergeCallback.EndPositions[1] = 0 + (stParMergeCallback.EndPositions[2] - 0) / 2;
            stParMergeCallback.EndPositions[0] = 0;

            // This will notify the worker threads of a new work item
            // Default thead wakeup to 7
            THREADER->WorkMain(pWorkItem, stParMergeCallback.MergeBlocks, 7, 1, FALSE);

        }

        // Free temp memory used
        WORKSPACE_FREE(pWorkSpace);
        return 0;
    }
    else {

        // single threaded sort
        return
            pSortFunction(
                pValues,
                arrayLength);
    }
}



//-----------------------------------------------------------------------------------------------
// Sorts in place
// TODO: Make multithreaded like
template <typename T>
static SORT_FUNCTION
SortInPlace(void* pDataIn1, int64_t arraySize1, SORT_MODE mode) {

    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        return quicksort_<T>;

    //case SORT_MODE::SORT_MODE_MERGE:
    //    result = mergesort_<T>;
    //    break;

    case SORT_MODE::SORT_MODE_HEAP:
        return  heapsort_<T>;

    }

    return NULL;
}




//-----------------------------------------------------------------------------------------------
template <typename T, typename UINDEX>
static int
SortIndex(
    int64_t* pCutOffs, int64_t cutOffLength, void* pDataIn1, UINDEX* toSort, int64_t arraySize1, SORT_MODE mode) {

    int result = 0;

    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        result = aquicksort_<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
        break;

        //case SORT_MODE::SORT_MODE_MERGE:
        //   result = amergesort_<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
        //   break;
    case SORT_MODE::SORT_MODE_MERGE:
        result = par_amergesort<T, UINDEX>(pCutOffs, cutOffLength, (T*)pDataIn1, (UINDEX*)toSort, arraySize1, 0, PAR_SORT_TYPE::Normal);
        break;

    case SORT_MODE::SORT_MODE_HEAP:
        result = aheapsort_<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, (UINDEX)arraySize1);
        break;

    }

    if (result != 0) {
        LOGGING("**Error sorting.  size %llu   mode %d\n", (int64_t)arraySize1, mode);
    }

    return result;
}




//-----------------------------------------------------------------------------------------------
template <typename UINDEX>
static int
SortIndexString(int64_t* pCutOffs, int64_t cutOffLength, const char* pDataIn1, UINDEX* toSort, int64_t arraySize1, SORT_MODE mode, int64_t strlen) {

    int result = 0;
    switch (mode) {
    default:
    case SORT_MODE::SORT_MODE_MERGE:
        result = par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, (UINDEX*)toSort, (int32_t)arraySize1, strlen, PAR_SORT_TYPE::String);
        break;

    }

    return result;
}


//-----------------------------------------------------------------------------------------------
template <typename UINDEX>
static int
SortIndexUnicode(int64_t* pCutOffs, int64_t cutOffLength, const char* pDataIn1, UINDEX* toSort, int64_t arraySize1, SORT_MODE mode, int64_t strlen) {

    int result = 0;
    switch (mode) {
    default:
    case SORT_MODE::SORT_MODE_MERGE:
        result = par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, (UINDEX*)toSort, arraySize1, strlen, PAR_SORT_TYPE::Unicode);
        break;

    }

    return result;
}

//-----------------------------------------------------------------------------------------------
template <typename UINDEX>
static int
SortIndexVoid(int64_t* pCutOffs, int64_t cutOffLength, const char* pDataIn1, UINDEX* toSort, int64_t arraySize1, SORT_MODE mode, int64_t strlen) {

    int result = 0;
    switch (mode) {
    default:
    case SORT_MODE::SORT_MODE_MERGE:
        result = par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, (UINDEX*)toSort, arraySize1, strlen, PAR_SORT_TYPE::Void);
        break;
    }

    return result;
}


//------------------------------------------------------------------------------------------
// Internal and can be called from groupby
// caller must allocate the pDataOut1 as int64_t with size arraySize1
// UINDEX = int32_t or int64_t
template <typename UINDEX>
static void SortIndex(
    int64_t*   pCutOffs,
    int64_t    cutOffLength,
    void*      pDataIn1,
    UINDEX     arraySize1,
    UINDEX*    pDataOut1,
    SORT_MODE  mode,
    int        arrayType1,
    int64_t    strlen) {

    switch (arrayType1) {
    case ATOP_UNICODE:
        SortIndexUnicode<UINDEX>(pCutOffs, cutOffLength, (const char*)pDataIn1, pDataOut1, arraySize1, mode, strlen);
        break;
    case ATOP_VOID:
        SortIndexVoid<UINDEX>(pCutOffs, cutOffLength, (const char*)pDataIn1, pDataOut1, arraySize1, mode, strlen);
        break;
    case ATOP_STRING:
        SortIndexString<UINDEX>(pCutOffs, cutOffLength, (const char*)pDataIn1, pDataOut1, arraySize1, mode, strlen);
        break;
    case ATOP_BOOL:
    case ATOP_INT8:
        SortIndex<int8_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_INT16:
        SortIndex<int16_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_INT32:
        SortIndex<int32_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_INT64:
        SortIndex<int64_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_UINT8:
        SortIndex<uint8_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_UINT16:
        SortIndex<uint16_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_UINT32:
        SortIndex<uint32_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_UINT64:
        SortIndex<uint64_t, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_FLOAT:
        SortIndex<float, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_DOUBLE:
        SortIndex<double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_LONGDOUBLE:
        SortIndex<long double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    default:
        LOGGING("SortIndex does not understand type %d\n", arrayType1);
    }

}

// Stub for lexsort to return int32
extern "C" void SortIndex32(
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
extern "C" void SortIndex64(
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
static SORT_FUNCTION SortArray(void* pDataIn1, int64_t arraySize1, int32_t atype, SORT_MODE mode) {
    switch (atype) {
    case ATOP_STRING:
        return SortInPlace<char>(pDataIn1, arraySize1, mode);
    case ATOP_BOOL:
        return SortInPlace<bool>(pDataIn1, arraySize1, mode);
    case ATOP_INT8:
        return SortInPlace<int8_t>(pDataIn1, arraySize1, mode);
    case ATOP_INT16:
        return SortInPlace<int16_t>(pDataIn1, arraySize1, mode);
    case ATOP_INT32:
        return SortInPlace<int32_t>(pDataIn1, arraySize1, mode);
    case ATOP_INT64:
        return SortInPlace<int64_t>(pDataIn1, arraySize1, mode);
    case ATOP_UINT8:
        return SortInPlace<uint8_t>(pDataIn1, arraySize1, mode);
    case ATOP_UINT16:
        return SortInPlace<uint16_t>(pDataIn1, arraySize1, mode);
    case ATOP_UINT32:
        return SortInPlace<uint32_t>(pDataIn1, arraySize1, mode);
    case ATOP_UINT64:
        return SortInPlace<uint64_t>(pDataIn1, arraySize1, mode);
    case ATOP_FLOAT:
        return SortInPlace<float>(pDataIn1, arraySize1, mode);
    case ATOP_DOUBLE:
        return SortInPlace<double>(pDataIn1, arraySize1, mode);
    case ATOP_LONGDOUBLE:
        return SortInPlace<long double>(pDataIn1, arraySize1, mode);
    default:
        LOGGING("SortArray does not understand type %d\n", atype);
        return NULL;
        break;
    }
    return NULL;
}

//===============================================================================

extern "C" int QuickSort(
    int atype,
    void* pDataIn1,
    int64_t arraySize1,
    int64_t strides,
    int64_t itemSize) {

    SORT_FUNCTION pSortFunction=
        SortArray(pDataIn1, arraySize1, atype, SORT_MODE::SORT_MODE_QSORT);

    if (pSortFunction) {
        switch (atype) {
        case ATOP_BOOL:
            return par_quicksort<bool>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_INT8:
            return par_quicksort<int8_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_INT16:
            return par_quicksort<int16_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_INT32:
            return par_quicksort<int32_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_INT64:
            return par_quicksort<int64_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_UINT8:
            return par_quicksort<uint8_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_UINT16:
            return par_quicksort<uint16_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_UINT32:
            return par_quicksort<uint32_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_UINT64:
            return par_quicksort<uint64_t>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_FLOAT:
            return par_quicksort<float>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_DOUBLE:
            return par_quicksort<double>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        case ATOP_LONGDOUBLE:
            return par_quicksort<long double>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        //case ATOP_STRING:
        //    return par_quicksort<char>(pDataIn1, arraySize1, strides, itemSize, pSortFunction);
        default:
            LOGGING("SortArray does not understand type %d\n", atype);
            return 0;
        }
    }
    return 0;

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

    while ((i > 0) && !(STRING_LT(&pData[i * strlen], &pData[(i - 1) * strlen], strlen))) {
        i--;
    }

    return i <= 0;
}


//-----------------------------------------------------------------------------------------------
static int
IsSortedUnicode(const char* pData, int64_t arraySize1, int64_t strlen) {

    int result = 0;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && !(UNICODE_LT(&pData[i * strlen], &pData[(i - 1) * strlen], strlen))) {
        i--;
    }

    return i <= 0;
}

//-----------------------------------------------------------------------------------------------
static int
IsSortedVoid(const char* pData, int64_t arraySize1, int64_t strlen) {

    int result = 0;

    int64_t i = arraySize1 - 1;

    while ((i > 0) && !(VOID_LT(&pData[i * strlen], &pData[(i - 1) * strlen], strlen))) {
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
