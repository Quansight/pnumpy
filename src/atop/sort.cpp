#include "common_inc.h"
#include "threads.h"
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

static __inline int npy_get_msb(uint64_t unum)
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

#define T_LT(_X_,_Y_) (_X_ < _Y_)

// Anything compared to a nan will return 0
#define FLOAT_LT(_X_,_Y_) (_X_ < _Y_ || (_Y_ != _Y_ && _X_ == _X_))  

#define INT32_LT(_X_,_Y_) (_X_ < _Y_)
#define INT32_SWAP(_X_,_Y_) { int32_t temp; temp=_X_; _X_=_Y_; _Y_=temp;}
#define INTP_SWAP(_X_,_Y_) { auto temp=_X_; _X_=_Y_; _Y_=temp;}

//#define T_SWAP(_X_, _Y_) { auto temp;  temp = _X_; _X_ = _Y_; _Y_ = temp; }
#define T_SWAP(_X_, _Y_)  std::swap(_X_,_Y_); 


__inline bool COMPARE_LT(float X, float Y) { return (X < Y || (Y != Y && X == X)); }
__inline bool COMPARE_LT(double X, double Y) { return (X < Y || (Y != Y && X == X)); }
__inline bool COMPARE_LT(long double X, long double Y) { return (X < Y || (Y != Y && X == X)); }
__inline bool COMPARE_LT(int32_t X, int32_t Y) { return (X < Y); }
__inline bool COMPARE_LT(int64_t X, int64_t Y) { return (X < Y); }
__inline bool COMPARE_LT(uint32_t X, uint32_t Y) { return (X < Y); }
__inline bool COMPARE_LT(uint64_t X, uint64_t Y) { return (X < Y); }
__inline bool COMPARE_LT(int8_t X, int8_t Y) { return (X < Y); }
__inline bool COMPARE_LT(int16_t X, int16_t Y) { return (X < Y); }
__inline bool COMPARE_LT(uint8_t X, uint8_t Y) { return (X < Y); }
__inline bool COMPARE_LT(uint16_t X, uint16_t Y) { return (X < Y); }

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
heapsort_(T* start, int64_t n)
{
    T     tmp, * a;
    int64_t i, j, l;

    /* The array needs to be offset by one for heapsort indexing */
    a = (T*)start - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n&& T_LT(a[j], a[j + 1])) {
                j += 1;
            }
            if (T_LT(tmp, a[j])) {
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
            if (j < n && T_LT(a[j], a[j + 1])) {
                j++;
            }
            if (T_LT(tmp, a[j])) {
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
            if (j < n&& T_LT(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (T_LT(v[tmp], v[a[j]])) {
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
            if (j < n && T_LT(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (T_LT(v[tmp], v[a[j]])) {
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
aheapsort_float(T* vv, UINDEX* tosort, UINDEX n)
{
    T* v = vv;
    UINDEX* a, i, j, l, tmp;
    /* The arrays need to be offset by one for heapsort indexing */
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n&& FLOAT_LT(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (FLOAT_LT(v[tmp], v[a[j]])) {
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
            if (j < n && FLOAT_LT(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (FLOAT_LT(v[tmp], v[a[j]])) {
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



//--------------------------------------------------------------------------------------
// N.B. This function isn't static like the others because it's used in a couple of places in GroupBy.cpp.
template <typename T>
/*static*/ int
quicksort_(T* start, int64_t num)
{
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


//-----------------------------------------------------------------------------------------------
template <typename T, typename UINDEX>
static int
aquicksort_float(T* vv, UINDEX* tosort, UINDEX num)
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
                if (FLOAT_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
                if (FLOAT_LT(v[*pr], v[*pm])) INTP_SWAP(*pr, *pm);
                if (FLOAT_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
                vp = v[*pm];
                pi = pl;
                pj = pr - 1;
                INTP_SWAP(*pm, *pj);
                for (;;) {
                    do ++pi; while (FLOAT_LT(v[*pi], vp));
                    do --pj; while (FLOAT_LT(vp, v[*pj]));
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
                while (pj > pl&& FLOAT_LT(vp, v[*pk])) {
                    *pj-- = *pk--;
                }
                *pj = vi;
            }
        } else {
            aheapsort_float<T, UINDEX>(vv, pl, (UINDEX)(pr - pl + 1));
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
            if (T_LT(*pm, *pj)) {
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
            while (pj > pl&& T_LT(vp, *pk)) {
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
    pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
    if (pw == NULL) {
        return -1;
    }
    mergesort0_(pl, pr, pw);

    WORKSPACE_FREE(pw);
    return 0;
}



//-----------------------------------------------------------------------------------------------
// binary mergesort with no recursion (num must be power of 2)
// TJD NOTE: Does not appear much faster
template <typename T>
static int
mergesort_binary_norecursion(T* start, int64_t num)
{
    T* pl, * pr, * pw, * pm, * pk, * pi, * pj;

    pl = start;
    pr = pl + num;
    pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
    if (pw == NULL) {
        return -1;
    }

    T* pEnd;

    pEnd = pr - SMALL_MERGESORT;

    while (pl < pEnd) {

        T* pMiddle;
        T  vp;

        pMiddle = pl + SMALL_MERGESORT;

        /* insertion sort */
        for (pi = pl + 1; pi < pMiddle; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& T_LT(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }

        pl += SMALL_MERGESORT;

    }

    //-- reamining for insertion sort
    pEnd = pr;
    while (pl < pEnd) {

        T* pMiddle;
        T  vp;

        pMiddle = pl + SMALL_MERGESORT;

        /* insertion sort */
        for (pi = pl + 1; pi < pMiddle; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl&& T_LT(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }

        pl += SMALL_MERGESORT;

    }

    // START MERGING ----------------------------------
    //----------------------------------------------------------------
    int64_t startSize = SMALL_MERGESORT;

    pEnd = start + num;
    pl = start;

    while ((pl + startSize) < pEnd) {

        while ((pl + startSize) < pEnd) {

            pr = pl + (2 * startSize);
            pm = pl + ((pr - pl) >> 1);

            // memcpy first half into workspace
            memcpy(pw, pl, (pm - pj) * sizeof(T));

            // merge
            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;

            while (pj < pi && pm < pr) {
                if (T_LT(*pm, *pj)) {
                    *pk++ = *pm++;
                }
                else {
                    *pk++ = *pj++;
                }
            }

            // memcpy 
            while (pj < pi) {
                *pk++ = *pj++;
            }

            // move to next segment to merger
            pl = pr;
        }

        // Now merge again
        pl = start;
        startSize <<= 1;
    }

    //printf("%llu  vs %llu\n", startSize, num);
    //mergesort0_(pl, pr, pw);

    WORKSPACE_FREE(pw);
    return 0;
}



//--------------------------------------------------------------------------------------
template <typename T>
static void
mergesort0_float(T* pl, T* pr, T* pw, T* head)
{
    T vp, * pi, * pj, * pk, * pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        mergesort0_float(pl, pm, pw, head);
        mergesort0_float(pm, pr, pw, head);


        pi = pw;
        pj = pl;

        // TD NOTE: Jan 2018 -- the memcpy improves float sorting slightly, it does not improve INT copying for some reason...
#ifndef USE_MEMCPY
        int64_t diff = pm - pj;
        memcpy(pi, pj, diff * sizeof(T));
        pi += diff;
        pj += diff;
#else
        while (pj < pm) {
            *pi++ = *pj++;
        }
#endif
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (FLOAT_LT(*pm, *pj)) {
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
            while (pj > pl&& FLOAT_LT(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    }
}


//-----------------------------------------------------------------------------------------------
template <typename T>
static int
mergesort_float(T* start, int64_t num)
{
    T* pl, * pr, * pw;

    pl = start;
    pr = pl + num;
    pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
    if (pw == NULL) {
        return -1;
    }
    mergesort0_float(pl, pr, pw, start);

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
        amergesort0_string(pl, pm, strItem, pw, strlen);
        amergesort0_string(pm, pr, strItem, pw, strlen);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
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
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
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
template <typename T, typename UINDEX>
static void
// T= data type == float32/float64
// UINDEX = int32_t or int64_t
amergesort0_float(UINDEX* pl, UINDEX* pr, T* v, UINDEX* pw, int64_t strlen = 0)
{
    T vp;
    UINDEX vi, * pi, * pj, * pk, * pm;

    //PLOGGING("merging %llu bytes of data at %p\n", pr - pl, v);

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_float(pl, pm, v, pw);
        amergesort0_float(pm, pr, v, pw);

        //if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {
        {
            // MERGES DATA, memcpy
#ifdef USE_MEMCPY
            //int64_t bytescopy = (pm - pl) * sizeof(UINDEX);
            //aligned_block_copy_backwards((int64_t*)pw, (int64_t*)pl, bytescopy);
            //   
            memcpy(pw, pl, (pm - pl) * sizeof(UINDEX));

            //pi = pw +(pm-pl) -1;
            //pj = pl + (pm-pl) -1;
            //while (pj >= pl) {
            //   *pi-- = *pj--;
            //}

#else
         // Copy left side into workspace
            pi = pw;
            pj = pl;
            while (pj < pm) {
                *pi++ = *pj++;
            }
#endif
            // merge
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
        if (T_LT(v[*pm], v[*(pm - 1)])) {

            for (pi = pw, pj = pl; pj < pm;) {
                *pi++ = *pj++;
            }
            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;
            while (pj < pi && pm < pr) {
                if (T_LT(v[*pm], v[*pj])) {
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
            while (pj > pl&& T_LT(vp, v[*pk])) {
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


template <typename T, typename UINDEX>
static int
amergesort_float(T* v, UINDEX* tosort, UINDEX num)
{
    UINDEX* pl, * pr, * pworkspace;

    pl = tosort;
    pr = pl + num;

    pworkspace = (UINDEX*)WORKSPACE_ALLOC((num / 2) * sizeof(UINDEX));
    if (pworkspace == NULL) {
        return -1;
    }
    amergesort0_float(pl, pr, v, pworkspace);
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
template <typename T, typename UINDEX>
static void
ParMergeFloat(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1) {
    UINDEX* pl, * pr;

    UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
    UINDEX* pToSort = (UINDEX*)pToSort1;
    T* pValue = (T*)pValue1;

    pl = pToSort;
    pr = pl + totalLen;

    PLOGGING("float calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
    amergesort0_float(pl, pr, pValue, pWorkSpace);
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


    //if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {

    //   for (pi = pw, pj = pl; pj < pm;) {
    //      *pi++ = *pj++;
    //   }
    //   pi = pw + (pm - pl);
    //   pj = pw;
    //   pk = pl;
    //   while (pj < pi && pm < pr) {
    //      if (COMPARE_LT(v[*pm], v[*pj])) {
    //         *pk++ = *pm++;
    //      }
    //      else {
    //         *pk++ = *pj++;
    //      }
    //   }
    //   while (pj < pi) {
    //      *pk++ = *pj++;
    //   }
    //}

    // BUG BUG doing lexsort on two arrays: string, int.  Once sorted, resorting does not work.
    if (TRUE || STRING_LT(pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, strlen)) {

        //printf("%lld %lld %lld %lld\n", (int64_t)pValue[*pl], (int64_t)pValue[*(pm - 1)], (int64_t)pValue[*pm], (int64_t)pValue[*(pr - 1)]);
        //printf("%lld %lld %lld %lld %lld\n", (int64_t)*pl, (int64_t)*(pm - 2), (int64_t)*(pm - 1), (int64_t)*pm, (int64_t)*(pr - 1));

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
        PLOGGING("**Already sorted string %lld\n", (int64_t)(*pm));

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

    if (TRUE || UNICODE_LT(pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, strlen)) {

        //printf("%lld %lld %lld %lld\n", (int64_t)pValue[*pl], (int64_t)pValue[*(pm - 1)], (int64_t)pValue[*pm], (int64_t)pValue[*(pr - 1)]);
        //printf("%lld %lld %lld %lld %lld\n", (int64_t)*pl, (int64_t)*(pm - 2), (int64_t)*(pm - 1), (int64_t)*pm, (int64_t)*(pr - 1));

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

    // BUG BUG doing lexsort on two arrays: string, int.  Once sorted, resorting does not work.
    if (TRUE || VOID_LT(pValue + (*pm) * strlen, pValue + (*pm - 1) * strlen, strlen)) {

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


    //if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {

    //   for (pi = pw, pj = pl; pj < pm;) {
    //      *pi++ = *pj++;
    //   }
    //   pi = pw + (pm - pl);
    //   pj = pw;
    //   pk = pl;
    //   while (pj < pi && pm < pr) {
    //      if (COMPARE_LT(v[*pm], v[*pj])) {
    //         *pk++ = *pm++;
    //      }
    //      else {
    //         *pk++ = *pj++;
    //      }
    //   }
    //   while (pj < pi) {
    //      *pk++ = *pj++;
    //   }
    //}

    // quickcheck to see if we have to copy
    // BUG BUG doing lexsort on two arrays: string, int.  Once sorted, resorting does not work.
    if (TRUE || COMPARE_LT(pValue[*pm], pValue[*(pm - 1)])) {

        //printf("%lld %lld %lld %lld\n", (int64_t)pValue[*pl], (int64_t)pValue[*(pm - 1)], (int64_t)pValue[*pm], (int64_t)pValue[*(pr - 1)]);
        //printf("%lld %lld %lld %lld %lld\n", (int64_t)*pl, (int64_t)*(pm - 2), (int64_t)*(pm - 1), (int64_t)*pm, (int64_t)*(pr - 1));

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
        PLOGGING("**Already sorted %lld\n", (int64_t)(*pm));
    }

}


typedef void(*MERGE_STEP_ONE)(void* pValue, void* pToSort, int64_t num, int64_t strlen, void* pWorkSpace);
typedef void(*MERGE_STEP_TWO)(void* pValue1, void* pToSort1, int64_t totalLen, int64_t strlen, void* pWorkSpace1);
//--------------------------------------------------------------------
struct MERGE_STEP_ONE_CALLBACK {
    MERGE_STEP_ONE MergeCallbackOne;
    MERGE_STEP_TWO MergeCallbackTwo;

    void* pValues;
    void* pToSort;
    int64_t ArrayLength;

    // set to 0 if not a string, otherwise the string length
    int64_t StrLen;

    void* pWorkSpace;
    int64_t MergeBlocks;
    int64_t TypeSizeInput;
    int64_t TypeSizeOutput;

    // used to synchronize parallel merges
    int64_t EndPositions[8];
    int64_t Level[3];

} stParMergeCallback;



//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static int64_t ParMergeThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    MERGE_STEP_ONE_CALLBACK* Callback = (MERGE_STEP_ONE_CALLBACK*)pstWorkerItem->WorkCallbackArg;
    int64_t didSomeWork = FALSE;

    int64_t index;
    int64_t workBlock;


    // As long as there is work to do
    while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0) {
        // First index is 1 so we subtract
        index--;

        //PLOGGING("[%d] DoWork start loop -- %lld  index: %lld\n", core, workIndex, index);

        // the very first index starts at 0
        int64_t pFirst = 0;

        if (index >= 1) {
            pFirst = Callback->EndPositions[index - 1];
        }

        int64_t pSecond = Callback->EndPositions[index];
        char* pToSort1 = (char*)(Callback->pToSort);

        int64_t MergeSize = (pSecond - pFirst);
        int64_t OffsetSize = pFirst;
        PLOGGING("%d : MergeOne index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);

        // Workspace uses half the size
        //char* pWorkSpace1 = (char*)pWorkSpace + (offsetAdjToSort / 2);
        char* pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);

        Callback->MergeCallbackOne(Callback->pValues, pToSort1 + (OffsetSize * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

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
        int64_t result = FMInterlockedOr(&Callback->Level[0], bitshift);

        // Check if our buddy was already set
        PLOGGING("index: %lld  %lld %lld -- %s\n", index, buddy, (result & buddy), buddy == (result & buddy) ? "GOOD" : "WAIT");

        if (buddy == (result & buddy)) {
            // Move to next level -- 4 things to sort
            index = index / 2;
            index = index * 2;
            index += 1;

            pFirst = 0;

            if (index >= 2) {
                pFirst = Callback->EndPositions[index - 2];
            }

            pSecond = Callback->EndPositions[index];
            pToSort1 = (char*)(Callback->pToSort);
            MergeSize = (pSecond - pFirst);
            OffsetSize = pFirst;

            PLOGGING("%d : MergeTwo index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);
            pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
            Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (OffsetSize * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

            index /= 2;
            bitshift = 1LL << index;

            // Now find the buddy bit (adjacent bit)
            buddy = 0;
            if (index & 1) {
                buddy = 1LL << (index - 1);
            }
            else {
                buddy = 1LL << (index + 1);
            }

            // Get back which bits were set before the OR operation
            result = FMInterlockedOr(&Callback->Level[1], bitshift);

            // Check if our buddy was already set
            PLOGGING("index -- LEVEL 2: %lld  %lld %lld -- %s\n", index, buddy, (result & buddy), buddy == (result & buddy) ? "GOOD" : "WAIT");

            if (buddy == (result & buddy)) {
                // Move to next level -- 2 things to sort
                index = index / 2;
                index = index * 4;
                index += 3;

                pFirst = 0;

                if (index >= 4) {
                    pFirst = Callback->EndPositions[index - 4];
                }

                pSecond = Callback->EndPositions[index];
                pToSort1 = (char*)(Callback->pToSort);
                MergeSize = (pSecond - pFirst);
                OffsetSize = pFirst;

                PLOGGING("%d : MergeThree index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);
                pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
                Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (OffsetSize * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);


                index /= 4;
                bitshift = 1LL << index;

                // Now find the buddy bit (adjacent bit)
                buddy = 0;
                if (index & 1) {
                    buddy = 1LL << (index - 1);
                }
                else {
                    buddy = 1LL << (index + 1);
                }

                // Get back which bits were set before the OR operation
                result = FMInterlockedOr(&Callback->Level[2], bitshift);

                if (buddy == (result & buddy)) {
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
    case PAR_SORT_TYPE::Float:
        amergesort0_float(pToSort, pToSort + arrayLength, pValues, pWorkSpace);
        break;
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
            // Allocate all memory up front?
            //UINDEX* pWorkSpace = (UINDEX*)WORKSPACE_FREE(((arrayLength / 2) + 256)* sizeof(UINDEX));
            void* pWorkSpace = NULL;
            uint64_t allocSize = arrayLength * sizeof(UINDEX);

            //(UINDEX*)WORKSPACE_ALLOC(arrayLength * sizeof(UINDEX));
            pWorkSpace = WORKSPACE_ALLOC(allocSize);

            if (pWorkSpace == NULL) {
                return -1;
            }

            MERGE_STEP_ONE mergeStepOne = NULL;

            switch (sortType) {
            case PAR_SORT_TYPE::Float:
                mergeStepOne = ParMergeFloat<T, UINDEX>;
                break;
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

                if (stParMergeCallback.MergeBlocks == 2) {

                    stParMergeCallback.EndPositions[1] = arrayLength;
                    stParMergeCallback.EndPositions[0] = arrayLength / 2;
                }
                else if (stParMergeCallback.MergeBlocks == 4) {

                    stParMergeCallback.EndPositions[3] = arrayLength;
                    stParMergeCallback.EndPositions[1] = arrayLength / 2;
                    stParMergeCallback.EndPositions[2] = stParMergeCallback.EndPositions[1] + (stParMergeCallback.EndPositions[3] - stParMergeCallback.EndPositions[1]) / 2;
                    stParMergeCallback.EndPositions[0] = 0 + (stParMergeCallback.EndPositions[1] - 0) / 2;
                }

                else {
                    // We use an 8 way merge, we need the size breakdown
                    stParMergeCallback.EndPositions[7] = arrayLength;
                    stParMergeCallback.EndPositions[3] = arrayLength / 2;
                    stParMergeCallback.EndPositions[5] = stParMergeCallback.EndPositions[3] + (stParMergeCallback.EndPositions[7] - stParMergeCallback.EndPositions[3]) / 2;
                    stParMergeCallback.EndPositions[1] = 0 + (stParMergeCallback.EndPositions[3] - 0) / 2;
                    stParMergeCallback.EndPositions[6] = stParMergeCallback.EndPositions[5] + (stParMergeCallback.EndPositions[7] - stParMergeCallback.EndPositions[5]) / 2;
                    stParMergeCallback.EndPositions[4] = stParMergeCallback.EndPositions[3] + (stParMergeCallback.EndPositions[5] - stParMergeCallback.EndPositions[3]) / 2;
                    stParMergeCallback.EndPositions[2] = stParMergeCallback.EndPositions[1] + (stParMergeCallback.EndPositions[3] - stParMergeCallback.EndPositions[1]) / 2;
                    stParMergeCallback.EndPositions[0] = 0 + (stParMergeCallback.EndPositions[1] - 0) / 2;
                }

                // This will notify the worker threads of a new work item
                THREADER->WorkMain(pWorkItem, stParMergeCallback.MergeBlocks, 0, 1, FALSE);

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




//-----------------------------------------------------------------------------------------------
// Sorts in place
// TODO: Make multithreaded like
template <typename T>
static int
SortInPlace(void* pDataIn1, int64_t arraySize1, SORT_MODE mode) {

    int result = 0;

    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        result = quicksort_((T*)pDataIn1, arraySize1);
        break;

    case SORT_MODE::SORT_MODE_MERGE:
        result = mergesort_((T*)pDataIn1, arraySize1);
        break;

    case SORT_MODE::SORT_MODE_HEAP:
        result = heapsort_<T>((T*)pDataIn1, arraySize1);
        break;

    }

    if (result != 0) {
        LOGGING("**Error sorting.  size %llu   mode %d\n", arraySize1, mode);
    }

    return result;
}



//-----------------------------------------------------------------------------------------------
// Sorts in place
template <typename T>
static int
SortInPlaceFloat(void* pDataIn1, int64_t arraySize1, SORT_MODE mode) {

    int result = 0;

    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        result = quicksort_((T*)pDataIn1, arraySize1);
        break;

    case SORT_MODE::SORT_MODE_MERGE:
        result = mergesort_float((T*)pDataIn1, arraySize1);
        break;

    case SORT_MODE::SORT_MODE_HEAP:
        result = heapsort_<T>((T*)pDataIn1, arraySize1);
        break;

    }

    if (result != 0) {
        LOGGING("**Error sorting.  size %llu   mode %d\n", (int64_t)arraySize1, mode);
    }

    return result;
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
template <typename T, typename UINDEX>
static int
SortIndexFloat(int64_t* pCutOffs, int64_t cutOffLength, void* pDataIn1, UINDEX* toSort, UINDEX arraySize1, SORT_MODE mode) {

    int result = 0;

    switch (mode) {
    case SORT_MODE::SORT_MODE_QSORT:
        result = aquicksort_float<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
        break;

    case SORT_MODE::SORT_MODE_MERGE:
        result = par_amergesort<T, UINDEX>(pCutOffs, cutOffLength, (T*)pDataIn1, (UINDEX*)toSort, arraySize1, 0, PAR_SORT_TYPE::Float);
        break;

    case SORT_MODE::SORT_MODE_HEAP:
        result = aheapsort_float<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
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
        SortIndexFloat<float, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_DOUBLE:
        SortIndexFloat<double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    case ATOP_LONGDOUBLE:
        SortIndexFloat<long double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
        break;
    default:
        LOGGING("SortIndex does not understand type %d\n", arrayType1);
    }

}

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
extern "C" BOOL SortArray(void* pDataIn1, int64_t arraySize1, int32_t arrayType1, SORT_MODE mode) {
    switch (arrayType1) {
    case ATOP_STRING:
        SortInPlace<char>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_BOOL:
        SortInPlace<bool>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_INT8:
        SortInPlace<int8_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_INT16:
        SortInPlace<int16_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_INT32:
        SortInPlace<int32_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_INT64:
        SortInPlace<int64_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_UINT8:
        SortInPlace<uint8_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_UINT16:
        SortInPlace<uint16_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_UINT32:
        SortInPlace<uint32_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_UINT64:
        SortInPlace<uint64_t>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_FLOAT:
        SortInPlaceFloat<float>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_DOUBLE:
        SortInPlaceFloat<double>(pDataIn1, arraySize1, mode);
        break;
    case ATOP_LONGDOUBLE:
        SortInPlaceFloat<long double>(pDataIn1, arraySize1, mode);
        break;
    default:
        LOGGING("SortArray does not understand type %d\n", arrayType1);
        return FALSE;
        break;
    }
    return TRUE;
}



//
//template <typename UINDEX>
//void LexSort(UINDEX* pDataOut, int64_t* pCutOffs, int64_t cutOffLength)
//
//{
//
//    // BUG? what if we have index= and cutoffs= ??
//    if (pCutOffs) {
//        LOGGING("Have cutoffs %lld\n", cutOffLength);
//
//        // For cutoffs, prep the indexes with 0:n for each partition
//        UINDEX* pCounter = pDataOut;
//
//        int64_t startPos = 0;
//        for (int64_t j = 0; j < cutOffLength; j++) {
//
//            int64_t endPos = pCutOffs[j];
//            int64_t partitionLength = endPos - startPos;
//            for (UINDEX i = 0; i < partitionLength; i++) {
//                *pCounter++ = i;
//            }
//            startPos = endPos;
//        }
//    }
//    else {
//
//        // If the user did not provide a start index, we make one
//        if (index == NULL) {
//            THREADER->DoMultiThreadedChunkWork(arraySize1, ARangeCallback<UINDEX>, pDataOut);
//        }
//    }
//
//    // When multiple arrays are passed, we sort in order of how it is passed
//    // Thus, the last array is the last sort, and therefore determines the primary sort order
//    for (UINDEX i = 0; i < mlp.tupleSize; i++) {
//        // For each array...
//        // TODO!! Convert DTYPE
//        if (sizeof(UINDEX) == 4) {
//            SortIndex32(pCutOffs, cutOffLength, mlp.aInfo[i].pData, arraySize1, pDataOut, SORT_MODE::SORT_MODE_MERGE, mlp.aInfo[i].NumpyDType, mlp.aInfo[i].ItemSize);
//        }
//        else {
//            SortIndex64(pCutOffs, cutOffLength, mlp.aInfo[i].pData, arraySize1, pDataOut, SORT_MODE::SORT_MODE_MERGE, mlp.aInfo[i].NumpyDType, mlp.aInfo[i].ItemSize);
//        }
//    }
//
//}




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
