#include "common.h"
#include <cmath>
#include "../atop/threads.h"

//------------------------------------------------------
// The routines below are largely copied from numpy
//
#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

#ifdef NPY_ALLOW_THREADS
#define NPY_BEGIN_THREADS_NDITER(iter) \
        do { \
            if (!NpyIter_IterationNeedsAPI(iter)) { \
                NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter)); \
            } \
        } while(0)
#else
#define NPY_BEGIN_THREADS_NDITER(iter)
#endif


//#define PY_SSIZE_T_CLEAN
//#include <Python.h>
//#include "structmember.h"

//#define NPY_NO_DEPRECATED_API NPY_API_VERSION
//#define _MULTIARRAYMODULE
//#include "numpy/arrayobject.h"
//#include "numpy/arrayscalars.h"
//
//#include "numpy/npy_math.h"
//#include "numpy/npy_cpu.h"

//#include "npy_config.h"
//
//#include "npy_pycompat.h"
//
//#include "multiarraymodule.h"
//#include "common.h"
//#include "arrayobject.h"
//#include "ctors.h"
//#include "lowlevel_strided_loops.h"
//#include "array_assign.h"
//
//#include "npy_sort.h"
//#include "npy_partition.h"
//#include "npy_binsearch.h"
//#include "alloc.h"
//#include "arraytypes.h"
//
//
//typedef enum {
//    NPY_CLIP = 0,
//    NPY_WRAP = 1,
//    NPY_RAISE = 2
//} NPY_CLIPMODE;
#define PyArray_TRIVIALLY_ITERABLE(arr) ( \
                    PyArray_NDIM(arr) <= 1 || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS) || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS) \
                    )

#define PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size, arr) ( \
        assert(PyArray_TRIVIALLY_ITERABLE(arr)), \
        size == 1 ? 0 : ((PyArray_NDIM(arr) == 1) ? \
                             PyArray_STRIDE(arr, 0) : PyArray_ITEMSIZE(arr)))


#define PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) (            \
                        PyArray_NDIM(arr1) == PyArray_NDIM(arr2) && \
                        PyArray_CompareLists(PyArray_DIMS(arr1), \
                                             PyArray_DIMS(arr2), \
                                             PyArray_NDIM(arr1)) && \
                        (PyArray_FLAGS(arr1)&(NPY_ARRAY_C_CONTIGUOUS| \
                                      NPY_ARRAY_F_CONTIGUOUS)) & \
                                (PyArray_FLAGS(arr2)&(NPY_ARRAY_C_CONTIGUOUS| \
                                              NPY_ARRAY_F_CONTIGUOUS)) \
                        )

#define PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2, arr1_read, arr2_read) ( \
                        PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) && \
                        PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK( \
                            arr1, arr2, arr1_read, arr2_read))

#define PyArray_PREPARE_TRIVIAL_ITERATION(arr, count, data, stride) \
                    count = PyArray_SIZE(arr); \
                    data = PyArray_BYTES(arr); \
                    stride = ((PyArray_NDIM(arr) == 0) ? 0 : \
                                    ((PyArray_NDIM(arr) == 1) ? \
                                            PyArray_STRIDE(arr, 0) : \
                                            PyArray_ITEMSIZE(arr)));

/*
 * memchr with stride and invert argument
 * intended for small searches where a call out to libc memchr is costly.
 * stride must be a multiple of size.
 * compared to memchr it returns one stride past end instead of NULL if needle
 * is not found.
 */
static NPY_INLINE char*
npy_memchr(char* haystack, char needle,
    npy_intp stride, npy_intp size, npy_intp * psubloopsize, int invert)
{
    char* p = haystack;
    npy_intp subloopsize = 0;

    if (!invert) {
        /*
         * this is usually the path to determine elements to process,
         * performance less important here.
         * memchr has large setup cost if 0 byte is close to start.
         */
        while (subloopsize < size && *p != needle) {
            subloopsize++;
            p += stride;
        }
    }
    else {
        /* usually find elements to skip path */
        if (NPY_CPU_HAVE_UNALIGNED_ACCESS && needle == 0 && stride == 1) {
            /* iterate until last multiple of 4 */
            char* block_end = haystack + size - (size % sizeof(unsigned int));
            while (p < block_end) {
                unsigned int  v = *(unsigned int*)p;
                if (v != 0) {
                    break;
                }
                p += sizeof(unsigned int);
            }
            /* handle rest */
            subloopsize = (p - haystack);
        }
        while (subloopsize < size && *p == needle) {
            subloopsize++;
            p += stride;
        }
    }

    *psubloopsize = subloopsize;

    return p;
}



///*
// * Sets the base object using PyArray_SetBaseObject
// */
//NPY_NO_EXPORT PyObject*
//PyArray_NewFromDescrAndBase(
//    PyTypeObject * subtype, PyArray_Descr * descr,
//    int nd, npy_intp const* dims, npy_intp const* strides, void* data,
//    int flags, PyObject * obj, PyObject * base)
//{
//    return PyArray_NewFromDescr_int(subtype, descr, nd,
//        dims, strides, data,
//        flags, obj, base, 0, 0);
//}

//===================================================================

#define NPY_MAX_PIVOT_STACK 50

#define NBUCKETS 1024 /* number of buckets for data*/
#define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
#define NCACHE 7 /* number of cache entries per bucket */
/* this structure fits neatly into a cacheline */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void* ptrs[NCACHE];
} cache_bucket;
static cache_bucket datacache[NBUCKETS];
static cache_bucket dimcache[NBUCKETS_DIM];


/* as the cache is managed in global variables verify the GIL is held */

/*
 * very simplistic small memory block cache to avoid more expensive libc
 * allocations
 * base function for data cache with 1 byte buckets and dimension cache with
 * sizeof(npy_intp) byte buckets
 */
static NPY_INLINE void*
_npy_alloc_cache(npy_uintp nelem, npy_uintp esz, npy_uint msz,
    cache_bucket* cache, void* (*alloc)(size_t))
{
    void* p;
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
    p = alloc(nelem * esz);
    if (p) {
#ifdef _PyPyGC_AddMemoryPressure
        _PyPyPyGC_AddMemoryPressure(nelem * esz);
#endif
#ifdef NPY_OS_LINUX
        /* allow kernel allocating huge pages for large arrays */
        if (NPY_UNLIKELY(nelem * esz >= ((1u << 22u))) && _madvise_hugepage) {
            npy_uintp offset = 4096u - (npy_uintp)p % (4096u);
            npy_uintp length = nelem * esz - offset;
            /**
             * Intentionally not checking for errors that may be returned by
             * older kernel versions; optimistically tries enabling huge pages.
             */
            madvise((void*)((npy_uintp)p + offset), length, MADV_HUGEPAGE);
        }
#endif
    }
    return p;
}

/*
 * return pointer p to cache, nelem is number of elements of the cache bucket
 * size (1 or sizeof(npy_intp)) of the block pointed too
 */
static NPY_INLINE void
_npy_free_cache(void* p, npy_uintp nelem, npy_uint msz,
    cache_bucket* cache, void (*dealloc)(void*))
{
    assert(PyGILState_Check());
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
    dealloc(p);
}

NPY_NO_EXPORT void
npy_free_cache(void* p, npy_uintp sz)
{
    _npy_free_cache(p, sz, NBUCKETS, datacache, &PyDataMem_FREE);
}

/*
 * array data cache, sz is number of bytes to allocate
 */
NPY_NO_EXPORT void*
npy_alloc_cache(npy_uintp sz)
{
    return _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
}
//------------------------------------------------------------------------------
/*
 * Returns -1 and sets an exception if *axis is an invalid axis for
 * an array of dimension ndim, otherwise adjusts it in place to be
 * 0 <= *axis < ndim, and returns 0.
 *
 * msg_prefix: borrowed reference, a string to prepend to the message
 */
static NPY_INLINE int
check_and_adjust_axis_msg(int* axis, int ndim, PyObject* msg_prefix)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        /*
         * Load the exception type, if we don't already have it. Unfortunately
         * we don't have access to npy_cache_import here
         */
        static PyObject* AxisError_cls = NULL;
        PyObject* exc;

        if (AxisError_cls == NULL) {
            PyObject* mod = PyImport_ImportModule("numpy.core._exceptions");

            if (mod != NULL) {
                AxisError_cls = PyObject_GetAttrString(mod, "AxisError");
                Py_DECREF(mod);
            }
        }

        /* Invoke the AxisError constructor */
        exc = PyObject_CallFunction(AxisError_cls, "iiO",
            *axis, ndim, msg_prefix);
        if (exc == NULL) {
            return -1;
        }
        PyErr_SetObject(AxisError_cls, exc);
        Py_DECREF(exc);

        return -1;
    }
    /* adjust negative indices */
    if (*axis < 0) {
        *axis += ndim;
    }
    return 0;
}
static NPY_INLINE int
check_and_adjust_axis(int* axis, int ndim)
{
    return check_and_adjust_axis_msg(axis, ndim, Py_None);
}

//------------------------------------------------------------------------------
/*
 * return true if pointer is aligned to 'alignment'
 */
static NPY_INLINE int
npy_is_aligned(const void* p, const npy_uintp alignment)
{
    /*
     * Assumes alignment is a power of two, as required by the C standard.
     * Assumes cast from pointer to uintp gives a sensible representation we
     * can use bitwise & on (not required by C standard, but used by glibc).
     * This test is faster than a direct modulo.
     * Note alignment value of 0 is allowed and returns False.
     */
    return ((npy_uintp)(p) & ((alignment)-1)) == 0;
}

/* See array_assign.h for parameter documentation */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, npy_intp const* shape,
    char* data, npy_intp const* strides, int alignment)
{

    /*
     * The code below expects the following:
     *  * that alignment is a power of two, as required by the C standard.
     *  * that casting from pointer to uintp gives a sensible representation
     *    we can use bitwise operations on (perhaps *not* req. by C std,
     *    but assumed by glibc so it should be fine)
     *  * that casting stride from intp to uintp (to avoid dependence on the
     *    signed int representation) preserves remainder wrt alignment, so
     *    stride%a is the same as ((unsigned intp)stride)%a. Req. by C std.
     *
     *  The code checks whether the lowest log2(alignment) bits of `data`
     *  and all `strides` are 0, as this implies that
     *  (data + n*stride)%alignment == 0 for all integers n.
     */

    if (alignment > 1) {
        npy_uintp align_check = (npy_uintp)data;
        int i;

        for (i = 0; i < ndim; i++) {
#if NPY_RELAXED_STRIDES_CHECKING
            /* skip dim == 1 as it is not required to have stride 0 */
            if (shape[i] > 1) {
                /* if shape[i] == 1, the stride is never used */
                align_check |= (npy_uintp)strides[i];
            }
            else if (shape[i] == 0) {
                /* an array with zero elements is always aligned */
                return 1;
            }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
            align_check |= (npy_uintp)strides[i];
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
        }

        return npy_is_aligned((void*)align_check, alignment);
    }
    else if (alignment == 1) {
        return 1;
    }
    else {
        /* always return false for alignment == 0, which means cannot-be-aligned */
        return 0;
    }
}

NPY_NO_EXPORT int
IsAligned(PyArrayObject* ap)
{
    return raw_array_is_aligned(PyArray_NDIM(ap), PyArray_DIMS(ap),
        (char *)PyArray_DATA(ap), PyArray_STRIDES(ap),
        PyArray_DESCR(ap)->alignment);
}

typedef enum {
    MEM_OVERLAP_NO = 0,        /* no solution exists */
    MEM_OVERLAP_YES = 1,       /* solution found */
    MEM_OVERLAP_TOO_HARD = -1, /* max_work exceeded */
    MEM_OVERLAP_OVERFLOW = -2, /* algorithm failed due to integer overflow */
    MEM_OVERLAP_ERROR = -3     /* invalid input */
} mem_overlap_t;


/* Gets a half-open range [start, end) of offsets from the data pointer */
NPY_VISIBILITY_HIDDEN void
offset_bounds_from_strides(const int itemsize, const int nd,
    const npy_intp* dims, const npy_intp* strides,
    npy_intp* lower_offset, npy_intp* upper_offset)
{
    npy_intp max_axis_offset;
    npy_intp lower = 0;
    npy_intp upper = 0;
    int i;

    for (i = 0; i < nd; i++) {
        if (dims[i] == 0) {
            /* If the array size is zero, return an empty range */
            *lower_offset = 0;
            *upper_offset = 0;
            return;
        }
        /* Expand either upwards or downwards depending on stride */
        max_axis_offset = strides[i] * (dims[i] - 1);
        if (max_axis_offset > 0) {
            upper += max_axis_offset;
        }
        else {
            lower += max_axis_offset;
        }
    }
    /* Return a half-open range */
    upper += itemsize;
    *lower_offset = lower;
    *upper_offset = upper;
}


/* Gets a half-open range [start, end) which contains the array data */
static void
get_array_memory_extents(PyArrayObject* arr,
    npy_uintp* out_start, npy_uintp* out_end,
    npy_uintp* num_bytes)
{
    npy_intp low, upper;
    int j;
    offset_bounds_from_strides(PyArray_ITEMSIZE(arr), PyArray_NDIM(arr),
        PyArray_DIMS(arr), PyArray_STRIDES(arr),
        &low, &upper);
    *out_start = (npy_uintp)PyArray_DATA(arr) + (npy_uintp)low;
    *out_end = (npy_uintp)PyArray_DATA(arr) + (npy_uintp)upper;

    *num_bytes = PyArray_ITEMSIZE(arr);
    for (j = 0; j < PyArray_NDIM(arr); ++j) {
        *num_bytes *= PyArray_DIM(arr, j);
    }
}



/**
 * Determine whether two arrays share some memory.
 *
 * Returns: 0 (no shared memory), 1 (shared memory), or < 0 (failed to solve).
 *
 * Note that failures to solve can occur due to integer overflows, or effort
 * required solving the problem exceeding max_work.  The general problem is
 * NP-hard and worst case runtime is exponential in the number of dimensions.
 * max_work controls the amount of work done, either exact (max_work == -1), only
 * a simple memory extent check (max_work == 0), or set an upper bound
 * max_work > 0 for the number of solution candidates considered.
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_share_memory(PyArrayObject* a, PyArrayObject* b)
{
    npy_uintp start1 = 0, end1 = 0, size1 = 0;
    npy_uintp start2 = 0, end2 = 0, size2 = 0;

    get_array_memory_extents(a, &start1, &end1, &size1);
    get_array_memory_extents(b, &start2, &end2, &size2);

    // TODO: Come up with better logic
    // This logic does not look correct.
    if (!(start1 < end2 && start2 < end1 && start1 < end1 && start2 < end2)) {
        /* Memory extents don't overlap */
        return MEM_OVERLAP_NO;
    }
    if (start1 > end1) {
        npy_uintp temp = start1;
        start1 = end1;
        end1 = temp;
    }
    if (start2 > end2) {
        npy_uintp temp = start2;
        start2 = end2;
        end2 = temp;
    }
    if ((end1 < start2) || (start1 > end2)) {
        return MEM_OVERLAP_NO;
    }

    return MEM_OVERLAP_TOO_HARD;
}



/* Returns 1 if the arrays have overlapping data, 0 otherwise */
static int
arrays_overlap(PyArrayObject* arr1, PyArrayObject* arr2)
{
    mem_overlap_t result;

    result = solve_may_share_memory(arr1, arr2);
    if (result == MEM_OVERLAP_NO) {
        return 0;
    }
    else {
        return 1;
    }
}

/*
 * Returns -1 and sets an exception if *index is an invalid index for
 * an array of size max_item, otherwise adjusts it in place to be
 * 0 <= *index < max_item, and returns 0.
 * 'axis' should be the array axis that is being indexed over, if known. If
 * unknown, use -1.
 * If _save is NULL it is assumed the GIL is taken
 * If _save is not NULL it is assumed the GIL is not taken and it
 * is acquired in the case of an error
 */
FORCE_INLINE static int
check_and_adjust_index(npy_intp* index, npy_intp max_item, int axis,
    PyThreadState* _save)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*index < -max_item) || (*index >= max_item))) {
        NPY_END_THREADS;
        /* Try to be as clear as possible about what went wrong. */
        if (axis >= 0) {
            PyErr_Format(PyExc_IndexError,
                "index lld is out of bounds for axis %d with size %lld",
                (long long)*index, axis, (long long)max_item);
        }
        else {
            PyErr_Format(PyExc_IndexError,
                "index %lld is out of bounds for size %lld", (long long)*index, (long long)max_item);
        }
        return -1;
    }
    /* adjust negative indices */
    if (*index < 0) {
        *index += max_item;
    }
    return 0;
}

FORCE_INLINE static int
npy_fasttake_impl(
    char* dest, char* src, const npy_intp* indices,
    npy_intp n, npy_intp m, npy_intp max_item,
    npy_intp nelem, npy_intp chunk,
    NPY_CLIPMODE clipmode, npy_intp itemsize, int needs_refcounting,
    PyArray_Descr* dtype, int axis)
{
    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS_DESCR(dtype);
    switch (clipmode) {
    case NPY_RAISE:
        for (npy_intp i = 0; i < n; i++) {
            for (npy_intp j = 0; j < m; j++) {
                npy_intp tmp = indices[j];
                if (check_and_adjust_index(&tmp, max_item, axis,
                    _save) < 0) {
                    return -1;
                }
                char* tmp_src = src + tmp * chunk;
                if (needs_refcounting) {
                    for (npy_intp k = 0; k < nelem; k++) {
                        PyArray_Item_INCREF(tmp_src, dtype);
                        PyArray_Item_XDECREF(dest, dtype);
                        memmove(dest, tmp_src, itemsize);
                        dest += itemsize;
                        tmp_src += itemsize;
                    }
                }
                else {
                    memmove(dest, tmp_src, chunk);
                    dest += chunk;
                }
            }
            src += chunk * max_item;
        }
        break;
    case NPY_WRAP:
        for (npy_intp i = 0; i < n; i++) {
            for (npy_intp j = 0; j < m; j++) {
                npy_intp tmp = indices[j];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                char* tmp_src = src + tmp * chunk;
                if (needs_refcounting) {
                    for (npy_intp k = 0; k < nelem; k++) {
                        PyArray_Item_INCREF(tmp_src, dtype);
                        PyArray_Item_XDECREF(dest, dtype);
                        memmove(dest, tmp_src, itemsize);
                        dest += itemsize;
                        tmp_src += itemsize;
                    }
                }
                else {
                    memmove(dest, tmp_src, chunk);
                    dest += chunk;
                }
            }
            src += chunk * max_item;
        }
        break;
    case NPY_CLIP:
        for (npy_intp i = 0; i < n; i++) {
            for (npy_intp j = 0; j < m; j++) {
                npy_intp tmp = indices[j];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                char* tmp_src = src + tmp * chunk;
                if (needs_refcounting) {
                    for (npy_intp k = 0; k < nelem; k++) {
                        PyArray_Item_INCREF(tmp_src, dtype);
                        PyArray_Item_XDECREF(dest, dtype);
                        memmove(dest, tmp_src, itemsize);
                        dest += itemsize;
                        tmp_src += itemsize;
                    }
                }
                else {
                    memmove(dest, tmp_src, chunk);
                    dest += chunk;
                }
            }
            src += chunk * max_item;
        }
        break;
    }

    NPY_END_THREADS;
    return 0;
}


/*
 * Helper function instantiating npy_fasttake_impl in different branches
 * to allow the compiler to optimize each to the specific itemsize.
 */
static int
npy_fasttake(
    char* dest, char* src, const npy_intp* indices,
    npy_intp n, npy_intp m, npy_intp max_item,
    npy_intp nelem, npy_intp chunk,
    NPY_CLIPMODE clipmode, npy_intp itemsize, int needs_refcounting,
    PyArray_Descr* dtype, int axis)
{
    if (!needs_refcounting) {
        if (chunk == 1) {
            return npy_fasttake_impl(
                dest, src, indices, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis);
        }
        if (chunk == 2) {
            return npy_fasttake_impl(
                dest, src, indices, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis);
        }
        if (chunk == 4) {
            return npy_fasttake_impl(
                dest, src, indices, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis);
        }
        if (chunk == 8) {
            return npy_fasttake_impl(
                dest, src, indices, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis);
        }
        if (chunk == 16) {
            return npy_fasttake_impl(
                dest, src, indices, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis);
        }
        if (chunk == 32) {
            return npy_fasttake_impl(
                dest, src, indices, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis);
        }
    }

    return npy_fasttake_impl(
        dest, src, indices, n, m, max_item, nelem, chunk,
        clipmode, itemsize, needs_refcounting, dtype, axis);
}


/*NUMPY_API
 * Take
 */
static 
PyObject* PyArray_PTakeFrom(PyArrayObject* self0, PyObject* indices0, int axis,
    PyArrayObject* out, NPY_CLIPMODE clipmode)
{
    printf("take from\n");
    PyArray_Descr* dtype;
    PyArrayObject* obj = NULL, * self, * indices;
    npy_intp nd, i, n, m, max_item, chunk, itemsize, nelem;
    npy_intp shape[NPY_MAXDIMS];

    npy_bool needs_refcounting;

    indices = NULL;
    self = (PyArrayObject*)PyArray_CheckAxis(self0, &axis,
        NPY_ARRAY_CARRAY_RO);
    if (self == NULL) {
        return NULL;
    }
    indices = (PyArrayObject*)PyArray_ContiguousFromAny(indices0,
        NPY_INTP,
        0, 0);
    if (indices == NULL) {
        goto fail;
    }

    n = m = chunk = 1;
    nd = PyArray_NDIM(self) + PyArray_NDIM(indices) - 1;
    for (i = 0; i < nd; i++) {
        if (i < axis) {
            shape[i] = PyArray_DIMS(self)[i];
            n *= shape[i];
        }
        else {
            if (i < axis + PyArray_NDIM(indices)) {
                shape[i] = PyArray_DIMS(indices)[i - axis];
                m *= shape[i];
            }
            else {
                shape[i] = PyArray_DIMS(self)[i - PyArray_NDIM(indices) + 1];
                chunk *= shape[i];
            }
        }
    }
    if (!out) {
        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject*)PyArray_NewFromDescr(Py_TYPE(self),
            dtype,
            nd, shape,
            NULL, NULL, 0,
            (PyObject*)self);

        if (obj == NULL) {
            goto fail;
        }

    }
    else {
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;

        if ((PyArray_NDIM(out) != nd) ||
            !PyArray_CompareLists(PyArray_DIMS(out), shape, nd)) {
            PyErr_SetString(PyExc_ValueError,
                "output array does not match result of ndarray.take");
            goto fail;
        }

        if (arrays_overlap(out, self)) {
            flags |= NPY_ARRAY_ENSURECOPY;
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ARRAY_ENSURECOPY;
        }
        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject*)PyArray_FromArray(out, dtype, flags);
        if (obj == NULL) {
            goto fail;
        }
    }

    {

        max_item = PyArray_DIMS(self)[axis];
        nelem = chunk;
        itemsize = PyArray_ITEMSIZE(obj);
        chunk = chunk * itemsize;
        char* src = (char*)PyArray_DATA(self);
        char* dest = (char*)PyArray_DATA(obj);
        needs_refcounting = PyDataType_REFCHK(PyArray_DESCR(self));
        npy_intp* indices_data = (npy_intp*)PyArray_DATA(indices);

        if (!((max_item == 0) && (PyArray_SIZE(obj) != 0))) {

            if (npy_fasttake(
                dest, src, indices_data, n, m, max_item, nelem, chunk,
                clipmode, itemsize, needs_refcounting, dtype, axis) >= 0) {

                Py_XDECREF(indices);
                Py_XDECREF(self);
                if (out != NULL && out != obj) {
                    Py_INCREF(out);
                    PyArray_ResolveWritebackIfCopy(obj);
                    Py_DECREF(obj);
                    obj = out;
                }
                // success
                return (PyObject*)obj;
            }
        }
    }
fail:
    PyArray_DiscardWritebackIfCopy(obj);
    Py_XDECREF(obj);
    Py_XDECREF(indices);
    Py_XDECREF(self);
    return NULL;
}

/*NUMPY_API
 * Put values into an array
 */
static PyObject*
PyArray_PPutTo(PyArrayObject* self, PyObject* values0, PyObject* indices0,
    NPY_CLIPMODE clipmode)
{
    printf("put to\n");
    PyArrayObject* indices, * values;
    npy_intp i, chunk, ni, max_item, nv, tmp;
    char* src, * dest;
    int copied = 0;
    int overlap = 0;

    indices = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
            "put: first argument must be an array");
        return NULL;
    }

    if (PyArray_FailUnlessWriteable(self, "put: output array") < 0) {
        return NULL;
    }

    indices = (PyArrayObject*)PyArray_ContiguousFromAny(indices0,
        NPY_INTP, 0, 0);
    if (indices == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(indices);
    Py_INCREF(PyArray_DESCR(self));
    values = (PyArrayObject*)PyArray_FromAny(values0, PyArray_DESCR(self), 0, 0,
        NPY_ARRAY_DEFAULT | NPY_ARRAY_FORCECAST, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values);
    if (nv <= 0) {
        goto finish;
    }

    overlap = arrays_overlap(self, values) || arrays_overlap(self, indices);
    if (overlap || !PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject* obj;
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY |
            NPY_ARRAY_ENSURECOPY;

        Py_INCREF(PyArray_DESCR(self));
        obj = (PyArrayObject*)PyArray_FromArray(self,
            PyArray_DESCR(self), flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }
    max_item = PyArray_SIZE(self);
    dest = (char*)PyArray_DATA(self);
    chunk = PyArray_DESCR(self)->elsize;

    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        switch (clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp*)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, NULL) < 0) {
                    goto fail;
                }
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest + tmp * chunk, PyArray_DESCR(self));
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp*)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest + tmp * chunk, PyArray_DESCR(self));
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp*)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest + tmp * chunk, PyArray_DESCR(self));
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(ni);
        switch (clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp*)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, _save) < 0) {
                    goto fail;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp*)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp*)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
        NPY_END_THREADS;
    }

finish:
    Py_XDECREF(values);
    Py_XDECREF(indices);
    if (copied) {
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    Py_RETURN_NONE;

fail:
    Py_XDECREF(indices);
    Py_XDECREF(values);
    if (copied) {
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    return NULL;
}


static NPY_GCC_OPT_3 NPY_INLINE void
npy_fastputmask_impl(
    char* dest, char* src, const npy_bool* mask_data,
    npy_intp ni, npy_intp nv, npy_intp chunk)
{
    if (nv == 1) {
        for (npy_intp i = 0; i < ni; i++) {
            if (mask_data[i]) {
                memmove(dest, src, chunk);
            }
            dest += chunk;
        }
    }
    else {
        char* tmp_src = src;
        for (npy_intp i = 0, j = 0; i < ni; i++, j++) {
            if (NPY_UNLIKELY(j >= nv)) {
                j = 0;
                tmp_src = src;
            }
            if (mask_data[i]) {
                memmove(dest, tmp_src, chunk);
            }
            dest += chunk;
            tmp_src += chunk;
        }
    }
}


/*
 * Helper function instantiating npy_fastput_impl in different branches
 * to allow the compiler to optimize each to the specific itemsize.
 */
static NPY_GCC_OPT_3 void
npy_fastputmask(
    char* dest, char* src, npy_bool* mask_data,
    npy_intp ni, npy_intp nv, npy_intp chunk)
{
    if (chunk == 1) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 2) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 4) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 8) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 16) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 32) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }

    return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
}


/*NUMPY_API
 * Put values into an array according to a mask.
 */
NPY_NO_EXPORT PyObject*
PyArray_PPutMask(PyArrayObject* self, PyObject* values0, PyObject* mask0)
{
    printf("put mask\n");
    PyArrayObject* mask, * values;
    PyArray_Descr* dtype;
    npy_intp chunk, ni, nv;
    char* src, * dest;
    npy_bool* mask_data;
    int copied = 0;
    int overlap = 0;

    mask = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
            "putmask: first argument must "
            "be an array");
        return NULL;
    }

    mask = (PyArrayObject*)PyArray_FROM_OTF(mask0, NPY_BOOL,
        NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (mask == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(mask);
    if (ni != PyArray_SIZE(self)) {
        PyErr_SetString(PyExc_ValueError,
            "putmask: mask and data must be "
            "the same size");
        goto fail;
    }
    mask_data = (npy_bool*)PyArray_DATA(mask);
    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    values = (PyArrayObject*)PyArray_FromAny(values0, dtype,
        0, 0, NPY_ARRAY_CARRAY, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values); /* zero if null array */
    if (nv <= 0) {
        Py_XDECREF(values);
        Py_XDECREF(mask);
        Py_RETURN_NONE;
    }
    src = (char *)PyArray_DATA(values);

    overlap = arrays_overlap(self, values) || arrays_overlap(self, mask);
    if (overlap || !PyArray_ISCONTIGUOUS(self)) {
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;
        PyArrayObject* obj;

        if (overlap) {
            flags |= NPY_ARRAY_ENSURECOPY;
        }

        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject*)PyArray_FromArray(self, dtype, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    chunk = PyArray_DESCR(self)->elsize;
    dest = (char *)PyArray_DATA(self);

    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        for (npy_intp i = 0, j = 0; i < ni; i++, j++) {
            if (j >= nv) {
                j = 0;
            }
            if (mask_data[i]) {
                char* src_ptr = src + j * chunk;
                char* dest_ptr = dest + i * chunk;

                PyArray_Item_INCREF(src_ptr, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest_ptr, PyArray_DESCR(self));
                memmove(dest_ptr, src_ptr, chunk);
            }
        }
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(self));
        npy_fastputmask(dest, src, mask_data, ni, nv, chunk);
        NPY_END_THREADS;
    }

    Py_XDECREF(values);
    Py_XDECREF(mask);
    if (copied) {
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    Py_RETURN_NONE;

fail:
    Py_XDECREF(mask);
    Py_XDECREF(values);
    if (copied) {
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    return NULL;
}

/*NUMPY_API
 * Repeat the array.
 */
NPY_NO_EXPORT PyObject*
PyArray_PRepeat(PyArrayObject* aop, PyObject* op, int axis)
{
    printf("repeat\n");
    npy_intp* counts;
    npy_intp n, n_outer, i, j, k, chunk;
    npy_intp total = 0;
    npy_bool broadcast = NPY_FALSE;
    PyArrayObject* repeats = NULL;
    PyObject* ap = NULL;
    PyArrayObject* ret = NULL;
    char* new_data, * old_data;

    repeats = (PyArrayObject*)PyArray_ContiguousFromAny(op, NPY_INTP, 0, 1);
    if (repeats == NULL) {
        return NULL;
    }

    /*
     * Scalar and size 1 'repeat' arrays broadcast to any shape, for all
     * other inputs the dimension must match exactly.
     */
    if (PyArray_NDIM(repeats) == 0 || PyArray_SIZE(repeats) == 1) {
        broadcast = NPY_TRUE;
    }

    counts = (npy_intp*)PyArray_DATA(repeats);

    if ((ap = PyArray_CheckAxis(aop, &axis, NPY_ARRAY_CARRAY)) == NULL) {
        Py_DECREF(repeats);
        return NULL;
    }

    aop = (PyArrayObject*)ap;
    n = PyArray_DIM(aop, axis);

    if (!broadcast && PyArray_SIZE(repeats) != n) {
        PyErr_Format(PyExc_ValueError,
            "operands could not be broadcast together "
            "with shape (%zd,) (%zd,)", n, PyArray_DIM(repeats, 0));
        goto fail;
    }
    if (broadcast) {
        total = counts[0] * n;
    }
    else {
        for (j = 0; j < n; j++) {
            if (counts[j] < 0) {
                PyErr_SetString(PyExc_ValueError,
                    "repeats may not contain negative values.");
                goto fail;
            }
            total += counts[j];
        }
    }

    /* Construct new array */
    PyArray_DIMS(aop)[axis] = total;
    Py_INCREF(PyArray_DESCR(aop));
    ret = (PyArrayObject*)PyArray_NewFromDescr(Py_TYPE(aop),
        PyArray_DESCR(aop),
        PyArray_NDIM(aop),
        PyArray_DIMS(aop),
        NULL, NULL, 0,
        (PyObject*)aop);
    PyArray_DIMS(aop)[axis] = n;
    if (ret == NULL) {
        goto fail;
    }
    new_data = (char *)PyArray_DATA(ret);
    old_data = (char *)PyArray_DATA(aop);

    chunk = PyArray_DESCR(aop)->elsize;
    for (i = axis + 1; i < PyArray_NDIM(aop); i++) {
        chunk *= PyArray_DIMS(aop)[i];
    }

    n_outer = 1;
    for (i = 0; i < axis; i++) {
        n_outer *= PyArray_DIMS(aop)[i];
    }
    for (i = 0; i < n_outer; i++) {
        for (j = 0; j < n; j++) {
            npy_intp tmp = broadcast ? counts[0] : counts[j];
            for (k = 0; k < tmp; k++) {
                memcpy(new_data, old_data, chunk);
                new_data += chunk;
            }
            old_data += chunk;
        }
    }

    Py_DECREF(repeats);
    PyArray_INCREF(ret);
    Py_XDECREF(aop);
    return (PyObject*)ret;

fail:
    Py_DECREF(repeats);
    Py_XDECREF(aop);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 */
NPY_NO_EXPORT PyObject*
PyArray_PChoose(PyArrayObject* ip, PyObject* op, PyArrayObject* out,
    NPY_CLIPMODE clipmode)
{
    printf("choose\n");
    PyArrayObject* obj = NULL;
    PyArray_Descr* dtype;
    int n, elsize;
    npy_intp i;
    char* ret_data;
    PyArrayObject** mps, * ap;
    PyArrayMultiIterObject* multi = NULL;
    npy_intp mi;
    ap = NULL;

    /*
     * Convert all inputs to arrays of a common type
     * Also makes them C-contiguous
     */
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    ap = (PyArrayObject*)PyArray_FROM_OT((PyObject*)ip, NPY_INTP);
    if (ap == NULL) {
        goto fail;
    }
    /* Broadcast all arrays to each other, index array at the end. */
    multi = (PyArrayMultiIterObject*)
        PyArray_MultiIterFromObjects((PyObject**)mps, n, 1, ap);
    if (multi == NULL) {
        goto fail;
    }
    /* Set-up return array */
    if (out == NULL) {
        dtype = PyArray_DESCR(mps[0]);
        Py_INCREF(dtype);
        obj = (PyArrayObject*)PyArray_NewFromDescr(Py_TYPE(ap),
            dtype,
            multi->nd,
            multi->dimensions,
            NULL, NULL, 0,
            (PyObject*)ap);
    }
    else {
        int flags = NPY_ARRAY_CARRAY |
            NPY_ARRAY_WRITEBACKIFCOPY |
            NPY_ARRAY_FORCECAST;

        if ((PyArray_NDIM(out) != multi->nd)
            || !PyArray_CompareLists(PyArray_DIMS(out),
                multi->dimensions,
                multi->nd)) {
            PyErr_SetString(PyExc_TypeError,
                "choose: invalid shape for output array.");
            goto fail;
        }

        for (i = 0; i < n; i++) {
            if (arrays_overlap(out, mps[i])) {
                flags |= NPY_ARRAY_ENSURECOPY;
            }
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ARRAY_ENSURECOPY;
        }
        dtype = PyArray_DESCR(mps[0]);
        Py_INCREF(dtype);
        obj = (PyArrayObject*)PyArray_FromArray(out, dtype, flags);
    }

    if (obj == NULL) {
        goto fail;
    }
    elsize = PyArray_DESCR(obj)->elsize;
    ret_data = (char *)PyArray_DATA(obj);

    while (PyArray_MultiIter_NOTDONE(multi)) {
        mi = *((npy_intp*)PyArray_MultiIter_DATA(multi, n));
        if (mi < 0 || mi >= n) {
            switch (clipmode) {
            case NPY_RAISE:
                PyErr_SetString(PyExc_ValueError,
                    "invalid entry in choice "\
                    "array");
                goto fail;
            case NPY_WRAP:
                if (mi < 0) {
                    while (mi < 0) {
                        mi += n;
                    }
                }
                else {
                    while (mi >= n) {
                        mi -= n;
                    }
                }
                break;
            case NPY_CLIP:
                if (mi < 0) {
                    mi = 0;
                }
                else if (mi >= n) {
                    mi = n - 1;
                }
                break;
            }
        }
        memmove(ret_data, PyArray_MultiIter_DATA(multi, mi), elsize);
        ret_data += elsize;
        PyArray_MultiIter_NEXT(multi);
    }

    PyArray_INCREF(obj);
    Py_DECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_DECREF(ap);
    npy_free_cache(mps, n * sizeof(mps[0]));
    if (out != NULL && out != obj) {
        Py_INCREF(out);
        PyArray_ResolveWritebackIfCopy(obj);
        Py_DECREF(obj);
        obj = out;
    }
    return (PyObject*)obj;

fail:
    Py_XDECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_XDECREF(ap);
    npy_free_cache(mps, n * sizeof(mps[0]));
    PyArray_DiscardWritebackIfCopy(obj);
    Py_XDECREF(obj);
    return NULL;
}


NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char* dst, npy_intp outstrides, char* src,
    npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char* tout = dst;
    char* tin = src;

#define _COPY_N_SIZE(size) \
    for(i=0; i<N; i++) { \
        memcpy(tout, tin, size); \
        tin += instrides; \
        tout += outstrides; \
    } \
    return

    switch (elsize) {
    case 8:
        _COPY_N_SIZE(8);
    case 4:
        _COPY_N_SIZE(4);
    case 1:
        _COPY_N_SIZE(1);
    case 2:
        _COPY_N_SIZE(2);
    case 16:
        _COPY_N_SIZE(16);
    default:
        _COPY_N_SIZE(elsize);
    }
#undef _COPY_N_SIZE

}


/* byte swapping functions */
static NPY_INLINE npy_uint16
npy_bswap2(npy_uint16 x)
{
    return ((x & 0xffu) << 8) | (x >> 8);
}

/*
 * treat as int16 and byteswap unaligned memory,
 * some cpus don't support unaligned access
 */
static NPY_INLINE void
npy_bswap2_unaligned(char* x)
{
    char a = x[0];
    x[0] = x[1];
    x[1] = a;
}

static NPY_INLINE npy_uint32
npy_bswap4(npy_uint32 x)
{
#ifdef HAVE___BUILTIN_BSWAP32
    return __builtin_bswap32(x);
#else
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8) |
        ((x & 0xff0000u) >> 8) | (x >> 24);
#endif
}

static NPY_INLINE void
npy_bswap4_unaligned(char* x)
{
    char a = x[0];
    x[0] = x[3];
    x[3] = a;
    a = x[1];
    x[1] = x[2];
    x[2] = a;
}

static NPY_INLINE npy_uint64
npy_bswap8(npy_uint64 x)
{
#ifdef HAVE___BUILTIN_BSWAP64
    return __builtin_bswap64(x);
#else
    return ((x & 0xffULL) << 56) |
        ((x & 0xff00ULL) << 40) |
        ((x & 0xff0000ULL) << 24) |
        ((x & 0xff000000ULL) << 8) |
        ((x & 0xff00000000ULL) >> 8) |
        ((x & 0xff0000000000ULL) >> 24) |
        ((x & 0xff000000000000ULL) >> 40) |
        (x >> 56);
#endif
}

static NPY_INLINE void
npy_bswap8_unaligned(char* x)
{
    char a = x[0]; x[0] = x[7]; x[7] = a;
    a = x[1]; x[1] = x[6]; x[6] = a;
    a = x[2]; x[2] = x[5]; x[5] = a;
    a = x[3]; x[3] = x[4]; x[4] = a;
}

NPY_NO_EXPORT void
_strided_byte_swap(void* p, npy_intp stride, npy_intp n, int size)
{
    char* a, * b, c = 0;
    int j, m;

    switch (size) {
    case 1: /* no byteswap necessary */
        break;
    case 4:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint32))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint32* a_ = (npy_uint32*)a;
                *a_ = npy_bswap4(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap4_unaligned(a);
            }
        }
        break;
    case 8:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint64))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint64* a_ = (npy_uint64*)a;
                *a_ = npy_bswap8(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap8_unaligned(a);
            }
        }
        break;
    case 2:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint16))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint16* a_ = (npy_uint16*)a;
                *a_ = npy_bswap2(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap2_unaligned(a);
            }
        }
        break;
    default:
        m = size / 2;
        for (a = (char*)p; n > 0; n--, a += stride - m) {
            b = a + (size - 1);
            for (j = 0; j < m; j++) {
                c = *a; *a++ = *b; *b-- = c;
            }
        }
        break;
    }
}


/*
 * These algorithms use special sorting.  They are not called unless the
 * underlying sort function for the type is available.  Note that axis is
 * already valid. The sort functions require 1-d contiguous and well-behaved
 * data.  Therefore, a copy will be made of the data if needed before handing
 * it to the sorting routine.  An iterator is constructed and adjusted to walk
 * over all but the desired sorting axis.
 */
static int
_new_sortlike(PyArrayObject* op, int axis, PyArray_SortFunc* sort,
    PyArray_PartitionFunc* part, npy_intp const* kth, npy_intp nkth)
{
    npy_intp N = PyArray_DIM(op, axis);
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    npy_intp astride = PyArray_STRIDE(op, axis);
    int swap = PyArray_ISBYTESWAPPED(op);
    int needcopy = !IsAligned(op) || swap || astride != elsize;
    int hasrefs = PyDataType_REFCHK(PyArray_DESCR(op));

    PyArray_CopySwapNFunc* copyswapn = PyArray_DESCR(op)->f->copyswapn;
    char* buffer = NULL;

    PyArrayIterObject* it;
    npy_intp size;

    int ret = 0;

    NPY_BEGIN_THREADS_DEF;

    /* Check if there is any sorting to do */
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        return 0;
    }

    it = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)op, &axis);
    if (it == NULL) {
        return -1;
    }
    size = it->size;

    if (needcopy) {
        buffer = (char*)npy_alloc_cache(N * elsize);
        if (buffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(op));

    while (size--) {
        char* bufptr = it->dataptr;

        if (needcopy) {
            if (hasrefs) {
                /*
                 * For dtype's with objects, copyswapn Py_XINCREF's src
                 * and Py_XDECREF's dst. This would crash if called on
                 * an uninitialized buffer, or leak a reference to each
                 * object if initialized.
                 *
                 * So, first do the copy with no refcounting...
                 */
                _unaligned_strided_byte_copy(buffer, elsize,
                    it->dataptr, astride, N, elsize);
                /* ...then swap in-place if needed */
                if (swap) {
                    copyswapn(buffer, elsize, NULL, 0, N, swap, op);
                }
            }
            else {
                copyswapn(buffer, elsize, it->dataptr, astride, N, swap, op);
            }
            bufptr = buffer;
        }
        /*
         * TODO: If the input array is byte-swapped but contiguous and
         * aligned, it could be swapped (and later unswapped) in-place
         * rather than after copying to the buffer. Care would have to
         * be taken to ensure that, if there is an error in the call to
         * sort or part, the unswapping is still done before returning.
         */

        if (part == NULL) {
            ret = sort(bufptr, N, op);
            if (hasrefs && PyErr_Occurred()) {
                ret = -1;
            }
            if (ret < 0) {
                goto fail;
            }
        }
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;
            npy_intp i;
            for (i = 0; i < nkth; ++i) {
                ret = part(bufptr, N, kth[i], pivots, &npiv, op);
                if (hasrefs && PyErr_Occurred()) {
                    ret = -1;
                }
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        if (needcopy) {
            if (hasrefs) {
                if (swap) {
                    copyswapn(buffer, elsize, NULL, 0, N, swap, op);
                }
                _unaligned_strided_byte_copy(it->dataptr, astride,
                    buffer, elsize, N, elsize);
            }
            else {
                copyswapn(it->dataptr, astride, buffer, elsize, N, swap, op);
            }
        }

        PyArray_ITER_NEXT(it);
    }

fail:
    NPY_END_THREADS_DESCR(PyArray_DESCR(op));
    npy_free_cache(buffer, N * elsize);
    if (ret < 0 && !PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    Py_DECREF(it);

    return ret;
}

static PyObject*
_new_argsortlike(PyArrayObject* op, int axis, PyArray_ArgSortFunc* argsort,
    PyArray_ArgPartitionFunc* argpart,
    npy_intp const* kth, npy_intp nkth)
{
    npy_intp N = PyArray_DIM(op, axis);
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    npy_intp astride = PyArray_STRIDE(op, axis);
    int swap = PyArray_ISBYTESWAPPED(op);
    int needcopy = !IsAligned(op) || swap || astride != elsize;
    int hasrefs = PyDataType_REFCHK(PyArray_DESCR(op));
    int needidxbuffer;

    PyArray_CopySwapNFunc* copyswapn = PyArray_DESCR(op)->f->copyswapn;
    char* valbuffer = NULL;
    npy_intp* idxbuffer = NULL;

    PyArrayObject* rop;
    npy_intp rstride;

    PyArrayIterObject* it, * rit;
    npy_intp size;

    int ret = 0;

    NPY_BEGIN_THREADS_DEF;

    rop = (PyArrayObject*)PyArray_NewFromDescr(
        Py_TYPE(op), PyArray_DescrFromType(NPY_INTP),
        PyArray_NDIM(op), PyArray_DIMS(op), NULL, NULL,
        0, (PyObject*)op);
    if (rop == NULL) {
        return NULL;
    }
    rstride = PyArray_STRIDE(rop, axis);
    needidxbuffer = rstride != sizeof(npy_intp);

    /* Check if there is any argsorting to do */
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        memset(PyArray_DATA(rop), 0, PyArray_NBYTES(rop));
        return (PyObject*)rop;
    }

    it = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)op, &axis);
    rit = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)rop, &axis);
    if (it == NULL || rit == NULL) {
        ret = -1;
        goto fail;
    }
    size = it->size;

    if (needcopy) {
        valbuffer = (char*)npy_alloc_cache(N * elsize);
        if (valbuffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    if (needidxbuffer) {
        // TJD: This is allocating int64 buffer
        idxbuffer = (npy_intp*)npy_alloc_cache(N * sizeof(npy_intp));
        if (idxbuffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(op));

    while (size--) {
        char* valptr = it->dataptr;
        npy_intp* idxptr = (npy_intp*)rit->dataptr;
        npy_intp* iptr, i;

        if (needcopy) {
            if (hasrefs) {
                /*
                 * For dtype's with objects, copyswapn Py_XINCREF's src
                 * and Py_XDECREF's dst. This would crash if called on
                 * an uninitialized valbuffer, or leak a reference to
                 * each object item if initialized.
                 *
                 * So, first do the copy with no refcounting...
                 */
                _unaligned_strided_byte_copy(valbuffer, elsize,
                    it->dataptr, astride, N, elsize);
                /* ...then swap in-place if needed */
                if (swap) {
                    copyswapn(valbuffer, elsize, NULL, 0, N, swap, op);
                }
            }
            else {
                copyswapn(valbuffer, elsize,
                    it->dataptr, astride, N, swap, op);
            }
            valptr = valbuffer;
        }

        if (needidxbuffer) {
            idxptr = idxbuffer;
        }

        // TJD: This code is like an arange
        iptr = idxptr;
        for (i = 0; i < N; ++i) {
            *iptr++ = i;
        }

        if (argpart == NULL) {
            ret = argsort(valptr, idxptr, N, op);
            /* Object comparisons may raise an exception in Python 3 */
            if (hasrefs && PyErr_Occurred()) {
                ret = -1;
            }
            if (ret < 0) {
                goto fail;
            }
        }
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;

            for (i = 0; i < nkth; ++i) {
                ret = argpart(valptr, idxptr, N, kth[i], pivots, &npiv, op);
                /* Object comparisons may raise an exception in Python 3 */
                if (hasrefs && PyErr_Occurred()) {
                    ret = -1;
                }
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        if (needidxbuffer) {
            char* rptr = rit->dataptr;
            iptr = idxbuffer;

            for (i = 0; i < N; ++i) {
                *(npy_intp*)rptr = *iptr++;
                rptr += rstride;
            }
        }

        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(rit);
    }

fail:
    NPY_END_THREADS_DESCR(PyArray_DESCR(op));
    npy_free_cache(valbuffer, N * elsize);
    npy_free_cache(idxbuffer, N * sizeof(npy_intp));
    if (ret < 0) {
        if (!PyErr_Occurred()) {
            /* Out of memory during sorting or buffer creation */
            PyErr_NoMemory();
        }
        Py_XDECREF(rop);
        rop = NULL;
    }
    Py_XDECREF(it);
    Py_XDECREF(rit);

    return (PyObject*)rop;
}


/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyArray_PSort(PyArrayObject* op, int axis, NPY_SORTKIND which)
{
    printf("array sort\n");
    PyArray_SortFunc* sort = NULL;
    int n = PyArray_NDIM(op);

    if (check_and_adjust_axis(&axis, n) < 0) {
        return -1;
    }

    if (PyArray_FailUnlessWriteable(op, "sort array") < 0) {
        return -1;
    }

    if (which < 0 || which >= NPY_NSORTS) {
        PyErr_SetString(PyExc_ValueError, "not a valid sort kind");
        return -1;
    }

    sort = PyArray_DESCR(op)->f->sort[which];

    if (sort == NULL) {
        return -1;
    }

    return _new_sortlike(op, axis, sort, NULL, NULL, 0);
}


///*
// * make kth array positive, ravel and sort it
// */
//static PyArrayObject*
//partition_prep_kth_array(PyArrayObject* ktharray,
//    PyArrayObject* op,
//    int axis)
//{
//    const npy_intp* shape = PyArray_SHAPE(op);
//    PyArrayObject* kthrvl;
//    npy_intp* kth;
//    npy_intp nkth, i;
//
//    if (!PyArray_CanCastSafely(PyArray_TYPE(ktharray), NPY_INTP)) {
//        PyErr_Format(PyExc_TypeError, "Partition index must be integer");
//        return NULL;
//    }
//
//    if (PyArray_NDIM(ktharray) > 1) {
//        PyErr_Format(PyExc_ValueError, "kth array must have dimension <= 1");
//        return NULL;
//    }
//    kthrvl = (PyArrayObject*)PyArray_Cast(ktharray, NPY_INTP);
//
//    if (kthrvl == NULL)
//        return NULL;
//
//    kth = (npy_intp*)PyArray_DATA(kthrvl);
//    nkth = PyArray_SIZE(kthrvl);
//
//    for (i = 0; i < nkth; i++) {
//        if (kth[i] < 0) {
//            kth[i] += shape[axis];
//        }
//        if (PyArray_SIZE(op) != 0 &&
//            (kth[i] < 0 || kth[i] >= shape[axis])) {
//            PyErr_Format(PyExc_ValueError, "kth(=%zd) out of bounds (%zd)",
//                kth[i], shape[axis]);
//            Py_XDECREF(kthrvl);
//            return NULL;
//        }
//    }
//
//    /*
//     * sort the array of kths so the partitions will
//     * not trample on each other
//     */
//    if (PyArray_SIZE(kthrvl) > 1) {
//        PyArray_Sort(kthrvl, -1, NPY_QUICKSORT);
//    }
//
//    return kthrvl;
//}


///*NUMPY_API
// * Partition an array in-place
// */
//NPY_NO_EXPORT int
//PyArray_PPartition(PyArrayObject* op, PyArrayObject* ktharray, int axis,
//    NPY_SELECTKIND which)
//{
//    printf("partition\n");
//    PyArrayObject* kthrvl;
//    PyArray_PartitionFunc* part;
//    PyArray_SortFunc* sort;
//    int n = PyArray_NDIM(op);
//    int ret;
//
//    if (check_and_adjust_axis(&axis, n) < 0) {
//        return -1;
//    }
//
//    if (PyArray_FailUnlessWriteable(op, "partition array") < 0) {
//        return -1;
//    }
//
//    if (which < 0 || which >= NPY_NSELECTS) {
//        PyErr_SetString(PyExc_ValueError, "not a valid partition kind");
//        return -1;
//    }
//    part = get_partition_func(PyArray_TYPE(op), which);
//    if (part == NULL) {
//        /* Use sorting, slower but equivalent */
//        if (PyArray_DESCR(op)->f->compare) {
//            sort = npy_quicksort;
//        }
//        else {
//            PyErr_SetString(PyExc_TypeError,
//                "type does not have compare function");
//            return -1;
//        }
//    }
//
//    /* Process ktharray even if using sorting to do bounds checking */
//    kthrvl = partition_prep_kth_array(ktharray, op, axis);
//    if (kthrvl == NULL) {
//        return -1;
//    }
//
//    ret = _new_sortlike(op, axis, sort, part,
//        PyArray_DATA(kthrvl), PyArray_SIZE(kthrvl));
//
//    Py_DECREF(kthrvl);
//
//    return ret;
//}


/*NUMPY_API
 * ArgSort an array
 */
//NPY_NO_EXPORT PyObject*
//PyArray_PArgSort(PyArrayObject* op, int axis, NPY_SORTKIND which)
//{
//    printf("argsort\n");
//    PyArrayObject* op2;
//    PyArray_ArgSortFunc* argsort = NULL;
//    PyObject* ret;
//
//    argsort = PyArray_DESCR(op)->f->argsort[which];
//
//    if (argsort == NULL) {
//        if (PyArray_DESCR(op)->f->compare) {
//            switch (which) {
//            default:
//            case NPY_QUICKSORT:
//                argsort = npy_aquicksort;
//                break;
//            case NPY_HEAPSORT:
//                argsort = npy_aheapsort;
//                break;
//            case NPY_STABLESORT:
//                argsort = npy_atimsort;
//                break;
//            }
//        }
//        else {
//            PyErr_SetString(PyExc_TypeError,
//                "type does not have compare function");
//            return NULL;
//        }
//    }
//
//    op2 = (PyArrayObject*)PyArray_CheckAxis(op, &axis, 0);
//    if (op2 == NULL) {
//        return NULL;
//    }
//
//    ret = _new_argsortlike(op2, axis, argsort, NULL, NULL, 0);
//
//    Py_DECREF(op2);
//    return ret;
//}


/*NUMPY_API
 * ArgPartition an array
 */
//NPY_NO_EXPORT PyObject*
//PyArray_PArgPartition(PyArrayObject* op, PyArrayObject* ktharray, int axis,
//    NPY_SELECTKIND which)
//{
//    PyArrayObject* op2, * kthrvl;
//    PyArray_ArgPartitionFunc* argpart;
//    PyArray_ArgSortFunc* argsort;
//    PyObject* ret;
//
//    /*
//     * As a C-exported function, enum NPY_SELECTKIND loses its enum property
//     * Check the values to make sure they are in range
//     */
//    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
//        PyErr_SetString(PyExc_ValueError,
//            "not a valid partition kind");
//        return NULL;
//    }
//
//    argpart = get_argpartition_func(PyArray_TYPE(op), which);
//    if (argpart == NULL) {
//        /* Use sorting, slower but equivalent */
//        if (PyArray_DESCR(op)->f->compare) {
//            argsort = npy_aquicksort;
//        }
//        else {
//            PyErr_SetString(PyExc_TypeError,
//                "type does not have compare function");
//            return NULL;
//        }
//    }
//
//    op2 = (PyArrayObject*)PyArray_CheckAxis(op, &axis, 0);
//    if (op2 == NULL) {
//        return NULL;
//    }
//
//    /* Process ktharray even if using sorting to do bounds checking */
//    kthrvl = partition_prep_kth_array(ktharray, op2, axis);
//    if (kthrvl == NULL) {
//        Py_DECREF(op2);
//        return NULL;
//    }
//
//    ret = _new_argsortlike(op2, axis, argsort, argpart,
//        PyArray_DATA(kthrvl), PyArray_SIZE(kthrvl));
//
//    Py_DECREF(kthrvl);
//    Py_DECREF(op2);
//
//    return ret;
//}


/*NUMPY_API
 *LexSort an array providing indices that will sort a collection of arrays
 *lexicographically.  The first key is sorted on first, followed by the second key
 *-- requires that arg"merge"sort is available for each sort_key
 *
 *Returns an index array that shows the indexes for the lexicographic sort along
 *the given axis.
 */
NPY_NO_EXPORT PyObject*
PyArray_PLexSort(PyObject* sort_keys, int axis)
{
    printf("lexsort\n");

    PyArrayObject** mps;
    PyArrayIterObject** its;
    PyArrayObject* ret = NULL;
    PyArrayIterObject* rit = NULL;
    npy_intp n, N, size, i, j;
    npy_intp astride, rstride, * iptr;
    int nd;
    int needcopy = 0;
    int elsize;
    int maxelsize;
    int object = 0;
    PyArray_ArgSortFunc* argsort;
    NPY_BEGIN_THREADS_DEF;

    if (!PySequence_Check(sort_keys)
        || ((n = PySequence_Size(sort_keys)) <= 0)) {
        PyErr_SetString(PyExc_TypeError,
            "need sequence of keys with len > 0 in lexsort");
        return NULL;
    }
    mps = (PyArrayObject**)PyArray_malloc(n * sizeof(PyArrayObject*));
    if (mps == NULL) {
        return PyErr_NoMemory();
    }
    its = (PyArrayIterObject**)PyArray_malloc(n * sizeof(PyArrayIterObject*));
    if (its == NULL) {
        PyArray_free(mps);
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        mps[i] = NULL;
        its[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        PyObject* obj;
        obj = PySequence_GetItem(sort_keys, i);
        if (obj == NULL) {
            goto fail;
        }
        mps[i] = (PyArrayObject*)PyArray_FROM_O(obj);
        Py_DECREF(obj);
        if (mps[i] == NULL) {
            goto fail;
        }
        if (i > 0) {
            if ((PyArray_NDIM(mps[i]) != PyArray_NDIM(mps[0]))
                || (!PyArray_CompareLists(PyArray_DIMS(mps[i]),
                    PyArray_DIMS(mps[0]),
                    PyArray_NDIM(mps[0])))) {
                PyErr_SetString(PyExc_ValueError,
                    "all keys need to be the same shape");
                goto fail;
            }
        }
        if (!PyArray_DESCR(mps[i])->f->argsort[NPY_STABLESORT]
            && !PyArray_DESCR(mps[i])->f->compare) {
            PyErr_Format(PyExc_TypeError,
                "item %zd type does not have compare function", i);
            goto fail;
        }
        if (!object
            && PyDataType_FLAGCHK(PyArray_DESCR(mps[i]), NPY_NEEDS_PYAPI)) {
            object = 1;
        }
    }

    /* Now we can check the axis */
    nd = PyArray_NDIM(mps[0]);
    /*
    * Special case letting axis={-1,0} slip through for scalars,
    * for backwards compatibility reasons.
    */
    if (nd == 0 && (axis == 0 || axis == -1)) {
        /* TODO: can we deprecate this? */
    }
    else if (check_and_adjust_axis(&axis, nd) < 0) {
        goto fail;
    }
    if ((nd == 0) || (PyArray_SIZE(mps[0]) <= 1)) {
        /* empty/single element case */
        ret = (PyArrayObject*)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
            0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        if (PyArray_SIZE(mps[0]) > 0) {
            *((npy_intp*)(PyArray_DATA(ret))) = 0;
        }
        goto finish;
    }

    for (i = 0; i < n; i++) {
        its[i] = (PyArrayIterObject*)PyArray_IterAllButAxis(
            (PyObject*)mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now do the sorting */
    ret = (PyArrayObject*)PyArray_NewFromDescr(
        &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
        PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
        0, NULL);
    if (ret == NULL) {
        goto fail;
    }
    rit = (PyArrayIterObject*)
        PyArray_IterAllButAxis((PyObject*)ret, &axis);
    if (rit == NULL) {
        goto fail;
    }
    if (!object) {
        NPY_BEGIN_THREADS;
    }
    size = rit->size;
    N = PyArray_DIMS(mps[0])[axis];
    rstride = PyArray_STRIDE(ret, axis);
    maxelsize = PyArray_DESCR(mps[0])->elsize;
    needcopy = (rstride != sizeof(npy_intp));
    for (j = 0; j < n; j++) {
        needcopy = needcopy
            || PyArray_ISBYTESWAPPED(mps[j])
            || !(PyArray_FLAGS(mps[j]) & NPY_ARRAY_ALIGNED)
            || (PyArray_STRIDES(mps[j])[axis] != (npy_intp)PyArray_DESCR(mps[j])->elsize);
        if (PyArray_DESCR(mps[j])->elsize > maxelsize) {
            maxelsize = PyArray_DESCR(mps[j])->elsize;
        }
    }

    if (needcopy) {
        char* valbuffer, * indbuffer;
        int* swaps;

        assert(N > 0);  /* Guaranteed and assumed by indbuffer */
        npy_intp valbufsize = N * maxelsize;
        if (NPY_UNLIKELY(valbufsize) == 0) {
            valbufsize = 1;  /* Ensure allocation is not empty */
        }

        valbuffer = (char*)PyDataMem_NEW(valbufsize);
        if (valbuffer == NULL) {
            goto fail;
        }
        indbuffer = (char*)PyDataMem_NEW(N * sizeof(npy_intp));
        if (indbuffer == NULL) {
            PyDataMem_FREE(valbuffer);
            goto fail;
        }
        swaps = (int*)malloc(NPY_LIKELY(n > 0) ? n * sizeof(int) : 1);
        if (swaps == NULL) {
            PyDataMem_FREE(valbuffer);
            PyDataMem_FREE(indbuffer);
            goto fail;
        }

        for (j = 0; j < n; j++) {
            swaps[j] = PyArray_ISBYTESWAPPED(mps[j]);
        }
        while (size--) {
            iptr = (npy_intp*)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                int rcode;
                elsize = PyArray_DESCR(mps[j])->elsize;
                astride = PyArray_STRIDES(mps[j])[axis];
                argsort = PyArray_DESCR(mps[j])->f->argsort[NPY_STABLESORT];
                if (argsort == NULL) {
                    goto fail;
                    //argsort = npy_atimsort;
                }
                _unaligned_strided_byte_copy(valbuffer, (npy_intp)elsize,
                    its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (npy_intp)elsize, N, elsize);
                }
                rcode = argsort(valbuffer, (npy_intp*)indbuffer, N, mps[j]);
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                    && PyErr_Occurred())) {
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
                    free(swaps);
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                sizeof(npy_intp), N, sizeof(npy_intp));
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            iptr = (npy_intp*)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                int rcode;
                argsort = PyArray_DESCR(mps[j])->f->argsort[NPY_STABLESORT];
                if (argsort == NULL) {
                    goto fail;
                    //argsort = npy_atimsort;
                }
                rcode = argsort(its[j]->dataptr,
                    (npy_intp*)rit->dataptr, N, mps[j]);
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                    && PyErr_Occurred())) {
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            PyArray_ITER_NEXT(rit);
        }
    }

    if (!object) {
        NPY_END_THREADS;
    }

finish:
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    Py_XDECREF(rit);
    PyArray_free(mps);
    PyArray_free(its);
    return (PyObject*)ret;

fail:
    NPY_END_THREADS;
    if (!PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    Py_XDECREF(rit);
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    PyArray_free(mps);
    PyArray_free(its);
    return NULL;
}


///*NUMPY_API
// *
// * Search the sorted array op1 for the location of the items in op2. The
// * result is an array of indexes, one for each element in op2, such that if
// * the item were to be inserted in op1 just before that index the array
// * would still be in sorted order.
// *
// * Parameters
// * ----------
// * op1 : PyArrayObject *
// *     Array to be searched, must be 1-D.
// * op2 : PyObject *
// *     Array of items whose insertion indexes in op1 are wanted
// * side : {NPY_SEARCHLEFT, NPY_SEARCHRIGHT}
// *     If NPY_SEARCHLEFT, return first valid insertion indexes
// *     If NPY_SEARCHRIGHT, return last valid insertion indexes
// * perm : PyObject *
// *     Permutation array that sorts op1 (optional)
// *
// * Returns
// * -------
// * ret : PyObject *
// *   New reference to npy_intp array containing indexes where items in op2
// *   could be validly inserted into op1. NULL on error.
// *
// * Notes
// * -----
// * Binary search is used to find the indexes.
// */
//NPY_NO_EXPORT PyObject*
//PyArray_PSearchSorted(PyArrayObject* op1, PyObject* op2,
//    NPY_SEARCHSIDE side, PyObject* perm)
//{
//    printf("searchsorted\n");
//
//    PyArrayObject* ap1 = NULL;
//    PyArrayObject* ap2 = NULL;
//    PyArrayObject* ap3 = NULL;
//    PyArrayObject* sorter = NULL;
//    PyArrayObject* ret = NULL;
//    PyArray_Descr* dtype;
//    int ap1_flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED;
//    PyArray_BinSearchFunc* binsearch = NULL;
//    PyArray_ArgBinSearchFunc* argbinsearch = NULL;
//    NPY_BEGIN_THREADS_DEF;
//
//    /* Find common type */
//    dtype = PyArray_DescrFromObject((PyObject*)op2, PyArray_DESCR(op1));
//    if (dtype == NULL) {
//        return NULL;
//    }
//    /* refs to dtype we own = 1 */
//
//    /* Look for binary search function */
//    if (perm) {
//        argbinsearch = get_argbinsearch_func(dtype, side);
//    }
//    else {
//        binsearch = get_binsearch_func(dtype, side);
//    }
//    if (binsearch == NULL && argbinsearch == NULL) {
//        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
//        /* refs to dtype we own = 1 */
//        Py_DECREF(dtype);
//        /* refs to dtype we own = 0 */
//        return NULL;
//    }
//
//    /* need ap2 as contiguous array and of right type */
//    /* refs to dtype we own = 1 */
//    Py_INCREF(dtype);
//    /* refs to dtype we own = 2 */
//    ap2 = (PyArrayObject*)PyArray_CheckFromAny(op2, dtype,
//        0, 0,
//        NPY_ARRAY_CARRAY_RO | NPY_ARRAY_NOTSWAPPED,
//        NULL);
//    /* refs to dtype we own = 1, array creation steals one even on failure */
//    if (ap2 == NULL) {
//        Py_DECREF(dtype);
//        /* refs to dtype we own = 0 */
//        return NULL;
//    }
//
//    /*
//     * If the needle (ap2) is larger than the haystack (op1) we copy the
//     * haystack to a contiguous array for improved cache utilization.
//     */
//    if (PyArray_SIZE(ap2) > PyArray_SIZE(op1)) {
//        ap1_flags |= NPY_ARRAY_CARRAY_RO;
//    }
//    ap1 = (PyArrayObject*)PyArray_CheckFromAny((PyObject*)op1, dtype,
//        1, 1, ap1_flags, NULL);
//    /* refs to dtype we own = 0, array creation steals one even on failure */
//    if (ap1 == NULL) {
//        goto fail;
//    }
//
//    if (perm) {
//        /* need ap3 as a 1D aligned, not swapped, array of right type */
//        ap3 = (PyArrayObject*)PyArray_CheckFromAny(perm, NULL,
//            1, 1,
//            NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED,
//            NULL);
//        if (ap3 == NULL) {
//            PyErr_SetString(PyExc_TypeError,
//                "could not parse sorter argument");
//            goto fail;
//        }
//        if (!PyArray_ISINTEGER(ap3)) {
//            PyErr_SetString(PyExc_TypeError,
//                "sorter must only contain integers");
//            goto fail;
//        }
//        /* convert to known integer size */
//        sorter = (PyArrayObject*)PyArray_FromArray(ap3,
//            PyArray_DescrFromType(NPY_INTP),
//            NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
//        if (sorter == NULL) {
//            PyErr_SetString(PyExc_ValueError,
//                "could not parse sorter argument");
//            goto fail;
//        }
//        if (PyArray_SIZE(sorter) != PyArray_SIZE(ap1)) {
//            PyErr_SetString(PyExc_ValueError,
//                "sorter.size must equal a.size");
//            goto fail;
//        }
//    }
//
//    /* ret is a contiguous array of intp type to hold returned indexes */
//    ret = (PyArrayObject*)PyArray_NewFromDescr(
//        &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
//        PyArray_NDIM(ap2), PyArray_DIMS(ap2), NULL, NULL,
//        0, (PyObject*)ap2);
//    if (ret == NULL) {
//        goto fail;
//    }
//
//    if (ap3 == NULL) {
//        /* do regular binsearch */
//        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
//        binsearch((const char*)PyArray_DATA(ap1),
//            (const char*)PyArray_DATA(ap2),
//            (char*)PyArray_DATA(ret),
//            PyArray_SIZE(ap1), PyArray_SIZE(ap2),
//            PyArray_STRIDES(ap1)[0], PyArray_DESCR(ap2)->elsize,
//            NPY_SIZEOF_INTP, ap2);
//        NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
//    }
//    else {
//        /* do binsearch with a sorter array */
//        int error = 0;
//        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
//        error = argbinsearch((const char*)PyArray_DATA(ap1),
//            (const char*)PyArray_DATA(ap2),
//            (const char*)PyArray_DATA(sorter),
//            (char*)PyArray_DATA(ret),
//            PyArray_SIZE(ap1), PyArray_SIZE(ap2),
//            PyArray_STRIDES(ap1)[0],
//            PyArray_DESCR(ap2)->elsize,
//            PyArray_STRIDES(sorter)[0], NPY_SIZEOF_INTP, ap2);
//        NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
//        if (error < 0) {
//            PyErr_SetString(PyExc_ValueError,
//                "Sorter index out of range.");
//            goto fail;
//        }
//        Py_DECREF(ap3);
//        Py_DECREF(sorter);
//    }
//    Py_DECREF(ap1);
//    Py_DECREF(ap2);
//    return (PyObject*)ret;
//
//fail:
//    Py_XDECREF(ap1);
//    Py_XDECREF(ap2);
//    Py_XDECREF(ap3);
//    Py_XDECREF(sorter);
//    Py_XDECREF(ret);
//    return NULL;
//}
//
///*NUMPY_API
// * Diagonal
// *
// * In NumPy versions prior to 1.7,  this function always returned a copy of
// * the diagonal array. In 1.7, the code has been updated to compute a view
// * onto 'self', but it still copies this array before returning, as well as
// * setting the internal WARN_ON_WRITE flag. In a future version, it will
// * simply return a view onto self.
// */
//NPY_NO_EXPORT PyObject*
//PyArray_PDiagonal(PyArrayObject* self, int offset, int axis1, int axis2)
//{
//    int i, idim, ndim = PyArray_NDIM(self);
//    npy_intp* strides;
//    npy_intp stride1, stride2, offset_stride;
//    npy_intp* shape, dim1, dim2;
//
//    char* data;
//    npy_intp diag_size;
//    PyArray_Descr* dtype;
//    PyObject* ret;
//    npy_intp ret_shape[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];
//
//    if (ndim < 2) {
//        PyErr_SetString(PyExc_ValueError,
//            "diag requires an array of at least two dimensions");
//        return NULL;
//    }
//
//    /* Handle negative axes with standard Python indexing rules */
//    if (check_and_adjust_axis_msg(&axis1, ndim, npy_ma_str_axis1) < 0) {
//        return NULL;
//    }
//    if (check_and_adjust_axis_msg(&axis2, ndim, npy_ma_str_axis2) < 0) {
//        return NULL;
//    }
//    if (axis1 == axis2) {
//        PyErr_SetString(PyExc_ValueError,
//            "axis1 and axis2 cannot be the same");
//        return NULL;
//    }
//
//    /* Get the shape and strides of the two axes */
//    shape = PyArray_SHAPE(self);
//    dim1 = shape[axis1];
//    dim2 = shape[axis2];
//    strides = PyArray_STRIDES(self);
//    stride1 = strides[axis1];
//    stride2 = strides[axis2];
//
//    /* Compute the data pointers and diag_size for the view */
//    data = PyArray_DATA(self);
//    if (offset >= 0) {
//        offset_stride = stride2;
//        dim2 -= offset;
//    }
//    else {
//        offset = -offset;
//        offset_stride = stride1;
//        dim1 -= offset;
//    }
//    diag_size = dim2 < dim1 ? dim2 : dim1;
//    if (diag_size < 0) {
//        diag_size = 0;
//    }
//    else {
//        data += offset * offset_stride;
//    }
//
//    /* Build the new shape and strides for the main data */
//    i = 0;
//    for (idim = 0; idim < ndim; ++idim) {
//        if (idim != axis1 && idim != axis2) {
//            ret_shape[i] = shape[idim];
//            ret_strides[i] = strides[idim];
//            ++i;
//        }
//    }
//    ret_shape[ndim - 2] = diag_size;
//    ret_strides[ndim - 2] = stride1 + stride2;
//
//    /* Create the diagonal view */
//    dtype = PyArray_DTYPE(self);
//    Py_INCREF(dtype);
//    ret = PyArray_NewFromDescrAndBase(
//        Py_TYPE(self), dtype,
//        ndim - 1, ret_shape, ret_strides, data,
//        PyArray_FLAGS(self), (PyObject*)self, (PyObject*)self);
//    if (ret == NULL) {
//        return NULL;
//    }
//
//    /*
//     * For numpy 1.9 the diagonal view is not writeable.
//     * This line needs to be removed in 1.10.
//     */
//    PyArray_CLEARFLAGS((PyArrayObject*)ret, NPY_ARRAY_WRITEABLE);
//
//    return ret;
//}
//
///*NUMPY_API
// * Compress
// */
//NPY_NO_EXPORT PyObject*
//PyArray_PCompress(PyArrayObject* self, PyObject* condition, int axis,
//    PyArrayObject* out)
//{
//    PyArrayObject* cond;
//    PyObject* res, * ret;
//
//    if (PyArray_Check(condition)) {
//        cond = (PyArrayObject*)condition;
//        Py_INCREF(cond);
//    }
//    else {
//        PyArray_Descr* dtype = PyArray_DescrFromType(NPY_BOOL);
//        if (dtype == NULL) {
//            return NULL;
//        }
//        cond = (PyArrayObject*)PyArray_FromAny(condition, dtype,
//            0, 0, 0, NULL);
//        if (cond == NULL) {
//            return NULL;
//        }
//    }
//
//    if (PyArray_NDIM(cond) != 1) {
//        Py_DECREF(cond);
//        PyErr_SetString(PyExc_ValueError,
//            "condition must be a 1-d array");
//        return NULL;
//    }
//
//    res = PyArray_Nonzero(cond);
//    Py_DECREF(cond);
//    if (res == NULL) {
//        return res;
//    }
//    ret = PyArray_TakeFrom(self, PyTuple_GET_ITEM(res, 0), axis,
//        out, NPY_RAISE);
//    Py_DECREF(res);
//    return ret;
//}

/*
 * count number of nonzero bytes in 48 byte block
 * w must be aligned to 8 bytes
 *
 * even though it uses 64 bit types its faster than the bytewise sum on 32 bit
 * but a 32 bit type version would make it even faster on these platforms
 */
static NPY_INLINE npy_intp
count_nonzero_bytes_384(const npy_uint64* w)
{
    const npy_uint64 w1 = w[0];
    const npy_uint64 w2 = w[1];
    const npy_uint64 w3 = w[2];
    const npy_uint64 w4 = w[3];
    const npy_uint64 w5 = w[4];
    const npy_uint64 w6 = w[5];
    npy_intp r;

    /*
     * last part of sideways add popcount, first three bisections can be
     * skipped as we are dealing with bytes.
     * multiplication equivalent to (x + (x>>8) + (x>>16) + (x>>24)) & 0xFF
     * multiplication overflow well defined for unsigned types.
     * w1 + w2 guaranteed to not overflow as we only have 0 and 1 data.
     */
    r = ((w1 + w2 + w3 + w4 + w5 + w6) * 0x0101010101010101ULL) >> 56ULL;

    /*
     * bytes not exclusively 0 or 1, sum them individually.
     * should only happen if one does weird stuff with views or external
     * buffers.
     * Doing this after the optimistic computation allows saving registers and
     * better pipelining
     */
    if (NPY_UNLIKELY(
        ((w1 | w2 | w3 | w4 | w5 | w6) & 0xFEFEFEFEFEFEFEFEULL) != 0)) {
        /* reload from pointer to avoid a unnecessary stack spill with gcc */
        const char* c = (const char*)w;
        npy_uintp i, count = 0;
        for (i = 0; i < 48; i++) {
            count += (c[i] != 0);
        }
        return count;
    }

    return r;
}


//=============================================================================================================


/* Start raw iteration */
#define NPY_RAW_ITER_START(idim, ndim, coord, shape) \
        memset((coord), 0, (ndim) * sizeof(coord[0])); \
        do {

/* Increment to the next n-dimensional coordinate for one raw array */
#define NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (data) -= ((shape)[idim] - 1) * (strides)[idim]; \
                } \
                else { \
                    (data) += (strides)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for two raw arrays */
#define NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, dataB, stridesB) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for three raw arrays */
#define NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for four raw arrays */
#define NPY_RAW_ITER_FOUR_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC, \
                              dataD, stridesD) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                    (dataD) -= ((shape)[idim] - 1) * (stridesD)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    (dataD) += (stridesD)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))


//=====================================================================
/*
 * Prepares shape and strides for a simple raw array iteration.
 * This sorts the strides into FORTRAN order, reverses any negative
 * strides, then coalesces axes where possible. The results are
 * filled in the output parameters.
 *
 * This is intended for simple, lightweight iteration over arrays
 * where no buffering of any kind is needed, and the array may
 * not be stored as a PyArrayObject.
 *
 * The arrays shape, out_shape, strides, and out_strides must all
 * point to different data.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp const* shape,
    char* data, npy_intp const* strides,
    int* out_ndim, npy_intp* out_shape,
    char** out_data, npy_intp* out_strides)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_data = data;
        out_shape[0] = 1;
        out_strides[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entry = strides[0], shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride */
        if (stride_entry >= 0) {
            *out_data = data;
            out_strides[0] = stride_entry;
        }
        else {
            *out_data = data + stride_entry * (shape_entry - 1);
            out_strides[0] = -stride_entry;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, strides, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_strides[i] = strides[iperm];
    }

    /* Reverse any negative strides */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entry = out_strides[i], shape_entry = out_shape[i];

        if (stride_entry < 0) {
            data += stride_entry * (shape_entry - 1);
            out_strides[i] = -stride_entry;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_data = data;
            out_shape[0] = 0;
            out_strides[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_strides[i] * out_shape[i] == out_strides[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
    }
    ndim = i + 1;

#if 0
    /* DEBUG */
    {
        printf("raw iter ndim %d\n", ndim);
        printf("shape: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_shape[i]);
        }
        printf("\n");
        printf("strides: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_strides[i]);
        }
        printf("\n");
    }
#endif

    * out_data = data;
    *out_ndim = ndim;
    return 0;
}


/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char* data, npy_intp const* ashape, npy_intp const* astrides)
{
    printf("boolean true\n");
    int idim;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp i, coord[NPY_MAXDIMS];
    npy_intp count = 0;
    NPY_BEGIN_THREADS_DEF;

    /* Use raw iteration with no heap memory allocation */
    if (PyArray_PrepareOneRawArrayIter(
        ndim, ashape,
        data, astrides,
        &ndim, shape,
        &data, strides) < 0) {
        return -1;
    }

    /* Handle zero-sized array */
    if (shape[0] == 0) {
        return 0;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(shape[0]);

    /* Special case for contiguous inner loop */
    if (strides[0] == 1) {
        NPY_RAW_ITER_START(idim, ndim, coord, shape) {
            /* Process the innermost dimension */
            const char* d = data;
            const char* e = data + shape[0];
            if (NPY_CPU_HAVE_UNALIGNED_ACCESS ||
                npy_is_aligned(d, sizeof(npy_uint64))) {
                npy_uintp stride = 6 * sizeof(npy_uint64);
                for (; d < e - (shape[0] % stride); d += stride) {
                    count += count_nonzero_bytes_384((const npy_uint64*)d);
                }
            }
            for (; d < e; ++d) {
                count += (*d != 0);
            }
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides);
    }
    /* General inner loop */
    else {
        NPY_RAW_ITER_START(idim, ndim, coord, shape) {
            char* d = data;
            /* Process the innermost dimension */
            for (i = 0; i < shape[0]; ++i, d += strides[0]) {
                count += (*d != 0);
            }
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides);
    }

    NPY_END_THREADS;

    return count;
}

/*NUMPY_API
 * Counts the number of non-zero elements in the array.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
PyArray_PCountNonzero(PyArrayObject* self)
{
    printf("count nonzero\n");
    PyArray_NonzeroFunc* nonzero;
    char* data;
    npy_intp stride, count;
    npy_intp nonzero_count = 0;
    int needs_api = 0;
    PyArray_Descr* dtype;

    NpyIter* iter;
    NpyIter_IterNextFunc* iternext;
    char** dataptr;
    npy_intp* strideptr, * innersizeptr;
    NPY_BEGIN_THREADS_DEF;

    /* Special low-overhead version specific to the boolean type */
    dtype = PyArray_DESCR(self);
    if (dtype->type_num == NPY_BOOL) {
        return count_boolean_trues(PyArray_NDIM(self), (char*)PyArray_DATA(self),
            PyArray_DIMS(self), PyArray_STRIDES(self));
    }
    nonzero = PyArray_DESCR(self)->f->nonzero;

    /* If it's a trivial one-dimensional loop, don't use an iterator */
    if (PyArray_TRIVIALLY_ITERABLE(self)) {
        needs_api = PyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI);
        PyArray_PREPARE_TRIVIAL_ITERATION(self, count, data, stride);

        if (needs_api) {
            while (count--) {
                if (nonzero(data, self)) {
                    ++nonzero_count;
                }
                if (PyErr_Occurred()) {
                    return -1;
                }
                data += stride;
            }
        }
        else {
            NPY_BEGIN_THREADS_THRESHOLDED(count);
            while (count--) {
                if (nonzero(data, self)) {
                    ++nonzero_count;
                }
                data += stride;
            }
            NPY_END_THREADS;
        }

        return nonzero_count;
    }

    /*
     * If the array has size zero, return zero (the iterator rejects
     * size zero arrays)
     */
    if (PyArray_SIZE(self) == 0) {
        return 0;
    }

    /*
     * Otherwise create and use an iterator to count the nonzeros.
     */
    iter = NpyIter_New(self, NPY_ITER_READONLY |
        NPY_ITER_EXTERNAL_LOOP |
        NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING,
        NULL);
    if (iter == NULL) {
        return -1;
    }
    needs_api = NpyIter_IterationNeedsAPI(iter);

    /* Get the pointers for inner loop iteration */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    NPY_BEGIN_THREADS_NDITER(iter);

    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* Iterate over all the elements to count the nonzeros */
    do {
        data = *dataptr;
        stride = *strideptr;
        count = *innersizeptr;

        while (count--) {
            if (nonzero(data, self)) {
                ++nonzero_count;
            }
            if (needs_api && PyErr_Occurred()) {
                nonzero_count = -1;
                goto finish;
            }
            data += stride;
        }

    } while (iternext(iter));

finish:
    NPY_END_THREADS;

    NpyIter_Deallocate(iter);

    return nonzero_count;
}

/*NUMPY_API
 * Nonzero
 *
 * TODO: In NumPy 2.0, should make the iteration order a parameter.
 */
NPY_NO_EXPORT PyObject*
PyArray_PNonzero(PyArrayObject* self)
{
    printf("nonzero\n");
    int i, ndim = PyArray_NDIM(self);
    PyArrayObject* ret = NULL;
    PyObject* ret_tuple;
    npy_intp ret_dims[2];

    PyArray_NonzeroFunc* nonzero;
    PyArray_Descr* dtype;

    npy_intp nonzero_count;
    npy_intp added_count = 0;
    int needs_api;
    int is_bool;

    NpyIter* iter;
    NpyIter_IterNextFunc* iternext;
    NpyIter_GetMultiIndexFunc* get_multi_index;
    char** dataptr;

    dtype = PyArray_DESCR(self);
    nonzero = dtype->f->nonzero;
    needs_api = PyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI);

    /* Special case - nonzero(zero_d) is nonzero(atleast_1d(zero_d)) */
    if (ndim == 0) {
        char const* msg;
        if (PyArray_ISBOOL(self)) {
            msg =
                "Calling nonzero on 0d arrays is deprecated, as it behaves "
                "surprisingly. Use `atleast_1d(cond).nonzero()` if the old "
                "behavior was intended. If the context of this warning is of "
                "the form `arr[nonzero(cond)]`, just use `arr[cond]`.";
        }
        else {
            msg =
                "Calling nonzero on 0d arrays is deprecated, as it behaves "
                "surprisingly. Use `atleast_1d(arr).nonzero()` if the old "
                "behavior was intended.";
        }
        if (DEPRECATE(msg) < 0) {
            return NULL;
        }

        static npy_intp const zero_dim_shape[1] = { 1 };
        static npy_intp const zero_dim_strides[1] = { 0 };

        Py_INCREF(PyArray_DESCR(self));  /* array creation steals reference */

        
        PyArrayObject* self_1d = (PyArrayObject*)PyArray_NewFromDescr(
            Py_TYPE(self), PyArray_DESCR(self),
            1, zero_dim_shape, zero_dim_strides, PyArray_BYTES(self),
            PyArray_FLAGS(self), (PyObject*)self);// , (PyObject*)self);

        //PyArrayObject* self_1d = (PyArrayObject*)PyArray_NewFromDescrAndBase(
        //    Py_TYPE(self), PyArray_DESCR(self),
        //    1, zero_dim_shape, zero_dim_strides, PyArray_BYTES(self),
        //    PyArray_FLAGS(self), (PyObject*)self, (PyObject*)self);

        if (self_1d == NULL) {
            return NULL;
        }
        ret_tuple = PyArray_Nonzero(self_1d);
        Py_DECREF(self_1d);
        return ret_tuple;
    }

    /*
     * First count the number of non-zeros in 'self'.
     */
    nonzero_count = PyArray_CountNonzero(self);
    if (nonzero_count < 0) {
        return NULL;
    }

    is_bool = PyArray_ISBOOL(self);

    /* Allocate the result as a 2D array */
    ret_dims[0] = nonzero_count;
    ret_dims[1] = ndim;
    ret = (PyArrayObject*)PyArray_NewFromDescr(
        &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
        2, ret_dims, NULL, NULL,
        0, NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* If it's a one-dimensional result, don't use an iterator */
    if (ndim == 1) {
        npy_intp* multi_index = (npy_intp*)PyArray_DATA(ret);
        char* data = PyArray_BYTES(self);
        npy_intp stride = PyArray_STRIDE(self, 0);
        npy_intp count = PyArray_DIM(self, 0);
        NPY_BEGIN_THREADS_DEF;

        /* nothing to do */
        if (nonzero_count == 0) {
            goto finish;
        }

        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(count);
        }

        /* avoid function call for bool */
        if (is_bool) {
            /*
             * use fast memchr variant for sparse data, see gh-4370
             * the fast bool count is followed by this sparse path is faster
             * than combining the two loops, even for larger arrays
             */
            if (((double)nonzero_count / count) <= 0.1) {
                npy_intp subsize;
                npy_intp j = 0;
                while (1) {
                    npy_memchr(data + j * stride, 0, stride, count - j,
                        &subsize, 1);
                    j += subsize;
                    if (j >= count) {
                        break;
                    }
                    *multi_index++ = j++;
                }
            }
            else {
                npy_intp j;
                for (j = 0; j < count; ++j) {
                    if (*data != 0) {
                        *multi_index++ = j;
                    }
                    data += stride;
                }
            }
        }
        else {
            npy_intp j;
            for (j = 0; j < count; ++j) {
                if (nonzero(data, self)) {
                    if (++added_count > nonzero_count) {
                        break;
                    }
                    *multi_index++ = j;
                }
                if (needs_api && PyErr_Occurred()) {
                    break;
                }
                data += stride;
            }
        }

        NPY_END_THREADS;

        goto finish;
    }

    /*
     * Build an iterator tracking a multi-index, in C order.
     */
    iter = NpyIter_New(self, NPY_ITER_READONLY |
        NPY_ITER_MULTI_INDEX |
        NPY_ITER_ZEROSIZE_OK |
        NPY_ITER_REFS_OK,
        NPY_CORDER, NPY_NO_CASTING,
        NULL);

    if (iter == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        npy_intp* multi_index;
        NPY_BEGIN_THREADS_DEF;
        /* Get the pointers for inner loop iteration */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            return NULL;
        }
        get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
        if (get_multi_index == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            return NULL;
        }

        needs_api = NpyIter_IterationNeedsAPI(iter);

        NPY_BEGIN_THREADS_NDITER(iter);

        dataptr = NpyIter_GetDataPtrArray(iter);

        multi_index = (npy_intp*)PyArray_DATA(ret);

        /* Get the multi-index for each non-zero element */
        if (is_bool) {
            /* avoid function call for bool */
            do {
                if (**dataptr != 0) {
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
            } while (iternext(iter));
        }
        else {
            do {
                if (nonzero(*dataptr, self)) {
                    if (++added_count > nonzero_count) {
                        break;
                    }
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
                if (needs_api && PyErr_Occurred()) {
                    break;
                }
            } while (iternext(iter));
        }

        NPY_END_THREADS;
    }

    NpyIter_Deallocate(iter);

finish:
    if (PyErr_Occurred()) {
        Py_DECREF(ret);
        return NULL;
    }

    /* if executed `nonzero()` check for miscount due to side-effect */
    if (!is_bool && added_count != nonzero_count) {
        PyErr_SetString(PyExc_RuntimeError,
            "number of non-zero array elements "
            "changed during function execution.");
        Py_DECREF(ret);
        return NULL;
    }

    ret_tuple = PyTuple_New(ndim);
    if (ret_tuple == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    /* Create views into ret, one for each dimension */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride = ndim * NPY_SIZEOF_INTP;
        /* the result is an empty array, the view must point to valid memory */
        npy_intp data_offset = nonzero_count == 0 ? 0 : i * NPY_SIZEOF_INTP;

        PyArrayObject* view = (PyArrayObject*)PyArray_NewFromDescr(
            Py_TYPE(ret), PyArray_DescrFromType(NPY_INTP),
            1, &nonzero_count, &stride, PyArray_BYTES(ret) + data_offset,
            PyArray_FLAGS(ret), (PyObject*)ret);

        //PyArrayObject* view = (PyArrayObject*)PyArray_NewFromDescrAndBase(
        //    Py_TYPE(ret), PyArray_DescrFromType(NPY_INTP),
        //    1, &nonzero_count, &stride, PyArray_BYTES(ret) + data_offset,
        //    PyArray_FLAGS(ret), (PyObject*)ret, (PyObject*)ret);

        if (view == NULL) {
            Py_DECREF(ret);
            Py_DECREF(ret_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(ret_tuple, i, (PyObject*)view);
    }
    Py_DECREF(ret);

    return ret_tuple;
}

///*
// * Gets a single item from the array, based on a single multi-index
// * array of values, which must be of length PyArray_NDIM(self).
// */
//NPY_NO_EXPORT PyObject*
//PyArray_PMultiIndexGetItem(PyArrayObject* self, const npy_intp* multi_index)
//{
//    int idim, ndim = PyArray_NDIM(self);
//    char* data = (char*)PyArray_DATA(self);
//    npy_intp* shape = PyArray_SHAPE(self);
//    npy_intp* strides = PyArray_STRIDES(self);
//
//    /* Get the data pointer */
//    for (idim = 0; idim < ndim; ++idim) {
//        npy_intp shapevalue = shape[idim];
//        npy_intp ind = multi_index[idim];
//
//        if (check_and_adjust_index(&ind, shapevalue, idim, NULL) < 0) {
//            return NULL;
//        }
//        data += ind * strides[idim];
//    }
//
//    return PyArray_GETITEM(self, data);
//}
//
///*
// * Sets a single item in the array, based on a single multi-index
// * array of values, which must be of length PyArray_NDIM(self).
// *
// * Returns 0 on success, -1 on failure.
// */
//NPY_NO_EXPORT int
//PyArray_PMultiIndexSetItem(PyArrayObject* self, const npy_intp* multi_index,
//    PyObject* obj)
//{
//    int idim, ndim = PyArray_NDIM(self);
//    char* data = (char*)PyArray_DATA(self);
//    npy_intp* shape = PyArray_SHAPE(self);
//    npy_intp* strides = PyArray_STRIDES(self);
//
//    /* Get the data pointer */
//    for (idim = 0; idim < ndim; ++idim) {
//        npy_intp shapevalue = shape[idim];
//        npy_intp ind = multi_index[idim];
//
//        if (check_and_adjust_index(&ind, shapevalue, idim, NULL) < 0) {
//            return -1;
//        }
//        data += ind * strides[idim];
//    }
//
//    return PyArray_SETITEM(self, data, obj);
//}
