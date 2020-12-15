// If this is not included, calling PY_ARRAY functions will have a null value
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API
#define NO_IMPORT_ARRAY

#include "common.h"
#include "../atop/atop.h"
#include "../atop/threads.h"

//#if defined(_WIN32) && !defined(__GNUC__)
//#include <../Lib/site-packages/numpy/core/include/numpy/arrayobject.h>
//#else
//#include <numpy/arrayobject.h>
//#endif

//#include "Python.h"
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include "numpy/ndarrayobject.h"
//#include <stdint.h>
//#include <stdio.h>
//#include "../atop/threads.h"
#define LOGGING(...)


#if defined(_WIN32) && !defined(__GNUC__)

#define CASE_NPY_INT32      case NPY_INT32:       case NPY_INT
#define CASE_NPY_UINT32     case NPY_UINT32:      case NPY_UINT
#define CASE_NPY_INT64      case NPY_INT64
#define CASE_NPY_UINT64     case NPY_UINT64
#define CASE_NPY_FLOAT64    case NPY_DOUBLE:     case NPY_LONGDOUBLE

#else

#define CASE_NPY_INT32      case NPY_INT32
#define CASE_NPY_UINT32     case NPY_UINT32
#define CASE_NPY_INT64      case NPY_INT64:    case NPY_LONGLONG
#define CASE_NPY_UINT64     case NPY_UINT64:   case NPY_ULONGLONG
#define CASE_NPY_FLOAT64    case NPY_DOUBLE
#endif


//int64_t default1 = -9223372036854775808L;
static int64_t  gDefaultInt64 = 0x8000000000000000;
static int32_t  gDefaultInt32 = 0x80000000;
static uint16_t gDefaultInt16 = 0x8000;
static uint8_t  gDefaultInt8 = 0x80;

static uint64_t gDefaultUInt64 = 0xFFFFFFFFFFFFFFFF;
static uint32_t gDefaultUInt32 = 0xFFFFFFFF;
static uint16_t gDefaultUInt16 = 0xFFFF;
static uint8_t  gDefaultUInt8 = 0xFF;

static float  gDefaultFloat = NAN;
static double gDefaultDouble = NAN;
static int8_t   gDefaultBool = 0;
static char   gString[1024] = { 0,0,0,0 };

//----------------------------------------------------
// returns pointer to a data type (of same size in memory) that holds the invalid value for the type
// does not yet handle strings
void* GetDefaultForType(int numpyInType) {
    void* pgDefault = &gDefaultInt64;

    switch (numpyInType) {
    case NPY_FLOAT:  pgDefault = &gDefaultFloat;
        break;
    case NPY_LONGDOUBLE:
    case NPY_DOUBLE: pgDefault = &gDefaultDouble;
        break;
        // BOOL should not really have an invalid value inhabiting the type
    case NPY_BOOL:   pgDefault = &gDefaultBool;
        break;
    case NPY_BYTE:   pgDefault = &gDefaultInt8;
        break;
    case NPY_INT16:  pgDefault = &gDefaultInt16;
        break;
    CASE_NPY_INT32:  pgDefault = &gDefaultInt32;
        break;
    CASE_NPY_INT64:  pgDefault = &gDefaultInt64;
        break;
    case NPY_UINT8:  pgDefault = &gDefaultUInt8;
        break;
    case NPY_UINT16: pgDefault = &gDefaultUInt16;
        break;
    CASE_NPY_UINT32: pgDefault = &gDefaultUInt32;
        break;
    CASE_NPY_UINT64: pgDefault = &gDefaultUInt64;
        break;
    case NPY_STRING: pgDefault = &gString;
        break;
    case NPY_UNICODE: pgDefault = &gString;
        break;
    default:
        printf("!!! likely problem in GetDefaultForType\n");
    }

    return pgDefault;
}

// Structs used to hold any type of AVX 256 bit registers
struct _m128comboi {
    __m128i  i1;
    __m128i  i2;
};

struct _m256all {
    union {
        __m256i  i;
        __m256d  d;
        __m256   s;
        _m128comboi ci;
    };
};


//----------------------------------------------------------------
// Calculate the total number of bytes used by the array.
// TODO: Need to extend this to accomodate strided arrays.
int64_t CalcArrayLength(int ndim, npy_intp* dims) {
    int64_t length = 1;

    // handle case of zero length array
    if (dims && ndim > 0) {
        for (int i = 0; i < ndim; i++) {
            length *= dims[i];
        }
    }
    else {
        // Want to set this to zero, but scalar issue?
        //length = 0;
    }
    return length;
}

//----------------------------------------------------------------
// calcluate the total number of bytes used
int64_t ArrayLength(PyArrayObject* inArr) {

    return CalcArrayLength(PyArray_NDIM(inArr), PyArray_DIMS(inArr));
}

//-----------------------------------------------------------------------------------
PyArrayObject* AllocateNumpyArray(int ndim, npy_intp* dims, int32_t numpyType, int64_t itemsize=0, int fortran_array=0, npy_intp* strides=nullptr) {

    PyArrayObject* returnObject = nullptr;
    const int64_t    len = CalcArrayLength(ndim, dims);

    // PyArray_New (and the functions it wraps) don't truly respect the 'flags' argument
    // passed into them; they only check whether it's zero or non-zero, and based on that they
    // set the NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY flags. Construct our flags value so we end
    // up with an array with the layout the caller requested.
    const int array_flags = fortran_array ? NPY_ARRAY_F_CONTIGUOUS : 0;

    // Make one dimension size on stack
    volatile int64_t dimensions[1] = { len };

    // This is the safest way...
    if (!dims) {
        // Happens with a=FA([]); 100*a;  or  FA([1])[0] / FA([2])
        ndim = 1;
        dims = (npy_intp*)dimensions;
    }

    PyTypeObject* const allocType =  pPyArray_Type;

    // probably runt object from matlab -- have to fix this up or it will fail
    // comes from empty strings in matlab - might need to
    if (PyTypeNum_ISFLEXIBLE(numpyType) && itemsize == 0) {
        itemsize = 1;
    }

    // NOTE: this path taken when we already have data in our own memory
    returnObject = (PyArrayObject*)PyArray_New(
        allocType,
        ndim,
        dims,
        numpyType,
        strides,      // Strides
        nullptr,
        (int)itemsize,
        array_flags,
        NULL);

    if (!returnObject) {
        printf("!!!out of memory allocating numpy array size:%lld  dims:%d  dtype:%d  itemsize:%lld  flags:%d  dim0:%lld\n", (long long)len, ndim, numpyType, (long long)itemsize, array_flags, (long long)dims[0]);
        return nullptr;
    }

    return returnObject;
}


//-----------------------------------------------------------------------------------
// NOTE: will only allocate 1 dim arrays
PyArrayObject* AllocateLikeResize(PyArrayObject* inArr, npy_intp rowSize) {
    int numpyType = PyArray_TYPE(inArr);

    PyArrayObject* result = NULL;

    int64_t itemSize = PyArray_ITEMSIZE(inArr);
    result = AllocateNumpyArray(1, &rowSize, numpyType, itemSize);

    return result;
}


/**
 * Count the number of 'True' (nonzero) 1-byte bool values in an array,
 * using an AVX2-based implementation.
 *
 * @param pData Array of 1-byte bool values.
 * @param length The number of elements in the array.
 * @return The number of nonzero 1-byte bool values in the array.
 */
 // TODO: When we support runtime CPU detection/dispatching, bring back the original popcnt-based implementation
 //       of this function for systems that don't support AVX2. Also consider implementing an SSE-based version
 //       of this function for the same reason (logic will be very similar, just using __m128i instead).
 // TODO: Consider changing `length` to uint64_t here so it agrees better with the result of sizeof().
int64_t SumBooleanMask(const int8_t* const pData, const int64_t length) {
    // Basic input validation.
    if (!pData)
    {
        return 0;
    }
    else if (length < 0)
    {
        return 0;
    }

    // Now that we know length is >= 0, it's safe to convert it to unsigned so it agrees with
    // the sizeof() math in the logic below.
    // Make sure to use this instead of 'length' in the code below to avoid signed/unsigned
    // arithmetic warnings.
    const size_t ulength = length;

    // Holds the accumulated result value.
    int64_t result = 0;

    // YMM (32-byte) vector packed with 32 byte values, each set to 1.
    // NOTE: The obvious thing here would be to use _mm256_set1_epi8(1),
    //       but many compilers (e.g. MSVC) store the data for this vector
    //       then load it here, which unnecessarily wastes cache space we could be
    //       using for something else.
    //       Generate the constants using a few intrinsics, it's faster than even an L1 cache hit anyway.
    const auto zeros_ = _mm256_setzero_si256();
    // compare 0 to 0 returns 0xFF; treated as an int8_t, 0xFF = -1, so abs(-1) = 1.
    const auto ones = _mm256_abs_epi8(_mm256_cmpeq_epi8(zeros_, zeros_));

    //
    // Convert each byte in the input to a 0 or 1 byte according to C-style boolean semantics.
    //

    // This first loop does the bulk of the processing for large vectors -- it doesn't use popcount
    // instructions and instead relies on the fact we can sum 0/1 values to acheive the same result,
    // up to CHAR_MAX. This allows us to use very inexpensive instructions for most of the accumulation
    // so we're primarily limited by memory bandwidth.
    const size_t vector_length = ulength / sizeof(__m256i);
    const auto pVectorData = (__m256i*)pData;
    for (size_t i = 0; i < vector_length;)
    {
        // Determine how much we can process in _this_ iteration of the loop.
        // The maximum number of "inner" iterations here is CHAR_MAX (255),
        // because otherwise our byte-sized counters would overflow.
        auto inner_loop_iters = vector_length - i;
        if (inner_loop_iters > 255) inner_loop_iters = 255;

        // Holds the current per-vector-lane (i.e. per-byte-within-vector) popcount.
        // PERF: If necessary, the loop below can be manually unrolled to ensure we saturate memory bandwidth.
        auto byte_popcounts = _mm256_setzero_si256();
        for (size_t j = 0; j < inner_loop_iters; j++)
        {
            // Use an unaligned load to grab a chunk of data;
            // then call _mm256_min_epu8 where one operand is the register we set
            // earlier containing packed byte-sized 1 values (e.g. 0x01010101...).
            // This effectively converts each byte in the input to a 0 or 1 byte value.
            const auto cstyle_bools = _mm256_min_epu8(ones, _mm256_loadu_si256(&pVectorData[i + j]));

            // Since each byte in the converted vector now contains either a 0 or 1,
            // we can simply add it to the running per-byte sum to simulate a popcount.
            byte_popcounts = _mm256_add_epi8(byte_popcounts, cstyle_bools);
        }

        // Sum the per-byte-lane popcounts, then add them to the overall result.
        // For the vectorized partial sums, it's important the 'zeros' argument is used as the second operand
        // so that the zeros are 'unpacked' into the high byte(s) of each packed element in the result.
        const auto zeros = _mm256_setzero_si256();

        // Sum 32x 1-byte counts -> 16x 2-byte counts
        const auto byte_popcounts_8a = _mm256_unpacklo_epi8(byte_popcounts, zeros);
        const auto byte_popcounts_8b = _mm256_unpackhi_epi8(byte_popcounts, zeros);
        const auto byte_popcounts_16 = _mm256_add_epi16(byte_popcounts_8a, byte_popcounts_8b);

        // Sum 16x 2-byte counts -> 8x 4-byte counts
        const auto byte_popcounts_16a = _mm256_unpacklo_epi16(byte_popcounts_16, zeros);
        const auto byte_popcounts_16b = _mm256_unpackhi_epi16(byte_popcounts_16, zeros);
        const auto byte_popcounts_32 = _mm256_add_epi32(byte_popcounts_16a, byte_popcounts_16b);

        // Sum 8x 4-byte counts -> 4x 8-byte counts
        const auto byte_popcounts_32a = _mm256_unpacklo_epi32(byte_popcounts_32, zeros);
        const auto byte_popcounts_32b = _mm256_unpackhi_epi32(byte_popcounts_32, zeros);
        const auto byte_popcounts_64 = _mm256_add_epi64(byte_popcounts_32a, byte_popcounts_32b);

        // perform the operation horizontally in m0
        union {
            volatile int64_t  horizontal[4];
            __m256i mathreg[1];
        };

        mathreg[0] = byte_popcounts_64;
        for (int j = 0; j < 4; j++) {
            result += horizontal[j];
        }

        // Increment the outer loop counter by the number of inner iterations we performed.
        i += inner_loop_iters;
    }

    // Handle the last few bytes, if any, that couldn't be handled with the vectorized loop.
    const auto vectorized_length = vector_length * sizeof(__m256i);
    for (size_t i = vectorized_length; i < ulength; i++)
    {
        if (pData[i])
        {
            result++;
        }
    }

    return result;
}


//===================================================
// Input: boolean array
// Output: chunk count and ppChunkCount
// NOTE: CALLER MUST FREE pChunkCount
//
int64_t BooleanCount(PyArrayObject* aIndex, int64_t** ppChunkCount) {

    // Pass one, count the values
    // Eight at a time
    const int64_t lengthBool = ArrayLength(aIndex);
    const int8_t* const pBooleanMask = (int8_t*)PyArray_BYTES(aIndex);

    // Count the number of chunks (of boolean elements).
    // It's important we handle the case of an empty array (zero length) when determining the number
    // of per-chunk counts to return; the behavior of malloc'ing zero bytes is undefined, and the code
    // below assumes there's always at least one entry in the count-per-chunk array. If we don't handle
    // the empty array case we'll allocate an empty count-per-chunk array and end up doing an
    // out-of-bounds write.
    const int64_t chunkSize = THREADER->WORK_ITEM_CHUNK;
    int64_t chunks = lengthBool > 1 ? lengthBool : 1;

    chunks = (chunks + (chunkSize - 1)) / chunkSize;

    // TOOD: divide up per core instead
    int64_t* const pChunkCount = (int64_t*)WORKSPACE_ALLOC(chunks * sizeof(int64_t));


    // MT callback
    struct BSCallbackStruct {
        int64_t* pChunkCount;
        const int8_t* pBooleanMask;
    };

    // This is the routine that will be called back from multiple threads
    // t64_t(*MTCHUNK_CALLBACK)(void* callbackArg, int core, int64_t start, int64_t length);
    auto lambdaBSCallback = [](void* callbackArgT, int core, int64_t start, int64_t length) -> int64_t {
        BSCallbackStruct* callbackArg = (BSCallbackStruct*)callbackArgT;

        const int8_t* pBooleanMask = callbackArg->pBooleanMask;
        int64_t* pChunkCount = callbackArg->pChunkCount;

        // Use the single-threaded implementation to sum the number of
        // 1-byte boolean TRUE values in the current chunk.
        // This means the current function is just responsible for parallelizing over the chunks
        // but doesn't do any real "math" itself.
        int64_t total = SumBooleanMask(&pBooleanMask[start], length);

        pChunkCount[start / THREADER->WORK_ITEM_CHUNK] = total;
        return TRUE;
    };

    BSCallbackStruct stBSCallback;
    stBSCallback.pChunkCount = pChunkCount;
    stBSCallback.pBooleanMask = pBooleanMask;

    BOOL didMtWork = THREADER->DoMultiThreadedChunkWork(lengthBool, lambdaBSCallback, &stBSCallback);


    *ppChunkCount = pChunkCount;
    // if multithreading turned off...
    return didMtWork ? chunks : 1;
}


//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be anything)
// Arg2: numpy array aIndex (must be BOOL)
//
PyObject*
BooleanIndex(PyObject* self, PyObject* args)
{
    PyArrayObject* aValues = NULL;
    PyArrayObject* aIndex = NULL;

    if (!PyArg_ParseTuple(
        args, "O!O!:BooleanIndex",
        &PyArray_Type, &aValues,
        &PyArray_Type, &aIndex
    )) {

        return NULL;
    }

    if (PyArray_TYPE(aIndex) != NPY_BOOL) {
        PyErr_Format(PyExc_ValueError, "Second argument must be a boolean array");
        return NULL;
    }

    // Pass one, count the values
    // Eight at a time
    int64_t lengthBool = ArrayLength(aIndex);
    int64_t lengthValue = ArrayLength(aValues);

    if (lengthBool != lengthValue) {
        PyErr_Format(PyExc_ValueError, "Array lengths must match %lld vs %lld", lengthBool, lengthValue);
        return NULL;
    }

    int64_t* pChunkCount = NULL;
    int64_t    chunks = BooleanCount(aIndex, &pChunkCount);

    int64_t totalTrue = 0;

    // Store the offset
    for (int64_t i = 0; i < chunks; i++) {
        int64_t temp = totalTrue;
        totalTrue += pChunkCount[i];

        // reassign to the cumulative sum so we know the offset
        pChunkCount[i] = temp;
    }

    LOGGING("boolindex total: %I64d  length: %I64d  type:%d\n", totalTrue, lengthBool, PyArray_TYPE(aValues));

    int8_t* pBooleanMask = (int8_t*)PyArray_BYTES(aIndex);

    // Now we know per chunk how many true there are... we can allocate the new array
    PyArrayObject* pReturnArray = AllocateLikeResize(aValues, totalTrue);

    if (pReturnArray) {

        // MT callback
        struct BICallbackStruct {
            int64_t* pChunkCount;
            int8_t* pBooleanMask;
            char* pValuesIn;
            char* pValuesOut;
            int64_t    itemSize;
        };


        //-----------------------------------------------
        //-----------------------------------------------
        // This is the routine that will be called back from multiple threads
        auto lambdaBICallback2 = [](void* callbackArgT, int core, int64_t start, int64_t length) -> int64_t {
            BICallbackStruct* callbackArg = (BICallbackStruct*)callbackArgT;

            int8_t* pBooleanMask = callbackArg->pBooleanMask;
            int64_t* pData = (int64_t*)&pBooleanMask[start];
            int64_t  chunkCount = callbackArg->pChunkCount[start / THREADER->WORK_ITEM_CHUNK];
            int64_t  itemSize = callbackArg->itemSize;
            char* pValuesIn = &callbackArg->pValuesIn[start * itemSize];
            char* pValuesOut = &callbackArg->pValuesOut[chunkCount * itemSize];

            int64_t  blength = length / 8;

            switch (itemSize) {
            case 1:
            {
                int8_t* pVOut = (int8_t*)pValuesOut;
                int8_t* pVIn = (int8_t*)pValuesIn;

                for (int64_t i = 0; i < blength; i++) {
                    uint64_t bitmask = *(uint64_t*)pData;
                    uint64_t mask = 0xff;

                    // NOTE: the below can be optimized with vector intrinsics
                    // little endian, so the first value is low bit (not high bit)
                    if (bitmask != 0) {
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++; 
                    }
                    else {
                        pVIn += 8;
                    }
                    pData++;
                }

                // Get last
                pBooleanMask = (int8_t*)pData;

                blength = length & 7;
                for (int64_t i = 0; i < blength; i++) {
                    if (*pBooleanMask++) {
                        *pVOut++ = *pVIn;
                    }
                    pVIn++;
                }
            }
            break;
            case 2:
            {
                int16_t* pVOut = (int16_t*)pValuesOut;
                int16_t* pVIn = (int16_t*)pValuesIn;

                for (int64_t i = 0; i < blength; i++) {

                    uint64_t bitmask = *(uint64_t*)pData;
                    uint64_t mask = 0xff;
                    // little endian, so the first value is low bit (not high bit)
                    if (bitmask != 0) {
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;
                    }
                    else {
                        pVIn += 8;
                    }
                    pData++;
                }

                // Get last
                pBooleanMask = (int8_t*)pData;

                blength = length & 7;
                for (int64_t i = 0; i < blength; i++) {
                    if (*pBooleanMask++) {
                        *pVOut++ = *pVIn;
                    }
                    pVIn++;
                }
            }
            break;
            case 4:
            {
                int32_t* pVOut = (int32_t*)pValuesOut;
                int32_t* pVIn = (int32_t*)pValuesIn;

                for (int64_t i = 0; i < blength; i++) {

                    // little endian, so the first value is low bit (not high bit)
                    uint64_t bitmask = *(uint64_t*)pData;
                    uint64_t mask = 0xff;
                    if (bitmask != 0) {
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;
                    }
                    else {
                        pVIn += 8;
                    }
                    pData++;
                }

                // Get last
                pBooleanMask = (int8_t*)pData;

                blength = length & 7;
                for (int64_t i = 0; i < blength; i++) {
                    if (*pBooleanMask++) {
                        *pVOut++ = *pVIn;
                    }
                    pVIn++;
                }
            }
            break;
            case 8:
            {
                int64_t* pVOut = (int64_t*)pValuesOut;
                int64_t* pVIn = (int64_t*)pValuesIn;

                for (int64_t i = 0; i < blength; i++) {

                    // little endian, so the first value is low bit (not high bit)
                    uint64_t bitmask = *(uint64_t*)pData;
                    uint64_t mask = 0xff;
                    if (bitmask != 0) {
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;  mask <<= 8;
                        if (bitmask & mask) { *pVOut++ = *pVIn; } pVIn++;
                    }
                    else {
                        pVIn += 8;
                    }
                    pData++;
                }

                // Get last
                pBooleanMask = (int8_t*)pData;

                blength = length & 7;
                for (int64_t i = 0; i < blength; i++) {
                    if (*pBooleanMask++) {
                        *pVOut++ = *pVIn;
                    }
                    pVIn++;
                }
            }
            break;

            default:
            {
                for (int64_t i = 0; i < blength; i++) {

                    // little endian, so the first value is low bit (not high bit)
                    uint64_t bitmask = *(uint64_t*)pData;
                    uint64_t mask = 0xff;
                    if (bitmask != 0) {
                        int counter = 8;
                        while (counter--) {
                            if (bitmask & mask) {
                                memcpy(pValuesOut, pValuesIn, itemSize);
                                pValuesOut += itemSize;
                            }
                            pValuesIn += itemSize;
                            bitmask >>= 8;
                        }
                    }
                    else {
                        pValuesIn += (itemSize * 8);
                    }
                    pData++;
                }

                // Get last
                pBooleanMask = (int8_t*)pData;

                blength = length & 7;
                for (int64_t i = 0; i < blength; i++) {
                    if (*pBooleanMask++) {
                        memcpy(pValuesOut, pValuesIn, itemSize);
                        pValuesOut += itemSize;
                    }
                    pValuesIn += itemSize;
                }
            }
            break;
            }

            return TRUE;
        };

        BICallbackStruct stBICallback;
        stBICallback.pChunkCount = pChunkCount;
        stBICallback.pBooleanMask = pBooleanMask;
        stBICallback.pValuesIn = (char*)PyArray_BYTES(aValues);
        stBICallback.pValuesOut = (char*)PyArray_BYTES(pReturnArray);
        stBICallback.itemSize = PyArray_ITEMSIZE(aValues);

        THREADER->DoMultiThreadedChunkWork(lengthBool, lambdaBICallback2, &stBICallback);
    }

    WORKSPACE_FREE(pChunkCount);
    return (PyObject*)pReturnArray;
}




typedef void(*MBGET_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t valSize, int64_t itemSize, int64_t start, int64_t len, void* pDefault);


//-------------------------------------------------------------------
// T = data type as input type for values
// U = data type as index type
// thus <float, int32> converts a float to an int32
template<typename T, typename U>
class MergeBase {
public:
    MergeBase() {};
    ~MergeBase() {};

    //----------------------------------------------------
    // In parallel mode aValues DOES NOT change
    // aValues  : remains constant
    // aIndex   : incremented each call
    // aDataOut : incremented each call
    // start    : incremented each call
    static void GetItemInt(void* aValues, void* aIndex, void* aDataOut, int64_t valSize, int64_t itemSize, int64_t start, int64_t len, void* pDefault) {
        const T* pValues = (T*)aValues;
        const U* pIndex = (U*)aIndex;
        T* pDataOut = (T*)aDataOut;
        T  defaultVal = *(T*)pDefault;

        LOGGING("mbget sizes %lld  start:%lld  len: %lld   def: %lld  or  %lf\n", valSize, start, len, (int64_t)defaultVal, (double)defaultVal);
        LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

        for (int64_t i = 0; i < len; i++)
        {
            const auto index = pIndex[i];
            pDataOut[i] =
                // Make sure the item is in range; if the index is negative -- but otherwise
                // still in range -- mimic Python's negative-indexing support.
                index >= -valSize && index < valSize
                ? pValues[index >= 0 ? index : index + valSize]

                // Index is out of range -- assign the invalid value.
                : defaultVal;
        }
    }

    //----------------------------------------------------------
    //
    static void GetItemIntU(void* aValues, void* aIndex, void* aDataOut, int64_t valSizeX, int64_t itemSize, int64_t start, int64_t len, void* pDefault) {
        const T* pValues = (T*)aValues;
        const U* pIndex = (U*)aIndex;
        T* pDataOut = (T*)aDataOut;
        T  defaultVal = *(T*)pDefault;

        uint64_t valSize = (uint64_t)valSizeX;

        LOGGING("mbgetu sizes %lld  start:%lld  len: %lld   def: %lld  or  %lf\n", valSize, start, len, (int64_t)defaultVal, (double)defaultVal);
        LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

        for (int64_t i = 0; i < len; i++) {
            const auto index = pIndex[i];
            pDataOut[i] =
                // Make sure the item is in range
                index >= 0 && index < valSize
                ? pValues[index]
                : defaultVal;
        }
    }


    //----------------------------------------------------------
    //
    static void GetItemIntF(void* aValues, void* aIndex, void* aDataOut, int64_t valSizeX, int64_t itemSize, int64_t start, int64_t len, void* pDefault) {
        const T* pValues = (T*)aValues;
        const U* pIndex = (U*)aIndex;
        T* pDataOut = (T*)aDataOut;
        T  defaultVal = *(T*)pDefault;

        uint64_t valSize = (uint64_t)valSizeX;

        LOGGING("mbgetf sizes %lld  start:%lld  len: %lld   def: %lld  or  %lf\n", valSize, start, len, (int64_t)defaultVal, (double)defaultVal);
        LOGGING("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

        for (int64_t i = 0; i < len; i++) {
            // Make sure float is not fractional value
            const auto index = (int64_t)pIndex[i];

            pDataOut[i] =
                // Make sure the item is in range
                (U)index == pIndex[i] && index >= 0 && index < (int64_t)valSize
                ? pValues[index]
                : defaultVal;
        }
    }


    //----------------------------------------------------
    // In parallel mode aValues DOES NOT change
    // aValues  : remains constant
    // aIndex   : incremented each call
    // aDataOut : incremented each call
    // start    : incremented each call
    static void GetItemString(void* aValues, void* aIndex, void* aDataOut, int64_t valSize, int64_t itemSize, int64_t start, int64_t len, void* pDefault) {
        const char* pValues = (char*)aValues;
        const U* pIndex = (U*)aIndex;
        char* pDataOut = (char*)aDataOut;
        char* defaultVal = (char*)pDefault;

        LOGGING("mbget string sizes %lld  %lld %lld   \n", valSize, len, itemSize);
        //printf("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

        for (int64_t i = 0; i < len; i++) {
            const U index = pIndex[i];
            char* const dest = pDataOut + (i * itemSize);

            // Make sure the item is in range
            if (index >= -valSize && index < valSize) {
                // Handle Python-style negative indexing.
                const int64_t newIndex = index >= 0 ? index : index + valSize;
                const char* const src = pValues + (newIndex * itemSize);

                for (int j = 0; j < itemSize; j++) {
                    dest[j] = src[j];
                }
            }
            else {
                // This is an out-of-bounds index -- set the result to the default value,
                // which for string types is all NUL (0x00) characters.
                // TODO: Consider using memset here instead -- need to benchmark before changing.
                for (int j = 0; j < itemSize; j++) {
                    dest[j] = 0;
                }
            }
        }
    }


    //----------------------------------------------------
    // In parallel mode aValues DOES NOT change
    // aValues  : remains constant
    // aIndex   : incremented each call
    // aDataOut : incremented each call
    // start    : incremented each call
    static void GetItemStringU(void* aValues, void* aIndex, void* aDataOut, int64_t valSizeX, int64_t itemSize, int64_t start, int64_t len, void* pDefault) {
        const char* pValues = (char*)aValues;
        const U* pIndex = (U*)aIndex;
        char* pDataOut = (char*)aDataOut;
        char* defaultVal = (char*)pDefault;

        uint64_t valSize = (uint64_t)valSizeX;

        LOGGING("mbgetu string sizes %lld  %lld %lld   \n", valSize, len, itemSize);
        //printf("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

        for (int64_t i = 0; i < len; i++) {
            U index = pIndex[i];
            char* const dest = pDataOut + (i * itemSize);

            // Make sure the item is in range
            if (index >= 0 && index < valSize) {
                const char* const src = pValues + (index * itemSize);

                for (int j = 0; j < itemSize; j++) {
                    dest[j] = src[j];
                }
            }
            else {
                // This is an out-of-bounds index -- set the result to the default value,
                // which for string types is all NUL (0x00) characters.
                // TODO: Consider using memset here instead -- need to benchmark before changing.
                for (int j = 0; j < itemSize; j++) {
                    dest[j] = 0;
                }
            }

        }
    }


    static void GetItemStringF(void* aValues, void* aIndex, void* aDataOut, int64_t valSizeX, int64_t itemSize, int64_t start, int64_t len, void* pDefault) {
        const char* pValues = (char*)aValues;
        const U* pIndex = (U*)aIndex;
        char* pDataOut = (char*)aDataOut;
        char* defaultVal = (char*)pDefault;

        uint64_t valSize = (uint64_t)valSizeX;

        LOGGING("mbgetf string sizes %lld  %lld %lld   \n", valSize, len, itemSize);
        //printf("**V %p    I %p    O  %p %llu \n", pValues, pIndex, pDataOut, valSize);

        for (int64_t i = 0; i < len; i++) {
            const int64_t index = (int64_t)pIndex[i];
            char* const dest = pDataOut + (i * itemSize);

            // Make sure the item is in range
            if ((U)index == pIndex[i] && index >= 0 && index < (int64_t)valSize) {
                const char* const src = pValues + (index * itemSize);

                for (int j = 0; j < itemSize; j++) {
                    dest[j] = src[j];
                }
            }
            else {
                // This is an out-of-bounds index -- set the result to the default value,
                // which for string types is all NUL (0x00) characters.
                // TODO: Consider using memset here instead -- need to benchmark before changing.
                for (int j = 0; j < itemSize; j++) {
                    dest[j] = 0;
                }
            }

        }
    }


};


//------------------------------------------------------------
// inputType is Values type
// inputType2 is Index type
static MBGET_FUNC GetConversionFunction(int inputType, int inputType2, int func) {

    switch (inputType2) {
    case NPY_INT8:
        switch (inputType) {
        case NPY_FLOAT:  return MergeBase<float, int8_t>::GetItemInt;
        case NPY_DOUBLE: return MergeBase<double, int8_t>::GetItemInt;
        case NPY_LONGDOUBLE: return MergeBase<long double, int8_t>::GetItemInt;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, int8_t>::GetItemInt;
        case NPY_INT16:  return MergeBase<int16_t, int8_t>::GetItemInt;
        CASE_NPY_INT32:  return MergeBase<int32_t, int8_t>::GetItemInt;
        CASE_NPY_INT64:  return MergeBase<int64_t, int8_t>::GetItemInt;
        case NPY_UINT8:  return MergeBase<uint8_t, int8_t>::GetItemInt;
        case NPY_UINT16: return MergeBase<uint16_t, int8_t>::GetItemInt;
        CASE_NPY_UINT32: return MergeBase<uint32_t, int8_t>::GetItemInt;
        CASE_NPY_UINT64: return MergeBase<uint64_t, int8_t>::GetItemInt;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, int8_t>::GetItemString;
        }
        break;

    case NPY_INT16:
        switch (inputType) {
        case NPY_FLOAT:  return MergeBase<float, int16_t>::GetItemInt;
        case NPY_DOUBLE: return MergeBase<double, int16_t>::GetItemInt;
        case NPY_LONGDOUBLE: return MergeBase<long double, int16_t>::GetItemInt;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, int16_t>::GetItemInt;
        case NPY_INT16:  return MergeBase<int16_t, int16_t>::GetItemInt;
        CASE_NPY_INT32:  return MergeBase<int32_t, int16_t>::GetItemInt;
        CASE_NPY_INT64:  return MergeBase<int64_t, int16_t>::GetItemInt;
        case NPY_UINT8:  return MergeBase<uint8_t, int16_t>::GetItemInt;
        case NPY_UINT16: return MergeBase<uint16_t, int16_t>::GetItemInt;
        CASE_NPY_UINT32: return MergeBase<uint32_t, int16_t>::GetItemInt;
        CASE_NPY_UINT64: return MergeBase<uint64_t, int16_t>::GetItemInt;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, int16_t>::GetItemString;
        }
        break;

    CASE_NPY_INT32:
        switch (inputType) {
        case NPY_FLOAT:  return MergeBase<float, int32_t>::GetItemInt;
        case NPY_DOUBLE: return MergeBase<double, int32_t>::GetItemInt;
        case NPY_LONGDOUBLE: return MergeBase<long double, int32_t>::GetItemInt;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, int32_t>::GetItemInt;
        case NPY_INT16:  return MergeBase<int16_t, int32_t>::GetItemInt;
        CASE_NPY_INT32:  return MergeBase<int32_t, int32_t>::GetItemInt;
        CASE_NPY_INT64:  return MergeBase<int64_t, int32_t>::GetItemInt;
        case NPY_UINT8:  return MergeBase<uint8_t, int32_t>::GetItemInt;
        case NPY_UINT16: return MergeBase<uint16_t, int32_t>::GetItemInt;
        CASE_NPY_UINT32: return MergeBase<uint32_t, int32_t>::GetItemInt;
        CASE_NPY_UINT64: return MergeBase<uint64_t, int32_t>::GetItemInt;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, int32_t>::GetItemString;

        }
        break;

    CASE_NPY_INT64:
        switch (inputType) {
            //case NPY_BOOL:   return MergeBase<bool, bool>::GetItemInt;
        case NPY_FLOAT:  return MergeBase<float, int64_t>::GetItemInt;
        case NPY_DOUBLE: return MergeBase<double, int64_t>::GetItemInt;
        case NPY_LONGDOUBLE: return MergeBase<long double, int64_t>::GetItemInt;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, int64_t>::GetItemInt;
        case NPY_INT16:  return MergeBase<int16_t, int64_t>::GetItemInt;
        CASE_NPY_INT32:  return MergeBase<int32_t, int64_t>::GetItemInt;
        CASE_NPY_INT64:  return MergeBase<int64_t, int64_t>::GetItemInt;
        case NPY_UINT8:  return MergeBase<uint8_t, int64_t>::GetItemInt;
        case NPY_UINT16: return MergeBase<uint16_t, int64_t>::GetItemInt;
        CASE_NPY_UINT32: return MergeBase<uint32_t, int64_t>::GetItemInt;
        CASE_NPY_UINT64: return MergeBase<uint64_t, int64_t>::GetItemInt;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, int64_t>::GetItemString;
        }
        break;


    case NPY_UINT8:
        switch (inputType) {
            //case NPY_BOOL:   return MergeBase<bool, bool>::GetConversionFunction(inputType, func);
        case NPY_FLOAT:  return MergeBase<float, uint8_t>::GetItemIntU;
        case NPY_DOUBLE: return MergeBase<double, uint8_t>::GetItemIntU;
        case NPY_LONGDOUBLE: return MergeBase<long double, uint8_t>::GetItemIntU;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, uint8_t>::GetItemIntU;
        case NPY_INT16:  return MergeBase<int16_t, uint8_t>::GetItemIntU;
        CASE_NPY_INT32:  return MergeBase<int32_t, uint8_t>::GetItemIntU;
        CASE_NPY_INT64:  return MergeBase<int64_t, uint8_t>::GetItemIntU;
        case NPY_UINT8:  return MergeBase<uint8_t, uint8_t>::GetItemIntU;
        case NPY_UINT16: return MergeBase<uint16_t, uint8_t>::GetItemIntU;
        CASE_NPY_UINT32: return MergeBase<uint32_t, uint8_t>::GetItemIntU;
        CASE_NPY_UINT64: return MergeBase<uint64_t, uint8_t>::GetItemIntU;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, uint8_t>::GetItemStringU;
        }
        break;

    case NPY_UINT16:
        switch (inputType) {
            //case NPY_BOOL:   return MergeBase<bool, bool>::GetItemInt;
        case NPY_FLOAT:  return MergeBase<float, uint16_t>::GetItemIntU;
        case NPY_DOUBLE: return MergeBase<double, uint16_t>::GetItemIntU;
        case NPY_LONGDOUBLE: return MergeBase<long double, uint16_t>::GetItemIntU;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, uint16_t>::GetItemIntU;
        case NPY_INT16:  return MergeBase<int16_t, uint16_t>::GetItemIntU;
        CASE_NPY_INT32:  return MergeBase<int32_t, uint16_t>::GetItemIntU;
        CASE_NPY_INT64:  return MergeBase<int64_t, uint16_t>::GetItemIntU;
        case NPY_UINT8:  return MergeBase<uint8_t, uint16_t>::GetItemIntU;
        case NPY_UINT16: return MergeBase<uint16_t, uint16_t>::GetItemIntU;
        CASE_NPY_UINT32: return MergeBase<uint32_t, uint16_t>::GetItemIntU;
        CASE_NPY_UINT64: return MergeBase<uint64_t, uint16_t>::GetItemIntU;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, uint16_t>::GetItemStringU;
        }
        break;

    CASE_NPY_UINT32:
        switch (inputType) {
        case NPY_FLOAT:  return MergeBase<float, uint32_t>::GetItemIntU;
        case NPY_DOUBLE: return MergeBase<double, uint32_t>::GetItemIntU;
        case NPY_LONGDOUBLE: return MergeBase<long double, uint32_t>::GetItemIntU;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, uint32_t>::GetItemIntU;
        case NPY_INT16:  return MergeBase<int16_t, uint32_t>::GetItemIntU;
        CASE_NPY_INT32:  return MergeBase<int32_t, uint32_t>::GetItemIntU;
        CASE_NPY_INT64:  return MergeBase<int64_t, uint32_t>::GetItemIntU;
        case NPY_UINT8:  return MergeBase<uint8_t, uint32_t>::GetItemIntU;
        case NPY_UINT16: return MergeBase<uint16_t, uint32_t>::GetItemIntU;
        CASE_NPY_UINT32: return MergeBase<uint32_t, uint32_t>::GetItemIntU;
        CASE_NPY_UINT64: return MergeBase<uint64_t, uint32_t>::GetItemIntU;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, uint32_t>::GetItemStringU;
        }
        break;

    CASE_NPY_UINT64:
        switch (inputType) {
            //case NPY_BOOL:   return MergeBase<bool, bool>::GetItemInt;
        case NPY_FLOAT:  return MergeBase<float, uint64_t>::GetItemIntU;
        case NPY_DOUBLE: return MergeBase<double, uint64_t>::GetItemIntU;
        case NPY_LONGDOUBLE: return MergeBase<long double, uint64_t>::GetItemIntU;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, uint64_t>::GetItemIntU;
        case NPY_INT16:  return MergeBase<int16_t, uint64_t>::GetItemIntU;
        CASE_NPY_INT32:  return MergeBase<int32_t, uint64_t>::GetItemIntU;
        CASE_NPY_INT64:  return MergeBase<int64_t, uint64_t>::GetItemIntU;
        case NPY_UINT8:  return MergeBase<uint8_t, uint64_t>::GetItemIntU;
        case NPY_UINT16: return MergeBase<uint16_t, uint64_t>::GetItemIntU;
        CASE_NPY_UINT32: return MergeBase<uint32_t, uint64_t>::GetItemIntU;
        CASE_NPY_UINT64: return MergeBase<uint64_t, uint64_t>::GetItemIntU;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, uint64_t>::GetItemStringU;
        }
        break;


    case NPY_FLOAT32:
        switch (inputType) {
            //case NPY_BOOL:   return MergeBase<bool, bool>::GetItemInt;
        case NPY_FLOAT:  return MergeBase<float, float>::GetItemIntF;
        case NPY_DOUBLE: return MergeBase<double, float>::GetItemIntF;
        case NPY_LONGDOUBLE: return MergeBase<long double, float>::GetItemIntF;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, float>::GetItemIntF;
        case NPY_INT16:  return MergeBase<int16_t, float>::GetItemIntF;
        CASE_NPY_INT32:  return MergeBase<int32_t, float>::GetItemIntF;
        CASE_NPY_INT64:  return MergeBase<int64_t, float>::GetItemIntF;
        case NPY_UINT8:  return MergeBase<uint8_t, float>::GetItemIntF;
        case NPY_UINT16: return MergeBase<uint16_t, float>::GetItemIntF;
        CASE_NPY_UINT32: return MergeBase<uint32_t, float>::GetItemIntF;
        CASE_NPY_UINT64: return MergeBase<uint64_t, float>::GetItemIntF;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, float>::GetItemStringF;

        }
        break;

    case NPY_FLOAT64:
        switch (inputType) {
            //case NPY_BOOL:   return MergeBase<bool, bool>::GetItemInt;
        case NPY_FLOAT:  return MergeBase<float, uint64_t>::GetItemIntF;
        case NPY_DOUBLE: return MergeBase<double, uint64_t>::GetItemIntF;
        case NPY_LONGDOUBLE: return MergeBase<long double, uint64_t>::GetItemIntF;
        case NPY_BOOL:
        case NPY_BYTE:   return MergeBase<int8_t, uint64_t>::GetItemIntF;
        case NPY_INT16:  return MergeBase<int16_t, uint64_t>::GetItemIntF;
        CASE_NPY_INT32:  return MergeBase<int32_t, uint64_t>::GetItemIntF;
        CASE_NPY_INT64:  return MergeBase<int64_t, uint64_t>::GetItemIntF;
        case NPY_UINT8:  return MergeBase<uint8_t, uint64_t>::GetItemIntF;
        case NPY_UINT16: return MergeBase<uint16_t, uint64_t>::GetItemIntF;
        CASE_NPY_UINT32: return MergeBase<uint32_t, uint64_t>::GetItemIntF;
        CASE_NPY_UINT64: return MergeBase<uint64_t, uint64_t>::GetItemIntF;
        case NPY_VOID:
        case NPY_UNICODE:
        case NPY_STRING: return MergeBase<char*, uint64_t>::GetItemStringF;
        }
        break;

    }
    printf("mbget cannot find type for %d\n", inputType);
    return NULL;
}



struct MBGET_CALLBACK {
    MBGET_FUNC GetItemCallback;

    void* pValues;
    void* pIndex;
    void* pDataOut;
    int64_t    valSize1;
    int64_t    aIndexSize;
    void* pDefault;
    int64_t    TypeSizeValues;
    int64_t    TypeSizeIndex;

} stMBGCallback;


//---------------------------------------------------------
// Used by GetItem
//  Concurrent callback from multiple threads
static int64_t AnyGetItem(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {

    int64_t didSomeWork = 0;
    MBGET_CALLBACK* Callback = &stMBGCallback; // (MBGET_CALLBACK*)&pstWorkerItem->WorkCallbackArg;

    char* aValues = (char*)Callback->pValues;
    char* aIndex = (char*)Callback->pIndex;

    int64_t typeSizeValues = Callback->TypeSizeValues;
    int64_t typeSizeIndex = Callback->TypeSizeIndex;

    LOGGING("check2 ** %lld %lld\n", typeSizeValues, typeSizeIndex);

    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        // Do NOT move aValues
        // Move aIndex
        // Move pDataOut (same type as Values)
        // move starting position

        // Calculate how much to adjust the pointers to get to the data for this work block
        int64_t blockStart = workBlock * pstWorkerItem->BlockSize;

        int64_t valueAdj = blockStart * typeSizeValues;
        int64_t indexAdj = blockStart * typeSizeIndex;

        LOGGING("%d : workBlock %lld   blocksize: %lld    lenx: %lld  %lld  %lld  %lld %lld\n", core, workBlock, pstWorkerItem->BlockSize, lenX, typeSizeValues, typeSizeIndex, valueAdj, indexAdj);

        Callback->GetItemCallback(aValues, aIndex + indexAdj, (char*)Callback->pDataOut + valueAdj, Callback->valSize1, typeSizeValues, blockStart, lenX, Callback->pDefault);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock(core);
    }

    return didSomeWork;
}





//---------------------------------------------------------------------------
// Input:
// Arg1: numpy array aValues (can be anything)
// Arg2: numpy array aIndex (must be int8/int16/int32 or int64)
// Arg3: default value
//
//def fixMbget(aValues, aIndex, result, default) :
//   """
//   A numba routine to speed up mbget for numerical values.
//   """
//   N = aIndex.shape[0]
//   valSize = aValues.shape[0]
//   for i in range(N) :
//      if (aIndex[i] >= 0 and aIndex[i] < valSize) :
//         result[i] = aValues[aIndex[i]]
//      else :
//   result[i] = default
extern "C" PyObject*
getitem(PyObject* self, PyObject* args)
{
    PyArrayObject* aValues = NULL;
    PyArrayObject* aIndex = NULL;
    PyObject* defaultValue = NULL;

    if (PyTuple_Size(args) == 2) {
        if (!PyArg_ParseTuple(
            args, "O!O!:getitem",
            pPyArray_Type, &aValues,
            pPyArray_Type, &aIndex
        )) {

            return NULL;
        }
        defaultValue = Py_None;

    }
    else
        if (!PyArg_ParseTuple(
            args, "O!O!O:getitem",
            pPyArray_Type, &aValues,
            pPyArray_Type, &aIndex,
            &defaultValue)) {

            return NULL;
        }

    int32_t numpyValuesType = PyArray_TYPE(aValues);
    int32_t numpyIndexType = PyArray_TYPE(aIndex);

    if (numpyIndexType > NPY_LONGDOUBLE) {
        PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d using index dtype: %d", numpyValuesType, numpyIndexType);
        return NULL;
    }

    //printf("numpy types %d %d\n", numpyValuesType, numpyIndexType);

    void* pValues = PyArray_BYTES(aValues);
    void* pIndex = PyArray_BYTES(aIndex);

    int ndim = PyArray_NDIM(aValues);
    npy_intp* dims = PyArray_DIMS(aValues);
    int64_t valSize1 = CalcArrayLength(ndim, dims);
    int64_t len = valSize1;

    MBGET_FUNC  pFunction = GetConversionFunction(numpyValuesType, numpyIndexType, 0);

    if (pFunction != NULL) {

        PyArrayObject* outArray = (PyArrayObject*)Py_None;
        int64_t aIndexSize = ArrayLength(aIndex);

        // Allocate the size of aIndex but the type is the value
        outArray = AllocateLikeResize(aValues, aIndexSize);

        if (outArray) {
            void* pDataOut = PyArray_BYTES(outArray);
            void* pDefault = GetDefaultForType(numpyValuesType);

            // reserve a full 16 bytes for default in case we have oneS
            _m256all tempDefault;

            // Check if a default value was passed in as third parameter
            if (defaultValue != Py_None) {
                // TJD disabled this on Dec. 2020
                //BOOL result;
                //int64_t itemSize;
                //void* pTempData = NULL;

                //// Try to convert the scalar
                //result = ConvertScalarObject(defaultValue, &tempDefault, numpyValuesType, &pTempData, &itemSize);

                //if (result) {
                //    // Assign the new default for out of range indexes
                //    pDefault = &tempDefault;
                //}
                pDefault = &tempDefault;
            }

            stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(aIndexSize);

            if (pWorkItem == NULL) {

                // Threading not allowed for this work item, call it directly from main thread
                pFunction(pValues, pIndex, pDataOut, valSize1, PyArray_ITEMSIZE(aValues), 0, aIndexSize, pDefault);

            }
            else {
                // Each thread will call this routine with the callbackArg
                // typedef int64_t(*DOWORK_CALLBACK)(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex);
                pWorkItem->DoWorkCallback = AnyGetItem;

                pWorkItem->WorkCallbackArg = &stMBGCallback;

                stMBGCallback.GetItemCallback = pFunction;
                stMBGCallback.pValues = pValues;
                stMBGCallback.pIndex = pIndex;
                stMBGCallback.pDataOut = pDataOut;

                // arraylength of values input array -- used to check array bounds
                stMBGCallback.valSize1 = valSize1;
                stMBGCallback.aIndexSize = aIndexSize;
                stMBGCallback.pDefault = pDefault;

                //
                stMBGCallback.TypeSizeValues = PyArray_ITEMSIZE(aValues);
                stMBGCallback.TypeSizeIndex = PyArray_ITEMSIZE(aIndex);

                //printf("**check %p %p %p %lld %lld\n", pValues, pIndex, pDataOut, stMBGCallback.TypeSizeValues, stMBGCallback.TypeSizeIndex);

                // This will notify the worker threads of a new work item
                THREADER->WorkMain(pWorkItem, aIndexSize, 0);
                //g_cMathWorker->WorkMain(pWorkItem, aIndexSize);
            }

            return (PyObject*)outArray;
        }
        PyErr_Format(PyExc_ValueError, "GetItem ran out of memory %d %d", numpyValuesType, numpyIndexType);
        return NULL;

    }

    PyErr_Format(PyExc_ValueError, "Dont know how to convert these types %d %d", numpyValuesType, numpyIndexType);
    return NULL;
}


