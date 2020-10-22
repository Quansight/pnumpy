#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include <stdint.h>
#include <stdio.h>
#include "../atop/atop.h"
#include "../atop/threads.h"
#define LOGGING(...)


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

    PyTypeObject* const allocType =  &PyArray_Type;

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
        printf("!!!out of memory allocating numpy array size:%llu  dims:%d  dtype:%d  itemsize:%lld  flags:%d  dim0:%lld\n", len, ndim, numpyType, itemsize, array_flags, (int64_t)dims[0]);
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

        // Sum 4x 8-byte counts -> 1x 32-byte count.
        const auto byte_popcount_256 =
            _mm256_extract_epi64(byte_popcounts_64, 0)
            + _mm256_extract_epi64(byte_popcounts_64, 1)
            + _mm256_extract_epi64(byte_popcounts_64, 2)
            + _mm256_extract_epi64(byte_popcounts_64, 3);

        // Add the accumulated popcount from this loop iteration (for 32*255 bytes) to the overall result.
        result += byte_popcount_256;

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
        PyErr_Format(PyExc_ValueError, "Second argument must be boolean array");
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

                    // little endian, so the first value is low bit (not high bit)
                    uint32_t bitmask = (uint32_t)(_pext_u64(*pData, 0x0101010101010101));
                    if (bitmask != 0) {
                        if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
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

                    // little endian, so the first value is low bit (not high bit)
                    uint32_t bitmask = (uint32_t)(_pext_u64(*pData, 0x0101010101010101));
                    if (bitmask != 0) {
                        if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
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
                    uint32_t bitmask = (uint32_t)(_pext_u64(*pData, 0x0101010101010101));
                    if (bitmask != 0) {
                        if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
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
                    uint32_t bitmask = (uint32_t)(_pext_u64(*pData, 0x0101010101010101));
                    if (bitmask != 0) {
                        if (bitmask & 1) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 2) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 4) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 8) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 16) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 32) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 64) { *pVOut++ = *pVIn; } pVIn++;
                        if (bitmask & 128) { *pVOut++ = *pVIn; } pVIn++;
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
                    uint32_t bitmask = (uint32_t)(_pext_u64(*pData, 0x0101010101010101));
                    if (bitmask != 0) {
                        int counter = 8;
                        while (counter--) {
                            if (bitmask & 1) {
                                memcpy(pValuesOut, pValuesIn, itemSize);
                                pValuesOut += itemSize;
                            }

                            pValuesIn += itemSize;
                            bitmask >>= 1;
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
