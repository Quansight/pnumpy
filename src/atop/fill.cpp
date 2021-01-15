#include "common_inc.h"
#include "threads.h"
#include "halffloat.h"
#include <cmath>
#include <algorithm>

//#define LOGGING printf
#define LOGGING(...)

// NOTES
// FillZeros calls
// PyArray_AssignRawScalar
// raw_array_assign_scalar
// which then calls PyArray_GetDTypeTransferFunction
//        /* Process the innermost dimension */
//stransfer(dst_data, dst_strides_it[0], src_data, 0,
//    shape_it[0], src_itemsize, transferdata);

////================================================================================
typedef int(*ARANGE_FILL)(char* pBufferV, void* pFirstV, void* pNextValueV, int64_t start, int64_t length);

// vector code disabled for now (does not seem much faster)
//template<typename TYPE>
//static int
//ArangeFillTypeInt32(char* pBufferV, void* pFirstV, void* pNextValueV, int64_t start, int64_t length)
//{
//    //printf("int32 fill\n");
//    TYPE* pBuffer = (TYPE*)pBufferV;
//
//    // The start/next are stored by numpy in first two values of array
//    TYPE first = *(TYPE*)pFirstV;
//    TYPE delta = *(TYPE*)pNextValueV;
//
//    delta -= first;
//
//    __m256i mstart = _mm256_set_epi32(7,6,5,4,3,2,1,0);
//    __m256i madd= _mm256_set1_epi32(sizeof(__m256i)/sizeof(TYPE)); // 8
//    __m256i mdelta = _mm256_set1_epi32((int32_t)delta);
//    madd = _mm256_mullo_epi32(madd, mdelta);
//    mstart = _mm256_add_epi32(mstart, _mm256_set1_epi32((int32_t)start));
//    mstart = _mm256_mullo_epi32(mstart, mdelta);
//
//    __m256i* pDest = (__m256i*)(pBuffer + start);
//    __m256i* pDestEnd = pDest + (length - start) / 8;
//
//    while (pDest != pDestEnd) {
//        _mm256_storeu_si256(pDest, mstart);
//        mstart = _mm256_add_epi32(mstart, madd);
//        pDest++;
//    }
//
//    start = start + length - (length & 7);
//    for (int64_t i = start; i < length; i++) {
//        pBuffer[i] = (TYPE)(first + i * delta);
//    }
//
//    return 0;
//}


template<typename TYPE>
static int
ArangeFillType(char *pBufferV, void* pFirstV, void* pNextValueV, int64_t start, int64_t length)
{
    TYPE* pBuffer = (TYPE*)pBufferV;

    // The start/next are stored by numpy in first two values of array
    TYPE first = *(TYPE*)pFirstV;
    TYPE delta = *(TYPE*)pNextValueV;

    delta -= first;

    // TOOD: vectorize this code
    for (int64_t i = start; i < length; i++) {
        pBuffer[i] = (TYPE)(first + i * delta);
    }
    // Path below is slower
    //TYPE* pBufferEnd = pBuffer + length;
    //while (pBuffer < pBufferEnd) {
    //    *pBuffer++ = start;
    //    start += delta;
    //}
    return 0;
}

ARANGE_FILL g_ArangeFill[ATOP_LAST] = {
    NULL, //ArangeFillType<bool>,
    ArangeFillType<int8_t>,  ArangeFillType<uint8_t>,
    ArangeFillType<int16_t>, ArangeFillType<uint16_t>,
    ArangeFillType<int32_t>, ArangeFillType<uint32_t>,
    ArangeFillType<int64_t>, ArangeFillType<uint64_t>,
    NULL, NULL, //int128
    NULL, ArangeFillType<float>, ArangeFillType<double>, ArangeFillType<long double>,
    NULL, NULL, NULL, NULL, // Complex
    NULL, NULL, NULL       // String, unicode, void
};

extern "C" int ArangeFill(
    int   atype,
    char* pBuffer,
    void* pFirstValue,
    void* pSecondValue,
    int64_t length,
    int32_t threadwakeup) {

    ARANGE_FILL pArangeFill = g_ArangeFill[atype];

    // check if we have the routine
    if (pArangeFill) {

        // Multithreaded callback
        struct ArangeCallbackStruct {
            ARANGE_FILL pArangeFill;
            char* pBuffer;
            void* pFirstValue;
            void* pSecondValue;
            int64_t length;
        } stArangeCallback{ pArangeFill, pBuffer, pFirstValue, pSecondValue, length };

        // This is the routine that will be called back from multiple threads
        auto lambdaArangeCallback = [](void* callbackArgT, int core, int64_t start, int64_t length) -> int64_t {
            LOGGING("[%d] Arange  %lld %lld\n", core, start, length);
            ArangeCallbackStruct* cb=(ArangeCallbackStruct * )callbackArgT;
            cb->pArangeFill(cb->pBuffer, cb->pFirstValue, cb->pSecondValue, start, start + length);
            return 1;
        };

        THREADER->DoMultiThreadedChunkWork(length, lambdaArangeCallback, &stArangeCallback, threadwakeup);
        return 0;
    }
    // fail
    return -1;
}

