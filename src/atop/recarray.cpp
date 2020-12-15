#include "common_inc.h"
#include <cmath>
#include "invalids.h"
#include "threads.h"

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

static const int64_t CHUNKSIZE = 16384;

// This is used to multiply the strides
const union
{
    INT32 i[8];
    __m256i m;
    //} __vindex8_strides = { 7, 6, 5, 4, 3, 2, 1, 0 };
} __vindex8_strides = { 0, 1, 2, 3, 4, 5, 6, 7 };

//-----------------------------------
//
void ConvertRecArray(char* pStartOffset, int64_t startRow, int64_t totalRows, stRecarrayOffsets* pstOffset, int64_t numArrays, int64_t itemSize)
{
    // Try to keep everything in L1Cache
    const int64_t L1CACHE = 32768;
    int64_t CHUNKROWS = L1CACHE / (itemSize * 2);
    if (CHUNKROWS < 1) {
        CHUNKROWS = 1;
    }

    __m256i vindex = _mm256_mullo_epi32(_mm256_set1_epi32((INT32)itemSize), _mm256_loadu_si256(&__vindex8_strides.m));
    __m128i vindex128 = _mm256_extracti128_si256(vindex, 0);

    while (startRow < totalRows) {

        // Calc how many rows to process in this pass
        int64_t endRow = startRow + CHUNKROWS;
        if (endRow > totalRows) {
            endRow = totalRows;
        }

        int64_t origRow = startRow;

        //printf("processing %lld\n", startRow);
        for (int64_t i = 0; i < numArrays; i++) {

            startRow = origRow;

            // Calculate place to read
            char* pRead = pStartOffset + pstOffset[i].readoffset;
            char* pWrite = pstOffset[i].pData;

            int64_t arrItemSize = pstOffset[i].itemsize;

            //printf("processing  start:%lld  end:%lld   pRead:%p  %p  itemsize: %lld\n", startRow, endRow, pRead, pWrite, arrItemSize);

            switch (pstOffset[i].itemsize) {
            case 1:
                while (startRow < endRow) {
                    INT8 data = *(INT8*)(pRead + (startRow * itemSize));
                    *(INT8*)(pWrite + startRow) = data;
                    startRow++;
                }
                break;
            case 2:
                while (startRow < endRow) {
                    INT16 data = *(INT16*)(pRead + (startRow * itemSize));
                    *(INT16*)(pWrite + startRow * arrItemSize) = data;
                    startRow++;
                }
                break;
            case 4:
                // ??? use _mm256_i32gather_epi32 to speed up
            {
                int64_t endSubRow = endRow - 8;
                while (startRow < endSubRow) {
                    __m256i m0 = _mm256_i32gather_epi32((INT32*)(pRead + (startRow * itemSize)), vindex, 1);
                    _mm256_storeu_si256((__m256i*)(pWrite + (startRow * arrItemSize)), m0);
                    startRow += 8;
                }
                while (startRow < endRow) {
                    INT32 data = *(INT32*)(pRead + (startRow * itemSize));
                    *(INT32*)(pWrite + startRow * arrItemSize) = data;
                    startRow++;
                }
            }
            break;
            case 8:
            {
                int64_t endSubRow = endRow - 4;
                while (startRow < endSubRow) {
                    __m256i m0 = _mm256_i32gather_epi64((int64_t*)(pRead + (startRow * itemSize)), vindex128, 1);
                    _mm256_storeu_si256((__m256i*)(pWrite + (startRow * arrItemSize)), m0);
                    startRow += 4;
                }
                while (startRow < endRow) {
                    int64_t data = *(int64_t*)(pRead + (startRow * itemSize));
                    *(int64_t*)(pWrite + startRow * arrItemSize) = data;
                    startRow++;
                }
            }
            break;
            default:
                while (startRow < endRow) {
                    char* pSrc = pRead + (startRow * itemSize);
                    char* pDest = pWrite + (startRow * arrItemSize);
                    char* pEnd = pSrc + arrItemSize;
                    while ((pSrc + 8) < pEnd) {
                        *(int64_t*)pDest = *(int64_t*)pSrc;
                        pDest += 8;
                        pSrc += 8;
                    }
                    while (pSrc < pEnd) {
                        *pDest++ = *pSrc++;
                    }
                    startRow++;
                }
                break;

            }

        }
    }
}


//==============================================
// totalRows = total number of record array rows
// 
extern "C" void RecArrayToColMajor(
    stRecarrayOffsets* pstOffset,
    char* pStartOffset,
    int64_t totalRows,
    int64_t numArrays,
    int64_t itemSize) {

    static const int64_t CHUNKSIZE = 16384;

    // Try to keep everything in L1Cache
    const int64_t L1CACHE = 32768;
    int64_t CHUNKROWS = L1CACHE / (itemSize * 2);
    if (CHUNKROWS < 1) {
        CHUNKROWS = 1;
    }

    LOGGING("Chunkrows is %I64d \n", CHUNKROWS);

    int64_t startRow = 0;

    if (THREADER && totalRows > 16384) {
        // Prepare for multithreading
        struct stConvertRec {
            char* pStartOffset;
            int64_t startRow;
            int64_t totalRows;
            stRecarrayOffsets* pstOffset;
            int64_t numArrays;
            int64_t itemSize;
            int64_t lastRow;
        } stConvert;

        int64_t items = (totalRows + (CHUNKSIZE - 1)) / CHUNKSIZE;

        stConvert.pStartOffset = pStartOffset;
        stConvert.startRow = startRow;
        stConvert.totalRows = totalRows;
        stConvert.pstOffset = pstOffset;
        stConvert.numArrays = numArrays;
        stConvert.itemSize = itemSize;
        stConvert.lastRow = items - 1;

        auto lambdaConvertRecCallback = [](void* callbackArgT, int core, int64_t workIndex) -> BOOL {
            stConvertRec* callbackArg = (stConvertRec*)callbackArgT;
            int64_t startRow = callbackArg->startRow + (workIndex * CHUNKSIZE);
            int64_t totalRows = startRow + CHUNKSIZE;

            if (totalRows > callbackArg->totalRows) {
                totalRows = callbackArg->totalRows;
            }

            ConvertRecArray(
                callbackArg->pStartOffset,
                startRow,
                totalRows,
                callbackArg->pstOffset,
                callbackArg->numArrays,
                callbackArg->itemSize);

            LOGGING("[%d] %lld completed\n", core, workIndex);
            return TRUE;
        };

        THREADER->DoMultiThreadedWork((int)items, lambdaConvertRecCallback, &stConvert);

    }
    else {
        ConvertRecArray(pStartOffset, startRow, totalRows, pstOffset, numArrays, itemSize);
    }
}

