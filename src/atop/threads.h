#pragma once

#include "common_inc.h"

#if defined(__GNUC__)
#if __GNUC_PREREQ(4, 4) || (__clang__ > 0 && __clang_major__ >= 3) || !defined(__GNUC__)
/* GCC >= 4.4 or clang or non-GCC compilers */
#include <x86intrin.h>
#elif __GNUC_PREREQ(4, 1)
/* GCC 4.1, 4.2, and 4.3 do not have x86intrin.h, directly include SSE2 header */
#include <emmintrin.h>
#endif
#endif

#define YieldProcessor _mm_pause

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <fcntl.h>

#if defined(__linux__)

#include <linux/futex.h>

static int
futex(int* uaddr, int futex_op, int val,
    const struct timespec* timeout, int* uaddr2, int val3)
{
    return syscall(SYS_futex, uaddr, futex_op, val,
        timeout, uaddr, val3);
}

#elif defined(__APPLE__)

// temp remove warnings
// #warning MathThreads does not yet support Darwin/macOS.
extern pthread_cond_t  g_WakeupCond;

#endif  // defined(__linux__)

#endif  // defined(__unix__) || defined(__unix) || defined(__APPLE__)

#define THREADLOGGING(...)
//#define THREADLOGGING printf


//-----------------------------------------------------
// Determines the CAP on threads
#define MAX_THREADS_WHEN_CANNOT_DETECT 5

// set this value lower to help windows wake up threads
#define MAX_THREADS_ALLOWED 31

#define FUTEX_WAKE_DEFAULT  11
#define FUTEX_WAKE_MAX      31


//--------------------------------------------------------------------
//BOOL
//WINAPI
//WaitOnAddress(
//   _In_reads_bytes_(AddressSize) volatile VOID * Address,
//   _In_reads_bytes_(AddressSize) PVOID CompareAddress,
//   _In_ SIZE_T AddressSize,
//   _In_opt_ DWORD dwMilliseconds
//);
//
//
//VOID
//WINAPI
//WakeByAddressSingle(
//   _In_ PVOID Address
//);
//
//
//VOID
//WINAPI
//WakeByAddressAll(
//   _In_ PVOID Address
//);

//-------------------------------------------------------------------
//
// global scope
#define VOID void

#if RT_COMPILER_MSVC
#if !defined(_WIN64)
typedef  int INT_PTR, * PINT_PTR;
typedef  unsigned int UINT_PTR, * PUINT_PTR;

typedef  long LONG_PTR, * PLONG_PTR;
typedef  unsigned long ULONG_PTR, * PULONG_PTR;

typedef ULONG_PTR SIZE_T, * PSIZE_T;
typedef LONG_PTR SSIZE_T, * PSSIZE_T;

#else
typedef __int64 INT_PTR, * PINT_PTR;
typedef unsigned __int64 UINT_PTR, * PUINT_PTR;

typedef __int64 LONG_PTR, * PLONG_PTR;
typedef unsigned __int64 ULONG_PTR, * PULONG_PTR;

typedef ULONG_PTR SIZE_T, * PSIZE_T;
typedef LONG_PTR SSIZE_T, * PSSIZE_T;

#endif
#else
typedef int64_t INT_PTR, * PINT_PTR;
typedef uint64_t UINT_PTR, * PUINT_PTR;

typedef int64_t LONG_PTR, * PLONG_PTR;
typedef uint64_t ULONG_PTR, * PULONG_PTR;

typedef ULONG_PTR SIZE_T, * PSIZE_T;
typedef LONG_PTR SSIZE_T, * PSSIZE_T;

#endif

typedef void* PVOID;
typedef unsigned long       DWORD;
typedef ULONG_PTR DWORD_PTR, * PDWORD_PTR;
typedef DWORD* LPDWORD;

typedef VOID(WINAPI* WakeSingleAddress)(PVOID);
typedef VOID(WINAPI* WakeAllAddress)(PVOID);
typedef BOOL(WINAPI* WaitAddress)(volatile VOID*, PVOID, SIZE_T, DWORD);

extern WakeSingleAddress g_WakeSingleAddress;
extern WakeAllAddress g_WakeAllAddress;
extern WaitAddress g_WaitAddress;

// Callback routine from worker thread
typedef int64_t(*DOWORK_CALLBACK)(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex);

// Callback routine from multithreaded worker thread (items just count up from 0,1,2,...)
typedef int64_t(*MTWORK_CALLBACK)(void* callbackArg, int core, int64_t workIndex);

// Callback routine from multithreaded chunk thread (0, 65536, 130000, etc.)
typedef int64_t(*MTCHUNK_CALLBACK)(void* callbackArg, int core, int64_t start, int64_t length);

// For auto binning we need to divide bins up amongst multiple thread
struct stBinCount {
    // Valid if ... > BinLow && <= BinHigh
    int64_t BinLow;
    int64_t BinHigh;
    int64_t BinNum;
    void* pUserMemory;
};

struct OLD_CALLBACK {

    // Args to call
    union {
        VOID* pDataInBase1;
        VOID* pValues;
    };

    union {
        VOID* pDataInBase2;
        VOID* pIndex;
        VOID* pToSort;
    };

    VOID* pDataInBase3;

    //-------------------------------------------------
    union {
        VOID* pDataOutBase1;
        VOID* pWorkSpace;
    };

    // Total number of array elements
    union {
        //INT64             TotalElements;
        int64_t             IndexSize;

        // strlen is for sorting strings
        int64_t             StrLen;
    };

    union {
        int32_t             ScalarMode;
        int32_t             MergeBlocks;
    };

    union {
        int64_t             TotalElements2;
        int64_t             ValSize;
    };

    // Default value to fill
    void* pDefault;


    void* pThreadWorkSpace;


};

//-----------------------------------------------------------
//
struct stMATH_WORKER_ITEM {

    // -----------------------------------
    // Tested with addition
    // %timeit global a; a+= 5

    // a=arange(100_000)
    // operation 0x4000  0x8000   0x10000   0x20000
    // -------   ------  -------  -------   -------
    // a+=5        51      52 us     27 us   27
    // a+=5 nowait 49
    // a+b         49      50 us     46 us   46
    //sqrt:        83     104 us    209 us   209
    // sum:        54                26 us   26
    //
    // arange(1_000_000)
    // operation 0x4000  0x8000   0x10000   0x20000 
    // -------   ------  -------  -------   -------
    // a+=5        114     120 us    118 us  133
    // a+b          91     121 us    128 us   46
    //sqrt:        252     293 us    293 us  209
    // sum:         50      51        52 us   68

    // a=arange(100_000.0)
    // operation 0x4000  0x8000   0x10000   0x20000
    // -------   ------  -------  -------   -------
    // a+b         69      xx      137 us    xx
    // sqrt:       85              209
    // sum:        52      xx      30 us     xx

    // Items larger than this might be worked on in parallel
    static const int64_t WORK_ITEM_CHUNK = 0x4000;
    static const int64_t WORK_ITEM_BIG = (WORK_ITEM_CHUNK * 2);
    static const int64_t WORK_ITEM_MASK = (WORK_ITEM_CHUNK - 1);

    // The callback to the thread routine that does work
    // with the argument to pass
    DOWORK_CALLBACK   DoWorkCallback;
    void* WorkCallbackArg;

    // How many threads to wake up (atomic decrement)
    int64_t             ThreadWakeup;

    // Used when calling MultiThreadedWork
    union {
        MTWORK_CALLBACK   MTWorkCallback;
        MTCHUNK_CALLBACK  MTChunkCallback;
    };

    // TotalElements is used on asymmetric last block
    int64_t             TotalElements;

    // How many elements per block to work on
    int64_t             BlockSize;


    // The last block to work on
    volatile int64_t    BlockLast;

    //-------------------------------------------------
    // The next block (atomic)
    // Incremented
    // If BlockNext > BlockLast -- no work to be done
    volatile int64_t    BlockNext;


    //-----------------------------------------------
    // Atomic access
    // When BlocksCompleted == BlockLast , the job is completed
    int64_t             BlocksCompleted;

    OLD_CALLBACK        OldCallback;



    //==============================================================
    FORCE_INLINE int64_t GetWorkBlock() {
        int64_t val = InterlockedIncrement64(&BlockNext);
        return val - 1;
    }

    //==============================================================
    FORCE_INLINE void CompleteWorkBlock() {
        // Indicate we completed a block
        InterlockedIncrement64(&BlocksCompleted);
    }

    //=============================================================
    // Called by routines that work by index 
    // returns 0 on failure
    // else returns length of workblock
    FORCE_INLINE int64_t GetNextWorkIndex(int64_t* workBlock) {
        int64_t wBlock = *workBlock = GetWorkBlock();

        //THREADLOGGING("working on block %llu\n", wBlock);

        // Make sure something to work on
        if (wBlock < BlockLast) {
            return wBlock;
        }

        return 0;
    }

    //=============================================================
    // Called by routines that work on chunks/blocks of memory
    // returns 0 on failure
    // else returns length of workblock
    FORCE_INLINE int64_t GetNextWorkBlock(int64_t* workBlock) {

        int64_t wBlock = *workBlock = GetWorkBlock();

        //THREADLOGGING("working on block %llu\n", wBlock);

        // Make sure something to work on
        if (wBlock < BlockLast) {
            int64_t  lenWorkBlock;
            lenWorkBlock = BlockSize;

            // Check if this is the last workblock
            if ((wBlock + 1) == BlockLast) {

                // check if ends on perfect boundary
                if ((TotalElements & WORK_ITEM_MASK) != 0) {

                    // This is the last block and may have an odd number of data to process
                    lenWorkBlock = TotalElements & WORK_ITEM_MASK;
                    //THREADLOGGING("last workblock %llu  %llu  MASK  %llu\n", lenWorkBlock, TotalElements, WORK_ITEM_MASK);
                }
            }
            return lenWorkBlock;
        }
        return 0;
    }


    //------------------------------------------------------------------------------
    // Call this to do work until no work left to do
    // Returns TRUE if it did some work
    // Returns FALSE if it did no work 
    // If core is -1, it is the main thread
    FORCE_INLINE int64_t DoWork(int core, int64_t workIndex) {

        return DoWorkCallback(this, core, workIndex);
    }

};


//-----------------------------------------------------------
// allocated on 64 byte alignment
struct stWorkerRing {
    static const int64_t   RING_BUFFER_SIZE = 1024;
    static const int64_t   RING_BUFFER_MASK = 1023;

    volatile int64_t       WorkIndex;
    volatile int64_t       WorkIndexCompleted;

    // incremented when worker thread start
    volatile int64_t       WorkThread;

    // Set to TRUE if hyperthreading is on, skipping every other core
    int32_t                HyperThreading;
    int32_t                SleepTime;

    int32_t                NumaNode;
    int32_t                Cancelled;

    // Change this value to wake up less workers
    int32_t                FutexWakeCount;

    stMATH_WORKER_ITEM   WorkerQueue[RING_BUFFER_SIZE];

    void Init() {
        WorkIndex = 0;
        HyperThreading = 0;
        WorkIndexCompleted = 0;
        WorkThread = 0;
        NumaNode = 0;
        Cancelled = 0;
        SleepTime = 1;
        // how many threads to wake up on Linux
        FutexWakeCount = FUTEX_WAKE_DEFAULT;

        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            WorkerQueue[i].BlockSize = 0;
            WorkerQueue[i].BlockLast = 0;
            WorkerQueue[i].TotalElements = 0;
            WorkerQueue[i].BlockNext = 0;
            WorkerQueue[i].BlocksCompleted = 0;
        }

    }

    FORCE_INLINE void Cancel() {
        Cancelled = 1;
    }

    FORCE_INLINE stMATH_WORKER_ITEM* GetWorkItem() {
        return  &WorkerQueue[WorkIndex & RING_BUFFER_MASK];
    }

    FORCE_INLINE stMATH_WORKER_ITEM* GetExistingWorkItem() {
        return  &WorkerQueue[(WorkIndex - 1) & RING_BUFFER_MASK];
    }

    FORCE_INLINE void SetWorkItem(int32_t maxThreadsToWake) {
        // This routine will wakup threads on Windows and Linux
        // Once we increment other threads will notice
        InterlockedIncrement64(&WorkIndex);

#if defined(RT_OS_WINDOWS)
        // Are we allowed to wake threads?
        if (g_WakeAllAddress != NULL) {

            if (maxThreadsToWake < 5) {
                // In windows faster to wake single if just a few threads
                for (int i = 0; i < maxThreadsToWake; i++) {
                    g_WakeSingleAddress((PVOID)&WorkIndex);
                }
            }
            else {
                // In windows the more threads we wake up, the longer it takes to return from this OS call
                g_WakeAllAddress((PVOID)&WorkIndex);
            }
        }

#elif defined(RT_OS_LINUX)
        // Linux thread wakeup
        int s = futex((int*)&WorkIndex, FUTEX_WAKE, maxThreadsToWake, NULL, NULL, 0);
        if (s == -1)
            THREADLOGGING("***error futex-FUTEX_WAKE\n");     // TODO: Change to use fprintf(stderr, msg) instead

#elif defined(RT_OS_DARWIN)
        // temp remove warning
        //#warning MathThreads does not yet support Darwin/macOS.
        pthread_cond_broadcast(&g_WakeupCond);
#else
#error riptide MathThreads support not implemented for this platform.

#endif

    }

    FORCE_INLINE void CompleteWorkItem() {
        InterlockedIncrement64(&WorkIndexCompleted);
    }
};

WakeSingleAddress InitWakeCalls();
//DWORD WINAPI WorkerThreadFunction(LPVOID lpParam);

//============================================================================================
//============================================================================================


//#define MATHLOGGING printf
#define MATHLOGGING(...)

// Windows typedefs needd to call Windows APIS------------------------------------------------
typedef INT_PTR(WINAPI* FARPROC)();

#ifndef DECLARE_HANDLE
#define DECLARE_HANDLE(name) struct name##__{int unused;}; typedef struct name##__ *name
DECLARE_HANDLE(HINSTANCE);
typedef HINSTANCE HMODULE;
#endif

typedef wchar_t WCHAR;    // wc,   16-bit UNICODE character

extern "C" {
    typedef INT_PTR(WINAPI* FARPROC)();
    typedef DWORD(WINAPI* PTHREAD_START_ROUTINE)(LPVOID lpThreadParameter);
    typedef PTHREAD_START_ROUTINE LPTHREAD_START_ROUTINE;


    extern void PrintCPUInfo(char * buffer, size_t buffercount);

#if defined(RT_OS_WINDOWS)
    typedef HANDLE THANDLE;
    extern int GetProcCount();

#else
    typedef pthread_t  THANDLE;

    int GetProcCount();
    VOID Sleep(DWORD dwMilliseconds);
    BOOL CloseHandle(THANDLE hObject);

    uint64_t SetThreadAffinityMask(pid_t hThread, uint64_t dwThreadAffinityMask);

    BOOL GetProcessAffinityMask(HANDLE hProcess, uint64_t* lpProcessAffinityMask, uint64_t* lpSystemAffinityMask);
    pid_t GetCurrentThread();

    HANDLE GetCurrentProcess(VOID);

    DWORD  GetLastError(VOID);

    HANDLE CreateThread(VOID* lpThreadAttributes, SIZE_T dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, LPVOID lpParameter, DWORD dwCreationFlags, LPDWORD lpThreadId);

    HMODULE LoadLibraryW(const WCHAR* lpLibFileName);
    FARPROC GetProcAddress(HMODULE hModule, const char* lpProcName);

#endif
};

THANDLE StartThread(stWorkerRing* pWorkerRing);

// Move to reduce
#define MINF(x,y) x < y ? x : y
#define MAXF(x,y) x > y ? x : y


class CMathWorker {

public:
    static const int64_t WORK_ITEM_CHUNK = stMATH_WORKER_ITEM::WORK_ITEM_CHUNK;
    static const int64_t WORK_ITEM_BIG = stMATH_WORKER_ITEM::WORK_ITEM_BIG;
    static const int64_t WORK_ITEM_MASK = stMATH_WORKER_ITEM::WORK_ITEM_MASK;
    static const int MAX_WORKER_HANDLES = 64;

    int   WorkerThreadCount;

    // Windows specific
    DWORD LogicalProcessorCount = 0;
    DWORD NumaNodeCount = 0;
    DWORD ProcessorCoreCount = 0;
    DWORD ProcessorL1CacheCount = 0;
    DWORD ProcessorL2CacheCount = 0;
    DWORD ProcessorL3CacheCount = 0;
    DWORD ProcessorPackageCount = 0;


    // Set to true to stop threading
    BOOL  NoThreading;

    // Set to true to stop allocating from a cache
    BOOL  NoCaching;

    //------------------------------------------------------------------------------
    // Data Members 
    stWorkerRing*   pWorkerRing;

    THANDLE         WorkerThreadHandles[MAX_WORKER_HANDLES];
    char            CPUString[512];
    //------------------------------------------------------------------------------
    // Data Members 
    CMathWorker() {

        PrintCPUInfo(CPUString, sizeof(CPUString));
        ProcessorInformation();

        WorkerThreadCount = GetProcCount();
        NoThreading = FALSE;
        NoCaching = FALSE;

        pWorkerRing = (stWorkerRing*)ALIGNED_ALLOC(sizeof(stWorkerRing), 64);
        if (pWorkerRing) {
            pWorkerRing->WorkIndex = 0;
            pWorkerRing->Init();
        }

        for (int i = 0; i < WorkerThreadCount; i++) {

            WorkerThreadHandles[i] = 0;
        }

    };

    ~CMathWorker() {
        if (pWorkerRing) pWorkerRing->Cancel();
        Sleep(100);
        KillWorkerThreads();
        // DO NOT DEALLOCATE DO TO threads not exiting 
        //ALIGNED_FREE(pWorkerRing);
    };


#if defined(RT_OS_WINDOWS)
    // Windows routine to get physical and logical processor count
    // Helper function to count set bits in the processor mask.
    DWORD CountSetBits(ULONG_PTR bitMask)
    {
        DWORD LSHIFT = sizeof(ULONG_PTR) * 8 - 1;
        DWORD bitSetCount = 0;
        ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;
        DWORD i;

        for (i = 0; i <= LSHIFT; ++i)
        {
            bitSetCount += ((bitMask & bitTest) ? 1 : 0);
            bitTest /= 2;
        }

        return bitSetCount;
    }

    int ProcessorInformation()
    {
        BOOL done = FALSE;
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
        DWORD returnLength = 0;
        DWORD byteOffset = 0;
        PCACHE_DESCRIPTOR Cache;


        while (!done)
        {
            DWORD rc = GetLogicalProcessorInformation(buffer, &returnLength);

            if (FALSE == rc)
            {
                if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
                {
                    if (buffer)
                        free(buffer);

                    buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
                        returnLength);

                    if (NULL == buffer)
                    {
                        MATHLOGGING(TEXT("\nError: Allocation failure\n"));
                        return (2);
                    }
                }
                else
                {
                    MATHLOGGING(TEXT("\nError %d\n"), GetLastError());
                    return (3);
                }
            }
            else
            {
                done = TRUE;
            }
        }

        ptr = buffer;

        while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength)
        {
            switch (ptr->Relationship)
            {
            case RelationNumaNode:
                // Non-NUMA systems report a single record of this type.
                NumaNodeCount++;
                break;

            case RelationProcessorCore:
                ProcessorCoreCount++;

                // A hyperthreaded core supplies more than one logical processor.
                LogicalProcessorCount += CountSetBits(ptr->ProcessorMask);
                break;

            case RelationCache:
                // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache. 
                Cache = &ptr->Cache;
                if (Cache->Level == 1)
                {
                    ProcessorL1CacheCount++;
                }
                else if (Cache->Level == 2)
                {
                    ProcessorL2CacheCount++;
                }
                else if (Cache->Level == 3)
                {
                    ProcessorL3CacheCount++;
                }
                break;

            case RelationProcessorPackage:
                // Logical processors share a physical package.
                ProcessorPackageCount++;
                break;

            default:
                MATHLOGGING(TEXT("\nError: Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.\n"));
                break;
            }
            byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            ptr++;
        }

        MATHLOGGING(TEXT("\nGetLogicalProcessorInformation results:\n"));
        MATHLOGGING(TEXT("Number of NUMA nodes: %d\n"), NumaNodeCount);
        MATHLOGGING(TEXT("Number of physical processor packages: %d\n"),  ProcessorPackageCount);
        MATHLOGGING(TEXT("Number of processor cores: %d\n"), ProcessorCoreCount);
        MATHLOGGING(TEXT("Number of logical processors: %d\n"), LogicalProcessorCount);
        MATHLOGGING(TEXT("Number of processor L1/L2/L3 caches: %d/%d/%d\n"),  ProcessorL1CacheCount,  ProcessorL2CacheCount,  ProcessorL3CacheCount);

        free(buffer);

        return 0;
    }
#else
    // Linux routine to be completed
    int ProcessorInformation() {

    }

#endif

    //------------------------------------------------------------------------------
    // Returns number of worker threads + main thread
    int GetNumCores() {
        // include main python thread
        return WorkerThreadCount + 1;
    }

    //---------------------------------
    // Changes how many threads wake up in Linux
    int SetFutexWakeup(int howManyToWake) {
        if (howManyToWake < 1) {
            // On Windows seem to need at least 1
            howManyToWake = 1;
        }

        if (howManyToWake > FUTEX_WAKE_MAX) {
            // see linux man page on futex
            howManyToWake = FUTEX_WAKE_MAX;
        }

        int previousVal = pWorkerRing->FutexWakeCount;

        pWorkerRing->FutexWakeCount = howManyToWake;
        return previousVal;
    }

    int GetFutexWakeup() {
        return pWorkerRing->FutexWakeCount;
    }

    //------------------------------------------------------------------------------
    // 
    void StartWorkerThreads(int numaNode) {

        MATHLOGGING("Start worker threads\n");

        if (ProcessorCoreCount < LogicalProcessorCount) {
            // Set flag that hyperthreading is on
            MATHLOGGING("Hyperthreading on\n");
            WorkerThreadCount = ProcessorCoreCount;
            pWorkerRing->HyperThreading = 1;
        }

        for (int i = 0; i < WorkerThreadCount; i++) {

            WorkerThreadHandles[i] = StartThread(pWorkerRing);
        }

        // Pin the main thread to a numa node?
        // TODO: work
        //uint64_t mask = ((uint64_t)1 << WorkerThreadCount);//core number starts from 0
        // uint64_t ret = SetThreadAffinityMask(GetCurrentThread(), (uint64_t)0xFFFF);
        // SetThreadAffinityMask(GetCurrentThread(), (uint64_t)0xFFFFFFFF);

    }

    //------------------------------------------------------------------------------
    // 
    void KillWorkerThreads() {
        for (int i = 0; i < WorkerThreadCount; i++) {
            CloseHandle(WorkerThreadHandles[i]);
        }
    }



    //------------------------------------------------------------------------------
    //  Concurrent callback from multiple threads
    static int64_t MultiThreadedCounterCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
        // -1 is the first core
        core = core + 1;
        int64_t didSomeWork = 0;

        int64_t index;
        int64_t workBlock;

        // As long as there is work to do
        while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0) {
            // First index is 1 so we subtract
            index--;

            pstWorkerItem->MTWorkCallback(pstWorkerItem->WorkCallbackArg, core, index);

            didSomeWork++;
            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();

        }
        return didSomeWork;

    }

    //-----------------------------------------------------------
    // Automatically handles threading vs no threading
    // Uses counters that start at 0 and go up from 1
    void DoMultiThreadedWork(int numItems, MTWORK_CALLBACK  doMTWorkCallback, void* workCallbackArg, int32_t threadWakeup = 0)
    {
        // See if we get a work item (threading might be off)
        stMATH_WORKER_ITEM* pWorkItem = GetWorkItemCount(numItems);

        if (pWorkItem) {
            //
            // Each thread will call this routine with the callbackArg
            //
            pWorkItem->DoWorkCallback = MultiThreadedCounterCallback;
            pWorkItem->WorkCallbackArg = workCallbackArg;
            pWorkItem->MTWorkCallback = doMTWorkCallback;

            MATHLOGGING("before compress threaded\n");

            // This will notify the worker threads of a new work item
            WorkMain(pWorkItem, numItems, threadWakeup, 1, FALSE);
            MATHLOGGING("after compress threaded\n");
        }
        else {
            // Just assume core 0 does all the work
            for (int t = 0; t < numItems; t++) {
                doMTWorkCallback(workCallbackArg, 0, t);
            }
        }
    }


    //------------------------------------------------------------------------------
    //  Concurrent callback from multiple threads
    //  Based on chunk size, each workIndex gets (0, 65536, 130000, etc.)
    // callback sig: typedef BOOL(*MTCHUNK_CALLBACK)(void* callbackArg, int core, int64_t start, int64_t length);

    static int64_t MultiThreadedChunkCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
        // -1 is the first core
        core = core + 1;
        int64_t didSomeWork = 0;

        int64_t lenX;
        int64_t workBlock;

        // As long as there is work to do
        while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {
            int64_t start = pstWorkerItem->BlockSize * workBlock;

            pstWorkerItem->MTChunkCallback(pstWorkerItem->WorkCallbackArg, core, start, lenX);

            didSomeWork++;
            // tell others we completed this work block
            pstWorkerItem->CompleteWorkBlock();
        }

        return didSomeWork;
    }


    //-----------------------------------------------------------
    // Automatically handles threading vs no threading
    // Used to divide up single array of data into chunks or sections
    // Returns TRUE if actually did multithreaded work, otherwise FALSE
    BOOL DoMultiThreadedChunkWork(int64_t lengthData, MTCHUNK_CALLBACK  doMTChunkCallback, void* workCallbackArg, int32_t threadWakeup = 0)
    {
        // See if we get a work item (threading might be off)
        stMATH_WORKER_ITEM* pWorkItem = GetWorkItem(lengthData);

        if (pWorkItem) {
            //
            //
            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = MultiThreadedChunkCallback;
            pWorkItem->WorkCallbackArg = workCallbackArg;
            pWorkItem->MTChunkCallback = doMTChunkCallback;

            // This will notify the worker threads of a new work item
            WorkMain(pWorkItem, lengthData, threadWakeup);
            return TRUE;
        }
        else {
            // Just assume core 0 does all the work
            doMTChunkCallback(workCallbackArg, 0, 0, lengthData);
            return FALSE;
        }
    }


    //------------------------------------------------------------------------------
    // Returns NULL if work item is too small or threading turned off
    // Otherwise returns a work item
    stMATH_WORKER_ITEM* GetWorkItemCount(int64_t len) {
        // If it is a small work item, process it immediately
        if (NoThreading) {
            return NULL;
        }

        // Otherwise allow parallel processing
        stMATH_WORKER_ITEM* pWorkItem = pWorkerRing->GetWorkItem();
        return pWorkItem;
    }


    //------------------------------------------------------------------------------
    // Returns NULL if work item is too small or threading turned off
    // Otherwise returns a work item
    inline stMATH_WORKER_ITEM* GetWorkItem(int64_t len) {
        // If it is a small work item, process it immediately
        if (len < WORK_ITEM_BIG || NoThreading) {
            return NULL;
        }

        // Otherwise allow parallel processing
        stMATH_WORKER_ITEM* pWorkItem = pWorkerRing->GetWorkItem();
        return pWorkItem;
    }

    //------------------------------------------------------------------------------
    // Called from main thread
    void WorkMain(
        stMATH_WORKER_ITEM* pWorkItem,
        int64_t len,
        int32_t  threadWakeup,
        int64_t BlockSize = WORK_ITEM_CHUNK,
        bool bGenericMode = TRUE) {

        pWorkItem->TotalElements = len;

        const int32_t   maxWakeup = GetFutexWakeup();

        MATHLOGGING("wakeup max:%d  requested:%d\n", maxWakeup, threadWakeup);
        // Only windows uses ThreadWakup
        // Linux uses the futex to wakup more threads
        // If the number of threads to wakeup is not specified, we use the default
        if (threadWakeup <= 0) {

            // use default number of threads
            threadWakeup = maxWakeup;
        }
        else {
            // use lower number to wake up threads
            threadWakeup = threadWakeup < maxWakeup ? threadWakeup : maxWakeup;
        }

        // only windows uses this for now
        pWorkItem->ThreadWakeup = threadWakeup;

        if (bGenericMode) {
            // WORK_ITEM_CHUNK at a time
            pWorkItem->BlockLast = (len + (BlockSize - 1)) / BlockSize;
        }
        else {
            // custom mode (called from groupby)
            // also can be called from parmerge
            pWorkItem->BlockLast = len + 1;
        }

        pWorkItem->BlocksCompleted = 0;
        pWorkItem->BlockNext = 0;
        pWorkItem->BlockSize = BlockSize;

        // Tell all worker threads about this new work item (futex or wakeall)
        // TODO: Consider waking a different number of threads based on complexity
        // NOTE: This is a common optimization point: how long it takes to wake up threads
        // uint64_t currentTSC = __rdtsc();

        pWorkerRing->SetWorkItem(threadWakeup);

        // MATHLOGGING("Took %lld cycles to wakeup\n", __rdtsc() - currentTSC);

        // Also do work
        pWorkItem->DoWork(-1, 0);

        if (bGenericMode) {
            // Check if all workers have completed
            while (pWorkItem->BlocksCompleted < pWorkItem->BlockLast) {
                MATHLOGGING("Waiting %llu  %llu \n", pWorkItem->BlocksCompleted, pWorkItem->BlockLast);
                YieldProcessor();
                // We yield instead of Sleep(0)
            }
        }
        else {
            // Check if all workers have completed
            while (pWorkItem->BlocksCompleted < len) {
                MATHLOGGING("Waiting %llu  %llu \n", pWorkItem->BlocksCompleted, pWorkItem->BlockLast);
                YieldProcessor();
                // We yield instead of Sleep(0)
            }
        }

        // Mark this as completed
        pWorkerRing->CompleteWorkItem();
    }


};

//------------------------------------------------------------
// declare the global math worker
extern void* g_cMathWorker;
#define THREADER    ((CMathWorker*)g_cMathWorker)

