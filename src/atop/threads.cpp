#include "threads.h"

#if defined(__GNUC__)
#pragma GCC target "arch=core-avx2,tune=core-avx2"
#if __GNUC_PREREQ(4, 4) || (__clang__ > 0 && __clang_major__ >= 3) || !defined(__GNUC__)
/* GCC >= 4.4 or clang or non-GCC compilers */
#include <x86intrin.h>
#elif __GNUC_PREREQ(4, 1)
/* GCC 4.1, 4.2, and 4.3 do not have x86intrin.h, directly include SSE2 header */
#include <emmintrin.h>
#endif
#endif

// to debug thread wakeup allow LOGGING to printf
//#define LOGGING printf
#define LOGGING(...)
#define LOGERROR printf

#if defined(RT_OS_DARWIN)
/* For MacOS use a conditional wakeup */
pthread_cond_t  g_WakeupCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t g_WakeupMutex = PTHREAD_MUTEX_INITIALIZER;
#endif


#if defined(RT_OS_WINDOWS)
WakeSingleAddress g_WakeSingleAddress = InitWakeCalls();
WakeAllAddress g_WakeAllAddress;
WaitAddress g_WaitAddress;


//-----------------------------------------------------------------
// Not every version of Windows has this useful API so we have to check for it dynamically
WakeSingleAddress InitWakeCalls()
{
    FARPROC fp;

    HMODULE hModule = LoadLibraryW(L"kernelbase.dll");

    if (hModule != NULL) {
        fp = GetProcAddress(hModule, "WakeByAddressSingle");
        if (fp != NULL) {
            //LogInform("**System supports WakeByAddressSingle ...\n");
            g_WakeSingleAddress = (VOID(WINAPI*)(PVOID)) fp;

            fp = GetProcAddress(hModule, "WakeByAddressAll");
            g_WakeAllAddress = (WakeAllAddress)fp;

            fp = GetProcAddress(hModule, "WaitOnAddress");
            g_WaitAddress = (WaitAddress)fp;

        }
        else {
            LOGERROR("**System does NOT support WakeByAddressSingle ...\n");
            g_WakeSingleAddress = NULL;
            g_WakeAllAddress = NULL;
            g_WaitAddress = NULL;

        }
    }

    return g_WakeSingleAddress;
}

#else
WakeSingleAddress g_WakeSingleAddress = NULL;
WakeAllAddress g_WakeAllAddress = NULL;
WaitAddress g_WaitAddress = NULL;
#endif


//-----------------------------------------------------------
// Main thread loop
// Threads will wait on an address then wake up when there is work
// Linux uses a futex to control how many threads wakeup
// Windows uses a counter
// Darwin (macOS) does not support futexes or WaitOnAddress, so it will need to use one of:
//   * POSIX condition variables
//   * C++11 condition variables from <atomic>
//   * libdispatch (GCD), using dispatch_semaphore_t (via dispatch_semaphore_create()) to control concurrency; include <dispatch/semaphore.h>
//   * BSD syscalls like __psynch_cvwait (and other __psynch functions). These are not externally documented -- need to look in github.com/apple/darwin-libpthread to see how things work.
//
#if defined(RT_OS_WINDOWS)
DWORD WINAPI WorkerThreadFunction(LPVOID lpParam)
#else
void*
WorkerThreadFunction(void* lpParam)
#endif
{
    stWorkerRing* pWorkerRing = (stWorkerRing*)lpParam;

    DWORD core = (DWORD)(InterlockedIncrement64(&pWorkerRing->WorkThread));
    core = core - 1;

    //if (core > 3) core += 16;
    //core += 16;

    LOGGING("Thread created with parameter: %d   %p\n", core, g_WaitAddress);

    // On windows we set the thread affinity mask
    if (g_WaitAddress != NULL) {
        uint64_t mask = (uint64_t)(1) << core;//core number starts from 0
        uint64_t ret = SetThreadAffinityMask(GetCurrentThread(), mask);
        //uint64_t ret = SetThreadAffinityMask(GetCurrentThread(), 0xFFFFFFFF);
    }

    int64_t lastWorkItemCompleted = -1;

    //
    // Setting Cancelled will stop all worker threads
    //
    while (pWorkerRing->Cancelled == 0) {
        int64_t workIndexCompleted;
        int64_t workIndex;

        workIndex = pWorkerRing->WorkIndex;
        workIndexCompleted = pWorkerRing->WorkIndexCompleted;

        BOOL didSomeWork = FALSE;

        // See if work to do
        if (workIndex > workIndexCompleted) {
            stMATH_WORKER_ITEM* pWorkItem = pWorkerRing->GetExistingWorkItem();

#if defined(RT_OS_WINDOWS)
            // Windows we check if the work was for our thread
            int64_t wakeup = InterlockedDecrement64(&pWorkItem->ThreadWakeup);
            if (wakeup >= 0) {
                didSomeWork = pWorkItem->DoWork(core, workIndex);
            }
            else {
                //printf("[%d] not doing work %lld.  %lld  %lld\n", core, wakeup, workIndex, workIndexCompleted);
                //workIndex++;
            }
#else
            didSomeWork = pWorkItem->DoWork(core, workIndex);

#endif
        }

        if (!didSomeWork) {
            workIndexCompleted = workIndex;

#if defined(RT_OS_WINDOWS)
            //printf("Sleeping %d", core);
            if (g_WaitAddress == NULL) {
                // For Windows 7 we just sleep
                Sleep(pWorkerRing->SleepTime);
            }
            else {
                if (!didSomeWork) {

                    //workIndexCompleted++;
                }

                LOGGING("[%d] WaitAddress %llu  %llu  %d\n", core, workIndexCompleted, pWorkerRing->WorkIndex, (int)didSomeWork);

                // Otherwise wake up using conditional variable
                g_WaitAddress(
                    &pWorkerRing->WorkIndex,
                    (PVOID)&workIndexCompleted,
                    8, // The size of the value being waited on (i.e. the number of bytes to read from the two pointers then compare).
                    1000000L);
            }
#elif defined(RT_OS_LINUX)

            LOGGING("[%d] WaitAddress %llu  %llu  %d\n", core, workIndexCompleted, pWorkerRing->WorkIndex, (int)didSomeWork);

            //int futex(int *uaddr, int futex_op, int val,
            //   const struct timespec *timeout,   /* or: uint32_t val2 */
            //   int *uaddr2, int val3);
            futex((int*)&pWorkerRing->WorkIndex, FUTEX_WAIT, (int)workIndexCompleted, NULL, NULL, 0);

#elif defined(RT_OS_DARWIN)
            LOGGING("[%lu] WaitAddress %llu  %llu  %d\n", core, workIndexCompleted, pWorkerRing->WorkIndex, (int)didSomeWork);

            pthread_mutex_lock(&g_WakeupMutex);
            pthread_cond_wait(&g_WakeupCond, &g_WakeupMutex);
            pthread_mutex_unlock(&g_WakeupMutex);

#else
#error riptide MathThreads support needs to be implemented for this platform.

#endif

            //printf("Waking %d", core);

            //YieldProcessor();
        }
        //YieldProcessor();
    }

    LOGERROR("Thread %d exiting!!!\n", (int)core);
#if defined(RT_OS_WINDOWS)
    return 0;
#else
    return NULL;
#endif
}


#if defined(RT_OS_WINDOWS)

//-----------------------------------------------------------
//
THANDLE StartThread(stWorkerRing* pWorkerRing)
{
    DWORD dwThreadId;
    THANDLE hThread;

    hThread = CreateThread(
        NULL, // default security attributes
        0, // use default stack size
        WorkerThreadFunction, // thread function
        pWorkerRing, // argument to thread function
        0, // use default creation flags
        &dwThreadId); // returns the thread identifier

                      //printf("The thread ID: %d.\n", dwThreadId);

                      // Check the return value for success. If something wrong...
    if (hThread == NULL) {
        LOGERROR("CreateThread() failed, error: %d.\n", GetLastError());
        return NULL;
    }

    return hThread;

}

#else

//-----------------------------------------------------------
//
THANDLE StartThread(stWorkerRing* pWorkerRing)
{
    int err;
    THANDLE hThread;

    err = pthread_create(&hThread, NULL, &WorkerThreadFunction, pWorkerRing);

    if (err != 0) {
        LOGERROR("*** Cannot create thread :[%s]\n", strerror(err));
    }

    return hThread;
}
#endif

//============================================================================================
#if defined(__GNUC__)
#  define MEM_STATIC static __inline __attribute__((unused))
#elif defined (__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#  define MEM_STATIC static inline
#elif defined(_MSC_VER)
#  define MEM_STATIC static __inline
#else
#  define MEM_STATIC static  /* this version may generate warnings for unused static functions; disable the relevant warning */
#endif

typedef unsigned int U32;

// Taken from the ZSTD project
MEM_STATIC ATOP_cpuid_t ATOP_cpuid(void) {
    U32 f1c = 0;
    U32 f1d = 0;
    U32 f7b = 0;
    U32 f7c = 0;
#ifdef _MSC_VER
    int reg[4];
    __cpuid((int*)reg, 0);
    {
        int const n = reg[0];
        if (n >= 1) {
            __cpuid((int*)reg, 1);
            f1c = (U32)reg[2];
            f1d = (U32)reg[3];
        }
        if (n >= 7) {
            __cpuidex((int*)reg, 7, 0);
            f7b = (U32)reg[1];
            f7c = (U32)reg[2];
        }
    }
#elif defined(__i386__) && defined(__PIC__) && !defined(__clang__) && defined(__GNUC__)
    /* The following block like the normal cpuid branch below, but gcc
    * reserves ebx for use of its pic register so we must specially
    * handle the save and restore to avoid clobbering the register
    */
    U32 n;
    __asm__(
        "pushl %%ebx\n\t"
        "cpuid\n\t"
        "popl %%ebx\n\t"
        : "=a"(n)
        : "a"(0)
        : "ecx", "edx");
    if (n >= 1) {
        U32 f1a;
        __asm__(
            "pushl %%ebx\n\t"
            "cpuid\n\t"
            "popl %%ebx\n\t"
            : "=a"(f1a), "=c"(f1c), "=d"(f1d)
            : "a"(1));
    }
    if (n >= 7) {
        __asm__(
            "pushl %%ebx\n\t"
            "cpuid\n\t"
            "movl %%ebx, %%eax\n\r"
            "popl %%ebx"
            : "=a"(f7b), "=c"(f7c)
            : "a"(7), "c"(0)
            : "edx");
    }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    U32 n;
    __asm__("cpuid" : "=a"(n) : "a"(0) : "ebx", "ecx", "edx");
    if (n >= 1) {
        U32 f1a;
        __asm__("cpuid" : "=a"(f1a), "=c"(f1c), "=d"(f1d) : "a"(1) : "ebx");
    }
    if (n >= 7) {
        U32 f7a;
        __asm__("cpuid"
            : "=a"(f7a), "=b"(f7b), "=c"(f7c)
            : "a"(7), "c"(0)
            : "edx");
    }
#endif
    {
        ATOP_cpuid_t cpuid;
        cpuid.f1c = f1c;
        cpuid.f1d = f1d;
        cpuid.f7b = f7b;
        cpuid.f7c = f7c;
        return cpuid;
    }
}

#define X(name, r, bit)                                                        \
  MEM_STATIC int ATOP_cpuid_##name(ATOP_cpuid_t const cpuid) {                 \
    return ((cpuid.r) & (1U << bit)) != 0;                                     \
  }

/* cpuid(1): Processor Info and Feature Bits. */
#define C(name, bit) X(name, f1c, bit)
C(sse3, 0)
C(pclmuldq, 1)
C(dtes64, 2)
C(monitor, 3)
C(dscpl, 4)
C(vmx, 5)
C(smx, 6)
C(eist, 7)
C(tm2, 8)
C(ssse3, 9)
C(cnxtid, 10)
C(fma, 12)
C(cx16, 13)
C(xtpr, 14)
C(pdcm, 15)
C(pcid, 17)
C(dca, 18)
C(sse41, 19)
C(sse42, 20)
C(x2apic, 21)
C(movbe, 22)
C(popcnt, 23)
C(tscdeadline, 24)
C(aes, 25)
C(xsave, 26)
C(osxsave, 27)
C(avx, 28)
C(f16c, 29)
C(rdrand, 30)
#undef C
#define D(name, bit) X(name, f1d, bit)
D(fpu, 0)
D(vme, 1)
D(de, 2)
D(pse, 3)
D(tsc, 4)
D(msr, 5)
D(pae, 6)
D(mce, 7)
D(cx8, 8)
D(apic, 9)
D(sep, 11)
D(mtrr, 12)
D(pge, 13)
D(mca, 14)
D(cmov, 15)
D(pat, 16)
D(pse36, 17)
D(psn, 18)
D(clfsh, 19)
D(ds, 21)
D(acpi, 22)
D(mmx, 23)
D(fxsr, 24)
D(sse, 25)
D(sse2, 26)
D(ss, 27)
D(htt, 28)
D(tm, 29)
D(pbe, 31)
#undef D

/* cpuid(7): Extended Features. */
#define B(name, bit) X(name, f7b, bit)
B(bmi1, 3)
B(hle, 4)
B(avx2, 5)
B(smep, 7)
B(bmi2, 8)
B(erms, 9)
B(invpcid, 10)
B(rtm, 11)
B(mpx, 14)
B(avx512f, 16)
B(avx512dq, 17)
B(rdseed, 18)
B(adx, 19)
B(smap, 20)
B(avx512ifma, 21)
B(pcommit, 22)
B(clflushopt, 23)
B(clwb, 24)
B(avx512pf, 26)
B(avx512er, 27)
B(avx512cd, 28)
B(sha, 29)
B(avx512bw, 30)
B(avx512vl, 31)
#undef B
#define C(name, bit) X(name, f7c, bit)
C(prefetchwt1, 0)
C(avx512vbmi, 1)
#undef C

#undef X

extern "C" {
    int g_bmi2 = 0;
    int g_avx2 = 0;
    ATOP_cpuid_t   g_cpuid;
};

#if defined(RT_OS_WINDOWS)

void PrintCPUInfo(char* buffer, size_t buffercount) {
    int CPUInfo[4] = { -1 };
    unsigned   nExIds, i = 0;
    char CPUBrandString[0x40];
    // Get the information associated with each extended ID.
    __cpuid(CPUInfo, 0x80000000);
    nExIds = CPUInfo[0];
    for (i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(CPUInfo, i);
        // Interpret CPU brand string
        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    // NEW CODE
    g_cpuid = ATOP_cpuid();

    g_bmi2 = ATOP_cpuid_bmi2(g_cpuid);
    g_avx2 = ATOP_cpuid_avx2(g_cpuid);

    snprintf(buffer, buffercount, "**CPU: %s  AVX2:%d  BMI2:%d\n", CPUBrandString, g_avx2, g_bmi2);
    if (g_avx2 == 0) {
        printf("!!!NOTE: this system does not support AVX2 or BMI2 instructions, and will not work!\n");
    }

}

#else
extern "C" {
#include <pthread.h>
#include <sys/types.h>
#include <sched.h>

#include <unistd.h>
#include <sys/syscall.h>

#ifdef RT_OS_FREEBSD
#include <sys/thr.h> // Use thr_self() syscall under FreeBSD to get thread id
#endif  // RT_OS_FREEBSD

    pid_t gettid(void) {
#if defined(RT_OS_LINUX)
        return syscall(SYS_gettid);

#elif defined(RT_OS_DARWIN)
        uint64_t thread_id;
        return pthread_threadid_np(NULL, &thread_id) ? 0 : (pid_t)thread_id;

#elif defined(RT_OS_FREEBSD)
        // https://www.freebsd.org/cgi/man.cgi?query=thr_self
        long thread_id;
        return thr_self(&thread_id) ? 0 : (pid_t)thread_id;

#else
#error Cannot determine how to get the identifier for the current thread on this platform.
#endif   // defined(RT_OS_LINUX)
    }


    VOID Sleep(DWORD dwMilliseconds) {
        usleep(dwMilliseconds * 1000);
    }

    BOOL CloseHandle(THANDLE hObject) {
        return TRUE;
    }

    pid_t GetCurrentThread() {
        return gettid();
    }

    uint64_t SetThreadAffinityMask(pid_t hThread, uint64_t dwThreadAffinityMask) {
#if defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)
        cpu_set_t cpuset;

        uint64_t bitpos = 1;
        int count = 0;

        while (!(bitpos & dwThreadAffinityMask)) {
            bitpos <<= 1;
            count++;
            if (count > 63) {
                break;
            }
        }

        //printf("**linux setting affinity %d\n", count);

        if (count <= 63) {

            CPU_ZERO(&cpuset);
            CPU_SET(count, &cpuset);
            //dwThreadAffinityMask
            sched_setaffinity(GetCurrentThread(), sizeof(cpuset), &cpuset);
        }

#else
        #warning No thread - affinity support implemented for this OS.This does not prevent riptide from running but overall performance may be reduced.
#endif   // defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)

            return 0;
    }

    BOOL GetProcessAffinityMask(HANDLE hProcess, uint64_t* lpProcessAffinityMask, uint64_t* lpSystemAffinityMask) {
#if defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)
        cpu_set_t cpuset;
        sched_getaffinity(getpid(), sizeof(cpuset), &cpuset);

        *lpProcessAffinityMask = 0;
        *lpSystemAffinityMask = 0;

        uint64_t bitpos = 1;
        for (int i = 0; i < 63; i++) {
            if (CPU_ISSET(i, &cpuset)) {
                *lpProcessAffinityMask |= bitpos;
                *lpSystemAffinityMask |= bitpos;
            }
            bitpos <<= 1;
        }

        if (*lpProcessAffinityMask == 0) {
            *lpSystemAffinityMask = 0xFF;
            *lpSystemAffinityMask = 0xFF;
        }

        //CPU_ISSET = 0xFF;
        return TRUE;

#else
        #warning No thread - affinity support implemented for this OS.This does not prevent riptide from running but overall performance may be reduced.
            return FALSE;

#endif   // defined(RT_OS_LINUX) || defined(RT_OS_FREEBSD)
    }


    HANDLE GetCurrentProcess(VOID) {
        return NULL;
    }

    DWORD  GetLastError(VOID) {
        return 0;
    }

    HANDLE CreateThread(VOID* lpThreadAttributes, SIZE_T dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, LPVOID lpParameter, DWORD dwCreationFlags, LPDWORD lpThreadId) {
        return NULL;
    }

    HMODULE LoadLibraryW(const WCHAR* lpLibFileName) {
        return NULL;
    }

    FARPROC GetProcAddress(HMODULE hModule, const char* lpProcName) {
        return NULL;
    }
}

#include <cpuid.h>

void PrintCPUInfo(char* buffer, size_t buffercount) {
    char CPUBrandString[0x40];
    unsigned int CPUInfo[4] = { 0,0,0,0 };

    __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    unsigned int nExIds = CPUInfo[0];

    memset(CPUBrandString, 0, sizeof(CPUBrandString));

    for (unsigned int i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);

        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    //printf("**CPU: %s\n", CPUBrandString);

    g_cpuid = ATOP_cpuid();

    g_bmi2 = ATOP_cpuid_bmi2(g_cpuid);
    g_avx2 = ATOP_cpuid_avx2(g_cpuid);

    snprintf(buffer, buffercount, "**CPU: %s  AVX2:%d  BMI2:%d\n", CPUBrandString, g_avx2, g_bmi2);
    if (g_avx2 == 0) {
        printf("!!!NOTE: this system does not support AVX2 or BMI2 instructions, and will not work!\n");
    }

}

#endif



int GetProcCount() {

    HANDLE proc = GetCurrentProcess();

    uint64_t mask1;
    uint64_t mask2;
    int count;

    count = 0;
    GetProcessAffinityMask(proc, &mask1, &mask2);

    while (mask1 != 0) {
        if (mask1 & 1) count++;
        mask1 = mask1 >> 1;
    }

    //printf("**Process count: %d   riptide_cpp build date and time: %s %s\n", count, __DATE__, __TIME__);

    if (count == 0) count = MAX_THREADS_WHEN_CANNOT_DETECT;

    if (count > MAX_THREADS_ALLOWED) count = MAX_THREADS_ALLOWED;

    return count;

}
