#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdint.h>
#include <stdio.h>
#include "../atop/atop.h"


#if defined(_WIN32)

// global scope
typedef VOID(WINAPI* FuncGetSystemTime)(LPFILETIME);
FuncGetSystemTime g_GetSystemTime;
FILETIME g_TimeStart;
static bool g_IsPreciseTime = false;


//------------------------------------
// Returns windows time in Nanos
__inline static uint64_t GetWindowsTime() {
   FILETIME timeNow;
   g_GetSystemTime(&timeNow);
   return (*(uint64_t*)&timeNow * 100) - 11644473600000000000L;
}

//-------------------------------------------------------------------
//
class CTimeStamp {
public:
   CTimeStamp()
   {
      FARPROC fp;

      g_GetSystemTime = GetSystemTimeAsFileTime;

      HMODULE hModule = LoadLibraryW(L"kernel32.dll");

      // Use printf instead of logging because logging is probably not up yet
      // Logging uses the timestamping, so timestamping loads first
      if (hModule != NULL) {
         fp = GetProcAddress(hModule, "GetSystemTimePreciseAsFileTime");
         if (fp != NULL) {
            g_IsPreciseTime = true;
            //printf("Using precise GetSystemTimePreciseAsFileTime time...\n");
            g_GetSystemTime = (VOID(WINAPI*)(LPFILETIME)) fp;
         }
         else {
            //LOGGING("**Using imprecise GetSystemTimeAsFileTime...\n");
         }
      }
      else {
         printf("!! error load kernel32\n");
      }

   }
};

static CTimeStamp* g_TimeStamp = new CTimeStamp();


//---------------------------------------------------------
// Returns and int64_t nanosecs since unix epoch
extern "C"
PyObject* timer_getutc(PyObject* self, PyObject* args) {

   // return nano time since Unix Epoch
   return PyLong_FromLongLong((long long)GetWindowsTime());
}

//---------------------------------------------------------
// Returns and uint64_t timestamp counter
extern "C"
PyObject* timer_gettsc(PyObject* self, PyObject* args) {

   // return tsc
   return PyLong_FromUnsignedLongLong(__rdtsc());
}



#else

#include <time.h>
#include <sys/time.h>
#include <unistd.h>

uint64_t GetTimeStamp() {
   //struct timeval tv;
   //gettimeofday(&tv, NULL);
   //return tv.tv_sec*(uint64_t)1000000 + tv.tv_usec;

   struct timespec x;
   clock_gettime(CLOCK_REALTIME, &x);
   return x.tv_sec * 1000000000L + x.tv_nsec;
}

static __inline__ uint64_t rdtsc(void)
{
   unsigned hi, lo;
   __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
   return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

//---------------------------------------------------------
// Returns and uint64_t timestamp counter
extern "C"
PyObject* timer_gettsc(PyObject* self, PyObject* args) {

   // return tsc
   return PyLong_FromUnsignedLongLong(rdtsc());
}

//---------------------------------------------------------
// Returns and int64_t nanosecs since unix epoch
extern "C"
PyObject* timer_getutc(PyObject* self, PyObject* args) {

   // return nano time since Unix Epoch
   return PyLong_FromLongLong(GetTimeStamp());
}

#endif

//---------------------------------------------------------
// Returns nanoseconds since utc epoch
uint64_t GetUTCNanos() {
#if defined(_WIN32)
   return GetWindowsTime();
#else
   return GetTimeStamp();
#endif
}


struct stLEDGER_ITEM {
    const char* StrName;
    int64_t     StartTime;
    int64_t     TotalTime;

    int64_t     ArraySize;

    int32_t     ArrayOp;
    int32_t     Type;

};

//-----------------------------------------------------------
// allocated on 64 byte alignment
struct stLedgerRing {
    static const int64_t   RING_BUFFER_SIZE = 8096;
    static const int64_t   RING_BUFFER_MASK = 8095;

    volatile int64_t       Index;

    stLEDGER_ITEM          LedgerQueue[RING_BUFFER_SIZE];

    void Init() {
        Index = 0;

        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            LedgerQueue[i].StrName = 0;
            LedgerQueue[i].StartTime = 0;
            LedgerQueue[i].TotalTime = 0;
            LedgerQueue[i].ArraySize = 0;
        }
    }
};

stLedgerRing    g_LedgerRing;

