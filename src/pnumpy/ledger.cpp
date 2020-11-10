#include "common.h"

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

// See ATOP_TYPES
static const char* gStrAtopTypes[]= {
    "bool",
    "int8", "uint8",
    "int16", "uint16",
    "int32", "uint32",
    "int64", "uint64",
    "int128", "uint128",
    "float16", "float32", "float64", "float80",
    "cfloat16", "cfloat32", "cfloat64", "cfloat80",
    "string", "unicode",
    "void",
    "last"
};


struct stLEDGER_ITEM {
    const char* StrName;
    int64_t     StartTime;
    int64_t     TotalTime;

    int64_t     ArrayLength1;
    int64_t     ArrayLength2;
    int64_t     ArrayLength3;  // not valid for unary

    int32_t     ArrayGroup;
    int32_t     ArrayOp;
    int32_t     AType;
    int32_t     Reserved1;

    const char* StrCatName;
    const char* StrOpName;
};

//-----------------------------------------------------------
// allocated on 64 byte alignment
struct stLedgerRing {
    // must be power of 2 for mask to work
    static const int64_t   RING_BUFFER_SIZE = 8096;
    static const int64_t   RING_BUFFER_MASK = 8095;

    volatile int64_t       Head;
    volatile int64_t       Tail;

    stLEDGER_ITEM          LedgerQueue[RING_BUFFER_SIZE];

    void Init() {
        Head = 0;
        Tail = 0;

        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            LedgerQueue[i].StrName = 0;
            LedgerQueue[i].StartTime = 0;
            LedgerQueue[i].TotalTime = 0;
        }
    }

    // Circular wrap around buffer
    // If (Head - Tail)  > RING_BUFFER_SIZE then buffer has overflowed
    stLEDGER_ITEM* GetNextEntry() {
        return &LedgerQueue[RING_BUFFER_MASK & Tail++];
    };
};

// Global ring buffer of last RING_BUFFER_SIZE math operations
static stLedgerRing    g_LedgerRing;

// rough estimate of last op code
#define MAX_FUNCOP 40
const char* g_str_ufunc_name[OPCAT_LAST][MAX_FUNCOP];

void LedgerInit() {

    // Init the ring buffer that holds entries
    g_LedgerRing.Init();

    // Build reverse lookup table
    for (int i = 0; i < OPCAT_LAST; i++) {
        stOpCategory* pstOpCategory = &gOpCategory[i];
        for (int j = 0; j < pstOpCategory->NumOps; j++) {
            int k = pstOpCategory->pUFuncToAtop[j].atop_op;
            if (k >= 0 && k < MAX_FUNCOP) {
                // NOTE: can print out everything we hook here
                //printf("%d %d %s\n", i, k, pstOpCategory->pUFuncToAtop[j].str_ufunc_name);
                g_str_ufunc_name[i][k] = pstOpCategory->pUFuncToAtop[j].str_ufunc_name;
            }
        }
    }
}

//--------------------------------------------------
// When the ufunc is hooked, if the ledger is turned on it can be recorded.
// The recording will go into the ring buffer for later retrieval.
// The ring buffer only holds so much and can overflow
void LedgerRecord(int32_t op_category, int64_t start_time, int64_t end_time, char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    int64_t deltaTime = end_time - start_time;

    stOpCategory* pstOpCategory = &gOpCategory[op_category];

    // Get the next slot in the ring buffer
    stLEDGER_ITEM* pEntry = g_LedgerRing.GetNextEntry();

    pEntry->ArrayGroup = op_category;
    pEntry->ArrayOp = funcop;
    pEntry->AType = atype;

    const char* strCatName = pstOpCategory->StrName;

    // Check for reduce operation
    if (op_category == OPCAT_BINARY && IS_BINARY_REDUCE) {
        strCatName = "Reduce";
    }

    pEntry->StrCatName = strCatName;
    pEntry->StrOpName = g_str_ufunc_name[op_category][funcop];
    pEntry->ArrayLength1 = (int64_t)dimensions[0];
    pEntry->ArrayLength2 = (int64_t)dimensions[1];

    // temporary for debugging print out results
    printf ("%lld \tlen: %lld   %s,  %s,  %s\n", (long long)deltaTime, (long long)dimensions[0], pEntry->StrOpName, gStrAtopTypes[atype], strCatName);
       
}


