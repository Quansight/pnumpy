#include "atop.h"
#include "threads.h"

#define LOGGING(...)
#define LOGERROR printf

//----------------------------------------------------------------------------------
// Lookup to go from 1 byte to 8 byte boolean values
int64_t gBooleanLUT64[256];
int32_t gBooleanLUT32[16];

int64_t gBooleanLUT64Inverse[256];
int32_t gBooleanLUT32Inverse[16];

void* g_cMathWorker = NULL;

// Keep track of stats
static int64_t g_TotalAllocs = 0;
static int64_t g_TotalFree = 0;
static int64_t g_TotalMemoryAllocated = 0;
static int64_t g_TotalMemoryFreed = 0;

#define MAGIC_PAGE_GUARD  0xDEADBEEFDEADBEEF
//-----------------------------------------------
void* FmAlloc(size_t _Size) {
    // make thread safe
    uint64_t* pageGuard = (uint64_t*)malloc(_Size + 16);
    if (pageGuard) {
        InterlockedIncrement64(&g_TotalAllocs);
        InterlockedAdd64(&g_TotalMemoryAllocated, _Size);
        pageGuard[0] = _Size;
        pageGuard[1] = MAGIC_PAGE_GUARD;

        // Skip past guard
        return &pageGuard[2];
    }
    return NULL;
}

void FmFree(void* _Block) {
    // The C standard requires that free() be a no-op when called with nullptr.
    // FmAlloc can return a nullptr, and since we want this function to behave
    // like free() we also need to handle the nullptr case here.
    if (!_Block) { return; }

    //LOGRECYCLE("Freeing %p\n", _Block);
    InterlockedIncrement64(&g_TotalFree);
    uint64_t* pageGuard = (uint64_t*)_Block;
    pageGuard--;
    pageGuard--;
    if (pageGuard[1] != MAGIC_PAGE_GUARD) {
        LOGERROR("!! User freed bad memory, no page guard %p\n", pageGuard);
    }
    else {
        InterlockedAdd64(&g_TotalMemoryFreed, pageGuard[0]);
        // mark so cannot free again
        pageGuard[1] = 0;
    }

    free(pageGuard);
}

//====================================================
// Must be called to initialize atop
// Will start threads and detect the CPU
// Will build runtime lookup tables
BOOL atop_init() {

    // Check if init already called
    if (g_cMathWorker) return FALSE;

    // Build LUTs used in comarisons after mask generated
    for (int i = 0; i < 256; i++) {
        BYTE* pDest = (BYTE*)&gBooleanLUT64[i];
        for (int j = 0; j < 8; j++) {
            *pDest++ = ((i >> j) & 1);
        }
    }
    // Build LUTs
    for (int i = 0; i < 16; i++) {
        BYTE* pDest = (BYTE*)&gBooleanLUT32[i];
        for (int j = 0; j < 4; j++) {
            *pDest++ = ((i >> j) & 1);
        }
    }

    // Build LUTs
    for (int i = 0; i < 256; i++) {
        gBooleanLUT64Inverse[i] = gBooleanLUT64[i] ^ 0x0101010101010101LL;
    }
    // Build LUTs
    for (int i = 0; i < 16; i++) {
        gBooleanLUT32Inverse[i] = gBooleanLUT32[i] ^ 0x01010101;
    }

    g_cMathWorker = new CMathWorker();

    // start up the worker threads now in case we use them
    THREADER->StartWorkerThreads(0);

    LOGGING("ATOP loaded\n");
    return TRUE;
}

