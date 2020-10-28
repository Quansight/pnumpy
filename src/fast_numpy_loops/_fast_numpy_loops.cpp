#include "common.h"
#include "../atop/threads.h"

#define LOGGING(...)

// Conversion from numpy dtype to atop dtype
int convert_dtype_to_atop[]={
     ATOP_BOOL,                         //NPY_BOOL = 0,
     ATOP_INT8, ATOP_UINT8,             //NPY_BYTE, NPY_UBYTE,
     ATOP_INT16, ATOP_UINT16,           //NPY_SHORT, NPY_USHORT,
     ATOP_INT32, ATOP_UINT32,           //NPY_INT, NPY_UINT,

#if !defined(RT_COMPILER_MSVC)
     ATOP_INT64, ATOP_UINT64,           //NPY_LONG, NPY_ULONG,
#else
     ATOP_INT32, ATOP_UINT32,           //NPY_LONG, NPY_ULONG,
#endif

     ATOP_INT64, ATOP_UINT64,           //NPY_LONGLONG, NPY_ULONGLONG,
     ATOP_FLOAT, ATOP_DOUBLE, ATOP_LONGDOUBLE,    //NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
     -1, -1, -1,                                  //NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
     -1,                                //NPY_OBJECT = 17,
     ATOP_STRING, ATOP_UNICODE,         //NPY_STRING, NPY_UNICODE,
     ATOP_VOID                          //NPY_VOID,
};

// Reverse conversion from atop dtype to numpy dtype
int convert_atop_to_dtype[] = {
     NPY_BOOL,                         //NPY_BOOL = 0,
     NPY_INT8, NPY_UINT8,              //NPY_BYTE, NPY_UBYTE,
     NPY_INT16, NPY_UINT16,            //NPY_SHORT, NPY_USHORT,
     NPY_INT32, NPY_UINT32,            //NPY_INT, NPY_UINT,
     NPY_INT64, NPY_UINT64,            //NPY_LONG, NPY_ULONG,
     NPY_LONGLONG, NPY_ULONGLONG,      // Really INT128
     NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,    //NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
     NPY_STRING, NPY_UNICODE,         //NPY_STRING, NPY_UNICODE,
     NPY_VOID                          //NPY_VOID,
};

// Find out the sizeof() for an atop dtype
int convert_atop_to_itemsize[] = {
    1,  1,  1,
    2,  2,
    4,  4,
    8,  8,
    16, 16,
    4,  8,  sizeof(long double),
    1,  4,
    0
};

//------------------------------------------------------------------------------------------
// A full list of ufuncs as of Oct 2020
// abs, absolute, add, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
// bitwise_and, bitwise_not, bitwise_or, bitwise_xor,
// cbrt, ceil, conj, conjugate, copysign, cos, cosh, deg2rad, degrees, divide, divmod, equal, exp, exp2, expm1,
// fabs, float_power, floor, floor_divide, fmax, fmin, fmod, frexp, gcd, greater, greater_equal,
// heaviside, hypot, invert, isfinite, isinf, isnan, isnat,
// lcm, ldexp, left_shift, less, less_equal, log, log10, log1p, log2, logaddexp, logaddexp2,
// logical_and, logical_not, logical_or, logical_xor
// matmul, maximum, minimum, mod, modf, multiply, negative, nextafter, not_equal, positive, power,
// rad2deg, radians, reciprocal, remainder, right_shift, rint
// sign, signbit, sin, sinh, spacing, sqrt, square, subtract, tan, tanh, true_divide, trunc

// Binary function mapping
static stUFuncToAtop gBinaryMapping[]={
    {"add",           BINARY_OPERATION::ADD},
    {"subtract",      BINARY_OPERATION::SUB },
    {"multiply",      BINARY_OPERATION::MUL },
    {"true_divide",   BINARY_OPERATION::DIV },
    {"floor_divide",  BINARY_OPERATION::FLOORDIV },
    //{"minimum",       BINARY_OPERATION::MIN },
    //{"maximum",       BINARY_OPERATION::MAX },
    {"power",         BINARY_OPERATION::POWER },
    {"remainder",     BINARY_OPERATION::REMAINDER },
    {"logical_and",   BINARY_OPERATION::LOGICAL_AND },
    {"logical_or",    BINARY_OPERATION::LOGICAL_OR },
    {"bitwise_and",   BINARY_OPERATION::BITWISE_AND },
    {"bitwise_or",    BINARY_OPERATION::BITWISE_OR },
    {"bitwise_xor",   BINARY_OPERATION::BITWISE_XOR },
    {"left_shift",    BINARY_OPERATION::BITWISE_LSHIFT },
    {"right_shift",   BINARY_OPERATION::BITWISE_RSHIFT },

};

// Compare function mapping
static stUFuncToAtop gCompareMapping[]={
    {"equal",         COMP_OPERATION::CMP_EQ},
    {"not_equal",     COMP_OPERATION::CMP_NE},
    {"greater",       COMP_OPERATION::CMP_GT},
    {"greater_equal", COMP_OPERATION::CMP_GTE},
    {"less",          COMP_OPERATION::CMP_LT},
    {"less_equal",    COMP_OPERATION::CMP_LTE } };

// Unary function mapping
static stUFuncToAtop gUnaryMapping[] = {
    {"abs",           UNARY_OPERATION::ABS},
    {"signbit",       UNARY_OPERATION::SIGNBIT},
    {"fabs",          UNARY_OPERATION::FABS},
    {"invert",        UNARY_OPERATION::INVERT},
    {"floor",         UNARY_OPERATION::FLOOR},
    {"ceil",          UNARY_OPERATION::CEIL},
    {"trunc",         UNARY_OPERATION::TRUNC},
    {"rint",          UNARY_OPERATION::ROUND},  
    {"negative",      UNARY_OPERATION::NEGATIVE},
    {"positive",      UNARY_OPERATION::POSITIVE},
    {"sign",          UNARY_OPERATION::SIGN},
    {"rint",          UNARY_OPERATION::RINT},
    {"sqrt",          UNARY_OPERATION::SQRT},
    {"square",        UNARY_OPERATION::SQUARE},
    {"reciprocal",    UNARY_OPERATION::RECIPROCAL},
    {"logical_not",   UNARY_OPERATION::LOGICAL_NOT},
    {"isinf",         UNARY_OPERATION::ISINF},
    {"isnan",         UNARY_OPERATION::ISNAN},
    {"isfinite",      UNARY_OPERATION::ISFINITE},
    //{"isnormal",      UNARY_OPERATION::ISNORMAL},  // not a ufunc
    // TODO numpy needs to add isnotinf, isnotnan, isnotfinite
    {"bitwise_not",   UNARY_OPERATION::BITWISE_NOT },
};

// Trigonemtric function mapping
// Includes exp, log functions also (since similar algo class)
// To be completed, add atan2, hypot
static stUFuncToAtop gTrigMapping[] = {
    {"sin",           TRIG_OPERATION::SIN},
    {"cos",           TRIG_OPERATION::COS},
    {"tan",           TRIG_OPERATION::TAN},
    {"arcsin",        TRIG_OPERATION::ASIN},
    {"arccos",        TRIG_OPERATION::ACOS},
    {"arctan",        TRIG_OPERATION::ATAN},
    {"sinh",          TRIG_OPERATION::SINH},
    {"cosh",          TRIG_OPERATION::COSH},
    {"tanh",          TRIG_OPERATION::TANH},
    {"arcsinh",       TRIG_OPERATION::ASINH},
    {"arccosh",       TRIG_OPERATION::ACOSH},
    {"arctanh",       TRIG_OPERATION::ATANH},
    {"cbrt",          TRIG_OPERATION::CBRT},
    {"exp",           TRIG_OPERATION::EXP},
    {"exp2",          TRIG_OPERATION::EXP2},
    {"expm1",         TRIG_OPERATION::EXPM1},
    {"log",           TRIG_OPERATION::LOG},
    {"log2",          TRIG_OPERATION::LOG2},
    {"log10",         TRIG_OPERATION::LOG10},
    {"log1p",         TRIG_OPERATION::LOG1P},
};


// Global table lookup to get to all loops, used by ledger
stOpCategory gOpCategory[OP_CATEGORY::OPCAT_LAST] = {
    {"Binary", sizeof(gBinaryMapping) / sizeof(stUFuncToAtop), OP_CATEGORY::OPCAT_BINARY, gBinaryMapping},
    {"Unary", sizeof(gUnaryMapping) / sizeof(stUFuncToAtop), OP_CATEGORY::OPCAT_UNARY, gUnaryMapping},
    {"Compare", sizeof(gCompareMapping) / sizeof(stUFuncToAtop), OP_CATEGORY::OPCAT_COMPARE, gCompareMapping},
    {"TrigLog", sizeof(gTrigMapping) / sizeof(stUFuncToAtop), OP_CATEGORY::OPCAT_TRIG, gTrigMapping}
};

//--------------------------------------------------------------------
// multithreaded struct used for calling unary op codes
struct UFUNC_CALLBACK {
    union {
        ANY_TWO_FUNC        pBinaryFunc;
        UNARY_FUNC          pUnaryFunc;
        REDUCE_FUNC         pReduceFunc;
        PyUFuncGenericFunction pOldFunc;
    };

    union {
        char* pDataIn1;
        char* pStartVal;    // Used for reduce
    };

    char* pDataIn2;
    char* pDataOut;

    // Keep itemsize in same order as numpy steps
    union {
        int64_t itemSizeIn1;
        int64_t steps[1];
    };

    int64_t itemSizeIn2;
    int64_t itemSizeOut;

    // mysterious innerloop used by numpy for tan
    void* innerloop;
};

struct stUFunc {
    union {
        ANY_TWO_FUNC            pBinaryFunc;
        UNARY_FUNC              pUnaryFunc;
    };

    PyUFuncGenericFunction  pOldFunc;
    REDUCE_FUNC             pReduceFunc;

    // the maximum threads to deploy
    int32_t                 MaxThreads;

    // the minimum number of elements in the array before threading allowed
    int32_t                 MinElementsToThread;
};

// global lookup tables for math opcode enum + dtype enum
stUFunc  g_UFuncLUT[BINARY_OPERATION::BINARY_LAST][ATOP_LAST];
stUFunc  g_CompFuncLUT[COMP_OPERATION::CMP_LAST][ATOP_LAST];
stUFunc  g_UnaryFuncLUT[UNARY_OPERATION::UNARY_LAST][ATOP_LAST];
stUFunc  g_TrigFuncLUT[TRIG_OPERATION::TRIG_LAST][ATOP_LAST];

// set to 0 to disable
stSettings g_Settings = { 1, 0, 0, 0 };

// Macro used just before call a ufunc
#define LEDGER_START()    g_Settings.LedgerEnabled = 0; int64_t ledgerStartTime = __rdtsc();

// Macro used just after ufunc call returns
#define LEDGER_END(_cat_) g_Settings.LedgerEnabled = 1; LedgerRecord(_cat_, ledgerStartTime, (int64_t)__rdtsc(), args, dimensions, steps, innerloop, funcop, atype);


//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static int64_t ReduceThreadCallbackStrided(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    int64_t didSomeWork = 0;
    UFUNC_CALLBACK* Callback = (UFUNC_CALLBACK*)pstWorkerItem->WorkCallbackArg;

    char* pDataIn2 = Callback->pDataIn2;
    char* pDataOut = Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        int64_t inputAdj2 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn2;
        int64_t outputAdj = workBlock * Callback->itemSizeOut;

        //printf("[%d] reduce on %lld with len %lld   block: %lld  itemsize: %lld\n", core, workIndex, lenX, workBlock, Callback->itemSizeIn2);
        Callback->pReduceFunc(pDataIn2 + inputAdj2, pDataOut + outputAdj, Callback->pStartVal, lenX, Callback->itemSizeIn2);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
    }

    return didSomeWork;
}

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static int64_t ReduceThreadCallbackNumpy(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    int64_t didSomeWork = 0;
    UFUNC_CALLBACK* Callback = (UFUNC_CALLBACK*)pstWorkerItem->WorkCallbackArg;

    char* pDataIn2 = Callback->pDataIn2;
    char* pDataOut = Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        int64_t inputAdj2 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn2;
        int64_t outputAdj = workBlock * Callback->itemSizeOut;

        char* args[3];
        npy_intp dimensions[1];
        npy_intp steps[3];

        args[0] = args[2] = pDataOut + outputAdj;
        args[1] = pDataIn2 + inputAdj2;
        dimensions[0] = lenX;
        steps[0] = 0;
        steps[2] = 0;
        steps[1] = Callback->itemSizeIn2;

        // this is also hackish
        // to set the start value, which is overloaded as first element in output value
        switch (Callback->itemSizeOut) {
        case 1:
            *(int8_t*)args[0] = *(int8_t*)(Callback->pStartVal);
            break;
        case 2:
            *(int16_t*)args[0] = *(int16_t*)(Callback->pStartVal);
            break;
        case 4:
            *(int32_t*)args[0] = *(int32_t*)(Callback->pStartVal);
            break;
        case 8:
            *(int64_t*)args[0] = *(int64_t*)(Callback->pStartVal);
            break;
        }

        LOGGING("[%d] numpy reduce on %lld with len %lld   block: %lld  itemsize: %lld\n", core, workIndex, lenX, workBlock, Callback->itemSizeIn2);
        Callback->pOldFunc(args, dimensions, steps, Callback->innerloop);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
    }

    return didSomeWork;
}

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
//  For new vectorized routines
static int64_t BinaryThreadCallbackStrided(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    int64_t didSomeWork = 0;
    const UFUNC_CALLBACK* Callback = (const UFUNC_CALLBACK*)pstWorkerItem->WorkCallbackArg;

    char* pDataIn1 = Callback->pDataIn1;
    char* pDataIn2 = Callback->pDataIn2;
    char* pDataOut = Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        int64_t inputAdj1 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn1;
        int64_t inputAdj2 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn2;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeOut;

        // LOGGING("[%d] working on %lld with len %lld   block: %lld\n", core, workIndex, lenX, workBlock);
        Callback->pBinaryFunc(pDataIn1 + inputAdj1, pDataIn2 + inputAdj2, pDataOut + outputAdj, lenX, Callback->itemSizeIn1, Callback->itemSizeIn2, Callback->itemSizeOut);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        //printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}


//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
//  For older numpy existing routines
static int64_t BinaryThreadCallbackNumpy(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    int64_t didSomeWork = 0;
    const UFUNC_CALLBACK* Callback = (const UFUNC_CALLBACK*)pstWorkerItem->WorkCallbackArg;

    char* pDataIn1 = Callback->pDataIn1;
    char* pDataIn2 = Callback->pDataIn2;
    char* pDataOut = Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        int64_t inputAdj1 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn1;
        int64_t inputAdj2 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn2;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeOut;

        char* args[3] = { pDataIn1 + inputAdj1, pDataIn2 + inputAdj2, pDataOut + outputAdj };
        npy_intp dimensions[3] = { lenX, lenX, lenX };

        LOGGING("[%d] orig numpy working on %lld with len %lld   block: %lld\n", core, workIndex, lenX, workBlock);
        Callback->pOldFunc(args, dimensions, (npy_intp*)(Callback->steps), Callback->innerloop);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        //printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}


//------------------------------------------------------------------------------
// Concurrent callback from multiple threads
// This routine is the multithreaded callback for existing numpy unary loops like abs, sqrt, etc.
static int64_t UnaryThreadCallbackNumpy(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    int64_t didSomeWork = 0;
    const UFUNC_CALLBACK* Callback = (const UFUNC_CALLBACK*)pstWorkerItem->WorkCallbackArg;

    char* pDataIn1 = Callback->pDataIn1;
    char* pDataOut = Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        int64_t inputAdj1 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn1;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeOut;

        char* args[2] = { pDataIn1 + inputAdj1,  pDataOut + outputAdj };
        npy_intp dimensions =  (npy_intp)lenX ;
        const npy_intp steps[2] = { Callback->itemSizeIn1 , Callback->itemSizeOut };

        LOGGING("[%d] working on %lld with len %lld   block: %lld\n", core, workIndex, lenX, workBlock);
        Callback->pOldFunc(args, &dimensions, steps, Callback->innerloop);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        //printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static int64_t UnaryThreadCallbackStrided(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    int64_t didSomeWork = 0;
    const UFUNC_CALLBACK* Callback = (const UFUNC_CALLBACK*)pstWorkerItem->WorkCallbackArg;

    char* pDataIn1 = Callback->pDataIn1;
    char* pDataOut = Callback->pDataOut;
    int64_t lenX;
    int64_t workBlock;

    // As long as there is work to do
    while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0) {

        int64_t inputAdj1 = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeIn1;
        int64_t outputAdj = pstWorkerItem->BlockSize * workBlock * Callback->itemSizeOut;

        // LOGGING("[%d] working on %lld with len %lld   block: %lld\n", core, workIndex, lenX, workBlock);
        Callback->pUnaryFunc(pDataIn1 + inputAdj1, pDataOut + outputAdj, lenX, Callback->itemSizeIn1, Callback->itemSizeOut);

        // Indicate we completed a block
        didSomeWork++;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        //printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}

//============================================================================
// For binary math functions like add, sbutract, multiply.
// 2 inputs and 1 output
static void AtopBinaryMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {

    if (!g_Settings.LedgerEnabled) {
        stUFunc* pstUFunc = &g_UFuncLUT[funcop][atype];
        npy_intp n = dimensions[0];
        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(n);
        LOGGING("called with %d %d   funcp: %p  len:%lld   inputs: %p %p %p  steps: %lld %lld %lld\n", funcop, atype, g_UFuncLUT[funcop][atype].pOldFunc, (long long)n, args[0], args[1], args[2], (long long)steps[0], (long long)steps[1], (long long)steps[2]);

        if (IS_BINARY_REDUCE) {
            // In a numpy binary reduce, the middle array is the real array
            REDUCE_FUNC pReduceFunc = pstUFunc->pReduceFunc;

            LOGGING("pReduce %p   opcode:%d   dtype:%d   %lld %lld %lld %lld\n", pReduceFunc, funcop, atype, (long long)dimensions[0], (long long)steps[0], (long long)steps[1], (long long)steps[2]);
            char* ip2 = args[1];
            char* op1 = args[0];
            if (!pWorkItem) {
                // Not threaded
                if (g_Settings.AtopEnabled && pReduceFunc) {
                    // Call fast vectorized function without any threading
                    pReduceFunc(ip2, op1, op1, n, steps[1]);
                }
                else {
                    // Call the original numpy function without any threading
                    pstUFunc->pOldFunc(args, dimensions, steps, innerloop);
                }
            }
            else {
                // Threaded
                int64_t itemsize = convert_atop_to_itemsize[atype];
                int64_t chunks = 1 + ((n - 1) / THREADER->WORK_ITEM_CHUNK);
                int64_t allocsize = chunks * itemsize;

                // try to alloc on stack for speed
                char* pReduceOfReduce = POSSIBLY_STACK_ALLOC(allocsize);

                UFUNC_CALLBACK stCallback;

                // Also pass in original data out since it is used as starting value
                stCallback.pStartVal = op1;

                // Create a data out for each work chunk
                stCallback.pDataOut = pReduceOfReduce;
                stCallback.pDataIn2 = ip2;
                stCallback.itemSizeIn2 = steps[1];
                stCallback.itemSizeOut = itemsize; // sizeof(T)

                pWorkItem->WorkCallbackArg = &stCallback;

                // Each thread will call this routine with the callbackArg
                if (g_Settings.AtopEnabled && pReduceFunc) {
                    stCallback.pReduceFunc = pReduceFunc;
                    pWorkItem->DoWorkCallback = ReduceThreadCallbackStrided;

                    // This will notify the worker threads of a new work item
                    // most functions are so fast, we do not need more than 4 worker threads
                    THREADER->WorkMain(pWorkItem, n, pstUFunc->MaxThreads);
                    pReduceFunc(pReduceOfReduce, op1, op1, chunks, itemsize);
                }
                else {
                    // A binary reduce for original numpy routine
                    //
                    stCallback.pOldFunc = pstUFunc->pOldFunc;
                    stCallback.innerloop = innerloop;
                    pWorkItem->DoWorkCallback = ReduceThreadCallbackNumpy;

                    // This will notify the worker threads of a new work item
                    // most functions are so fast, we do not need more than 4 worker threads
                    THREADER->WorkMain(pWorkItem, n, pstUFunc->MaxThreads);

                    // Finish it...
                    // Now perform same function over all the threaded results
                    char* args[3];

                    // possible bug -- check how big dimensions is for reduce
                    npy_intp dimensions[1];
                    npy_intp steps[3];

                    args[0] = args[2] = op1;
                    args[1] = pReduceOfReduce;

                    // todo, check if only 1 dimension for reduce
                    dimensions[0] = chunks;
                    steps[0] = 0;
                    steps[2] = 0;
                    steps[1] = itemsize;

                    pstUFunc->pOldFunc(args, dimensions, steps, innerloop);
                }
                // Free if not on the stack
                POSSIBLY_STACK_FREE(allocsize, pReduceOfReduce);
            }
        }
        else {
            // NOT a binary reduce
            ANY_TWO_FUNC pBinaryFunc = pstUFunc->pBinaryFunc;

            char* pInput1 = args[0];
            char* pInput2 = args[1];
            char* pOutput = args[2];

            // This code needs review: this is only an issue if array is contiguous and output is within
            // a cacheline of the input (that is, 32 or 64 bytes) because vector intrinsics process in chunks
            // Check for address overlap where the output memory lies inside input1 or input2
            if ((pOutput > pInput1&& pOutput < (pInput1 + 64)) || (pOutput > pInput2&& pOutput < (pInput2 + 64))) {
                pBinaryFunc = NULL;
            }

            // Check if threading allowed
            if (!pWorkItem) {
                // Threading not allowed
                if (g_Settings.AtopEnabled && pBinaryFunc) {
                    // For a scalar first is1 ==0  or steps[0] ==0
                    // For a scalar second is2 == 0  or steps[1] == 0
                    pBinaryFunc(args[0], args[1], args[2], (int64_t)n, (int64_t)steps[0], (int64_t)steps[1], (int64_t)steps[2]);
                }
                else {
                    // Call the original numpy function without any threading
                    pstUFunc->pOldFunc(args, dimensions, steps, innerloop);
                }
            }
            else {
                // Threading allowed
                UFUNC_CALLBACK stCallback;

                stCallback.pDataIn1 = args[0];
                stCallback.pDataIn2 = args[1];
                stCallback.pDataOut = args[2];
                stCallback.itemSizeIn1 = steps[0];
                stCallback.itemSizeIn2 = steps[1];
                stCallback.itemSizeOut = steps[2];

                if (g_Settings.AtopEnabled && pBinaryFunc) {
                    stCallback.pBinaryFunc = pBinaryFunc;

                    // Each thread will call this routine with the callbackArg
                    pWorkItem->DoWorkCallback = BinaryThreadCallbackStrided;
                }
                else {
                    stCallback.pOldFunc = pstUFunc->pOldFunc;
                    stCallback.innerloop = innerloop;

                    // Each thread will call this routine with the callbackArg
                    pWorkItem->DoWorkCallback = BinaryThreadCallbackNumpy;
                }

                pWorkItem->WorkCallbackArg = &stCallback;

                // This will notify the worker threads of a new work item
                // most functions are so fast, we do not need more than 4 worker threads
                THREADER->WorkMain(pWorkItem, n, pstUFunc->MaxThreads);
            }
        }
        return;
    }

    // Ledger is on, turn it off and call back to ourselves to time it    
    LEDGER_START();
    AtopBinaryMathFunction(args, dimensions, steps, innerloop, funcop, atype);
    LEDGER_END(OP_CATEGORY::OPCAT_BINARY);

};


// For binary math functions like add, sbutract, multiply.
// 2 inputs and 1 output
static void AtopCompareMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    if (!g_Settings.LedgerEnabled) {
        // LOGGING("comparison called with %d %d   funcp: %p  len: %lld\n", funcop, atype, g_CompFuncLUT[funcop][atype].pOldFunc, (long long)dimensions[0]);
        stUFunc* pstUFunc = &g_CompFuncLUT[funcop][atype];
        npy_intp n = dimensions[0];
        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(n);
        ANY_TWO_FUNC pBinaryFunc = pstUFunc->pBinaryFunc;

        // Check if threading allowed
        if (!pWorkItem) {
            // Threading not allowed
            if (g_Settings.AtopEnabled && pBinaryFunc) {
                // For a scalar first is1 ==0  or steps[0] ==0
                // For a scalar second is2 == 0  or steps[1] == 0
                pBinaryFunc(args[0], args[1], args[2], (int64_t)n, (int64_t)steps[0], (int64_t)steps[1], (int64_t)steps[2]);
            }
            else {
                // Call the original numpy function without any threading
                pstUFunc->pOldFunc(args, dimensions, steps, innerloop);
            }
        }
        else {
            // Threading allowed
            UFUNC_CALLBACK stCallback;

            stCallback.pDataIn1 = args[0];
            stCallback.pDataIn2 = args[1];
            stCallback.pDataOut = args[2];
            stCallback.itemSizeIn1 = steps[0];
            stCallback.itemSizeIn2 = steps[1];
            stCallback.itemSizeOut = steps[2];

            if (g_Settings.AtopEnabled && pBinaryFunc) {
                stCallback.pBinaryFunc = pBinaryFunc;

                // Each thread will call this routine with the callbackArg
                pWorkItem->DoWorkCallback = BinaryThreadCallbackStrided;
            }
            else {
                stCallback.pOldFunc = pstUFunc->pOldFunc;
                stCallback.innerloop = innerloop;

                // Each thread will call this routine with the callbackArg
                // Use the original numpy ufunc loop
                pWorkItem->DoWorkCallback = BinaryThreadCallbackNumpy;
            }

            pWorkItem->WorkCallbackArg = &stCallback;

            // This will notify the worker threads of a new work item
            // most functions are so fast, we do not need more than 4 worker threads
            THREADER->WorkMain(pWorkItem, n, pstUFunc->MaxThreads);
        }
        return;
    }

    // Ledger is on, turn it off and call back to ourselves to time it    
    LEDGER_START();
    AtopCompareMathFunction(args, dimensions, steps, innerloop, funcop, atype);
    LEDGER_END(OP_CATEGORY::OPCAT_COMPARE);

};



// For unary math functions like abs, sqrt
// 1 input and 1 output
static void AtopUnaryMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    if (!g_Settings.LedgerEnabled) {
        npy_intp n = dimensions[0];
        stUFunc* pstUFunc = &g_UnaryFuncLUT[funcop][atype];
        UNARY_FUNC pUnaryFunc = pstUFunc->pUnaryFunc;
        LOGGING("unary called with %d %d   funcp: %p  len: %lld  inputs: %p %p  steps: %lld %lld\n", funcop, atype, g_UFuncLUT[funcop][atype].pOldFunc, n, args[0], args[1], (int64_t)steps[0], (int64_t)steps[1]);

        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(n);
        int64_t strideOut = steps[1];
        if (strideOut == 0) {
            pUnaryFunc = NULL;
            //        strideOut = convert_atop_to_itemsize[atype];
            //        if (n != 1) printf("!!!! error unary with no strides but len != 1  %lld\n", n);
        }
        if (!pWorkItem) {
            // Threading not allowed
            if (g_Settings.AtopEnabled && pUnaryFunc) {
                pUnaryFunc(args[0], args[1], (int64_t)n, (int64_t)steps[0], strideOut);
            }
            else {
                // Do it the old way, threading not allowed
                pstUFunc->pOldFunc(args, dimensions, steps, innerloop);
            }
        }
        else {
            // Threading allowed
            UFUNC_CALLBACK stCallback;

            stCallback.pDataIn1 = args[0];
            stCallback.pDataOut = args[1];
            stCallback.itemSizeIn1 = steps[0];
            stCallback.itemSizeOut = strideOut;

            // Each thread will call this routine with the callbackArg
            pWorkItem->WorkCallbackArg = &stCallback;

            if (g_Settings.AtopEnabled && pUnaryFunc) {
                // Call the new replacement routine
                stCallback.pUnaryFunc = pUnaryFunc;
                pWorkItem->DoWorkCallback = UnaryThreadCallbackStrided;
            }
            else {
                // Call the original numpy routine
                stCallback.pOldFunc = pstUFunc->pOldFunc;
                stCallback.innerloop = innerloop;
                pWorkItem->DoWorkCallback = UnaryThreadCallbackNumpy;
            }
            // This will notify the worker threads of a new work item
            // most functions are so fast, we do not need more than 4 worker threads
            THREADER->WorkMain(pWorkItem, n, pstUFunc->MaxThreads);
        }
        return;
    }
    // Ledger is on, turn it off and call back to ourselves to time it    
    LEDGER_START();
    AtopUnaryMathFunction(args, dimensions, steps, innerloop, funcop, atype);
    LEDGER_END(OP_CATEGORY::OPCAT_UNARY);

};



static void AtopTrigMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    if (!g_Settings.LedgerEnabled) {
        npy_intp n = dimensions[0];
        stUFunc* pstUFunc = &g_TrigFuncLUT[funcop][atype];
        UNARY_FUNC pUnaryFunc = pstUFunc->pUnaryFunc;
        //printf("trig called with %d %d   funcp: %p  len: %lld  inputs: %p %p  steps: %lld %lld\n", funcop, atype, pstUFunc->pOldFunc, n, args[0], args[1], (int64_t)steps[0], (int64_t)steps[1]);

        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(n);
        int64_t strideOut = steps[1];
        if (strideOut == 0) {
            pUnaryFunc = NULL;
            //        strideOut = convert_atop_to_itemsize[atype];
            //        if (n != 1) printf("!!!! error unary with no strides but len != 1  %lld\n", n);
        }
        if (!pWorkItem) {
            // Threading not allowed
            if (g_Settings.AtopEnabled && pUnaryFunc) {
                pUnaryFunc(args[0], args[1], (int64_t)n, (int64_t)steps[0], strideOut);
            }
            else {
                // Do it the old way, threading not allowed
                pstUFunc->pOldFunc(args, dimensions, steps, innerloop);
            }
        }
        else {
            // Threading allowed
            UFUNC_CALLBACK stCallback;

            stCallback.pDataIn1 = args[0];
            stCallback.pDataOut = args[1];
            stCallback.itemSizeIn1 = steps[0];
            stCallback.itemSizeOut = strideOut;

            // Each thread will call this routine with the callbackArg
            pWorkItem->WorkCallbackArg = &stCallback;

            if (g_Settings.AtopEnabled && pUnaryFunc) {
                // Call the new replacement routine
                stCallback.pUnaryFunc = pUnaryFunc;
                pWorkItem->DoWorkCallback = UnaryThreadCallbackStrided;
            }
            else {
                // Call the original numpy routine
                stCallback.pOldFunc = pstUFunc->pOldFunc;
                stCallback.innerloop = innerloop;
                pWorkItem->DoWorkCallback = UnaryThreadCallbackNumpy;
            }
            // This will notify the worker threads of a new work item
            // most functions are so fast, we do not need more than 4 worker threads
            THREADER->WorkMain(pWorkItem, n, pstUFunc->MaxThreads);
        }
        return;
    }
    // Ledger is on, turn it off and call back to ourselves to time it    
    LEDGER_START();
    AtopTrigMathFunction(args, dimensions, steps, innerloop, funcop, atype);
    LEDGER_END(OP_CATEGORY::OPCAT_TRIG);

};

// the inclusion of this file is because there is no callback argument
#include "stubs.h"

template <class T>
void add_T(T **args, npy_intp const *dimensions, npy_intp const *steps,
          void *innerloopdata) {
    // steps is in bytes, so cast args to char** to allow strange steps.
    
    if (IS_BINARY_REDUCE) {
        char *iop1 = (char *)args[0]; 
        T io1 = *(T *)iop1; 
        char *ip2 = (char *)args[1]; 
        npy_intp is2 = steps[1]; 
        npy_intp n = dimensions[0]; 
        npy_intp i; 

        for(i = 0; i < n; i++, ip2 += is2)
        {
            io1 += *(T *)ip2;
        }
        *((T *)iop1) = io1;
    }
    else {
        char *ip1 = (char *)args[0];
        char *ip2 = (char *)args[1];
        char *op1 = (char *)args[2];
        // For a scalar first is1 ==0
        // For a scalar second is2 == 0
        npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
        npy_intp n = dimensions[0];
        npy_intp i;

        for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)
        {
            const T in1 = *(T *)ip1;
            const T in2 = *(T *)ip2;
            *(T *)op1 = in1 + in2;
        }
    }
}


extern "C"
PyObject* oldinit(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject* result = NULL;
    PyObject *ufunc = NULL;
    const char * uname = NULL;
    int ret = 0;
    int signature[3] = {NPY_INT32, NPY_INT32, NPY_INT32};
    PyUFuncGenericFunction oldfunc, newfunc;
    // C++ warns on assigning const char * to char *

    if (!PyArg_ParseTuple(args, "s:oldinit", &uname)) {
        return NULL;
    }

    // Initialize numpy's C-API.
    import_array();
    import_umath();

    PyObject *numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot import numpy");
        return NULL;
    }
    ufunc = PyObject_GetAttrString(numpy_module, uname);
    Py_DECREF(numpy_module);
    if (ufunc == NULL || (!PyObject_TypeCheck(ufunc, &PyUFunc_Type))) {
        if (ufunc) Py_XDECREF(ufunc);
        return PyErr_Format(PyExc_TypeError, "func %s must be the name of a ufunc", uname);
    }

    // TODO: parse requested dtype into the templating type
    newfunc = (PyUFuncGenericFunction)add_T<int32_t>;
    ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc,
                                             newfunc, signature, &oldfunc);

    if (ret < 0) {
        PyErr_SetString(PyExc_ValueError, "signature int,int->int not found");
    }
    if (oldfunc == newfunc) {
        result = PyUnicode_FromString("int32,int32->int32 (repeated initialize)");
    }
    else {
        result = PyUnicode_FromString("int32,int32->int32");
    }
    return result;
}


extern "C"
PyObject* newinit(PyObject* self, PyObject* args, PyObject* kwargs) {
    int dtypes[] = { NPY_BOOL, NPY_INT8, NPY_UINT8,  NPY_INT16, NPY_UINT16,  NPY_INT32, NPY_UINT32,  NPY_INT64, NPY_UINT64, NPY_FLOAT32, NPY_FLOAT64 };
    //int dtypes[] = {  NPY_INT32,  NPY_INT64};

    // Init atop: array threading operations
    if (atop_init() && g_avx2) {
        memset(g_UFuncLUT, 0, sizeof(g_UFuncLUT));

        // Initialize numpy's C-API.
        import_array();
        import_umath();
        PyObject* numpy_module = PyImport_ImportModule("numpy");
        if (numpy_module == NULL) {
            return NULL;
        }

        // call np.setbufsize()
        // note: could be done in __init__
        PyObject* setbufsize = PyObject_GetAttrString(numpy_module, "setbufsize");
        if (setbufsize && PyCallable_Check(setbufsize)) {
            PyObject* buffersize = PyTuple_New(1);
            PyTuple_SetItem(buffersize, 0, PyLong_FromLongLong(8192 * 1024));
            PyObject_CallObject(setbufsize, buffersize);
            Py_XDECREF(buffersize);
        }

        // Loop over all binary ufuncs we want to replace
        int64_t num_ufuncs = sizeof(gBinaryMapping) / sizeof(stUFuncToAtop);
        for (int64_t i = 0; i < num_ufuncs; i++) {
            PyObject* result = NULL;
            PyObject* ufunc = NULL;
            const char* ufunc_name = gBinaryMapping[i].str_ufunc_name;
            int atop = gBinaryMapping[i].atop_op;

            ufunc = PyObject_GetAttrString(numpy_module, ufunc_name);

            if (ufunc == NULL) {
                Py_XDECREF(ufunc);
                return PyErr_Format(PyExc_TypeError, "func %s must be the name of a ufunc", ufunc_name);
            }

            //printf("taking over func %s\n", ufunc_name);

            // Loop over all dtypes we support for the ufunc
            int64_t num_dtypes = sizeof(dtypes) / sizeof(int);
            for (int64_t j = 0; j < num_dtypes; j++) {
                PyUFuncGenericFunction oldFunc;
                int dtype = dtypes[j];
                int signature[3] = { dtype, dtype, dtype };

                int atype = convert_dtype_to_atop[dtype];

                signature[2] = -1;
                ANY_TWO_FUNC pBinaryFunc = GetSimpleMathOpFast(atop, atype, atype, &signature[2]);
                REDUCE_FUNC  pReduceFunc = GetReduceMathOpFast(atop, atype);

                if (signature[2] != -1) {
                    signature[2] = convert_atop_to_dtype[signature[2]];

                    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncGenericLUT[atop][atype], signature, &oldFunc);

                    if (ret < 0) {
                        return PyErr_Format(PyExc_TypeError, "Math failed with %d. func %s must be the name of a ufunc.  atop:%d   atype:%d  sigs:%d, %d, %d", ret, ufunc_name, atop, atype, signature[0], signature[1], signature[2]);
                    }

                    stUFunc* pstUFunc = &g_UFuncLUT[atop][atype];
                    // Store the new function to call and the previous ufunc
                    pstUFunc->pOldFunc = oldFunc;
                    pstUFunc->pBinaryFunc = pBinaryFunc;
                    pstUFunc->pReduceFunc = pReduceFunc;
                    pstUFunc->MaxThreads = 4;
                }
            }
        }

        // Loop over all compare ufuncs we want to replace
        num_ufuncs = sizeof(gCompareMapping) / sizeof(stUFuncToAtop);
        for (int64_t i = 0; i < num_ufuncs; i++) {
            PyObject* result = NULL;
            PyObject* ufunc = NULL;
            const char* ufunc_name = gCompareMapping[i].str_ufunc_name;
            int atop = gCompareMapping[i].atop_op;

            ufunc = PyObject_GetAttrString(numpy_module, ufunc_name);

            if (ufunc == NULL) {
                Py_XDECREF(ufunc);
                return PyErr_Format(PyExc_TypeError, "func %s must be the name of a ufunc", ufunc_name);
            }

            // Loop over all dtypes we support for the ufunc
            int64_t num_dtypes = sizeof(dtypes) / sizeof(int);
            for (int64_t j = 0; j < num_dtypes; j++) {
                PyUFuncGenericFunction oldFunc;
                int dtype = dtypes[j];
                int signature[3] = { dtype, dtype, dtype };

                int atype = convert_dtype_to_atop[dtype];

                ANY_TWO_FUNC pBinaryFunc = GetComparisonOpFast(atop, atype, atype, &signature[2]);
                signature[2] = convert_atop_to_dtype[signature[2]];

                int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncCompareLUT[atop][atype], signature, &oldFunc);

                if (ret < 0) {
                    return PyErr_Format(PyExc_TypeError, "Comparison failed with %d. func %s must be the name of a ufunc.  atop:%d   atype:%d  sigs:%d, %d, %d", ret, ufunc_name, atop, atype, signature[0], signature[1], signature[2]);
                }

                stUFunc* pstUFunc = &g_CompFuncLUT[atop][atype];

                // Store the new function to call and the previous ufunc
                pstUFunc->pOldFunc = oldFunc;
                pstUFunc->pBinaryFunc = pBinaryFunc;
                pstUFunc->MaxThreads = 4;
            }
        }

        // Loop over all unary ufuncs we want to replace
        num_ufuncs = sizeof(gUnaryMapping) / sizeof(stUFuncToAtop);
        for (int64_t i = 0; i < num_ufuncs; i++) {
            PyObject* result = NULL;
            PyObject* ufunc = NULL;
            const char* ufunc_name = gUnaryMapping[i].str_ufunc_name;
            int atop = gUnaryMapping[i].atop_op;

            ufunc = PyObject_GetAttrString(numpy_module, ufunc_name);

            if (ufunc == NULL) {
                Py_XDECREF(ufunc);
                return PyErr_Format(PyExc_TypeError, "func %s must be the name of a ufunc", ufunc_name);
            }

            // Loop over all dtypes we support for the ufunc
            int64_t num_dtypes = sizeof(dtypes) / sizeof(int);
            for (int64_t j = 0; j < num_dtypes; j++) {
                PyUFuncGenericFunction oldFunc;
                int dtype = dtypes[j];
                int signature[3] = { dtype, dtype, dtype };

                int atype = convert_dtype_to_atop[dtype];

                // For unary it only has a signature of 2
                signature[1] = -1;
                UNARY_FUNC pUnaryFunc = GetUnaryOpFast(atop, atype, &signature[1]);

                if (signature[1] != -1) {
                    signature[1] = convert_atop_to_dtype[signature[1]];

                    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncUnaryLUT[atop][atype], signature, &oldFunc);

                    if (ret < 0) {
                        return PyErr_Format(PyExc_TypeError, "Unary failed with %d. func %s must be the name of a ufunc.  atop:%d   atype:%d  sigs:%d, %d, %d", ret, ufunc_name, atop, atype, signature[0], signature[1], signature[2]);
                    }
                    stUFunc* pstUFunc = &g_UnaryFuncLUT[atop][atype];
                    // Store the new function to call and the previous ufunc
                    pstUFunc->pOldFunc = oldFunc;
                    pstUFunc->pUnaryFunc = pUnaryFunc;
                    pstUFunc->MaxThreads = 4;
                }
            }
        }


        // Loop over all trig ufuncs we want to replace
        int trig_dtypes[] = { NPY_FLOAT32, NPY_FLOAT64 };
        num_ufuncs = sizeof(gTrigMapping) / sizeof(stUFuncToAtop);
        for (int64_t i = 0; i < num_ufuncs; i++) {
            PyObject* result = NULL;
            PyObject* ufunc = NULL;
            const char* ufunc_name = gTrigMapping[i].str_ufunc_name;
            int atop = gTrigMapping[i].atop_op;

            ufunc = PyObject_GetAttrString(numpy_module, ufunc_name);

            if (ufunc == NULL) {
                Py_XDECREF(ufunc);
                return PyErr_Format(PyExc_TypeError, "func %s must be the name of a ufunc", ufunc_name);
            }

            // Loop over all dtypes we support for the ufunc
            int64_t num_dtypes = sizeof(trig_dtypes) / sizeof(int);
            for (int64_t j = 0; j < num_dtypes; j++) {
                PyUFuncGenericFunction oldFunc;
                int dtype = trig_dtypes[j];
                int signature[3] = { dtype, dtype, dtype };

                int atype = convert_dtype_to_atop[dtype];

                // For unary it only has a signature of 2
                UNARY_FUNC pUnaryFunc = NULL;

                // TODO: Call atop's fast log library
                // GetLogOpFast(atop, atype, &signature[1]);

                if (!pUnaryFunc) {
                    pUnaryFunc = GetTrigOpFast(atop, atype, &signature[1]);
                }

                // TODO: trig operations on ints can be internally upcast
                //if (!pUnaryFunc) {
                //    pUnaryFunc = GetTrigOpSlow(atop, atype, &signature[1]);
                //}
                signature[1] = convert_atop_to_dtype[signature[1]];

                // Even if pUnaryFunc is NULL, still hook it since we can thread it
                int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncTrigLUT[atop][atype], signature, &oldFunc);

                if (ret < 0) {
                    return PyErr_Format(PyExc_TypeError, "Trig failed with %d. func %s must be the name of a ufunc.  atop:%d   atype:%d  sigs:%d, %d, %d", ret, ufunc_name, atop, atype, signature[0], signature[1], signature[2]);
                }
                stUFunc* pstUFunc = &g_TrigFuncLUT[atop][atype];
                // Store the new function to call and the previous ufunc
                pstUFunc->pOldFunc = oldFunc;

                // NULL allowed here, it will use the default
                pstUFunc->pUnaryFunc = pUnaryFunc;
                pstUFunc->MaxThreads = 5;
            }
        }

        LedgerInit();

        RETURN_NONE;
    }
    return PyErr_Format(PyExc_ImportError, "atop was either already loaded or failed to load");
}

extern "C"
PyObject * atop_enable(PyObject * self, PyObject * args) {
    g_Settings.AtopEnabled = TRUE;
    RETURN_NONE;
}

extern "C"
PyObject * atop_disable(PyObject * self, PyObject * args) {
    g_Settings.AtopEnabled = FALSE;
    RETURN_NONE;
}

extern "C"
PyObject* atop_isenabled(PyObject* self, PyObject* args) {
    if (g_Settings.AtopEnabled) {
        RETURN_TRUE;
    }
    RETURN_FALSE;
}

extern "C"
PyObject * thread_enable(PyObject * self, PyObject * args) {
    if (THREADER) THREADER->NoThreading= FALSE;
    RETURN_NONE;
}

extern "C"
PyObject * thread_disable(PyObject * self, PyObject * args) {
    if (THREADER) THREADER->NoThreading = TRUE;
    RETURN_NONE;
}

extern "C"
PyObject * thread_isenabled(PyObject * self, PyObject * args) {
    if (THREADER) {
        if (!THREADER->NoThreading) {
            RETURN_TRUE;
        }
    }
    RETURN_FALSE;
}

// Returns previous val
extern "C"
PyObject * thread_setworkers(PyObject * self, PyObject * args) {
    if (THREADER) {
        int workers = 0;
        if (!PyArg_ParseTuple(args, "i:thread_setworkers", &workers)) {
            return NULL;
        }
        int previousVal = THREADER->SetFutexWakeup(workers);
        return PyLong_FromLong((long)previousVal);
    }
    // error
    RETURN_NONE;
}

extern "C"
PyObject * thread_getworkers(PyObject * self, PyObject * args) {
    if (THREADER) {
        int previousVal = THREADER->GetFutexWakeup();
        return PyLong_FromLong((long)previousVal);
    }
    RETURN_NONE;
}


extern "C"
PyObject * cpustring(PyObject * self, PyObject * args) {
    // threading collects the cpu string
    if (THREADER) return PyUnicode_FromString(THREADER->CPUString);
    RETURN_NONE;
}


extern "C"
PyObject * ledger_enable(PyObject * self, PyObject * args) {
    g_Settings.LedgerEnabled = TRUE;
    RETURN_NONE;
}

extern "C"
PyObject * ledger_disable(PyObject * self, PyObject * args) {
    g_Settings.LedgerEnabled = FALSE;
    RETURN_NONE;
}

extern "C"
PyObject * ledger_isenabled(PyObject * self, PyObject * args) {
    if (g_Settings.LedgerEnabled) {
        RETURN_TRUE;
    }
    RETURN_FALSE;
}

extern "C"
PyObject * ledger_info(PyObject * self, PyObject * args) {
    RETURN_NONE;
}

extern "C"
PyObject * recycler_enable(PyObject * self, PyObject * args) {
    g_Settings.RecyclerEnabled = TRUE;
    RETURN_NONE;
}

extern "C"
PyObject * recycler_disable(PyObject * self, PyObject * args) {
    g_Settings.RecyclerEnabled = FALSE;
    RETURN_NONE;
}

extern "C"
PyObject * recycler_isenabled(PyObject * self, PyObject * args) {
    if (g_Settings.RecyclerEnabled) {
        RETURN_TRUE;
    }
    RETURN_FALSE;
}

extern "C"
PyObject * recycler_info(PyObject * self, PyObject * args) {
    RETURN_NONE;
}
