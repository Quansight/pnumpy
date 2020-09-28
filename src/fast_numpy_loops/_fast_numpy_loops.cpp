
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include <stdint.h>
#include <stdio.h>
#include "../atop/atop.h"
#include "../atop/threads.h"

#define RETURN_NONE Py_INCREF(Py_None); return Py_None;

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

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

struct stUFuncToAtop {
    const char*     str_ufunc_name;
    const int       atop_op;
};

// Binary function mapping
static stUFuncToAtop gBinaryMapping[]={
    {"add",           MATH_OPERATION::ADD},
    {"subtract",      MATH_OPERATION::SUB } };

// Compare function mapping
static stUFuncToAtop gCompareMapping[]={
    {"equal",         COMP_OPERATION::CMP_EQ},
    {"not_equal",     COMP_OPERATION::CMP_NE},
    {"greater",       COMP_OPERATION::CMP_GT},
    {"greater_equal", COMP_OPERATION::CMP_GTE},
    {"less",          COMP_OPERATION::CMP_LT},
    {"less_equal",    COMP_OPERATION::CMP_LTE } };


//--------------------------------------------------------------------
// multithreaded struct used for calling unary op codes
struct COMPARE_CALLBACK {
    union {
        ANY_TWO_FUNC        pBinaryFunc;
        UNARY_FUNC_STRIDED  pUnaryCallbackStrided;
    };

    char* pDataIn1;
    char* pDataIn2;
    char* pDataOut;

    int64_t itemSizeIn1;
    int64_t itemSizeIn2;
    int64_t itemSizeOut;
    int32_t scalarmode;
};

struct stUFunc {
    union {
        ANY_TWO_FUNC            pBinaryFunc;
        UNARY_FUNC              pUnaryFunc;
    };

    PyUFuncGenericFunction  pOldFunc;
    int32_t                 MaxThreads;
    int32_t                 Reserved;
};

// global lookup tables for math opcode enum + dtype enum
stUFunc  g_UFuncLUT[MATH_OPERATION::MATH_LAST][ATOP_LAST];
stUFunc  g_CompFuncLUT[COMP_OPERATION::CMP_LAST][ATOP_LAST];

// set to 0 to disable
int32_t  g_AtopEnabled = 1;

// For binary math functions like add, sbutract, multiply.
// 2 inputs and 1 output
void AtopBinaryMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    //LOGGING("called with %d %d   funcp: %p\n", funcop, atype, g_UFuncLUT[funcop][atype].pOldFunc);
    if (g_AtopEnabled) {

        if (IS_BINARY_REDUCE) {
            g_UFuncLUT[funcop][atype].pOldFunc(args, dimensions, steps, innerloop);
        }
        else {
            char* ip1 = (char*)args[0];
            char* ip2 = (char*)args[1];
            char* op1 = (char*)args[2];
            // For a scalar first is1 ==0
            // For a scalar second is2 == 0
            npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
            npy_intp n = dimensions[0];
            g_UFuncLUT[funcop][atype].pBinaryFunc(ip1, ip2, op1, (int64_t)n, is1 == 0 ? SCALAR_MODE::FIRST_ARG_SCALAR : is2 == 0 ? SCALAR_MODE::SECOND_ARG_SCALAR : SCALAR_MODE::NO_SCALARS);

        }
    }
    else {
        g_UFuncLUT[funcop][atype].pOldFunc(args, dimensions, steps, innerloop);
    }
};

//------------------------------------------------------------------------------
//  Concurrent callback from multiple threads
static BOOL CompareThreadCallbackStrided(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, int64_t workIndex) {
    BOOL didSomeWork = FALSE;
    COMPARE_CALLBACK* Callback = (COMPARE_CALLBACK*)pstWorkerItem->WorkCallbackArg;

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
        Callback->pBinaryFunc(pDataIn1 + inputAdj1, pDataIn2 + inputAdj2, pDataOut + outputAdj, lenX, Callback->scalarmode);

        // Indicate we completed a block
        didSomeWork = TRUE;

        // tell others we completed this work block
        pstWorkerItem->CompleteWorkBlock();
        //printf("|%d %d", core, (int)workBlock);
    }

    return didSomeWork;
}


// For binary math functions like add, sbutract, multiply.
// 2 inputs and 1 output
void AtopCompareMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    // LOGGING("comparison called with %d %d   funcp: %p  len: %lld\n", funcop, atype, g_CompFuncLUT[funcop][atype].pOldFunc, (long long)dimensions[0]);

    if (g_AtopEnabled) {
        ANY_TWO_FUNC pBinaryFunc = g_CompFuncLUT[funcop][atype].pBinaryFunc;
        int32_t      maxThreads = g_CompFuncLUT[funcop][atype].MaxThreads;

        char* ip1 = (char*)args[0];
        char* ip2 = (char*)args[1];
        char* op1 = (char*)args[2];

        // For a scalar first is1 ==0
        // For a scalar second is2 == 0
        npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
        int64_t n = (int64_t)dimensions[0];
        int scalarmode = is1 == 0 ? SCALAR_MODE::FIRST_ARG_SCALAR : is2 == 0 ? SCALAR_MODE::SECOND_ARG_SCALAR : SCALAR_MODE::NO_SCALARS;

        stMATH_WORKER_ITEM* pWorkItem = THREADER->GetWorkItem(n);

        if (pWorkItem == NULL) {

            // Threading not allowed for this work item, call it directly from main thread
            pBinaryFunc(ip1, ip2, op1, n, scalarmode);
        }
        else {
            COMPARE_CALLBACK stCallback;

            // Each thread will call this routine with the callbackArg
            pWorkItem->DoWorkCallback = CompareThreadCallbackStrided;
            pWorkItem->WorkCallbackArg = &stCallback;

            stCallback.pBinaryFunc = pBinaryFunc;
            stCallback.pDataOut = op1;
            stCallback.pDataIn1 = ip1;
            stCallback.pDataIn2 = ip2;
            stCallback.itemSizeIn1 = is1;
            stCallback.itemSizeIn2 = is2;
            stCallback.itemSizeOut = os1;
            stCallback.scalarmode = scalarmode;

            // This will notify the worker threads of a new work item
            // most functions are so fast, we do not need more than 4 worker threads
            THREADER->WorkMain(pWorkItem, n, maxThreads);
        }

    }
    else {
        g_CompFuncLUT[funcop][atype].pOldFunc(args, dimensions, steps, innerloop);

    }
};


// For unary math functions like abs, sqrt
// 1 input and 1 output
void AtopUnaryMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    char* ip1 = (char*)args[0];
    char* op1 = (char*)args[1];
    npy_intp is1 = steps[0], os1 = steps[1];
    npy_intp n = dimensions[0];
    //g_UFuncLUT[funcop][atype].pBinaryFunc(ip1, ip2, op1, (int64_t)n, is1 == 0 ? SCALAR_MODE::FIRST_ARG_SCALAR : is2 == 0 ? SCALAR_MODE::SECOND_ARG_SCALAR : SCALAR_MODE::NO_SCALARS);

    
}


// Ugly macro exapnsion to handle which ufunc
// TODO: the ufunc callback needs to use enums
#define DEF_BINARY_USTUB(_FUNC_, _ATYPE_) void F##_ATYPE_##_FUNC_(char **args, const npy_intp *dimensions, const npy_intp *steps, void*innerloop) { \
    return AtopBinaryMathFunction(args, dimensions, steps, innerloop, _FUNC_, _ATYPE_);}

#define DEF_BINARY_USTUB_EXPAND(_FUNC_) \
    DEF_BINARY_USTUB(_FUNC_, ATOP_BOOL); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_INT8); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_UINT8); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_INT16); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_UINT16); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_INT32); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_UINT32); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_INT64); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_UINT64); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_INT128); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_UINT128); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_FLOAT); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_DOUBLE); \
    DEF_BINARY_USTUB(_FUNC_, ATOP_LONGDOUBLE);

// For however many functions there are
DEF_BINARY_USTUB_EXPAND(0)
DEF_BINARY_USTUB_EXPAND(1)
DEF_BINARY_USTUB_EXPAND(2)
DEF_BINARY_USTUB_EXPAND(3)
DEF_BINARY_USTUB_EXPAND(4)
DEF_BINARY_USTUB_EXPAND(5)
DEF_BINARY_USTUB_EXPAND(6)
DEF_BINARY_USTUB_EXPAND(7)
DEF_BINARY_USTUB_EXPAND(8)
DEF_BINARY_USTUB_EXPAND(9)
DEF_BINARY_USTUB_EXPAND(10)
DEF_BINARY_USTUB_EXPAND(11)
DEF_BINARY_USTUB_EXPAND(12)
DEF_BINARY_USTUB_EXPAND(13)
DEF_BINARY_USTUB_EXPAND(14)
DEF_BINARY_USTUB_EXPAND(15)
DEF_BINARY_USTUB_EXPAND(16)
DEF_BINARY_USTUB_EXPAND(17)
DEF_BINARY_USTUB_EXPAND(18)
DEF_BINARY_USTUB_EXPAND(19)

#define DEF_BINARY_USTUB_NAME(_FUNC_) \
    FATOP_BOOL##_FUNC_, \
    FATOP_INT8##_FUNC_, \
    FATOP_UINT8##_FUNC_, \
    FATOP_INT16##_FUNC_, \
    FATOP_UINT16##_FUNC_, \
    FATOP_INT32##_FUNC_, \
    FATOP_UINT32##_FUNC_, \
    FATOP_INT64##_FUNC_, \
    FATOP_UINT64##_FUNC_, \
    FATOP_INT128##_FUNC_, \
    FATOP_UINT128##_FUNC_, \
    FATOP_FLOAT##_FUNC_, \
    FATOP_DOUBLE##_FUNC_, \
    FATOP_LONGDOUBLE##_FUNC_ 

PyUFuncGenericFunction g_UFuncGenericLUT[MATH_OPERATION::MATH_LAST][ATOP_LAST] =
{
{DEF_BINARY_USTUB_NAME(0)},
{DEF_BINARY_USTUB_NAME(1)},
{DEF_BINARY_USTUB_NAME(2)},
{DEF_BINARY_USTUB_NAME(3)},
{DEF_BINARY_USTUB_NAME(4)},
{DEF_BINARY_USTUB_NAME(5)},
{DEF_BINARY_USTUB_NAME(6)},
{DEF_BINARY_USTUB_NAME(7)},
{DEF_BINARY_USTUB_NAME(8)},
{DEF_BINARY_USTUB_NAME(9)},
{DEF_BINARY_USTUB_NAME(10)},
{DEF_BINARY_USTUB_NAME(11)},
{DEF_BINARY_USTUB_NAME(12)},
{DEF_BINARY_USTUB_NAME(13)},
{DEF_BINARY_USTUB_NAME(14)},
{DEF_BINARY_USTUB_NAME(15)},
{DEF_BINARY_USTUB_NAME(16)},
{DEF_BINARY_USTUB_NAME(17)},
{DEF_BINARY_USTUB_NAME(18)},
{DEF_BINARY_USTUB_NAME(19)},
};


#define DEF_COMP_USTUB(_FUNC_, _ATYPE_) void COMPF##_ATYPE_##_FUNC_(char **args, const npy_intp *dimensions, const npy_intp *steps, void*innerloop) { \
    return AtopCompareMathFunction(args, dimensions, steps, innerloop, _FUNC_, _ATYPE_);}

#define DEF_COMP_USTUB_EXPAND(_FUNC_) \
    DEF_COMP_USTUB(_FUNC_, ATOP_BOOL); \
    DEF_COMP_USTUB(_FUNC_, ATOP_INT8); \
    DEF_COMP_USTUB(_FUNC_, ATOP_UINT8); \
    DEF_COMP_USTUB(_FUNC_, ATOP_INT16); \
    DEF_COMP_USTUB(_FUNC_, ATOP_UINT16); \
    DEF_COMP_USTUB(_FUNC_, ATOP_INT32); \
    DEF_COMP_USTUB(_FUNC_, ATOP_UINT32); \
    DEF_COMP_USTUB(_FUNC_, ATOP_INT64); \
    DEF_COMP_USTUB(_FUNC_, ATOP_UINT64); \
    DEF_COMP_USTUB(_FUNC_, ATOP_INT128); \
    DEF_COMP_USTUB(_FUNC_, ATOP_UINT128); \
    DEF_COMP_USTUB(_FUNC_, ATOP_FLOAT); \
    DEF_COMP_USTUB(_FUNC_, ATOP_DOUBLE); \
    DEF_COMP_USTUB(_FUNC_, ATOP_LONGDOUBLE);

// For however many functions there are
DEF_COMP_USTUB_EXPAND(0)
DEF_COMP_USTUB_EXPAND(1)
DEF_COMP_USTUB_EXPAND(2)
DEF_COMP_USTUB_EXPAND(3)
DEF_COMP_USTUB_EXPAND(4)
DEF_COMP_USTUB_EXPAND(5)

#define DEF_COMP_USTUB_NAME(_FUNC_) \
    COMPFATOP_BOOL##_FUNC_, \
    COMPFATOP_INT8##_FUNC_, \
    COMPFATOP_UINT8##_FUNC_, \
    COMPFATOP_INT16##_FUNC_, \
    COMPFATOP_UINT16##_FUNC_, \
    COMPFATOP_INT32##_FUNC_, \
    COMPFATOP_UINT32##_FUNC_, \
    COMPFATOP_INT64##_FUNC_, \
    COMPFATOP_UINT64##_FUNC_, \
    COMPFATOP_INT128##_FUNC_, \
    COMPFATOP_UINT128##_FUNC_, \
    COMPFATOP_FLOAT##_FUNC_, \
    COMPFATOP_DOUBLE##_FUNC_, \
    COMPFATOP_LONGDOUBLE##_FUNC_ 

PyUFuncGenericFunction g_UFuncCompareLUT[COMP_OPERATION::CMP_LAST][ATOP_LAST] =
{
{DEF_COMP_USTUB_NAME(0)},
{DEF_COMP_USTUB_NAME(1)},
{DEF_COMP_USTUB_NAME(2)},
{DEF_COMP_USTUB_NAME(3)},
{DEF_COMP_USTUB_NAME(4)},
{DEF_COMP_USTUB_NAME(5)}};


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
        printf("hello %lld   steps2:%lld  \n", (long long)n, (long long)is2);

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
        printf("hello %lld   steps1:%lld   steps2:%lld  \n", (long long)n, (long long)is1, (long long)is2);
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

    if (!PyArg_ParseTuple(args, "s:initialize", &uname)) {
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

            // Loop over all dtypes we support for the ufunc
            int64_t num_dtypes = sizeof(dtypes) / sizeof(int);
            for (int64_t j = 0; j < num_dtypes; j++) {
                PyUFuncGenericFunction oldFunc;
                int dtype = dtypes[j];
                int signature[3] = { dtype, dtype, dtype };

                int atype = convert_dtype_to_atop[dtype];

                ANY_TWO_FUNC pBinaryFunc = GetSimpleMathOpFast(atop, atype, atype, &signature[2]);
                signature[2] = convert_atop_to_dtype[signature[2]];

                if (pBinaryFunc) {
                    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncGenericLUT[atop][atype], signature, &oldFunc);

                    if (ret < 0) {
                        return PyErr_Format(PyExc_TypeError, "Math failed with %d. func %s must be the name of a ufunc.  atop:%d   atype:%d  sigs:%d, %d, %d", ret, ufunc_name, atop, atype, signature[0], signature[1], signature[2]);
                    }

                    // Store the new function to call and the previous ufunc
                    g_UFuncLUT[atop][atype].pOldFunc = oldFunc;
                    g_UFuncLUT[atop][atype].pBinaryFunc = pBinaryFunc;
                    g_UFuncLUT[atop][atype].MaxThreads = 4;
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

                if (pBinaryFunc) {
                    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncCompareLUT[atop][atype], signature, &oldFunc);

                    if (ret < 0) {
                        return PyErr_Format(PyExc_TypeError, "Comparison failed with %d. func %s must be the name of a ufunc.  atop:%d   atype:%d  sigs:%d, %d, %d", ret, ufunc_name, atop, atype, signature[0], signature[1], signature[2]);
                    }

                    // Store the new function to call and the previous ufunc
                    g_CompFuncLUT[atop][atype].pOldFunc = oldFunc;
                    g_CompFuncLUT[atop][atype].pBinaryFunc = pBinaryFunc;
                    g_CompFuncLUT[atop][atype].MaxThreads = 4;
                }
            }
        }

        //printf("going for abs\n");
        //PyObject* ufunc = PyObject_GetAttrString(numpy_module, "abs");
        //if (ufunc) {
        //    PyUFuncGenericFunction oldFunc;
        //    int signature[2] = { NPY_FLOAT, NPY_INT32 };
        //    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, NULL, signature, &oldFunc);
        //    printf("got abs %d\n" ,ret);

        //    if (ret < 0) {
        //        return PyErr_Format(PyExc_TypeError, "Failed with %d. func %s must be the name of a ufunc", ret, "abs");
        //    }

        //}

        RETURN_NONE;
    }
    return PyErr_Format(PyExc_ImportError, "atop was either already loaded or failed to load");
}

extern "C"
PyObject * enable(PyObject * self, PyObject * args) {
    g_AtopEnabled = TRUE;
    RETURN_NONE;
}

extern "C"
PyObject * disable(PyObject * self, PyObject * args) {
    g_AtopEnabled = FALSE;
    RETURN_NONE;
}

extern "C"
PyObject* isenabled(PyObject* self, PyObject* args) {
    if (g_AtopEnabled) {
        Py_XINCREF(Py_True);
        return Py_True;
    }
    Py_XINCREF(Py_False);
    return Py_False;
}

extern "C"
PyObject * cpustring(PyObject * self, PyObject * args) {
    return PyUnicode_FromString(THREADER->CPUString);
}
