
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include <stdint.h>
#include <stdio.h>
#include "../atop/atop.h"

#define RETURN_NONE Py_INCREF(Py_None); return Py_None;

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

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

struct stUFunc {
    ANY_TWO_FUNC            pTwoFunc;
    PyUFuncGenericFunction  pOldFunc;
};

// 
stUFunc  g_UFuncLUT[MATH_OPERATION::ADD][ATOP_LAST];

void AtopBinaryMathFunction(char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype) {
    //LOGGING("called with %d %d   funcp: %p\n", funcop, atype, g_UFuncLUT[funcop][atype].pOldFunc);

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
        g_UFuncLUT[funcop][atype].pTwoFunc(ip1, ip2, op1, (int64_t)n, is1 == 0 ? SCALAR_MODE::FIRST_ARG_SCALAR : is2 == 0 ? SCALAR_MODE::SECOND_ARG_SCALAR : SCALAR_MODE::NO_SCALARS);

    }
};

// Ugly macro exapnsion to handle which ufunc
// TODO: the ufunc callback needs to use enums
#define DEF_GEN_USTUB(_FUNC_, _ATYPE_) void F##_ATYPE_##_FUNC_(char **args, const npy_intp *dimensions, const npy_intp *steps, void*innerloop) { \
    return AtopBinaryMathFunction(args, dimensions, steps, innerloop, _FUNC_, _ATYPE_);}

#define DEF_GEN_USTUB_EXPAND(_FUNC_) \
    DEF_GEN_USTUB(_FUNC_, ATOP_BOOL); \
    DEF_GEN_USTUB(_FUNC_, ATOP_INT8); \
    DEF_GEN_USTUB(_FUNC_, ATOP_UINT8); \
    DEF_GEN_USTUB(_FUNC_, ATOP_INT16); \
    DEF_GEN_USTUB(_FUNC_, ATOP_UINT16); \
    DEF_GEN_USTUB(_FUNC_, ATOP_INT32); \
    DEF_GEN_USTUB(_FUNC_, ATOP_UINT32); \
    DEF_GEN_USTUB(_FUNC_, ATOP_INT64); \
    DEF_GEN_USTUB(_FUNC_, ATOP_UINT64); \
    DEF_GEN_USTUB(_FUNC_, ATOP_INT128); \
    DEF_GEN_USTUB(_FUNC_, ATOP_UINT128); \
    DEF_GEN_USTUB(_FUNC_, ATOP_FLOAT); \
    DEF_GEN_USTUB(_FUNC_, ATOP_DOUBLE); \
    DEF_GEN_USTUB(_FUNC_, ATOP_LONGDOUBLE);

// For however many functions there are
DEF_GEN_USTUB_EXPAND(0)
DEF_GEN_USTUB_EXPAND(1)
DEF_GEN_USTUB_EXPAND(2)
DEF_GEN_USTUB_EXPAND(3)
DEF_GEN_USTUB_EXPAND(4)
DEF_GEN_USTUB_EXPAND(5)
DEF_GEN_USTUB_EXPAND(6)
DEF_GEN_USTUB_EXPAND(7)
DEF_GEN_USTUB_EXPAND(8)
DEF_GEN_USTUB_EXPAND(9)
DEF_GEN_USTUB_EXPAND(10)
DEF_GEN_USTUB_EXPAND(11)
DEF_GEN_USTUB_EXPAND(12)
DEF_GEN_USTUB_EXPAND(13)
DEF_GEN_USTUB_EXPAND(14)
DEF_GEN_USTUB_EXPAND(15)
DEF_GEN_USTUB_EXPAND(16)
DEF_GEN_USTUB_EXPAND(17)
DEF_GEN_USTUB_EXPAND(18)
DEF_GEN_USTUB_EXPAND(19)

#define DEF_GEN_USTUB_NAME(_FUNC_) \
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

PyUFuncGenericFunction g_UFuncGenericLUT[MATH_OPERATION::LAST][ATOP_LAST] =
{
{DEF_GEN_USTUB_NAME(0)},
{DEF_GEN_USTUB_NAME(1)},
{DEF_GEN_USTUB_NAME(2)},
{DEF_GEN_USTUB_NAME(3)},
{DEF_GEN_USTUB_NAME(4)},
{DEF_GEN_USTUB_NAME(5)},
{DEF_GEN_USTUB_NAME(6)},
{DEF_GEN_USTUB_NAME(7)},
{DEF_GEN_USTUB_NAME(8)},
{DEF_GEN_USTUB_NAME(9)},
{DEF_GEN_USTUB_NAME(10)},
{DEF_GEN_USTUB_NAME(11)},
{DEF_GEN_USTUB_NAME(12)},
{DEF_GEN_USTUB_NAME(13)},
{DEF_GEN_USTUB_NAME(14)},
{DEF_GEN_USTUB_NAME(15)},
{DEF_GEN_USTUB_NAME(16)},
{DEF_GEN_USTUB_NAME(17)},
{DEF_GEN_USTUB_NAME(18)},
{DEF_GEN_USTUB_NAME(19)},
};


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
    //int dtypes[] = { NPY_BOOL, NPY_INT8, NPY_UINT8,  NPY_INT16, NPY_UINT16,  NPY_INT32, NPY_UINT32,  NPY_INT64, NPY_UINT64, NPY_FLOAT32, NPY_FLOAT64 };
    int dtypes[] = {  NPY_INT32,  NPY_INT64};
    const char* str_ufunc[] = { "add","subtract" };
    int   atop_mathop[] = { MATH_OPERATION::ADD, MATH_OPERATION::SUB };

    // Init atop: array threading operations
    if (atop_init()) {
        memset(g_UFuncLUT, 0, sizeof(g_UFuncLUT));

        // Initialize numpy's C-API.
        import_array();
        import_umath();
        PyObject* numpy_module = PyImport_ImportModule("numpy");
        if (numpy_module == NULL) {
            return NULL;
        }

        // Loop over all ufuncs we want to replace
        int64_t num_ufuncs = sizeof(str_ufunc) / sizeof(char*);
        for (int64_t i = 0; i < num_ufuncs; i++) {
            PyObject* result = NULL;
            PyObject* ufunc = NULL;
            const char* ufunc_name = str_ufunc[i];
            int atop = atop_mathop[i];

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
                int wantedOut = -1;

                int atype = convert_dtype_to_atop[dtype];

                ANY_TWO_FUNC pTwoFunc = GetSimpleMathOpFast(atop, atype, atype, atype, &wantedOut);

                if (pTwoFunc) {
                    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc, g_UFuncGenericLUT[atop][atype], signature, &oldFunc);

                    if (ret < 0) {
                        return PyErr_Format(PyExc_TypeError, "Failed with %d. func %s must be the name of a ufunc", ret, ufunc_name);
                    }

                    g_UFuncLUT[atop][atype].pOldFunc = oldFunc;
                    g_UFuncLUT[atop][atype].pTwoFunc = pTwoFunc;
                }
            }
        }
        RETURN_NONE;
    }
    return PyErr_Format(PyExc_ImportError, "atop was either already loaded or failed to load");
}

