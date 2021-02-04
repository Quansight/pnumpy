#pragma once
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// NOTE: See PY_ARRAY_UNIQUE_SYMBOL
// If this is not included, calling PY_ARRAY functions will have a null value
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API

#ifndef SHAREDATA_MAIN_C_FILE
#define NO_IMPORT_ARRAY
#endif

#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include <stdint.h>
#include <stdio.h>
#include "../atop/atop.h"


int dtype_to_atop(int dtype);

// Global user settings controlled by python functions
// set to 0 to disable
struct stSettings {
    int32_t  AtopEnabled;
    int32_t  LedgerEnabled;
    int32_t  RecyclerEnabled;
    int32_t  ZigZag;  // set to 0 to disable
    int32_t  Initialized;
    int32_t  Reserved;
    binaryfunc NumpyGetItem;  // optional hook
};

extern stSettings g_Settings;

struct stUFuncToAtop {
    const char* str_ufunc_name;
    const int       atop_op;
};

enum OP_CATEGORY:int32_t {
    OPCAT_BINARY = 0,
    OPCAT_UNARY = 1,
    OPCAT_COMPARE = 2,
    OPCAT_TRIG = 3,
    OPCAT_CONVERT = 4,
    OPCAT_SORT = 5,
    OPCAT_ARGSORT = 6,
    OPCAT_ARANGE = 7,
    OPCAT_ARGMINMAX = 8,
    OPCAT_LAST = 9,
};

struct stOpCategory {
    const char*     StrName;
    int32_t         NumOps;
    OP_CATEGORY     CatEnum;    // 
    stUFuncToAtop*  pUFuncToAtop;
};


//---------------------------------------------------------------------
// NOTE: See SDSArrayInfo and keep same
struct ArrayInfo {

    // Numpy object
    PyArrayObject* pObject;

    // First bytes
    char* pData;

    // Width in bytes of one row
    int64_t      ItemSize;

    // total number of items
    int64_t       ArrayLength;

    int64_t       NumBytes;

    int           NumpyDType;
    int           NDim;

    // When calling ensure contiguous, we might make a copy
    // if so, pObject is the copy and must be deleted.  pOriginal was passed in
    PyArrayObject* pOriginalObject;

};

extern void* GetDefaultForType(int numpyInType);
extern int64_t CalcArrayLength(int ndim, npy_intp* dims);
extern int64_t ArrayLength(PyArrayObject* inArr);
extern PyArrayObject* AllocateNumpyArray(int ndim, npy_intp* dims, int32_t numpyType, int64_t itemsize = 0, int fortran_array = 0, npy_intp* strides = nullptr);
extern PyArrayObject* AllocateLikeResize(PyArrayObject* inArr, npy_intp rowSize);
extern PyArrayObject* AllocateLikeNumpyArray(PyArrayObject* inArr, int numpyType);
extern BOOL ConvertScalarObject(PyObject* inObject1, void* pDest, int16_t numpyOutType, void** ppDataIn, int64_t* pItemSize);
extern int GetStridesAndContig(PyArrayObject* inArray, int& ndim, int64_t& stride);

// defined in pnumpy
extern stOpCategory gOpCategory[OPCAT_LAST];

extern void LedgerRecord(int32_t op_category, int64_t start_time, int64_t end_time, char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype);
extern void LedgerRecord2(int32_t op_category, int64_t start_time, int64_t end_time, int atype, int64_t length);
extern void LedgerInit();
extern int64_t CalcArrayLength(int ndim, npy_intp* dims);
extern int64_t ArrayLength(PyArrayObject* inArr);
extern PyArrayObject* AllocateNumpyArray(int ndim, npy_intp* dims, int32_t numpyType, int64_t itemsize, int fortran_array, npy_intp* strides);
extern PyArrayObject* AllocateLikeResize(PyArrayObject* inArr, npy_intp rowSize);
extern PyArrayObject* AllocateLikeNumpyArray(PyArrayObject* inArr, int numpyType);
extern ArrayInfo* BuildArrayInfo(
    PyObject* listObject,
    int64_t* pTupleSize,
    int64_t* pTotalItemSize,
    BOOL checkrows = TRUE,
    BOOL convert = TRUE);

extern void FreeArrayInfo(ArrayInfo* pAlloc);

extern PyObject* BooleanIndexInternal(PyArrayObject* aValues, PyArrayObject* aIndex);
extern "C" PyObject *getitem(PyObject * self, PyObject * args);

#define RETURN_NONE Py_INCREF(Py_None); return Py_None;
#define RETURN_FALSE Py_XINCREF(Py_False); return Py_False;
#define RETURN_TRUE Py_XINCREF(Py_True); return Py_True;

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

extern PyTypeObject* pPyArray_Type;

#if defined(_WIN32) && !defined(__GNUC__)

#define CASE_NPY_INT32      case NPY_INT32:       case NPY_INT
#define CASE_NPY_UINT32     case NPY_UINT32:      case NPY_UINT
#define CASE_NPY_INT64      case NPY_INT64
#define CASE_NPY_UINT64     case NPY_UINT64
#define CASE_NPY_FLOAT64    case NPY_DOUBLE:     case NPY_LONGDOUBLE

#else

#define CASE_NPY_INT32      case NPY_INT32
#define CASE_NPY_UINT32     case NPY_UINT32
#define CASE_NPY_INT64      case NPY_INT64:    case NPY_LONGLONG
#define CASE_NPY_UINT64     case NPY_UINT64:   case NPY_ULONGLONG
#define CASE_NPY_FLOAT64    case NPY_DOUBLE
#endif


