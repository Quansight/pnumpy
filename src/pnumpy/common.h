#pragma once
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdint.h>
#include <stdio.h>
#include "../atop/atop.h"

// Global user settings controlled by python functions
// set to 0 to disable
struct stSettings {
    int32_t  AtopEnabled;
    int32_t  LedgerEnabled;
    int32_t  RecyclerEnabled;
    int32_t  Reserved;
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
    OPCAT_LAST = 4,
};

struct stOpCategory {
    const char*     StrName;
    int32_t         NumOps;
    OP_CATEGORY     CatEnum;    // 
    stUFuncToAtop*  pUFuncToAtop;
};

// defined in pnumpy
extern stOpCategory gOpCategory[OPCAT_LAST];

extern void LedgerRecord(int32_t op_category, int64_t start_time, int64_t end_time, char** args, const npy_intp* dimensions, const npy_intp* steps, void* innerloop, int funcop, int atype);
extern void LedgerInit();

#define RETURN_NONE Py_INCREF(Py_None); return Py_None;
#define RETURN_FALSE Py_XINCREF(Py_False); return Py_False;
#define RETURN_TRUE Py_XINCREF(Py_True); return Py_True;

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

