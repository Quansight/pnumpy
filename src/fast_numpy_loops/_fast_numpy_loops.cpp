
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include <stdint.h>

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))


template <class T>
void add_T(T **args, npy_intp const *dimensions, npy_intp const *steps,
          void *innerloopdata) {
    // steps is in bytes, so cast args to char** to allow strange steps.
    if (IS_BINARY_REDUCE) {
        char *iop1 = (char *)args[0]; \
        T io1 = *(T *)iop1; \
        char *ip2 = (char *)args[1]; \
        npy_intp is2 = steps[1]; \
        npy_intp n = dimensions[0]; \
        npy_intp i; \
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
        npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
        npy_intp n = dimensions[0];
        npy_intp i;
        for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)
        {
            const T in1 = *(T *)ip1;
            const T in2 = *(T *)ip2;
            T *out = (T *)op1;
            *out = in1 + in2;
        }
    }
}

extern "C"
PyObject* initialize(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *result = NULL, *dtype=NULL;
    PyObject *ufunc = NULL;
    const char * uname = NULL;
    int ret = 0;
    int signature[3] = {NPY_INT, NPY_INT, NPY_INT}; // This is actually int32
    PyUFuncGenericFunction oldfunc, newfunc;
    // C++ warns on assigning const char * to char *
    const char *kwlist[] = {"dtypein", "dtypeout", NULL};
    char **_kwlist = (char **)kwlist;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O:initialize", _kwlist,
                                     &uname, &dtype)) {
        return NULL;
    } 
    // Initialize numpy's C-API. 
    import_array();
    import_umath();

    PyObject *numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL) {
        goto error;
    }
    ufunc = PyObject_GetAttrString(numpy_module, uname);
    Py_DECREF(numpy_module);
    if (ufunc == NULL) {
        goto error;
    }
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError, "func must be the name of a ufunc");
        goto error;
    }
    // TODO: parse requested dtype into the templating type
    newfunc = (PyUFuncGenericFunction)add_T<int32_t>;
    Py_XDECREF(dtype);
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
error:
    Py_XDECREF(ufunc);
    Py_XDECREF(dtype);
    return NULL; 
}


