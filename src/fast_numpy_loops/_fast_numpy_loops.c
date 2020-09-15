
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

/*
 * largest simd vector size in bytes numpy supports
 * it is currently a extremely large value as it is only used for memory
 * overlap checks
 */
#ifndef NPY_MAX_SIMD_SIZE
#define NPY_MAX_SIMD_SIZE 1024
#endif

static NPY_INLINE npy_uintp
abs_ptrdiff(char *a, char *b)
{
    return (a > b) ? (a - b) : (b - a);
}

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

/** (ip1, ip2) -> (op1) */
#define BINARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)


#define BINARY_REDUCE_LOOP(TYPE)\
    char *iop1 = args[0]; \
    TYPE io1 = *(TYPE *)iop1; \
    char *ip2 = args[1]; \
    npy_intp is2 = steps[1]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip2 += is2)

#define BASE_BINARY_LOOP(tin, tout, op) \
    BINARY_LOOP { \
        const tin in1 = *(tin *)ip1; \
        const tin in2 = *(tin *)ip2; \
        tout *out = (tout *)op1; \
        op; \
    }

void add_int(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *innerloopdata) {
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(int) {
            io1 += *(int *)ip2;
        }
        *((int *)iop1) = io1;
    }
    else {
        BASE_BINARY_LOOP(int, int, *out = in1 + in2);
    }
}


static PyObject* initialize(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *result = NULL, *dtype=NULL;
    PyObject *ufunc = NULL;
    const char * uname = NULL;
    int signature[3] = {NPY_INT, NPY_INT, NPY_INT}; // This is actually int32
    PyUFuncGenericFunction oldfunc;
    static char *kwlist[] = {"dtypein", "dtypeout", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O:initialize", kwlist,
                                     &uname, &dtype)) {
        return NULL;
    } 
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
    result = PyUnicode_FromString("int32,int32->int32");
    Py_XDECREF(dtype);
    int ret = PyUFunc_ReplaceLoopBySignature((PyUFuncObject*)ufunc,
                                             (PyUFuncGenericFunction)add_int,
                                             signature, &oldfunc);
    if (ret < 0) {
        PyErr_SetString(PyExc_ValueError, "signature int,int->int not found");
    }
    return result;
error:
    Py_XDECREF(ufunc);
    Py_XDECREF(dtype);
    return NULL; 
}

static char m_doc[] = "Provide methods to override NumPy ufuncs";


PyDoc_STRVAR(initialize_doc,
     "initialize(ufunc_name:");

static struct PyMethodDef module_functions[] = {
    {"initialize", (PyCFunction)initialize, METH_VARARGS | METH_KEYWORDS,
     initialize_doc},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "fast_numpy_loops._fast_numpy_loops",
    .m_doc = m_doc,
    .m_size = -1,
    .m_methods = module_functions,
};
#endif

static PyObject* moduleinit(void) {
    PyObject *module;

#if PY_MAJOR_VERSION >= 3
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule3(m_doc, module_functions, NULL);
#endif

    if (module == NULL)
        return NULL;

    // Initialize numpy's C-API
    import_array(); 
    import_umath();
    return module;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_fast_numpy_loops(void) {
    moduleinit();
}
#else
PyMODINIT_FUNC PyInit__fast_numpy_loops(void) {
    return moduleinit();
}
#endif

