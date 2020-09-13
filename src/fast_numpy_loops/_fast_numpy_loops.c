
#include "Python.h"

static PyObject* initialize(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *result = NULL, *obj, *dtype=NULL;
    static char *kwlist[] = {"dtypein", "dtypeout", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|O:initialize", kwlist,
            &obj, &dtype)) {
        return NULL;
    }
    result = PyUnicode_FromString("int,int->int");
    Py_XDECREF(obj);     
    Py_XDECREF(dtype);
    return result;
}

PyDoc_STRVAR(initialize_doc, "Docstring for initialize function.");

static struct PyMethodDef module_functions[] = {
    {"initialize", (PyCFunction)initialize, METH_VARARGS | METH_VARARGS,
     initialize_doc},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fast_numpy_loops._fast_numpy_loops", /* m_name */
    NULL,             /* m_doc */
    -1,               /* m_size */
    module_functions, /* m_methods */
    NULL,             /* m_reload */
    NULL,             /* m_traverse */
    NULL,             /* m_clear */
    NULL,             /* m_free */
};
#endif

static PyObject* moduleinit(void) {
    PyObject *module;

#if PY_MAJOR_VERSION >= 3
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule3("fast_numpy_loops._fast_numpy_loops", module_functions, NULL);
#endif

    if (module == NULL)
        return NULL;

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

