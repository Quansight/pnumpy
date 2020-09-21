#include "Python.h"

/*
 * Some C++ compilers do not like mixin non-designated-initializers
 * like PyModuleDef_HEAD_INIT with designated-initializers like
 * .m_doc, so break this part out into a C file
 */
  

PyObject* initialize(PyObject *self, PyObject *args, PyObject *kwargs);

static char m_doc[] = "Provide methods to override NumPy ufuncs";


PyDoc_STRVAR(initialize_doc,
     "initialize(ufunc_name:");

static struct PyMethodDef module_functions[] = {
    {"initialize", (PyCFunction)initialize, METH_VARARGS | METH_KEYWORDS,
     initialize_doc},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "fast_numpy_loops._fast_numpy_loops",
    .m_doc = m_doc,
    .m_size = -1,
    .m_methods = module_functions,
};

PyMODINIT_FUNC PyInit__fast_numpy_loops(void) {
    PyObject *module;

    module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    return module;
}
