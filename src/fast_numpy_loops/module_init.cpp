#include "Python.h"

/*
 * Some C++ compilers do not like mixin non-designated-initializers
 * like PyModuleDef_HEAD_INIT with designated-initializers like
 * .m_doc, so break this part out into a C file
 */
  

extern "C" PyObject* oldinit(PyObject *self, PyObject *args, PyObject *kwargs);
extern "C" PyObject* newinit(PyObject* self, PyObject* args, PyObject* kwargs);

static char m_doc[] = "Provide methods to override NumPy ufuncs";


PyDoc_STRVAR(initialize_doc,
     "oldinit(ufunc_name:");

static PyMethodDef module_functions[] = {
    {"initialize",       (PyCFunction)newinit, METH_VARARGS | METH_KEYWORDS, "init the atop"},
    {"oldinit",          (PyCFunction)oldinit, METH_VARARGS | METH_KEYWORDS, initialize_doc},
    {NULL, NULL}
};


static PyModuleDef moduledef = {
   PyModuleDef_HEAD_INIT,
   "fast_numpy_loops._fast_numpy_loops", // Module name
   m_doc,  // Module description
   0,
   module_functions,                     // Structure that defines the methods
   NULL,                                 // slots
   NULL,                                 // GC traverse
   NULL,                                 // GC
   NULL                                  // freefunc
};
PyMODINIT_FUNC PyInit__fast_numpy_loops(void) {
    PyObject *module;

    module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    return module;
}
