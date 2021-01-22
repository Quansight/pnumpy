#include "common.h"
#include <cmath>
#include "../atop/threads.h"

//------------------------------------------------------
// The arange routine is largely copied from numpy
//
#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure. In case of failure, set a PyExc_Overflow
 * exception
 */
static npy_intp
_arange_safe_ceil_to_intp(double value)
{
    double ivalue;

    ivalue = ceil(value);
    /* condition inverted to handle NaN */
    if (isnan(ivalue)) {
        PyErr_SetString(PyExc_ValueError,
            "arange: cannot compute length");
        return -1;
    }
    if (!(NPY_MIN_INTP <= ivalue && ivalue <= NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
            "arange: overflow while computing length");
        return -1;
    }

    return (npy_intp)ivalue;
}

/*NUMPY_API
  Arange,
*/
PyObject*
PArange(double start, double stop, double step, int type_num)
{
    npy_intp length;
    PyArrayObject* range;
    PyArray_ArrFuncs* funcs;
    PyObject* obj;
    int ret;
    double delta, tmp_len;
    NPY_BEGIN_THREADS_DEF;

    delta = stop - start;
    tmp_len = delta / step;

    /* Underflow and divide-by-inf check */
    if (tmp_len == 0.0 && delta != 0.0) {
        if (signbit(tmp_len)) {
            length = 0;
        }
        else {
            length = 1;
        }
    }
    else {
        length = _arange_safe_ceil_to_intp(tmp_len);
        if (error_converting(length)) {
            return NULL;
        }
    }

    if (length <= 0) {
        length = 0;
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
            NULL, NULL, 0, 0, NULL);
    }
    range = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, &length, type_num,
        NULL, NULL, 0, 0, NULL);
    if (range == NULL) {
        return NULL;
    }
    funcs = PyArray_DESCR(range)->f;

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    obj = PyFloat_FromDouble(start);
    ret = funcs->setitem(obj, PyArray_DATA(range), range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 1) {
        return (PyObject*)range;
    }
    obj = PyFloat_FromDouble(start + step);
    ret = funcs->setitem(obj, PyArray_BYTES(range) + PyArray_ITEMSIZE(range),
        range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 2) {
        return (PyObject*)range;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError,
            "no fill-function for data-type.");
        Py_DECREF(range);
        return NULL;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));
    funcs->fill(PyArray_DATA(range), length, range);
    NPY_END_THREADS;
    if (PyErr_Occurred()) {
        goto fail;
    }
    return (PyObject*)range;

fail:
    Py_DECREF(range);
    return NULL;
}
