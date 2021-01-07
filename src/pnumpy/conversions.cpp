#include "common.h"

#define LOGGING(...)

// TODO: look at casting in ndarraytypes.h, convert_datatype.c
//NPY_NO_EXPORT PyArray_VectorUnaryFunc*
//PyArray_GetCastFunc(PyArray_Descr* descr, int type_num)
//{
//    PyArray_VectorUnaryFunc* castfunc = NULL;
//
//    if (type_num < NPY_NTYPES_ABI_COMPATIBLE) {
//        castfunc = descr->f->cast[type_num];

//-----------------------------------
// Converts (in parallel) a numpy recarray (void type)
// Caller must have PREALLOCATE the colmajor arrays to copy data into
// Caller must also pass the struct offsets (within the recarray)
//
// Input1: the recordarray to convert
// Input2: int64 array of offsets
// Input3: object array of numpy arrays pre allocated that match in order the recarray
extern "C"
PyObject*
recarray_to_colmajor(PyObject* self, PyObject* args) {

    PyArrayObject* inArr = NULL;
    PyArrayObject* offsetArr = NULL;
    PyArrayObject* arrArr = NULL;

    //if (!PyArg_ParseTuple(args, "O!O!O!:recarray_to_colmajor",
    //    &PyArray_Type, &inArr,
    //    &PyArray_Type, &offsetArr,
    //    &PyArray_Type, &arrArr)) {
    //    return NULL;
    //}

    if (PyTuple_Size(args) == 3) {
        inArr = (PyArrayObject*)PyTuple_GetItem(args, 0);
        offsetArr = (PyArrayObject*)PyTuple_GetItem(args, 1);
        arrArr = (PyArrayObject*)PyTuple_GetItem(args, 2);
    }
    else {
        PyErr_Format(PyExc_ValueError, "recarray_to_colmajor must input 3 numpy arrays");
        return NULL;
    }

    int64_t itemSize = PyArray_ITEMSIZE(inArr);

    if (itemSize != PyArray_STRIDE(inArr, 0)) {
        PyErr_Format(PyExc_ValueError, "recarray_to_colmajor cannot yet handle strides");
        return NULL;
    }

    if (NPY_VOID != PyArray_TYPE(inArr)) {
        PyErr_Format(PyExc_ValueError, "recarray_to_colmajor must be void type");
        return NULL;
    }

    if (NPY_OBJECT != PyArray_TYPE(arrArr)) {
        PyErr_Format(PyExc_ValueError, "recarray_to_colmajor third param must be object array");
        return NULL;
    }

    int64_t length = ArrayLength(inArr);
    int64_t numArrays = ArrayLength(arrArr);

    if (numArrays != ArrayLength(offsetArr)) {
        PyErr_Format(PyExc_ValueError, "recarray_to_colmajor inputs do not match");
        return NULL;
    }

    int64_t totalRows = length;
    int64_t* pOffsets = (int64_t*)PyArray_BYTES(offsetArr);
    PyArrayObject** ppArrays = (PyArrayObject**)PyArray_BYTES(arrArr);

    stRecarrayOffsets* pstOffset;

    // TODO allocate this on the stack
    pstOffset = (stRecarrayOffsets*)WORKSPACE_ALLOC(sizeof(stRecarrayOffsets) * numArrays);

    for (int64_t i = 0; i < numArrays; i++) {
        // Consider adding pOffsets here
        pstOffset[i].pData = PyArray_BYTES(ppArrays[i]);
        pstOffset[i].readoffset = pOffsets[i];
        pstOffset[i].itemsize = PyArray_ITEMSIZE(ppArrays[i]);
    }

    char* pStartOffset = PyArray_BYTES(inArr);

    // Call atop to finish the work
    RecArrayToColMajor(
        pstOffset,
        pStartOffset,
        totalRows,
        numArrays,
        itemSize);

    WORKSPACE_FREE(pstOffset);

    RETURN_NONE;
}

