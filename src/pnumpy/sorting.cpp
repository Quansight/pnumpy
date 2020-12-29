#include "common.h"
#include "../atop/atop.h"
#include "../atop/threads.h"

#define LOGGING(...)
#define PLOGGING(...)

//===============================================================================
// Pass in a numpy array, it will be sorted in place
// the same array is returned
// Arg1: iKey (array of uniques)
// Arg2: iUniqueSort (resorting of iUnique)
//
// NOTE: every element of iKey must be an index into iUniqueSort (the max element = max(iUniqueSort)-1)
// Output:
//   Arg1 = iUniqueSort[Arg1[i]]
PyObject* sort_indirect(PyObject* self, PyObject* args) {
    PyArrayObject* inArr1 = NULL;
    PyArrayObject* inSort = NULL;

    // THIS CODE IS NOT FINSIHED

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArr1, &PyArray_Type, &inSort)) return NULL;

    int32_t arrayType1 = PyArray_TYPE(inArr1);
    int32_t sortType = PyArray_TYPE(inSort);

    int64_t arraySize1 = ArrayLength(inArr1);
    int64_t sortSize = ArrayLength(inSort);


    if (arrayType1 == NPY_INT32 && sortType == NPY_INT32) {
        int32_t* pDataIn = (int32_t*)PyArray_BYTES(inArr1);
        int32_t* pSort = (int32_t*)PyArray_BYTES(inSort);

        int32_t* inverseSort = (int32_t*)WORKSPACE_ALLOC(sortSize * sizeof(int32_t));
        for (int i = 0; i < sortSize; i++) {
            inverseSort[pSort[i]] = i;
        }

        for (int i = 0; i < arraySize1; i++) {
            pDataIn[i] = inverseSort[pDataIn[i]];
        }

        WORKSPACE_FREE(inverseSort);
    }
    else if (arrayType1 == NPY_INT32 && sortType == NPY_INT64) {
        int32_t* pDataIn = (int32_t*)PyArray_BYTES(inArr1);
        int64_t* pSort = (int64_t*)PyArray_BYTES(inSort);

        int64_t* inverseSort = (int64_t*)WORKSPACE_ALLOC(sortSize * sizeof(int64_t));
        for (int64_t i = 0; i < sortSize; i++) {
            inverseSort[pSort[i]] = i;
        }

        for (int i = 0; i < arraySize1; i++) {
            pDataIn[i] = (int32_t)inverseSort[pDataIn[i]];
        }
        WORKSPACE_FREE(inverseSort);

    }
    else if (arrayType1 == NPY_INT64 && sortType == NPY_INT64) {
        int64_t* pDataIn = (int64_t*)PyArray_BYTES(inArr1);
        int64_t* pSort = (int64_t*)PyArray_BYTES(inSort);

        int64_t* inverseSort = (int64_t*)WORKSPACE_ALLOC(sortSize * sizeof(int64_t));
        for (int64_t i = 0; i < sortSize; i++) {
            inverseSort[pSort[i]] = i;
        }

        for (int64_t i = 0; i < arraySize1; i++) {
            pDataIn[i] = inverseSort[pDataIn[i]];
        }
        WORKSPACE_FREE(inverseSort);

    }
    else {
        printf("**SortInplaceIndirect failure!  arrays must be int32 or int64\n");
    }

    Py_IncRef((PyObject*)inArr1);
    return (PyObject*)(inArr1);
}


////===============================================================================
//// Pass in a numpy array, it will be copied and sorted in the return array
//// the same array is returned
//PyObject* Sort(PyObject* self, PyObject* args) {
//    PyArrayObject* inArr1 = NULL;
//    int sortMode;
//
//    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &inArr1, &sortMode)) return NULL;
//
//    int32_t arrayType1 = PyArray_TYPE(inArr1);
//    int ndim = PyArray_NDIM(inArr1);
//    npy_intp* dims = PyArray_DIMS(inArr1);
//
//    int64_t arraySize1 = CalcArrayLength(ndim, dims);
//
//    // The output is a boolean where the nth item was found
//    PyArrayObject* duplicateArray = AllocateNumpyArray(ndim, dims, arrayType1);
//
//    if (duplicateArray == NULL) {
//        PyErr_Format(PyExc_ValueError, "Sort out of memory");
//        return NULL;
//    }
//
//    void* pDataIn1 = PyArray_BYTES(inArr1);
//    void* pDataOut1 = PyArray_BYTES(duplicateArray);
//
//    int64_t itemSize = NpyItemSize((PyObject*)inArr1);
//
//    memcpy(pDataOut1, pDataIn1, arraySize1 * itemSize);
//
//    SORT_MODE mode = (SORT_MODE)sortMode;
//
//    SortArray(pDataOut1, arraySize1, arrayType1, mode);
//
//    return SetFastArrayView(duplicateArray);
//}

//===============================================================================
// returns True or False
// Nans at the end are fine and still considered sorted
PyObject* is_sorted(PyObject* self, PyObject* args) {
    PyArrayObject* inArr1 = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr1)) return NULL;

    int32_t arrayType1 = PyArray_TYPE(inArr1);
    int ndim = PyArray_NDIM(inArr1);
    npy_intp* dims = PyArray_DIMS(inArr1);

    int64_t itemSize = PyArray_ITEMSIZE(inArr1);

    if (ndim != 1 || itemSize != PyArray_STRIDE(inArr1, 0)) {
        PyErr_Format(PyExc_ValueError, "IsSorted arrays must be one dimensional and contiguous.  ndim is %d\n", ndim);
        return NULL;
    }

    int64_t arraySize1 = CalcArrayLength(ndim, dims);
    void* pDataIn1 = PyArray_BYTES(inArr1);

    int64_t result = IsSorted(pDataIn1, arraySize1, arrayType1, itemSize);

    if (result == -1) {
        PyErr_Format(PyExc_ValueError, "IsSorted does not understand type %d\n", arrayType1);
        return NULL;
    }

    if (result) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else {
        Py_INCREF(Py_False);
        return Py_False;

    }

}


//===============================================================================
// checks for kwargs cutoff
// if exists, and is int64_t, returns pointer and length of cutoffs
// cutoffLength of -1 indicates an error
int64_t* GetCutOffs(PyObject* kwargs, int64_t& cutoffLength) {
    // Check for cutoffs kwarg to see if going into parallel mode
    if (kwargs && PyDict_Check(kwargs)) {
        PyArrayObject* pCutOffs = NULL;
        // Borrowed reference
        // Returns NULL if key not present
        pCutOffs = (PyArrayObject*)PyDict_GetItemString(kwargs, "cutoffs");

        if (pCutOffs != NULL && PyArray_Check(pCutOffs)) {
            switch (PyArray_TYPE(pCutOffs)) {
            CASE_NPY_INT64:
                cutoffLength = ArrayLength(pCutOffs);
                return (int64_t*)PyArray_BYTES(pCutOffs);
            default:
                printf("Bad cutoff dtype... make sure int64_t\n");
                cutoffLength = -1;
                return NULL;
            }
        }
    }
    cutoffLength = 0;
    return NULL;
}



// index must be int32_t or int64_t
static PyArrayObject* GetKwargIndex(PyObject* kwargs, int64_t& indexLength, int& dtype) {
    // Check for 'index' kwarg to see if prime lexsort
    if (kwargs && PyDict_Check(kwargs)) {
        PyArrayObject* pStartIndex = NULL;
        // Borrowed reference
        // Returns NULL if key not present
        pStartIndex = (PyArrayObject*)PyDict_GetItemString(kwargs, "index");

        if (pStartIndex != NULL && PyArray_Check(pStartIndex)) {
            indexLength = ArrayLength(pStartIndex);

            switch (PyArray_TYPE(pStartIndex)) {
            CASE_NPY_INT64:
                dtype = NPY_INT64;
                return pStartIndex;
            CASE_NPY_INT32:
                dtype = NPY_INT32;
                return pStartIndex;
            default:
                printf("Bad index dtype... make sure int64_t or int32_t\n");
                indexLength = -1;
                return NULL;
            }
        }
    }
    indexLength = 0;
    return NULL;
}


//===============================================================================

template<typename UINDEX>
static int64_t ARangeCallback(void* callbackArgT, int core, int64_t start, int64_t length) {

    UINDEX* pDataOut = (UINDEX*)callbackArgT;
    UINDEX istart = (UINDEX)start;
    UINDEX iend = istart + (UINDEX)length;

    for (UINDEX i = istart; i < iend; i++) {
        pDataOut[i] = i;
    }

    return (iend-istart);
}


//===============================================================================
// Helper class for one or more arrays
class CMultiListPrepare {

public:

    Py_ssize_t tupleSize;  // or number of arrays
    ArrayInfo* aInfo;
    int64_t totalItemSize;
    int64_t totalRows;

    CMultiListPrepare(PyObject* args) {
        aInfo = NULL;
        totalItemSize = 0;
        totalRows = 0;

        tupleSize = PyTuple_GET_SIZE(args);

        //MLPLOGGING("Tuple size %llu\n", tupleSize);

        if (tupleSize >= 1) {
            // Check if they passed in a list
            PyObject* listObject = PyTuple_GetItem(args, 0);
            if (PyList_Check(listObject)) {
                args = listObject;
                tupleSize = PyList_GET_SIZE(args);
                //MLPLOGGING("Found list inside tuple size %llu\n", tupleSize);
            }
        }

        int64_t listSize = 0;
        aInfo = BuildArrayInfo(args, &listSize, &totalItemSize);

        if (aInfo) {

            totalRows = aInfo[0].ArrayLength;

            for (int64_t i = 0; i < listSize; i++) {
                if (aInfo[i].ArrayLength != totalRows) {
                    PyErr_Format(PyExc_ValueError, "MultiListPrepare all arrays must be same number of rows %llu", totalRows);
                    totalRows = 0;
                }
            }
            if (totalRows != 0) {
                //printf("row width %llu   rows %llu\n", totalItemSize, totalRows);
            }
        }
    }

    ~CMultiListPrepare() {
        if (aInfo != NULL) {
            FreeArrayInfo(aInfo);
            aInfo = NULL;
        }
    }

};

//===============================================================================
// LexSort32 and LexSort64 funnel into here
// Kwargs:
//--------
// cutoffs=
// index=  specify start index instead of doing arange
// groups=True (also group)
// base_index=
// ascending??
//
// Returns
// -------
// Fancy index in desired sort order
// when group=True
// Adds iKey
// Adds iFirstKey
// Add  nCount
// NOTE: From nCount, and iFirstKey can build
//
template<typename UINDEX>
PyObject* lexsort(PyObject* self, PyObject* args, PyObject* kwargs) {
    CMultiListPrepare mlp(args);

    if (mlp.aInfo && mlp.tupleSize > 0) {
        int64_t    arraySize1 = mlp.totalRows;

        int64_t    cutOffLength = 0;
        int64_t* pCutOffs = GetCutOffs(kwargs, cutOffLength);

        if (pCutOffs && pCutOffs[cutOffLength - 1] != arraySize1) {
            PyErr_Format(PyExc_ValueError, "LexSort last cutoff length does not match array length %lld", arraySize1);
            return NULL;
        }
        if (cutOffLength == -1) {
            PyErr_Format(PyExc_ValueError, "LexSort 'cutoffs' must be an array of type int64_t");
            return NULL;
        }

        // Check types that we can sort
        for (UINDEX i = 0; i < mlp.tupleSize; i++) {
            int dtype = mlp.aInfo[i].NumpyDType;
            int atype = dtype_to_atop(dtype);
            if (atype == -1 || (atype > ATOP_LONGDOUBLE && (atype != ATOP_STRING && atype != ATOP_UNICODE))) {
                PyErr_Format(PyExc_ValueError, "LexSort cannot handle type %d\n",dtype);
                return NULL;
            }
        }


        int indexDType = 0;
        int64_t indexLength = 0;

        PyArrayObject* index = GetKwargIndex(kwargs, indexLength, indexDType);

        PyArrayObject* result = NULL;

        if (indexLength == -1) {
            PyErr_Format(PyExc_ValueError, "LexSort 'index' must be an array of type int64_t or int32_t");
            return NULL;
        }

        if (indexLength > 0) {
            if (indexLength > arraySize1) {
                PyErr_Format(PyExc_ValueError, "LexSort 'index' is larger than value array");
                return NULL;
            }

            if (sizeof(UINDEX) == 8 && indexDType != NPY_INT64) {
                PyErr_Format(PyExc_ValueError, "LexSort 'index' is not int64_t");
                return NULL;
            }

            if (sizeof(UINDEX) == 4 && indexDType != NPY_INT32) {
                PyErr_Format(PyExc_ValueError, "LexSort 'index' is not int32_t");
                return NULL;
            }

            // reduce what we sort to the startindex
            arraySize1 = indexLength;
            result = index;
            Py_IncRef((PyObject*)result);
        }
        else {
            result = AllocateLikeNumpyArray(mlp.aInfo[0].pObject, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
        }

        if (result) {
            // Return the sorted index
            UINDEX* pDataOut = (UINDEX*)PyArray_BYTES(result);

            // BUG? what if we have index= and cutoffs= ??
            if (pCutOffs) {
                LOGGING("Have cutoffs %lld\n", cutOffLength);

                // For cutoffs, prep the indexes with 0:n for each partition
                UINDEX* pCounter = pDataOut;

                int64_t startPos = 0;
                for (int64_t j = 0; j < cutOffLength; j++) {

                    int64_t endPos = pCutOffs[j];
                    int64_t partitionLength = endPos - startPos;
                    for (UINDEX i = 0; i < partitionLength; i++) {
                        *pCounter++ = i;
                    }
                    startPos = endPos;
                }
            }
            else {

                // If the user did not provide a start index, we make one
                if (index == NULL) {
                    THREADER->DoMultiThreadedChunkWork(arraySize1, ARangeCallback<UINDEX>, pDataOut);
                }
            }


            // When multiple arrays are passed, we sort in order of how it is passed
            // Thus, the last array is the last sort, and therefore determines the primary sort order
            for (UINDEX i = 0; i < mlp.tupleSize; i++) {
                int atype = dtype_to_atop(mlp.aInfo[i].NumpyDType);
                // For each array...
                if (sizeof(UINDEX) == 4) {
                    SortIndex32(pCutOffs, cutOffLength, mlp.aInfo[i].pData, arraySize1, (int32_t*)pDataOut, SORT_MODE::SORT_MODE_MERGE, atype, mlp.aInfo[i].ItemSize);
                }
                else {
                    SortIndex64(pCutOffs, cutOffLength, mlp.aInfo[i].pData, arraySize1, (int64_t*)pDataOut, SORT_MODE::SORT_MODE_MERGE, atype, mlp.aInfo[i].ItemSize);
                }
            }

            return (PyObject*)result;
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}


//===============================================================================
// Returns int32_t
extern "C" PyObject* lexsort32(PyObject* self, PyObject* args, PyObject* kwargs) {

    return lexsort<int32_t>(self, args, kwargs);
}


//===============================================================================
// Returns int64_t
extern "C" PyObject* lexsort64(PyObject* self, PyObject* args, PyObject* kwargs) {

    return lexsort<int64_t>(self, args, kwargs);
}


//===============================================================================
//===============================================================================
// checks for kwargs filter
// if exists, and is BOOL, returns pointer and length of bool
static bool* GetFilter(PyObject* kwargs, int64_t& filterLength) {
    // Check for cutoffs kwarg to see if going into parallel mode
    if (kwargs && PyDict_Check(kwargs)) {
        PyArrayObject* pFilter = NULL;
        // Borrowed reference
        // Returns NULL if key not present
        pFilter = (PyArrayObject*)PyDict_GetItemString(kwargs, "filter");

        if (pFilter != NULL && PyArray_Check(pFilter)) {
            switch (PyArray_TYPE(pFilter)) {
            case NPY_BOOL:
                filterLength = ArrayLength(pFilter);
                return (bool*)PyArray_BYTES(pFilter);
            }
        }
    }
    filterLength = 0;
    return NULL;
}

//===============================================================================
//
static int64_t GetBaseIndex(PyObject* kwargs) {
    if (kwargs && PyDict_Check(kwargs)) {
        PyObject* pBaseIndex = NULL;
        // Borrowed reference
        // Returns NULL if key not present
        pBaseIndex = PyDict_GetItemString(kwargs, "base_index");
        if (pBaseIndex != NULL && PyLong_Check(pBaseIndex)) {

            long baseindex = PyLong_AsLong(pBaseIndex);

            // only zero or one allowed
            if (baseindex == 0) return 0;
        }
    }
    return 1;
}



//===============================================================================
// TODO: Need to add checks for some array allocations in this function (to see if they succeeded,
//       and if not, free any other allocated arrays before setting a PyExc_MemoryError and returning).
template<typename UINDEX>
static PyObject* GroupFromLexSortInternal(
    PyObject* kwargs,
    UINDEX* pIndex,
    npy_intp    indexLength,
    npy_intp    indexLengthValues,
    void* pValues,
    npy_intp    itemSizeValues) {

    int64_t    cutOffLength = 0;
    int64_t* pCutOffs = GetCutOffs(kwargs, cutOffLength);

    int64_t    filterLength = 0;
    bool* pFilter = GetFilter(kwargs, filterLength);

    int64_t    base_index = GetBaseIndex(kwargs);


    if (pCutOffs && pCutOffs[cutOffLength - 1] != indexLength) {
        return PyErr_Format(PyExc_ValueError, "GroupFromLexSort last cutoff length does not match array length %lld", indexLength);
    }
    if (cutOffLength == -1) {
        return PyErr_Format(PyExc_ValueError, "GroupFromLexSort 'cutoffs' must be an array of type int64");
    }
    if (pFilter && filterLength != indexLength) {
        return PyErr_Format(PyExc_ValueError, "GroupFromLexSort filter length does not match array length %lld", indexLength);
    }

    // The countout always reserves the zero bin (even for when base_index =0) for filtering out
    // TODO: Change this to use type npy_intp and check for overflow.
    int64_t worstCase = indexLength + 1 + cutOffLength;

    PyArrayObject* const keys = AllocateNumpyArray(1, (npy_intp*)&indexLengthValues, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
    PyArrayObject* const first = AllocateNumpyArray(1, (npy_intp*)&indexLength, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
    PyArrayObject* const count = AllocateNumpyArray(1, (npy_intp*)&worstCase, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);

    // Make sure allocations succeeded
    if (!keys || !first || !count)
    {
        // Release/recycle any of the arrays which _were_ successfully allocated so they're not leaked.
        if (keys) { Py_XDECREF(keys); }
        if (first) { Py_XDECREF(first); }
        if (count) { Py_XDECREF(count); }

        return PyErr_Format(PyExc_MemoryError, "GroupFromLexSort out of memory length %lld", indexLength);
    }

    UINDEX* pKeyOut = (UINDEX*)PyArray_BYTES(keys);
    UINDEX* pFirstOut = (UINDEX*)PyArray_BYTES(first);
    UINDEX* pCountOut = (UINDEX*)PyArray_BYTES(count);

    int64_t uniqueCount = 0;
    GROUP_INDEX_FUNC  gpfunc = GroupIndex64;

    if (sizeof(UINDEX) == 4) gpfunc = GroupIndex32;

    if (pCutOffs) {
        PyArrayObject* uniqueCounts = AllocateNumpyArray(1, (npy_intp*)&cutOffLength, NPY_INT64);
        int64_t* pUniqueCounts = (int64_t*)PyArray_BYTES(uniqueCounts);

        PLOGGING("partition version col: %lld  %p  %p  %p\n", cutOffLength, pToSort, pToSort + arrayLength, pValues);

        struct stPGROUP {
            GROUP_INDEX_FUNC  funcSingleGroup;
            int64_t* pUniqueCounts;

            int64_t* pCutOffs;
            int64_t             cutOffLength;

            char* pValues;
            char* pIndex;
            char* pKeyOut;
            char* pFirstOut;
            char* pCountOut;
            bool* pFilter;
            int64_t             base_index;
            int64_t             strlen;
            int64_t             sizeofUINDEX;

        } pgroup;

        pgroup.funcSingleGroup = gpfunc;
        pgroup.pUniqueCounts = pUniqueCounts;

        pgroup.pCutOffs = pCutOffs;
        pgroup.cutOffLength = cutOffLength;
        pgroup.pValues = (char*)pValues;
        pgroup.pIndex = (char*)pIndex;

        pgroup.pKeyOut = (char*)pKeyOut;
        pgroup.pFirstOut = (char*)pFirstOut;
        pgroup.pCountOut = (char*)pCountOut;
        pgroup.pFilter = pFilter;
        pgroup.base_index = base_index;
        pgroup.strlen = itemSizeValues;
        const int64_t INDEX_SIZE = (int64_t)sizeof(UINDEX);

        pgroup.sizeofUINDEX = INDEX_SIZE;

        // Use threads per partition
        auto lambdaPSCallback = [](void* callbackArgT, int core, int64_t workIndex) -> int64_t {
            stPGROUP* callbackArg = (stPGROUP*)callbackArgT;
            int64_t t = workIndex;
            int64_t partLength;
            int64_t partStart;

            if (t == 0) {
                partStart = 0;
            }
            else {
                partStart = callbackArg->pCutOffs[t - 1];
            }

            partLength = callbackArg->pCutOffs[t] - partStart;

            PLOGGING("[%lld] start: %lld  length: %lld\n", t, partStart, partLength);

            int64_t shift =
                // shift the data pointers to match the partition
                // call a single threaded merge
                callbackArg->pUniqueCounts[t] =
                callbackArg->funcSingleGroup(
                    callbackArg->pValues + (partStart * callbackArg->strlen),
                    partLength,
                    callbackArg->pIndex + (partStart * callbackArg->sizeofUINDEX),
                    callbackArg->pKeyOut + (partStart * callbackArg->sizeofUINDEX),
                    callbackArg->pFirstOut + (partStart * callbackArg->sizeofUINDEX),
                    callbackArg->pCountOut + (partStart * callbackArg->sizeofUINDEX),
                    callbackArg->pFilter + partStart,
                    0,       //callbackArg->base_index, fix for countout
                    callbackArg->strlen);

            return 1;
        };

        THREADER->DoMultiThreadedWork((int)cutOffLength, lambdaPSCallback, &pgroup);

        // TODO: make global routine
        int64_t totalUniques = 0;
        for (int i = 0; i < cutOffLength; i++) {
            totalUniques += pUniqueCounts[i];
            pUniqueCounts[i] = totalUniques;
        }

        //printf("Total uniques %lld\n", totalUniques);

        // TODO: fix up keys
        //parallel add?
        PyArrayObject* firstReduced = AllocateNumpyArray(1, (npy_intp*)&totalUniques, INDEX_SIZE == 4 ? NPY_INT32 : NPY_INT64);

        totalUniques++;
        PyArrayObject* countReduced = AllocateNumpyArray(1, (npy_intp*)&totalUniques, INDEX_SIZE == 4 ? NPY_INT32 : NPY_INT64);

        // ANOTHER PARALEL ROUTINE to copy
        struct stPGROUPADD {
            int64_t* pUniqueCounts;

            int64_t* pCutOffs;       // May be NULL (if so no partitions)
            int64_t             cutOffLength;

            char* pIndex;
            char* pKeyOut;
            char* pFirstOut;
            char* pCountOut;

            char* pFirstReduced;
            char* pCountReduced;
            bool* pFilter;

            int64_t             base_index;
            int64_t             sizeofUINDEX;

        } pgroupadd;

        pgroupadd.pUniqueCounts = pUniqueCounts;
        pgroupadd.pCutOffs = pCutOffs;
        pgroupadd.cutOffLength = cutOffLength;

        pgroupadd.pIndex = (char*)pIndex;

        pgroupadd.pKeyOut = (char*)pKeyOut;
        pgroupadd.pFirstOut = (char*)pFirstOut;
        pgroupadd.pCountOut = (char*)pCountOut;
        pgroupadd.pFilter = pFilter;

        pgroupadd.pFirstReduced = (char*)PyArray_BYTES(firstReduced);
        pgroupadd.pCountReduced = (char*)PyArray_BYTES(countReduced);

        // skip first value since reserved for zero bin (and assign it 0)
        for (int64_t c = 0; c < INDEX_SIZE; c++) {
            *pgroupadd.pCountReduced++ = 0;
        }

        pgroupadd.base_index = base_index;
        pgroupadd.sizeofUINDEX = sizeof(UINDEX);

        // Use threads per partition
        auto lambdaPGADDCallback = [](void* callbackArgT, int core, int64_t workIndex) -> int64_t {
            stPGROUPADD* callbackArg = (stPGROUPADD*)callbackArgT;
            int64_t t = workIndex;
            int64_t partLength;
            int64_t partStart;
            int64_t uniquesBefore;

            if (t == 0) {
                partStart = 0;
                uniquesBefore = 0;
            }
            else {
                partStart = callbackArg->pCutOffs[t - 1];
                uniquesBefore = callbackArg->pUniqueCounts[t - 1];
            }

            partLength = callbackArg->pCutOffs[t] - partStart;
            //printf("[%lld] start: %lld  length: %lld  ubefore: %lld\n", t, partStart, partLength, uniquesBefore);

            if (callbackArg->sizeofUINDEX == 4) {
                int32_t* pKey = (int32_t*)callbackArg->pKeyOut;

                // the iGroup is fixed up
                int32_t* pIndex = (int32_t*)callbackArg->pIndex;

                // pFirst is reduced to iFirstKey (only the uniques)
                int32_t* pFirst = (int32_t*)callbackArg->pFirstOut;
                int32_t* pFirstReduced = (int32_t*)callbackArg->pFirstReduced;

                // becomes nCount and the very first is reserved for zero bin
                // holds all the uniques + 1 for the zero bin.
                int32_t* pCount = (int32_t*)callbackArg->pCountOut;
                int32_t* pCountReduced = (int32_t*)callbackArg->pCountReduced;

                int32_t  ubefore = (int32_t)uniquesBefore;

                if (t != 0) {
                    pKey += partStart;
                    pIndex += partStart;

                    for (int64_t i = 0; i < partLength; i++) {
                        pKey[i] += ((int32_t)ubefore + 1); // start at 1 (to reserve zero bin), becomes ikey
                        pIndex[i] += (int32_t)partStart;
                    }
                }
                else {
                    pKey += partStart;

                    for (int64_t i = 0; i < partLength; i++) {
                        pKey[i] += ((int32_t)partStart + 1); // start at 1, becomes ikey
                    }
                }

                int64_t uniqueLength = callbackArg->pUniqueCounts[t] - uniquesBefore;
                //printf("first reduced %d %lld\n", ubefore, uniqueLength);
                pFirst += partStart;
                pFirstReduced += ubefore;

                pCount += partStart;
                pCountReduced += ubefore;

                // very first [0] is for zero bin
                //pCount++;

                for (int64_t i = 0; i < uniqueLength; i++) {
                    pFirstReduced[i] = pFirst[i] + (int32_t)partStart;
                    //printf("setting %lld ", (int64_t)pCount[i]);
                    pCountReduced[i] = pCount[i];
                }

            }
            else {

                int64_t* pKey = (int64_t*)callbackArg->pKeyOut;

                // the iGroup is fixed up
                int64_t* pIndex = (int64_t*)callbackArg->pIndex;

                // pFirst is reduced to iFirstKey (only the uniques)
                int64_t* pFirst = (int64_t*)callbackArg->pFirstOut;
                int64_t* pFirstReduced = (int64_t*)callbackArg->pFirstReduced;

                // becomes nCount and the very first is reserved for zero bin
                // holds all the uniques + 1 for the zero bin.
                int64_t* pCount = (int64_t*)callbackArg->pCountOut;
                int64_t* pCountReduced = (int64_t*)callbackArg->pCountReduced;

                int64_t  ubefore = (int64_t)uniquesBefore;

                if (t != 0) {
                    pKey += partStart;
                    pIndex += partStart;

                    for (int64_t i = 0; i < partLength; i++) {
                        pKey[i] += ((int64_t)ubefore + 1); // start at 1 (to reserve zero bin), becomes ikey
                        pIndex[i] += (int64_t)partStart;
                    }
                }
                else {
                    pKey += partStart;

                    for (int64_t i = 0; i < partLength; i++) {
                        pKey[i] += ((int64_t)partStart + 1); // start at 1, becomes ikey
                    }
                }

                int64_t uniqueLength = callbackArg->pUniqueCounts[t] - uniquesBefore;
                //printf("first reduced %d %lld\n", ubefore, uniqueLength);
                pFirst += partStart;
                pFirstReduced += ubefore;

                pCount += partStart;
                pCountReduced += ubefore;

                // very first [0] is for zero bin
                //pCount++;

                for (int64_t i = 0; i < uniqueLength; i++) {
                    pFirstReduced[i] = pFirst[i] + (int64_t)partStart;
                    //printf("setting %lld ", (int64_t)pCount[i]);
                    pCountReduced[i] = pCount[i];
                }

            }

            return 1;
        };

        THREADER->DoMultiThreadedWork((int)cutOffLength, lambdaPGADDCallback, &pgroupadd);

        Py_DecRef((PyObject*)first);
        //Py_DecRef((PyObject*)count);

        PyObject* returnObject = PyList_New(4);
        PyList_SET_ITEM(returnObject, 0, (PyObject*)keys);
        PyList_SET_ITEM(returnObject, 1, (PyObject*)firstReduced); // iFirstKey
        PyList_SET_ITEM(returnObject, 2, (PyObject*)countReduced); // nCountGroup

        PyList_SET_ITEM(returnObject, 3, (PyObject*)uniqueCounts);
        return returnObject;
    }
    else {

        // When multiple arrays are passed, we sort in order of how it is passed
        // Thus, the last array is the last sort, and therefore determines the primary sort order
        uniqueCount =
            gpfunc(
                pValues,
                indexLength,
                pIndex,
                pKeyOut,
                pFirstOut,
                pCountOut,
                pFilter,
                base_index,
                itemSizeValues);

        // prior we allocate based on worst case
        // now we know the actual unique counts
        // memcpy..
        // also count invalid bin
        PyArrayObject* firstReduced = AllocateNumpyArray(1, (npy_intp*)&uniqueCount, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
        int64_t copySize = sizeof(UINDEX) * uniqueCount;
        memcpy(PyArray_BYTES(firstReduced), pFirstOut, copySize);

        uniqueCount++;
        PyArrayObject* countReduced = AllocateNumpyArray(1, (npy_intp*)&uniqueCount, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
        copySize = sizeof(UINDEX) * uniqueCount;
        // reduced
        memcpy(PyArray_BYTES(countReduced), pCountOut, copySize);

        Py_DecRef((PyObject*)first);
        Py_DecRef((PyObject*)count);

        PyObject* returnObject = PyList_New(3);
        PyList_SET_ITEM(returnObject, 0, (PyObject*)keys);
        PyList_SET_ITEM(returnObject, 1, (PyObject*)firstReduced);
        PyList_SET_ITEM(returnObject, 2, (PyObject*)countReduced);
        return returnObject;
    }
}


//===============================================================================
// Args:
//     Checks for cutoffs
//     If no cutoffs
//     
//     Arg1: lex=int32_t/int64_t result from lexsort
//     Arg2: value array that was sorted -- if it came from a list, convert it to a void type
//
// Returns 3 arrays
// iGroup (from lexsort)
// iFirstKey
// nCount -- from which iFirst can be derived
//
// if filtering was used then arrLength != arrLengthValues
// when this happens, the iGroup will have unfilled values for filtered out locations
PyObject* GroupFromLexSort(PyObject* self, PyObject* args, PyObject* kwargs) {

    PyArrayObject* inArrSortIndex = NULL;
    PyArrayObject* inArrValues = NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArrSortIndex, &PyArray_Type, &inArrValues))
    {
        return PyErr_Format(PyExc_TypeError, "Invalid argument types and/or count for GroupFromLexSort.");
    }

    const auto arrLength = ArrayLength(inArrSortIndex);
    const auto arrLengthValues = ArrayLength(inArrValues);

    // Due to filtering, now allow a smaller array length which might index the entire
    // size of inArrValues
    if (arrLength > arrLengthValues) {
        return PyErr_Format(PyExc_ValueError, "GroupFromLexSort input array lengths do not match: %lld vs %lld", arrLength, arrLengthValues);
    }

    const auto itemSize = PyArray_ITEMSIZE(inArrValues);

    const auto dtype = PyArray_TYPE(inArrSortIndex);

    void* pIndex = PyArray_BYTES(inArrSortIndex);
    void* pValues = PyArray_BYTES(inArrValues);

    switch (dtype) {
    CASE_NPY_INT32:
        return GroupFromLexSortInternal<int32_t>(kwargs, (int32_t*)pIndex, arrLength, arrLengthValues, pValues, itemSize);
        break;

    CASE_NPY_INT64:
        return GroupFromLexSortInternal<int64_t>(kwargs, (int64_t*)pIndex, arrLength, arrLengthValues, pValues, itemSize);
        break;

    default:
        return PyErr_Format(PyExc_ValueError, "GroupFromLexSort does not support index type of %d", dtype);
    }
}


