// If this is not included, calling PY_ARRAY functions will have a null value
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API
#define NO_IMPORT_ARRAY

#include "common.h"
#include "../atop/atop.h"
#include "../atop/threads.h"

#define LOGGING(...)

// For detecting npy scalar bools
typedef struct {
    PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

PyArray_Descr* g_pDescrLongLong = NULL;
PyArray_Descr* g_pDescrULongLong = NULL;

//int64_t default1 = -9223372036854775808L;
static int64_t  gDefaultInt64 = 0x8000000000000000;
static int32_t  gDefaultInt32 = 0x80000000;
static uint16_t gDefaultInt16 = 0x8000;
static uint8_t  gDefaultInt8 = 0x80;

static uint64_t gDefaultUInt64 = 0xFFFFFFFFFFFFFFFF;
static uint32_t gDefaultUInt32 = 0xFFFFFFFF;
static uint16_t gDefaultUInt16 = 0xFFFF;
static uint8_t  gDefaultUInt8 = 0xFF;

static float  gDefaultFloat = NAN;
static double gDefaultDouble = NAN;
static int8_t   gDefaultBool = 0;
static char   gString[1024] = { 0,0,0,0 };

//----------------------------------------------------
// returns pointer to a data type (of same size in memory) that holds the invalid value for the type
// does not yet handle strings
void* GetDefaultForType(int numpyInType) {
    void* pgDefault = &gDefaultInt64;

    switch (numpyInType) {
    case NPY_FLOAT:  pgDefault = &gDefaultFloat;
        break;
    case NPY_LONGDOUBLE:
    case NPY_DOUBLE: pgDefault = &gDefaultDouble;
        break;
        // BOOL should not really have an invalid value inhabiting the type
    case NPY_BOOL:   pgDefault = &gDefaultBool;
        break;
    case NPY_BYTE:   pgDefault = &gDefaultInt8;
        break;
    case NPY_INT16:  pgDefault = &gDefaultInt16;
        break;
    CASE_NPY_INT32:  pgDefault = &gDefaultInt32;
        break;
    CASE_NPY_INT64:  pgDefault = &gDefaultInt64;
        break;
    case NPY_UINT8:  pgDefault = &gDefaultUInt8;
        break;
    case NPY_UINT16: pgDefault = &gDefaultUInt16;
        break;
    CASE_NPY_UINT32: pgDefault = &gDefaultUInt32;
        break;
    CASE_NPY_UINT64: pgDefault = &gDefaultUInt64;
        break;
    case NPY_STRING: pgDefault = &gString;
        break;
    case NPY_UNICODE: pgDefault = &gString;
        break;
    default:
        printf("!!! likely problem in GetDefaultForType\n");
    }

    return pgDefault;
}



//----------------------------------------------------------------
// Calculate the total number of bytes used by the array.
// TODO: Need to extend this to accomodate strided arrays.
int64_t CalcArrayLength(int ndim, npy_intp* dims) {
    int64_t length = 1;

    // handle case of zero length array
    if (dims && ndim > 0) {
        for (int i = 0; i < ndim; i++) {
            length *= dims[i];
        }
    }
    else {
        // Want to set this to zero, but scalar issue?
        //length = 0;
    }
    return length;
}

//----------------------------------------------------------------
// calcluate the total number of bytes used
int64_t ArrayLength(PyArrayObject* inArr) {

    return CalcArrayLength(PyArray_NDIM(inArr), PyArray_DIMS(inArr));
}

//-----------------------------------------------------------------------------------
PyArrayObject* AllocateNumpyArray(int ndim, npy_intp* dims, int32_t numpyType, int64_t itemsize, int fortran_array, npy_intp* strides) {

    PyArrayObject* returnObject = nullptr;
    const int64_t    len = CalcArrayLength(ndim, dims);

    // PyArray_New (and the functions it wraps) don't truly respect the 'flags' argument
    // passed into them; they only check whether it's zero or non-zero, and based on that they
    // set the NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY flags. Construct our flags value so we end
    // up with an array with the layout the caller requested.
    const int array_flags = fortran_array ? NPY_ARRAY_F_CONTIGUOUS : 0;

    // Make one dimension size on stack
    volatile int64_t dimensions[1] = { len };

    // This is the safest way...
    if (!dims) {
        // Happens with a=FA([]); 100*a;  or  FA([1])[0] / FA([2])
        ndim = 1;
        dims = (npy_intp*)dimensions;
    }

    PyTypeObject* const allocType =  pPyArray_Type;

    // probably runt object from matlab -- have to fix this up or it will fail
    // comes from empty strings in matlab - might need to
    if (PyTypeNum_ISFLEXIBLE(numpyType) && itemsize == 0) {
        itemsize = 1;
    }

    // NOTE: this path taken when we already have data in our own memory
    returnObject = (PyArrayObject*)PyArray_New(
        allocType,
        ndim,
        dims,
        numpyType,
        strides,      // Strides
        nullptr,
        (int)itemsize,
        array_flags,
        NULL);

    if (!returnObject) {
        printf("!!!out of memory allocating numpy array size:%lld  dims:%d  dtype:%d  itemsize:%lld  flags:%d  dim0:%lld\n", (long long)len, ndim, numpyType, (long long)itemsize, array_flags, (long long)dims[0]);
        return nullptr;
    }

    return returnObject;
}


//-----------------------------------------------------------------------------------
// NOTE: will only allocate 1 dim arrays
PyArrayObject* AllocateLikeResize(PyArrayObject* inArr, npy_intp rowSize) {
    int numpyType = PyArray_TYPE(inArr);

    PyArrayObject* result = NULL;

    int64_t itemSize = PyArray_ITEMSIZE(inArr);
    result = AllocateNumpyArray(1, &rowSize, numpyType, itemSize);

    return result;
}

//-----------------------------------------------------------------------------------
// Check recycle pool
PyArrayObject* AllocateLikeNumpyArray(PyArrayObject* inArr, int numpyType) {
    const int ndim = PyArray_NDIM(inArr);
    npy_intp* const dims = PyArray_DIMS(inArr);

    // If the strides are all "normal", the array is C_CONTIGUOUS,
    // and this is _not_ a string / flexible array, try to re-use an array
    // from the recycler (array pool).
    if ((PyArray_ISNUMBER(inArr) || PyArray_ISBOOL(inArr)) && PyArray_ISCARRAY(inArr))
    {
        return AllocateNumpyArray(ndim, dims, numpyType, 0, false, nullptr);
    }

    // If we couldn't re-use an array from the recycler (for whatever reason),
    // allocate a new one based on the old one but override the type.
    // TODO: How to handle the case where either the prototype array is a string array xor numpyType is a string type?
    //       (For the latter, we don't have the target itemsize available here, so we don't know how to allocate the array.)
    PyArray_Descr* descr = NULL;

    if (numpyType == NPY_STRING || numpyType == NPY_UNICODE || numpyType == NPY_VOID) {
        descr = PyArray_DescrFromObject((PyObject*)inArr, NULL);
    }

    if (!descr) {
        descr = PyArray_DescrFromType(numpyType);
    }

    if (!descr) {
        return nullptr;
    }

    // new code for strings
    //descr->elsize = PyArray_ITEMSIZE(inArr);
    
    PyArrayObject* returnObject = (PyArrayObject*)PyArray_NewLikeArray(inArr, NPY_KEEPORDER, descr, 1);

    if (!returnObject) {
        return nullptr;
    }

    return returnObject;
}

//---------------------------------------------------------------------------
// Takes as input a scalar object that is a bool, float, or int
// Take as input the numpyOutType you want
// The output of pDest holds the value represented in a 256bit AVX2 register
//
// RETURNS:
// TRUE on success
// *ppDataIn points to scalar object
// pItemSize set to 0 unless a string or unicode
//
// CONVERTS scalar inObject1 to --> numpyOutType filling in pDest
// If inObject1 is a string or unicode, then ppDataIn is filled in with the itemsize
//
// NOTE: Cannot handle numpy scalar types like numpy.int32
BOOL ConvertScalarObject(PyObject* inObject1,  void* pDestVoid, int16_t numpyOutType, void** ppDataIn, int64_t* pItemSize) {
    // defined in common.cpp
    // Structs used to hold any type of AVX 256 bit registers
    struct _m128comboi {
        __m128i  i1;
        __m128i  i2;
    };

    struct _m256all {
        union {
            __m256i  i;
            __m256d  d;
            __m256   s;
            _m128comboi ci;
        };
    };

    _m256all* pDest = (_m256all * )pDestVoid;
    *pItemSize = 0;
    *ppDataIn = pDest;

    BOOL isNumpyScalarInteger = PyArray_IsScalar((inObject1), Integer);
    bool isPyBool = PyBool_Check(inObject1);
    LOGGING("In convert scalar object!  %d %d %d\n", numpyOutType, isNumpyScalarInteger, isPyBool);

    if (isPyBool || PyArray_IsScalar((inObject1), Bool)) {
        int64_t value = 1;
        if (isPyBool) {
            if (inObject1 == Py_False)
                value = 0;
        }
        else {

            // Must be a numpy scalar array type, pull the value (see scalartypes.c.src)
            value = ((PyBoolScalarObject*)inObject1)->obval;
        }

        switch (numpyOutType) {
        case NPY_BOOL:
        case NPY_INT8:
        case NPY_UINT8:
            pDest->i = _mm256_set1_epi8((int8_t)value);
            break;
        case NPY_INT16:
        case NPY_UINT16:
            pDest->i = _mm256_set1_epi16((int16_t)value);
            break;
        CASE_NPY_UINT32:
        CASE_NPY_INT32:
            pDest->i = _mm256_set1_epi32((int32_t)value);
            break;
        CASE_NPY_UINT64:
        CASE_NPY_INT64:
            pDest->ci.i1 = _mm_set1_epi64x(value);
            pDest->ci.i2 = _mm_set1_epi64x(value);
            break;
        case NPY_FLOAT32:
            pDest->s = _mm256_set1_ps((float)value);
            break;
        case NPY_FLOAT64:
            pDest->d = _mm256_set1_pd((double)value);
            break;
        default:
            printf("unknown bool scalar type in convertScalarObject %d\n", numpyOutType);
            return FALSE;
        }
    }
    else
        if (PyLong_Check(inObject1) || isNumpyScalarInteger) {

            int overflow = 0;
            int64_t value;
            uint64_t value2;

            if (!isNumpyScalarInteger) {
                value = PyLong_AsLongLongAndOverflow(inObject1, &overflow);

                // overflow of 1 indicates past LONG_MAX
                // overflow of -1 indicate past LONG_MIN which we do not handle
                // PyLong_AsLongLong will RAISE an overflow exception

                // If the value is negative, conversion to unsigned not allowed
                if (value >= 0 || overflow == 1) {
                    value2 = PyLong_AsUnsignedLongLongMask(inObject1);
                }
                else {
                    value2 = (uint64_t)value;
                }
            }
            else {
                PyArray_Descr* dtype = PyArray_DescrFromScalar(inObject1);
                //// NOTE: memory leak here?
                if (dtype->type_num <= NPY_LONGDOUBLE) {
                    if (g_pDescrLongLong == NULL) {
                        g_pDescrLongLong = PyArray_DescrNewFromType(NPY_LONGLONG);
                        g_pDescrULongLong = PyArray_DescrNewFromType(NPY_ULONGLONG);
                    }

                    PyArray_CastScalarToCtype(inObject1, &value, g_pDescrLongLong);
                    PyArray_CastScalarToCtype(inObject1, &value2, g_pDescrULongLong);
                }
                else {
                    // datetime64 falls into here
                    LOGGING("!!punting on scalar type is %d\n", dtype->type_num);
                    return FALSE;
                }
            }

            switch (numpyOutType) {
            case NPY_BOOL:
            case NPY_INT8:
                pDest->i = _mm256_set1_epi8((int8_t)value);
                break;
            case NPY_UINT8:
                pDest->i = _mm256_set1_epi8((uint8_t)value2);
                break;
            case NPY_INT16:
                pDest->i = _mm256_set1_epi16((int16_t)value);
                break;
            case NPY_UINT16:
                pDest->i = _mm256_set1_epi16((uint16_t)value2);
                break;
            CASE_NPY_INT32:
                pDest->i = _mm256_set1_epi32((int32_t)value);
                break;
            CASE_NPY_UINT32:
                pDest->i = _mm256_set1_epi32((uint32_t)value2);
                break;
            CASE_NPY_INT64:
                pDest->ci.i1 = _mm_set1_epi64x(value);
                pDest->ci.i2 = _mm_set1_epi64x(value);
                break;
            CASE_NPY_UINT64:
                pDest->ci.i1 = _mm_set1_epi64x(value2);
                pDest->ci.i2 = _mm_set1_epi64x(value2);
                break;
            case NPY_FLOAT32:
                pDest->s = _mm256_set1_ps((float)value);
                break;
            case NPY_FLOAT64:
                pDest->d = _mm256_set1_pd((double)value);
                break;
            default:
                printf("unknown long scalar type in convertScalarObject %d\n", numpyOutType);
                return FALSE;
            }
        }
        else if (PyFloat_Check(inObject1) || PyArray_IsScalar((inObject1), Floating)) {

            double value = PyFloat_AsDouble(inObject1);

            switch (numpyOutType) {
            case NPY_BOOL:
            case NPY_INT8:
                pDest->i = _mm256_set1_epi8((int8_t)value);
                break;
            case NPY_UINT8:
                pDest->i = _mm256_set1_epi8((uint8_t)value);
                break;
            case NPY_INT16:
                pDest->i = _mm256_set1_epi16((int16_t)value);
                break;
            case NPY_UINT16:
                pDest->i = _mm256_set1_epi16((uint16_t)value);
                break;
            CASE_NPY_UINT32:
                pDest->i = _mm256_set1_epi32((uint32_t)value);
                break;
            CASE_NPY_INT32:
                pDest->i = _mm256_set1_epi32((int32_t)value);
                break;
            CASE_NPY_UINT64:
                pDest->ci.i1 = _mm_set1_epi64x((uint64_t)value);
                pDest->ci.i2 = _mm_set1_epi64x((uint64_t)value);
                break;
            CASE_NPY_INT64:
                pDest->ci.i1 = _mm_set1_epi64x((int64_t)value);
                pDest->ci.i2 = _mm_set1_epi64x((int64_t)value);
                break;
            case NPY_FLOAT32:
                pDest->s = _mm256_set1_ps((float)value);
                break;
            case NPY_FLOAT64:
                pDest->d = _mm256_set1_pd((double)value);
                break;
            case NPY_LONGDOUBLE:
                pDest->d = _mm256_set1_pd((double)(long double)value);
                break;
            default:
                printf("unknown float scalar type in convertScalarObject %d\n", numpyOutType);
                return FALSE;
            }

        }
        else if (PyBytes_Check(inObject1)) {
            // happens when pass in b'test'
            *pItemSize = Py_SIZE(inObject1);
            *ppDataIn = ((PyBytesObject*)(inObject1))->ob_sval;
            return TRUE;
        }
        else if (PyUnicode_Check(inObject1)) {
            // happens when pass in 'test'
            *pItemSize = PyUnicode_GET_SIZE(inObject1) * 4;
            // memory leak needs to be deleted
            *ppDataIn = PyUnicode_AsUCS4Copy(inObject1);
            return TRUE;
        }
        else if (PyArray_IsScalar(inObject1, Generic)) {

            // only integers are not subclassed in numpy world
            if (PyArray_IsScalar((inObject1), Integer)) {
                PyArray_Descr* dtype = PyArray_DescrFromScalar(inObject1);

                // NOTE: memory leak here?
                printf("!!integer scalar type is %d\n", dtype->type_num);
                return FALSE;
            }
            else {
                printf("!!unknown numpy scalar type in convertScalarObject %d --", numpyOutType);
                return FALSE;
            }

        }

        else {
            // Complex types hits here
            LOGGING("!!unknown scalar type in convertScalarObject %d --", numpyOutType);
            PyTypeObject* type = inObject1->ob_type;
            LOGGING("type name is %s\n", type->tp_name);
            return FALSE;
        }

    //printf("returning from scalar conversion\n");
    return TRUE;
}



//===================================================
//------------------------------------------------------------------------------
// Determines the if array is contiguous, which allows for one loop
// The stride of the loop is returned
// Each array has 3 possible properties:
// 1) Itemsize contiguous (vector math and threading possible)
//    Example: a=arange(20)  or a=arange(20).reshape(5,4) or a=arange(20).reshape((5,2,2), order='F')
// 2) Strided contiguous (threading possible -- vector math possible only with gather)
//    Example: a=arange(20)[::-1] or a=arange(20)[::2]
// 3) Neither contiguous (must be 2 or more dimensions and at least one dimension is strided contiguous)
//    Requires multiple loops to process data
//    Example: a=arange(20).reshape(5,4)[::-1] or a=arange(20).reshape(5,4)[::2]
// Returns:
//  ndim:   number of dimensions
//  stride: stride to use if contig is TRUE
//  direction: 0 - neither RowMajor or ColMajor (fully contiguous)
//     > 0 RowMajor with value being the dimension where contiguous breaks
//     < 0 ColMajor with -value being the dimension where contiguous breaks
//  return value 0: one loop can process all data, FALSE = multiple loops
//  NOTE: if return value is 0 and itemsze == stride, then vector math possible
//
int GetStridesAndContig(PyArrayObject* inArray, int& ndim, int64_t& stride) {
    stride = PyArray_ITEMSIZE(inArray);
    int direction = 0;
    ndim = PyArray_NDIM(inArray);
    if (ndim > 0) {
        stride = PyArray_STRIDE(inArray, 0);
        if (ndim > 1) {
            // at least two strides
            int ndims = PyArray_NDIM(inArray);
            int64_t lastStride = PyArray_STRIDE(inArray, ndims - 1);
            if (lastStride == stride) {
                // contiguous with one of the dimensions having length 1
            }
            else
                if (std::abs(lastStride) < std::abs(stride)) {
                    // Row Major - 'C' Style
                    // work backwards
                    int currentdim = ndims - 1;
                    int64_t curStrideLen = lastStride;
                    while (currentdim != 0) {
                        curStrideLen *= PyArray_DIM(inArray, currentdim);
                        LOGGING("'C' %lld vs %lld  dim: %lld  stride: %lld \n", curStrideLen, PyArray_STRIDE(inArray, currentdim - 1), PyArray_DIM(inArray, currentdim - 1), lastStride);
                        if (PyArray_STRIDE(inArray, currentdim - 1) != curStrideLen)
                            break;
                        currentdim--;
                    }
                    stride = lastStride;
                    direction = currentdim;
                }
                else {
                    // Col Major - 'F' Style
                    int currentdim = 0;
                    int64_t curStrideLen = stride;
                    while (currentdim != (ndims - 1)) {
                        curStrideLen *= PyArray_DIM(inArray, currentdim);
                        LOGGING("'F' %lld vs %lld  dim:  %lld   stride: %lld \n", curStrideLen, PyArray_STRIDE(inArray, currentdim + 1), PyArray_DIM(inArray, currentdim + 1), stride);
                        if (PyArray_STRIDE(inArray, currentdim + 1) != curStrideLen)
                            break;
                        currentdim++;
                    }
                    // think!
                    //direction = (ndims - 1) - currentdim;
                    direction = currentdim - (ndims - 1);
                    //contig = currentdim == (ndims - 1);
                }
        }
    }
    return direction;
}




//-------------------------------------------------------------------
// TODO: Make this a class
// Free what was allocated with AllocArrayInfo
void
FreeArrayInfo(ArrayInfo* pAlloc) {
    if (pAlloc) {
        int64_t* pRawAlloc = (int64_t*)pAlloc;

        // go back one to find where we stuffed the array size
        --pRawAlloc;

        int64_t tupleSize = *pRawAlloc;
        // The next entry is the arrayInfo
        ArrayInfo* aInfo = (ArrayInfo*)&pRawAlloc[1];
        for (int64_t i = 0; i < tupleSize; i++) {
            if (aInfo[i].pOriginalObject) {
                Py_DecRef((PyObject*)aInfo[i].pObject);
            }
        }
        WORKSPACE_FREE(pRawAlloc);
    }
}

//---------------------------------------------------------
// Allocate array info object we can free laater
ArrayInfo*
AllocArrayInfo(int64_t tupleSize) {
    int64_t* pRawAlloc = (int64_t*)WORKSPACE_ALLOC((sizeof(ArrayInfo) * tupleSize) + sizeof(int64_t));
    if (pRawAlloc) {
        // store in first 8 bytes the count
        *pRawAlloc = tupleSize;

        // The next entry is the arrayInfo
        ArrayInfo* aInfo = (ArrayInfo*)&pRawAlloc[1];

        // make sure we clear out pOriginalObject
        for (int64_t i = 0; i < tupleSize; i++) {
            aInfo[i].pOriginalObject = NULL;
        }
        return aInfo;
    }
    return NULL;
}

//-----------------------------------------------------------
// Checks to see if an array is contiguous, if not, it makes a copy
// If a copy is made, the caller is responsible for decrementing the ref count
// Returns NULL on failure
// On Success returns back a contiguous array
PyArrayObject* EnsureContiguousArray(PyArrayObject* inObject) {
    int arrFlags = PyArray_FLAGS(inObject);

    // make sure C or F contiguous
    if (!(arrFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
        // Have to make a copy (which needs to be deleted later)
        inObject = (PyArrayObject*)PyArray_FromAny((PyObject*)inObject, NULL, 0, 0, NPY_ARRAY_ENSURECOPY, NULL);
        if (!inObject) {
            PyErr_Format(PyExc_ValueError, "RipTide: Error converting non-contiguous array");
            return NULL;
        }
    }
    return inObject;
}

// Pass in a list or tuple of arrays of the same size
// Returns an array of info (which must be freed later)
// checkrows:
// convert: whether or not to convert non-contiguous arrays
ArrayInfo* BuildArrayInfo(
    PyObject* listObject,
    int64_t* pTupleSize,
    int64_t* pTotalItemSize,
    BOOL checkrows,
    BOOL convert) {

    bool isTuple = false;
    bool isArray = false;
    bool isList = false;
    int64_t tupleSize = 0;


    if (PyArray_Check(listObject)) {
        isArray = true;
        tupleSize = 1;
    }
    else

        if (PyTuple_Check(listObject)) {
            isTuple = true;
            tupleSize = PyTuple_GET_SIZE(listObject);
        }
        else
            if (PyList_Check(listObject))
            {
                isList = true;
                tupleSize = PyList_GET_SIZE(listObject);
            }
            else {

                PyErr_Format(PyExc_ValueError, "BuildArrayInfo must pass in a list or tuple");
                return NULL;
            }

    // NOTE: If the list is empty, this will allocate 0 memory (which C99 says can return NULL
    ArrayInfo* aInfo = AllocArrayInfo(tupleSize);

    int64_t totalItemSize = 0;

    // Build a list of array information so we can rotate it
    for (int64_t i = 0; i < tupleSize; i++) {
        PyObject* inObject = NULL;

        if (isTuple) {
            inObject = PyTuple_GET_ITEM(listObject, i);
        }

        if (isList) {
            inObject = PyList_GetItem(listObject, i);
        }

        if (isArray) {
            inObject = listObject;
        }

        if (inObject == Py_None) {
            // NEW Code to handle none
            aInfo[i].pObject = NULL;
            aInfo[i].ItemSize = 0;
            aInfo[i].NDim = 0;
            aInfo[i].NumpyDType = 0;
            aInfo[i].ArrayLength = 0;
            aInfo[i].pData = NULL;
            aInfo[i].NumBytes = 0;

        }
        else
            if (PyArray_Check(inObject)) {

                aInfo[i].pObject = (PyArrayObject*)inObject;

                // Check if we need to convert non-contiguous
                if (convert) {
                    // If we copy, we have an extra ref count
                    inObject = (PyObject*)EnsureContiguousArray((PyArrayObject*)inObject);
                    if (!inObject) {
                        goto EXIT_ROUTINE;
                    }

                    if ((PyArrayObject*)inObject != aInfo[i].pObject) {
                        // the pObject was copied and needs to be deleted
                        // pOriginalObject is the original object
                        aInfo[i].pOriginalObject = aInfo[i].pObject;
                        aInfo[i].pObject = (PyArrayObject*)inObject;
                    }
                }

                aInfo[i].ItemSize = PyArray_ITEMSIZE((PyArrayObject*)inObject);
                aInfo[i].NDim = PyArray_NDIM((PyArrayObject*)inObject);
                aInfo[i].NumpyDType = PyArray_TYPE((PyArrayObject*)inObject);
                aInfo[i].ArrayLength = ArrayLength(aInfo[i].pObject);

                //if (aInfo[i].NumpyDType == -1) {
                //    PyErr_Format(PyExc_ValueError, "BuildArrayInfo array has bad dtype of %d", PyArray_TYPE((PyArrayObject*)inObject));
                //    goto EXIT_ROUTINE;
                //}

                int64_t stride0 = PyArray_STRIDE((PyArrayObject*)inObject, 0);
                int64_t itemSize = aInfo[i].ItemSize;

                if (checkrows) {
                    if (aInfo[i].NDim != 1) {
                        PyErr_Format(PyExc_ValueError, "BuildArrayInfo array must have ndim ==1 instead of %d", aInfo[i].NDim);
                        goto EXIT_ROUTINE;
                    }
                    if (itemSize != stride0) {
                        PyErr_Format(PyExc_ValueError, "BuildArrayInfo array strides must match itemsize -- %lld %lld", itemSize, stride0);
                        goto EXIT_ROUTINE;
                    }
                }
                else {

                    if (itemSize != stride0) {
                        // If 2 dims and Fortran, then strides will not match
                        // TODO: better check
                        if (aInfo[i].NDim == 1) {
                            PyErr_Format(PyExc_ValueError, "BuildArrayInfo without checkows, array strides must match itemsize for 1 dim -- %lld %lld", itemSize, stride0);
                            goto EXIT_ROUTINE;
                        }
                    }
                }

                if (aInfo[i].ItemSize == 0 || aInfo[i].ArrayLength == 0) {
                    LOGGING("**zero size warning BuildArrayInfo: %lld %lld\n", aInfo[i].ItemSize, aInfo[i].ArrayLength);
                    //PyErr_Format(PyExc_ValueError, "BuildArrayInfo array must have size");
                    //goto EXIT_ROUTINE;
                }
                aInfo[i].pData = (char*)PyArray_BYTES(aInfo[i].pObject);
                aInfo[i].NumBytes = aInfo[i].ArrayLength * aInfo[i].ItemSize;

                LOGGING("Array %llu has %llu bytes  %llu size\n", i, aInfo[i].NumBytes, aInfo[i].ItemSize);
                totalItemSize += aInfo[i].ItemSize;
            }
            else {
                PyErr_Format(PyExc_ValueError, "BuildArrayInfo only accepts numpy arrays");
                goto EXIT_ROUTINE;
            }

    }

    // Don't perform checks for an empty list of arrays;
    // otherwise we'll dereference an empty 'aInfo'.
    if (checkrows && tupleSize > 0) {
        const int64_t totalRows = aInfo[0].ArrayLength;

        for (int64_t i = 0; i < tupleSize; i++) {
            if (aInfo[i].ArrayLength != totalRows) {
                PyErr_Format(PyExc_ValueError, "BuildArrayInfo all arrays must be same number of rows %llu", totalRows);
                goto EXIT_ROUTINE;
            }
        }
    }

    *pTupleSize = tupleSize;
    *pTotalItemSize = totalItemSize;
    return aInfo;

EXIT_ROUTINE:
    *pTupleSize = 0;
    *pTotalItemSize = 0;
    FreeArrayInfo(aInfo);
    return NULL;

}
