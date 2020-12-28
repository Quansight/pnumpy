#pragma once

#include <cstdlib>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32) && !defined(__GNUC__)
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define NOMINMAX
// Windows Header Files:
#include <Windows.h>
#include <winnt.h>
#endif


/*
Macro symbol definitions to simplify conditional code compilation within riptide.

References:
* https://sourceforge.net/p/predef/wiki/Compilers/

*/

/*
Platform/OS detection
*/

#if defined(_WIN32)
// Target OS is Windows
#   define RT_OS_WINDOWS 1

#elif defined(__linux__)
// Target OS is Linux
#   define RT_OS_LINUX 1

    // Target OS is UNIX-like
#   define RT_OS_FAMILY_UNIX 1

#elif defined(__APPLE__)
// Target OS is macOS or iOS
#   define RT_OS_DARWIN 1

    // Target OS is UNIX-like
#   define RT_OS_FAMILY_UNIX 1

    // Target OS is BSD-like
#   define RT_OS_FAMILY_BSD 1

#elif __FreeBSD__
// Target OS is FreeBSD
#   define RT_OS_FREEBSD 1

    // Target OS is UNIX-like
#   define RT_OS_FAMILY_UNIX 1

    // Target OS is BSD-like
#   define RT_OS_FAMILY_BSD 1

#else
// If we can't detect the OS, make it a compiler error; compilation is likely to fail anyway due to
// not having any working implementations of some functions, so at least we can make it obvious why
// the compilation is failing.
#   error Unable to detect/classify the target OS.

#endif  /* Platform/OS detection */


/*
Compiler detection.
The order these detection checks operate in is IMPORTANT -- use CAUTION if changing or reordering them!
*/

#if defined(__clang__)
// Compiler is Clang/LLVM.
#   define RT_COMPILER_CLANG 1

#elif defined(__GNUC__)
// Compiler is GCC/g++.
#   define RT_COMPILER_GCC 1

#elif defined(__INTEL_COMPILER) || defined(_ICC)
// Compiler is the Intel C/C++ compiler.
#   define RT_COMPILER_INTEL 1

#elif defined(_MSC_VER)
/*
This check needs to be towards the end; a number of compilers (e.g. clang, Intel C/C++)
define the _MSC_VER symbol when running on Windows, so putting this check last means we
should have caught any of those already and this should be bona-fide MSVC.
*/
// Compiler is the Microsoft C/C++ compiler.
#   define RT_COMPILER_MSVC 1

#else
// Couldn't detect the compiler.
// We could allow compilation to proceed anyway, but the compiler/platform behavior detection
// below won't pass and it's important for correctness so this is an error.
#   error Unable to detect/classify the compiler being used.

#endif  /* compiler detection */


/*
Compiler behavior detection.
For conciseness/correctness in riptide code, we define some additional symbols here specifying certain
compiler behaviors. This way any code depending on these behaviors expresses it in terms of the behavior
rather than whether it's being compiled under a specific compiler(s) and/or platforms; this in turn
makes it easier to support new compilers and platforms just by adding the necessary defines here.
*/

#if !defined(RT_COMPILER_MSVC)
// Indicates whether the targeted compiler/platform defaults to emitting vector load/store operations
// requiring an aligned pointer when a vector pointer is dereferenced (so any such pointers must be
// aligned to prevent segfaults). When zero/false, the targeted compiler/platform emits unaligned
// vector load/store instructions by default.
#   define RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED 1
#else
// Indicates whether the targeted compiler/platform defaults to emitting vector load/store operations
// requiring an aligned pointer when a vector pointer is dereferenced (so any such pointers must be
// aligned to prevent segfaults). When zero/false, the targeted compiler/platform emits unaligned
// vector load/store instructions by default.
#   define RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED 0
#endif  /* RT_TARGET_VECTOR_MEMOP_DEFAULT_ALIGNED */

//-------------------------------------------
//-------------------------------------------
#define VOID void
typedef void* PVOID;
typedef void* LPVOID;
typedef void* HANDLE;

#define TRUE 1
#define FALSE 0
typedef int BOOL;
typedef unsigned char       BYTE;


#if defined(_WIN32) && !defined(__GNUC__)
#define WINAPI      __stdcall
#define InterlockedCompareExchange128 _InterlockedCompareExchange128
#ifndef InterlockedAdd64
#define InterlockedAdd64 _InterlockedAdd64
#endif
#ifndef InterlockedDecrement64
#define InterlockedDecrement64 _InterlockedDecrement64
#define InterlockedIncrement64 _InterlockedIncrement64
#endif
#define InterlockedIncrement _InterlockedIncrement

#define FMInterlockedOr(X,Y) InterlockedOr64((int64_t*)X,Y)

#include <intrin.h>
#ifndef MEM_ALIGN
#define MEM_ALIGN(x) __declspec(align(x))
#define ALIGN(x) __declspec(align(64))

#define FORCEINLINE __forceinline
#define FORCE_INLINE __forceinline

#define ALIGNED_ALLOC(Size,Alignment) _aligned_malloc(Size,Alignment)
#define ALIGNED_FREE(block) _aligned_free(block)

#define lzcnt_64 _lzcnt_u64

#endif
#else

#define WINAPI
#include <pthread.h>

// consider sync_add_and_fetch
#define InterlockedAdd64(val, len) (__sync_fetch_and_add(val, len) + len)
#define InterlockedIncrement64(val) (__sync_fetch_and_add(val, 1) + 1)
#define InterlockedIncrement(val) (__sync_fetch_and_add(val, 1) + 1)
#define FMInterlockedOr(val, bitpos) (__sync_fetch_and_or(val, bitpos))


#ifndef __GNUC_PREREQ
#define __GNUC_PREREQ(major, minor) ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((major) << 16) + (minor)))
#endif
#ifndef MEM_ALIGN
#define MEM_ALIGN(x) __attribute__((aligned(x)))
#endif

#define FORCEINLINE inline __attribute__((always_inline))
#define FORCE_INLINE inline __attribute__((always_inline))
#define ALIGN(x) x __attribute__((aligned(64)))

// Workaround for platforms/compilers which don't support C11 aligned_alloc
// but which do have posix_memalign().
#ifndef aligned_alloc

#ifdef posix_memalign
FORCEINLINE void* aligned_alloc(size_t alignment, size_t size)
{
    void* buffer = NULL;
    posix_memalign(&buffer, alignment, size);
    return buffer;
}

#else
// clang compiler does not support so we default to malloc
//#warning Unable to determine how to perform aligned allocations on this platform.
#define aligned_alloc(alignment, size) malloc(size)
#endif  // defined(posix_memalign)

#endif  // !defined(aligned_alloc)

#define ALIGNED_ALLOC(Size,Alignment) aligned_alloc(Alignment,Size)
#define ALIGNED_FREE(block) free(block)

#define lzcnt_64 __builtin_clzll

#endif

// To detect CPU features like AVX-256
typedef struct {
    uint32_t f1c;
    uint32_t f1d;
    uint32_t f7b;
    uint32_t f7c;
} ATOP_cpuid_t;

// Missing types include
// Half Float
// A bool that takes up one bit
// 2 byte unicode
// pointers to variable length strings of 1,2,4 itemsize
enum ATOP_TYPES {
    ATOP_BOOL = 0,
    ATOP_INT8, ATOP_UINT8,
    ATOP_INT16, ATOP_UINT16,
    ATOP_INT32, ATOP_UINT32,
    ATOP_INT64, ATOP_UINT64,
    ATOP_INT128, ATOP_UINT128,
    ATOP_HALF_FLOAT, ATOP_FLOAT, ATOP_DOUBLE, ATOP_LONGDOUBLE,  // 11, 12, 13, 14
    ATOP_CHALF_FLOAT, ATOP_CFLOAT, ATOP_CDOUBLE, ATOP_CLONGDOUBLE,
    ATOP_STRING, ATOP_UNICODE,
    ATOP_VOID,
    ATOP_LAST
};

enum COMP_OPERATION {
    // Two inputs, Always return a bool
    CMP_EQ = 0,
    CMP_NE = 1,
    CMP_LT = 2,
    CMP_GT = 3,
    CMP_LTE = 4,
    CMP_GTE = 5,
    CMP_LAST = 6,
};

enum UNARY_OPERATION {
    UNARY_INVALID = 0,

    // One input, returns same data type
    ABS = 1,
    SIGNBIT = 2,
    FABS = 3,
    INVERT = 4,
    FLOOR = 5,
    CEIL = 6,
    TRUNC = 7,
    ROUND = 8,
    NEGATIVE = 9,
    POSITIVE = 10,
    SIGN = 11,
    RINT = 12,

    // One input, always return a float one input
    SQRT = 15,
    SQUARE = 16,
    RECIPROCAL = 17,

    // one input, output bool
    LOGICAL_NOT = 18,
    ISINF = 19,
    ISNAN = 20,
    ISFINITE = 21,
    ISNORMAL = 22,

    ISNOTINF = 23,
    ISNOTNAN = 24,
    ISNOTFINITE = 25,
    ISNOTNORMAL = 26,
    ISNANORZERO = 27,

    // One input, does not allow floats
    BITWISE_NOT = 28,      // same as invert?

    UNARY_LAST = 35,
};

enum BINARY_OPERATION {
    BINARY_INVALID = 0,

    // Two ops, returns same type
    ADD = 1,
    SUB = 2,
    MUL = 3,
    MOD = 4,   // Warning: there are two mods - C,Java mod  and Python mod

    MIN = 5,
    MAX = 6,
    NANMIN = 7,
    NANMAX = 8,
    FLOORDIV = 9,
    POWER = 10,
    REMAINDER = 11,
    FMOD = 12,

    // Two ops, always return a double
    DIV = 13,
    SUBDATETIMES = 14,  // returns double
    SUBDATES = 15,   // returns int

    // Two inputs, Always return a bool
    LOGICAL_AND = 16,
    LOGICAL_XOR = 17,
    LOGICAL_OR = 18,

    // Two inputs, second input must be int based
    BITWISE_LSHIFT = 19,    //left_shift
    BITWISE_RSHIFT = 20,
    BITWISE_AND = 21,
    BITWISE_XOR = 22,
    BITWISE_OR = 23,
    BITWISE_ANDNOT = 24,
    BITWISE_NOTAND = 25,
    BITWISE_XOR_SPECIAL = 26,

    ATAN2 = 27,
    HYPOT = 28,

    BINARY_LAST = 29,
};

enum TRIG_OPERATION {
    TRIG_INVALID = 0,
    // One op, returns same type
    SIN = 1,
    COS = 2,
    TAN = 3,
    ASIN = 4,
    ACOS = 5,
    ATAN = 6,
    SINH = 7,
    COSH = 8,
    TANH = 9,
    ASINH = 10,
    ACOSH = 11,
    ATANH = 12,

    LOG = 13,
    LOG2 = 14,
    LOG10 = 15,
    EXP = 16,
    EXP2 = 17,
    EXPM1 = 18,
    LOG1P = 19,
    CBRT = 20,

    TRIG_LAST = 21
};

//----------------------------------------------------------------------------------
// Lookup to go from 1 byte to 8 byte boolean values
extern int64_t gBooleanLUT64[256];
extern int32_t gBooleanLUT32[16];

extern int64_t gBooleanLUT64Inverse[256];
extern int32_t gBooleanLUT32Inverse[16];


enum SCALAR_MODE {
    NO_SCALARS = 0,
    FIRST_ARG_SCALAR = 1,
    SECOND_ARG_SCALAR = 2,
    BOTH_SCALAR = 3   // not used
};


typedef void(*UNARY_FUNC)(void* pDataIn, void* pDataOut, int64_t len, int64_t strideIn, int64_t strideOut);
// Pass in two vectors and return one vector
// Used for operations like C = A + B
typedef void(*ANY_TWO_FUNC)(void* pDataIn, void* pDataIn2, void* pDataOut, int64_t len, int64_t strideIn1, int64_t strideIn2, int64_t strideOut);
typedef void(*GROUPBY_FUNC)(void* pstGroupBy, int64_t index);
typedef void(*REDUCE_FUNC)(void* pDataIn1X, void* pDataOutX, void* pStartVal, int64_t datalen, int64_t strideIn);


//======================================================
// Unary
//------------------------------------------------------
// Macro stub for returning None
#define STRIDE_NEXT(_TYPE_, _MEM_, _STRIDE_) (_TYPE_*)((char*)_MEM_ + _STRIDE_)


//--------------------------------------------------------------------
// multithreaded struct used for calling unary op codes
struct UNARY_CALLBACK {
    UNARY_FUNC pUnaryCallback;

    char* pDataIn;
    char* pDataOut;

    int64_t itemSizeIn;
    int64_t itemSizeOut;
};


//====================================================================
void* FmAlloc(size_t _Size);
void FmFree(void* _Block);

#define WORKSPACE_ALLOC FmAlloc
#define WORKSPACE_FREE FmFree

#define MAX_STACK_ALLOC (1024 * 1024)

// For small buffers that can be allocated on the stack
#if defined(_WIN32) && !defined(__GNUC__)
#define POSSIBLY_STACK_ALLOC(_alloc_size_) _alloc_size_ > MAX_STACK_ALLOC ? (char*)WORKSPACE_ALLOC(_alloc_size_) : (char*)_malloca(_alloc_size_);
#else
#define POSSIBLY_STACK_ALLOC(_alloc_size_) _alloc_size_ > MAX_STACK_ALLOC ? (char*)WORKSPACE_ALLOC(_alloc_size_) : (char*)alloca(_alloc_size_);
#endif
#define POSSIBLY_STACK_FREE(_alloc_size_, _mem_ptr_) if (_alloc_size_ > MAX_STACK_ALLOC) WORKSPACE_FREE(_mem_ptr_);

//=======================================================================
// Conversions
struct stRecarrayOffsets {
    char* pData;
    int64_t    readoffset;
    int64_t    itemsize;
};

extern "C" void RecArrayToColMajor(
    stRecarrayOffsets* pstOffset,
    char* pStartOffset,
    int64_t totalRows,
    int64_t numArrays,
    int64_t itemSize);

//=====================================================================
// Sorting
enum SORT_MODE {
    SORT_MODE_QSORT = 1,
    SORT_MODE_MERGE = 2,
    SORT_MODE_HEAP = 3
};

extern "C" BOOL SortArray(void* pDataIn1, int64_t arraySize1, int32_t arrayType1, SORT_MODE mode);
extern "C" int64_t IsSorted(void* pDataIn1,int64_t arraySize1, int32_t arrayType1, int64_t itemSize);
extern "C" void SortIndex32(
    int64_t *   pCutOffs,
    int64_t     cutOffLength,
    void*       pDataIn1,
    int64_t     arraySize1,
    int32_t *   pDataOut1,
    SORT_MODE   mode,
    int         arrayType1,
    int64_t     strlen);

extern "C" void SortIndex64(
    int64_t * pCutOffs,
    int64_t     cutOffLength,
    void*       pDataIn1,
    int64_t     arraySize1,
    int64_t *   pDataOut1,
    SORT_MODE   mode,
    int         arrayType1,
    int64_t     strlen);


typedef int64_t(*GROUP_INDEX_FUNC)(
    void* pDataIn1,
    int64_t       arraySize1V,
    void* pDataIndexInV,
    void* pGroupOutV,
    void* pFirstOutV,
    void* pCountOutV,
    bool* pFilter,       // optional
    int64_t       base_index,
    int64_t       strlen);


extern "C" int64_t GroupIndex32(
    void* pDataIn1,
    int64_t    arraySize1V,
    void* pDataIndexInV,
    void* pGroupOutV,
    void* pFirstOutV,
    void* pCountOutV,
    bool* pFilter,       // optional
    int64_t       base_index,
    int64_t       strlen);

extern "C" int64_t GroupIndex64(
    void* pDataIn1,
    int64_t    arraySize1V,
    void* pDataIndexInV,
    void* pGroupOutV,
    void* pFirstOutV,
    void* pCountOutV,
    bool* pFilter,       // optional
    int64_t       base_index,
    int64_t       strlen);



