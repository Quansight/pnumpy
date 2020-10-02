#pragma once
#include "common_inc.h"

// Export DLL section
#if defined(_WIN32) && !defined(__GNUC__)

#define DllExport __declspec(dllexport)

#else 

#define DllExport

#endif


extern "C" {

    // defined in atop.cpp
    DllExport BOOL atop_init();

    // defined in ops_binary.cpp
    DllExport ANY_TWO_FUNC GetSimpleMathOpFast(int func, int atopInType1, int atopInType2, int* wantedOutType);
    DllExport REDUCE_FUNC GetReduceMathOpFast(int func, int atopInType1);
    DllExport ANY_TWO_FUNC GetComparisonOpFast(int func, int atopInType1, int atopInType2, int* wantedOutType);
    DllExport UNARY_FUNC GetUnaryOpFast(int func, int atopInType1, int* wantedOutType);

    // CPUID capabilities
    extern DllExport int g_bmi2;
    extern DllExport int g_avx2;
    extern DllExport ATOP_cpuid_t   g_cpuid;

}

