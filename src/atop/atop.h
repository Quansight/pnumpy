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

    DllExport ANY_TWO_FUNC GetComparisonOpFast(int func, int atopInType1, int atopInType2, int* wantedOutType);
}

