#pragma once
#include "common_inc.h"

// For when integers have an invalid predefined
static const int8_t  GetInvalid(int8_t x) { return (int8_t)(0x80); };
static const int16_t GetInvalid(int16_t x) { return (int16_t)(0x8000); };
static const int32_t GetInvalid(int32_t x) { return (int32_t)(0x800000000000); };
static const int64_t GetInvalid(int64_t x) { return (int64_t)(0x8000000000000000); };

static const uint8_t  GetInvalid(uint8_t x) { return (uint8_t)(0xFF); };
static const uint16_t GetInvalid(uint16_t x) { return (uint16_t)(0xFFFF); };
static const uint32_t GetInvalid(uint32_t x) { return (uint32_t)(0xFFFFFFFF); };
static const uint64_t GetInvalid(uint64_t x) { return (uint64_t)(0xFFFFFFFFFFFFFFFF); };

static const float GetInvalid(float x) { return std::numeric_limits<float>::quiet_NaN(); };
static const double GetInvalid(double x) { return std::numeric_limits<double>::quiet_NaN(); };
static const long double GetInvalid(long double x) { return std::numeric_limits<long double>::quiet_NaN(); };


//-------------------------------------------------------------------------
