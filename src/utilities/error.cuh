#pragma once
#include <fstream>
#include <string>
#include <vector>

#include "gpu_macro.cuh"

#include <stdio.h>

#define CHECK(call)                                                                 \
    do {                                                                            \
        const gpuError_t error_code = call;                                         \
        if (error_code != gpuSuccess) {                                             \
            fprintf(stderr, "CUDA Error:\n");                                       \
            fprintf(stderr, "    File:       %s\n", __FILE__);                      \
            fprintf(stderr, "    Line:       %d\n", __LINE__);                      \
            fprintf(stderr, "    Error code: %d\n", error_code);                    \
            fprintf(stderr, "    Error text: %s\n", gpuGetErrorString(error_code)); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

#define CHECK_FFT(call)                                          \
    do {                                                         \
        const cufftResult error_code = call;                     \
        if (error_code != CUFFT_SUCCESS) {                       \
            fprintf(stderr, "cuFFT Error:\n");                   \
            fprintf(stderr, "    File:       %s\n", __FILE__);   \
            fprintf(stderr, "    Line:       %d\n", __LINE__);   \
            fprintf(stderr, "    Error code: %d\n", error_code); \
            exit(1);                                             \
        }                                                        \
    } while (0)

#ifdef STRONG_DEBUG
#define GPU_CHECK_KERNEL               \
    {                                  \
        CHECK(gpuGetLastError());      \
        CHECK(gpuDeviceSynchronize()); \
    }
#else
#define GPU_CHECK_KERNEL \
    { CHECK(gpuGetLastError()); }
#endif
