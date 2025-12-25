#pragma once

const int NUM_ELEMENTS = 94;
#define PI 3.14159265358979323846
#define HBAR 1.0
#define K_B 1.0
#define K_C 1.0
#define K_C_SP 1.0f

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
