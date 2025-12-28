#pragma once
#include "utilities/gpu_macro.cuh"

namespace dftcu {

void __attribute__((visibility("default")))
real_to_complex(size_t size, const double* real, gpufftComplex* complex,
                cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
complex_to_real(size_t size, const gpufftComplex* complex, double* real,
                cudaStream_t stream = nullptr);
double __attribute__((visibility("default")))
dot_product(size_t size, const double* a, const double* b, cudaStream_t stream = nullptr);

// Vector operations
void __attribute__((visibility("default")))
v_add(size_t n, const double* a, const double* b, double* out, cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_sub(size_t n, const double* a, const double* b, double* out, cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_mul(size_t n, const double* a, const double* b, double* out, cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_axpy(size_t n, double alpha, const double* x, double* y, cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_scale(size_t n, double alpha, const double* x, double* out, cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_sqrt(size_t n, const double* x, double* out, cudaStream_t stream = nullptr);
double __attribute__((visibility("default")))
v_sum(size_t n, const double* x, cudaStream_t stream = nullptr);

}  // namespace dftcu
