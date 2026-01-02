#pragma once
#include <complex>

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
v_axpy(size_t n, std::complex<double> alpha, const gpufftComplex* x, gpufftComplex* y,
       cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_scale(size_t n, double alpha, const double* x, double* out, cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_scale(size_t n, double alpha, const gpufftComplex* x, gpufftComplex* out,
        cudaStream_t stream = nullptr);
void __attribute__((visibility("default")))
v_sqrt(size_t n, const double* x, double* out, cudaStream_t stream = nullptr);
/** @brief Add a constant shift to a real vector */
void __attribute__((visibility("default")))
v_shift(size_t n, double shift, const double* in, double* out, cudaStream_t stream = nullptr);
/** @brief Sum all elements of a real vector on the GPU */
double v_sum(size_t n, const double* x, cudaStream_t stream = nullptr);

/** @brief Swap the first two axes of a 3D grid (X and Y) */
void swap_xy(int nr0, int nr1, int nr2, const double* in, double* out,
             cudaStream_t stream = nullptr);
void swap_xy(int nr0, int nr1, int nr2, const gpufftComplex* in, gpufftComplex* out,
             cudaStream_t stream = nullptr);

}  // namespace dftcu
