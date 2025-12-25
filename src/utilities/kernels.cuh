#pragma once
#include "utilities/gpu_macro.cuh"

namespace dftcu {

void real_to_complex(size_t size, const double* real, gpufftComplex* complex);
void complex_to_real(size_t size, const gpufftComplex* complex, double* real);
double dot_product(size_t size, const double* a, const double* b);

// Vector operations
void v_add(size_t n, const double* a, const double* b, double* out);
void v_sub(size_t n, const double* a, const double* b, double* out);
void v_mul(size_t n, const double* a, const double* b, double* out);
void v_axpy(size_t n, double alpha, const double* x, double* y);
void v_scale(size_t n, double alpha, const double* x, double* out);
void v_sqrt(size_t n, const double* x, double* out);

}  // namespace dftcu
