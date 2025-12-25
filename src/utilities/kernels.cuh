#pragma once
#include "utilities/gpu_macro.cuh"

namespace dftcu {

void real_to_complex(size_t size, const double* real, gpufftComplex* complex);
void complex_to_real(size_t size, const gpufftComplex* complex, double* real);
double dot_product(size_t size, const double* a, const double* b);

}  // namespace dftcu
