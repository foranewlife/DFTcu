#pragma once

#include <cufft.h>

#include <iostream>

#include "model/field.cuh"
#include "model/grid.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

#define CUFFT_CHECK(condition)                                                              \
    {                                                                                       \
        cufftResult status = condition;                                                     \
        if (status != CUFFT_SUCCESS) {                                                      \
            std::cerr << "cuFFT error: " << status << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                         \
            exit(1);                                                                        \
        }                                                                                   \
    }

namespace {
__global__ void scale_kernel(size_t size, gpufftComplex* data, double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}
}  // namespace

class FFTSolver {
  public:
    FFTSolver(Grid& grid) : grid_(grid) {
        int nr[3] = {grid.nr()[0], grid.nr()[1], grid.nr()[2]};
        CUFFT_CHECK(cufftPlan3d(&plan_, nr[0], nr[1], nr[2], CUFFT_Z2Z));
        CUFFT_CHECK(cufftSetStream(plan_, grid.stream()));
    }

    ~FFTSolver() { cufftDestroy(plan_); }

    void forward(ComplexField& field) {
        CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)field.data(),
                                 (cufftDoubleComplex*)field.data(), CUFFT_FORWARD));
        // Add 1/N scaling to get physical quantity in G-space
        size_t n = grid_.nnr();
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        scale_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(n, field.data(),
                                                                   1.0 / (double)n);
    }

    void backward(ComplexField& field) {
        CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)field.data(),
                                 (cufftDoubleComplex*)field.data(), CUFFT_INVERSE));
    }

  private:
    Grid& grid_;
    cufftHandle plan_;
};

// Real-to-Complex FFT Solver for real-valued potentials (Gamma-point optimization)
// Uses CUFFT_D2Z (real-to-complex) and CUFFT_Z2D (complex-to-real)
// This exploits the Hermitian symmetry: v(-G) = v(G)* for real v(r)
class RealFFTSolver {
  public:
    RealFFTSolver(Grid& grid) : grid_(grid) {
        int nr[3] = {grid.nr()[0], grid.nr()[1], grid.nr()[2]};

        // For R2C transform
        CUFFT_CHECK(cufftPlan3d(&plan_r2c_, nr[0], nr[1], nr[2], CUFFT_D2Z));
        CUFFT_CHECK(cufftSetStream(plan_r2c_, grid.stream()));

        // For C2R transform
        CUFFT_CHECK(cufftPlan3d(&plan_c2r_, nr[0], nr[1], nr[2], CUFFT_Z2D));
        CUFFT_CHECK(cufftSetStream(plan_c2r_, grid.stream()));
    }

    ~RealFFTSolver() {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }

    // Forward transform: Real space → Complex (Gamma-optimized)
    // Input: RealField (nnr real values)
    // Output: ComplexField (reduced size due to Hermitian symmetry)
    void forward(RealField& real_in, ComplexField& complex_out) {
        CUFFT_CHECK(
            cufftExecD2Z(plan_r2c_, real_in.data(), (cufftDoubleComplex*)complex_out.data()));
    }

    // Backward transform: Complex (Gamma-optimized) → Real space
    // Input: ComplexField (Gamma-optimized)
    // Output: RealField (nnr real values)
    void backward(ComplexField& complex_in, RealField& real_out) {
        CUFFT_CHECK(
            cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)complex_in.data(), real_out.data()));
    }

  private:
    Grid& grid_;
    cufftHandle plan_r2c_;
    cufftHandle plan_c2r_;
};

}  // namespace dftcu
