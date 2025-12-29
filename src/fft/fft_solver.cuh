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
    }

    void backward(ComplexField& field) {
        CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)field.data(),
                                 (cufftDoubleComplex*)field.data(), CUFFT_INVERSE));
    }

  private:
    Grid& grid_;
    cufftHandle plan_;
};

}  // namespace dftcu
