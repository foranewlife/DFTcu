#pragma once
#include "model/field.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

namespace dftcu {

static void __global__ scale_kernel(size_t size, gpufftComplex* data, double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

class FFTSolver {
  public:
    FFTSolver(const Grid& grid) : grid_(grid) {
        const int* nr = grid_.nr();
        CHECK_FFT(gpufftPlan3d(&plan_, nr[0], nr[1], nr[2], GPUFFT_C2C));
    }

    ~FFTSolver() { gpufftDestroy(plan_); }

    void forward(ComplexField& field) {
        CHECK_FFT(gpufftExecC2C(plan_, field.data(), field.data(), GPUFFT_FORWARD));
        // Scaling: DFTpy scales by dV
        size_t size = field.size();
        const int block_size = 256;
        const int grid_size = (size + block_size - 1) / block_size;
        scale_kernel<<<grid_size, block_size>>>(size, field.data(), grid_.dv());
        GPU_CHECK_KERNEL
    }

    void backward(ComplexField& field) {
        CHECK_FFT(gpufftExecC2C(plan_, field.data(), field.data(), GPUFFT_INVERSE));
        // Scaling: cuFFT inverse is not normalized (scales by N).
        // DFTpy inverse scales by 1/dV.
        // Total scaling should be 1 / (N * dV) ?
        // Wait, DFTpy ifft: griddata_3d = self.ifft_object(self) / direct_grid.dV
        // cuFFT ifft result is already scaled by N.
        // So we need to scale by 1 / (N * dV).
        size_t size = field.size();
        const int block_size = 256;
        const int grid_size = (size + block_size - 1) / block_size;
        double scale = 1.0 / (grid_.nnr() * grid_.dv());
        scale_kernel<<<grid_size, block_size>>>(size, field.data(), scale);
        GPU_CHECK_KERNEL
    }

  private:
    const Grid& grid_;
    gpufftHandle plan_;
};

}  // namespace dftcu
