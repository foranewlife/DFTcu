#include "hartree.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
void __global__ hartree_kernel(size_t size, gpufftComplex* rho_g, const double* gg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        // Convert G² from Å⁻² to Bohr⁻²
        const double BOHR_TO_ANGSTROM = 0.529177210903;
        double g2 = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

        if (g2 > 1e-12) {
            double factor = 4.0 * constants::D_PI / g2;
            rho_g[i].x *= factor;
            rho_g[i].y *= factor;
        } else {
            rho_g[i].x = 0.0;
            rho_g[i].y = 0.0;
        }
    }
}

void __global__ scale_and_copy_kernel(size_t size, const gpufftComplex* complex, double* real,
                                      double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        real[i] = complex[i].x * scale;
    }
}
}  // namespace

void scale_and_copy_complex_to_real(size_t size, const gpufftComplex* complex, double* real,
                                    double scale, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    scale_and_copy_kernel<<<grid_size, block_size, 0, stream>>>(size, complex, real, scale);
    GPU_CHECK_KERNEL;
}

Hartree::Hartree() {}

void Hartree::initialize_buffers(Grid& grid) {
    if (grid_ == &grid)
        return;

    grid_ = &grid;
    if (!fft_) {
        fft_ = std::make_unique<FFTSolver>(grid);
    }
    if (!rho_g_) {
        rho_g_ = std::make_unique<ComplexField>(grid);
    }
    if (!v_tmp_) {
        v_tmp_ = std::make_unique<RealField>(grid);
    }
}

void Hartree::compute(const RealField& rho, RealField& vh, double& energy) {
    Grid& grid = rho.grid();
    initialize_buffers(grid);
    size_t nnr = grid.nnr();

    // 1. Copy Real to Complex
    real_to_complex(nnr, rho.data(), rho_g_->data());

    // 2. Forward FFT: rho(r) -> FFT(rho)
    fft_->forward(*rho_g_);

    // 3. Multiply by 4*pi/G^2. Note: FFT is unscaled, so no extra factors needed here.
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;
    hartree_kernel<<<grid_size, block_size, 0, grid.stream()>>>(nnr, rho_g_->data(), grid.gg());
    GPU_CHECK_KERNEL;

    // 4. Backward FFT: V_h(G) -> V_h_unnorm(r)
    fft_->backward(*rho_g_);

    // 5. Scale by 1/N to get physical potential and copy to output
    scale_and_copy_complex_to_real(nnr, rho_g_->data(), vh.data(), 1.0 / (double)nnr,
                                   grid.stream());

    // 6. Hartree energy: E_H = 0.5 * integral( rho(r) * v_h(r) ) dV
    v_mul(nnr, rho.data(), vh.data(), v_tmp_->data(), grid.stream());
    energy = 0.5 * v_sum(nnr, v_tmp_->data(), grid.stream()) * grid.dv();
}

}  // namespace dftcu
