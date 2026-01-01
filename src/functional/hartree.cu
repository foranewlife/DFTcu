#include "hartree.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void hartree_kernel(size_t size, gpufftComplex* rho_g, const double* gg, double gcut) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        const double BOHR_TO_ANGSTROM = 0.529177210903;
        // Units: |G|^2 in Bohr^-2 is equivalent to Energy in Rydberg
        double g2_ang = gg[i];
        double g2 = g2_ang * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

        // Poisson equation: V(G) = 4pi * rho(G) / G^2 (in Hartree)
        // gcut is input in Rydberg.
        if (g2 > 1e-12 && (gcut < 0 || g2 <= gcut)) {
            double factor = 4.0 * constants::D_PI / g2;
            rho_g[i].x *= factor;
            rho_g[i].y *= factor;
        } else {
            rho_g[i].x = 0.0;
            rho_g[i].y = 0.0;
        }
    }
}
}  // namespace

Hartree::Hartree() {}

void Hartree::initialize_buffers(Grid& grid) {
    if (grid_ == &grid)
        return;
    grid_ = &grid;
    if (!fft_)
        fft_ = std::make_unique<FFTSolver>(grid);
    if (!rho_g_)
        rho_g_ = std::make_unique<ComplexField>(grid);
    if (!v_tmp_)
        v_tmp_ = std::make_unique<RealField>(grid);
}

void Hartree::compute(const RealField& rho, RealField& vh, double& energy) {
    Grid& grid = rho.grid();
    initialize_buffers(grid);
    size_t nnr = grid.nnr();

    real_to_complex(nnr, rho.data(), rho_g_->data());
    fft_->forward(*rho_g_);

    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;
    hartree_kernel<<<grid_size, block_size, 0, grid.stream()>>>(nnr, rho_g_->data(), grid.gg(),
                                                                gcut_);
    GPU_CHECK_KERNEL;

    fft_->backward(*rho_g_);
    complex_to_real(nnr, rho_g_->data(), vh.data(), grid.stream());

    v_mul(nnr, rho.data(), vh.data(), v_tmp_->data(), grid.stream());
    energy = 0.5 * v_sum(nnr, v_tmp_->data(), grid.stream()) * grid.dv_bohr();
}

}  // namespace dftcu
