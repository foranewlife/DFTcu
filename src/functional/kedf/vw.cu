#include "vw.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include <cmath>

namespace dftcu {

namespace {
__global__ void sqrt_rho_kernel(size_t size, const double* rho, double* sqrt_rho) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        sqrt_rho[i] = (rho[i] > 1e-30) ? sqrt(rho[i]) : 0.0;
    }
}

__global__ void vw_reciprocal_kernel(size_t size, gpufftComplex* phi_g, const double* gg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        // -nabla^2 -> gg in reciprocal space
        // V_vW = 0.5 * (-nabla^2 sqrt(rho)) / sqrt(rho)
        // Here we multiply by 0.5 * gg
        double factor = 0.5 * gg[i];
        phi_g[i].x *= factor;
        phi_g[i].y *= factor;
    }
}

__global__ void vw_potential_kernel(size_t size, const double* lap_phi, const double* phi, double* v_vw, double coeff) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (phi[i] > 1e-15) {
            v_vw[i] = coeff * lap_phi[i] / phi[i];
        } else {
            v_vw[i] = 0.0;
        }
    }
}
} // namespace

vonWeizsacker::vonWeizsacker(double coeff) : coeff_(coeff) {}

double vonWeizsacker::compute(const RealField& rho, RealField& v_kedf) {
    const Grid& grid = rho.grid();
    size_t nnr = grid.nnr();
    FFTSolver solver(grid);

    GPU_Vector<double> sqrt_rho(nnr);
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    sqrt_rho_kernel<<<grid_size, block_size>>>(nnr, rho.data(), sqrt_rho.data());
    GPU_CHECK_KERNEL

    ComplexField phi_g(grid);
    real_to_complex(nnr, sqrt_rho.data(), phi_g.data());

    solver.forward(phi_g);

    vw_reciprocal_kernel<<<grid_size, block_size>>>(nnr, phi_g.data(), grid.gg());
    GPU_CHECK_KERNEL

    solver.backward(phi_g);

    GPU_Vector<double> lap_phi(nnr);
    complex_to_real(nnr, phi_g.data(), lap_phi.data());

    vw_potential_kernel<<<grid_size, block_size>>>(nnr, lap_phi.data(), sqrt_rho.data(), v_kedf.data(), coeff_);
    GPU_CHECK_KERNEL

    // Energy = coeff * integral(rho * V_vW) dV
    // Note: the v_kedf already includes coeff_
    double energy = dot_product(nnr, rho.data(), v_kedf.data()) * grid.dv();

    return energy;
}

} // namespace dftcu
