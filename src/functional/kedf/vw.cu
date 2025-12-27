#include <cmath>

#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "vw.cuh"

namespace dftcu {

namespace {
__global__ void vw_sqrt_kernel(size_t size, const double* rho, double* phi,
                               vonWeizsacker::Parameters params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        phi[i] = (rho[i] > params.rho_threshold) ? sqrt(rho[i]) : 0.0;
    }
}

__global__ void vw_laplacian_kernel(size_t size, gpufftComplex* phi_g, const double* gg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        // vW reciprocal space kernel: 0.5 * G^2
        double factor = 0.5 * gg[i];
        phi_g[i].x *= factor;
        phi_g[i].y *= factor;
    }
}

__global__ void vw_potential_kernel(size_t size, const double* lap_phi, const double* phi,
                                    double* v_vw, double coeff, vonWeizsacker::Parameters params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (phi[i] > params.phi_threshold) {
            v_vw[i] = coeff * lap_phi[i] / phi[i];
        } else {
            v_vw[i] = 0.0;
        }
    }
}
}  // namespace

vonWeizsacker::vonWeizsacker(double coeff) : coeff_(coeff) {}

double vonWeizsacker::compute(const RealField& rho, RealField& v_kedf) {
    Grid& grid = rho.grid();
    size_t nnr = grid.nnr();
    FFTSolver solver(grid);

    GPU_Vector<double> phi(nnr);
    vw_sqrt_kernel<<<(nnr + 255) / 256, 256, 0, grid.stream()>>>(nnr, rho.data(), phi.data(),
                                                                 params_);
    GPU_CHECK_KERNEL;

    ComplexField phi_g(grid);
    real_to_complex(nnr, phi.data(), phi_g.data());
    solver.forward(phi_g);

    vw_laplacian_kernel<<<(nnr + 255) / 256, 256, 0, grid.stream()>>>(nnr, phi_g.data(), grid.gg());
    GPU_CHECK_KERNEL;

    solver.backward(phi_g);
    GPU_Vector<double> lap_phi(nnr);
    complex_to_real(nnr, phi_g.data(), lap_phi.data());

    vw_potential_kernel<<<(nnr + 255) / 256, 256, 0, grid.stream()>>>(
        nnr, lap_phi.data(), phi.data(), v_kedf.data(), coeff_, params_);
    GPU_CHECK_KERNEL;

    grid.synchronize();
    double energy = dot_product(nnr, rho.data(), v_kedf.data()) * grid.dv();

    return energy;
}

}  // namespace dftcu
