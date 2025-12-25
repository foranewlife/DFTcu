#include "wt.cuh"
#include "kernel.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include <cmath>

namespace dftcu {

namespace {
__global__ void power_rho_kernel(size_t size, const double* rho, double* rho_p, double p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        rho_p[i] = (rho[i] > 1e-30) ? pow(rho[i], p) : 0.0;
    }
}

__global__ void wt_reciprocal_kernel(size_t size, gpufftComplex* data_g, const double* gg, double tkf, double factor) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        double q = sqrt(gg[i]);
        double eta = q / tkf;
        double K = lindhard_function(eta, 1.0, 1.0) * factor;
        data_g[i].x *= K;
        data_g[i].y *= K;
    }
}

__global__ void wt_potential_kernel(size_t size, const double* rho, const double* v_conv, double* v_wt, double alpha, double beta, double coeff) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (rho[i] > 1e-30) {
            // For WT, alpha = beta = 5/6, so we assume alpha == beta here.
            // General V_NL = coeff * (alpha * rho^(alpha-1) * (K*rho^beta) + beta * rho^(beta-1) * (K*rho^alpha))
            // If alpha == beta, it simplifies to:
            v_wt[i] = coeff * 2.0 * alpha * pow(rho[i], alpha - 1.0) * v_conv[i];
            (void)beta; // Silence unused parameter
        } else {
            v_wt[i] = 0.0;
        }
    }
}
} // namespace

WangTeter::WangTeter(double coeff, double alpha, double beta) 
    : coeff_(coeff), alpha_(alpha), beta_(beta) {}

double WangTeter::compute(const RealField& rho, RealField& v_kedf) {
    const Grid& grid = rho.grid();
    size_t nnr = grid.nnr();
    FFTSolver solver(grid);

    // 1. Compute average density rho0 and Fermi wavevector tkf
    RealField ones(grid);
    ones.fill(1.0);
    double total_electrons = dot_product(nnr, rho.data(), ones.data()) * grid.dv();
    double rho0 = total_electrons / grid.volume();
    if (rho0 < 1e-12) return 0.0;

    const double pi = 3.14159265358979323846;
    double tkf = 2.0 * pow(3.0 * pi * pi * rho0, 1.0 / 3.0);

    // 2. Compute factor = (4/5) * C_TF (for alpha=beta=5/6)
    // General: factor = 5 / (9 * alpha * beta) * C_TF
    double c_tf = (3.0 / 10.0) * pow(3.0 * pi * pi, 2.0 / 3.0);
    double factor = (5.0 / (9.0 * alpha_ * beta_)) * c_tf;

    // 3. Compute rho^beta
    GPU_Vector<double> rho_beta(nnr);
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    power_rho_kernel<<<grid_size, block_size>>>(nnr, rho.data(), rho_beta.data(), beta_);
    GPU_CHECK_KERNEL

    // 4. FFT(rho^beta)
    ComplexField rho_beta_g(grid);
    real_to_complex(nnr, rho_beta.data(), rho_beta_g.data());
    solver.forward(rho_beta_g);

    // 5. Multiply by Kernel in reciprocal space
    wt_reciprocal_kernel<<<grid_size, block_size>>>(nnr, rho_beta_g.data(), grid.gg(), tkf, factor);
    GPU_CHECK_KERNEL

    // 6. IFFT to get convolution result v_conv
    solver.backward(rho_beta_g);
    GPU_Vector<double> v_conv(nnr);
    complex_to_real(nnr, rho_beta_g.data(), v_conv.data());

    // 7. Compute final potential
    wt_potential_kernel<<<grid_size, block_size>>>(nnr, rho.data(), v_conv.data(), v_kedf.data(), alpha_, beta_, coeff_);
    GPU_CHECK_KERNEL

    // 8. Compute Energy: E = coeff * integral( rho^alpha * v_conv ) dV
    // Note: v_kedf = coeff * 2 * alpha * rho^(alpha-1) * v_conv
    // So rho^alpha * v_conv = v_kedf * rho / (2 * alpha)
    // Energy = integral( v_kedf * rho / (2 * alpha) ) dV
    double energy = (dot_product(nnr, rho.data(), v_kedf.data()) * grid.dv()) / (2.0 * alpha_);

    return energy;
}

} // namespace dftcu
