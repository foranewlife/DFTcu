#include <cmath>

#include "kernel.cuh"
#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "wt.cuh"

namespace dftcu {

namespace {
__global__ void power_rho_kernel(size_t size, const double* rho, double* rho_p, double p,
                                 WangTeter::Parameters params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        rho_p[i] = (rho[i] > params.rho_threshold) ? pow(rho[i], p) : 0.0;
    }
}

__global__ void wt_reciprocal_kernel(size_t size, gpufftComplex* data_g, const double* gg,
                                     double tkf, double factor) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        double q = sqrt(gg[i]);
        double eta = q / tkf;
        double K = lindhard_function(eta, 1.0, 1.0) * factor;
        data_g[i].x *= K;
        data_g[i].y *= K;
    }
}

__global__ void wt_potential_kernel(size_t size, const double* rho, const double* v_conv,
                                    double* v_wt, double alpha, double beta, double coeff,
                                    WangTeter::Parameters params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (rho[i] > params.rho_threshold) {
            v_wt[i] = coeff * 2.0 * alpha * pow(rho[i], alpha - 1.0) * v_conv[i];
            (void)beta;
        } else {
            v_wt[i] = 0.0;
        }
    }
}
}  // namespace

WangTeter::WangTeter(double coeff, double alpha, double beta)
    : coeff_(coeff), alpha_(alpha), beta_(beta) {}

void WangTeter::initialize_buffers(Grid& grid) {
    if (grid_ == &grid)
        return;

    grid_ = &grid;
    rho_beta_ = std::make_unique<RealField>(grid);
    rho_beta_g_ = std::make_unique<ComplexField>(grid);
    v_conv_ = std::make_unique<RealField>(grid);
}

double WangTeter::compute(const RealField& rho, RealField& v_kedf) {
    Grid& grid = rho.grid();
    initialize_buffers(grid);
    size_t nnr = grid.nnr();
    FFTSolver solver(grid);

    double total_electrons = rho.integral();
    double rho0 = total_electrons / grid.volume();
    if (rho0 < params_.rho0_threshold)
        return 0.0;

    const double pi = constants::D_PI;
    double tkf = 2.0 * pow(3.0 * pi * pi * rho0, 1.0 / 3.0);
    double c_tf = constants::C_TF_BASE;
    double factor = (5.0 / (9.0 * alpha_ * beta_)) * c_tf;

    power_rho_kernel<<<(nnr + 255) / 256, 256, 0, grid.stream()>>>(
        nnr, rho.data(), rho_beta_->data(), beta_, params_);
    GPU_CHECK_KERNEL;

    real_to_complex(nnr, rho_beta_->data(), rho_beta_g_->data());
    solver.forward(*rho_beta_g_);

    // Apply Kernel in reciprocal space
    wt_reciprocal_kernel<<<(nnr + 255) / 256, 256, 0, grid.stream()>>>(nnr, rho_beta_g_->data(),
                                                                       grid.gg(), tkf, factor);
    GPU_CHECK_KERNEL;

    solver.backward(*rho_beta_g_);
    complex_to_real(nnr, rho_beta_g_->data(), v_conv_->data());

    wt_potential_kernel<<<(nnr + 255) / 256, 256, 0, grid.stream()>>>(
        nnr, rho.data(), v_conv_->data(), v_kedf.data(), alpha_, beta_, coeff_, params_);
    GPU_CHECK_KERNEL;

    grid.synchronize();
    double energy = (dot_product(nnr, rho.data(), v_kedf.data()) * grid.dv()) / (2.0 * alpha_);
    return energy;
}

}  // namespace dftcu
