#include <cmath>
#include <vector>

#include "pbe.cuh"
#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

// Kernel to compute h = 2 * v2 * grad_rho
__global__ void compute_h_kernel(int n, const double* v2, const double* grad_x,
                                 const double* grad_y, const double* grad_z, double* h_x,
                                 double* h_y, double* h_z) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double v2_val = v2[i];
        h_x[i] = 2.0 * v2_val * grad_x[i];
        h_y[i] = 2.0 * v2_val * grad_y[i];
        h_z[i] = 2.0 * v2_val * grad_z[i];
    }
}

// Kernel to compute d(G) = i * G * h(G) and sum components for divergence
__global__ void compute_div_g_kernel(int n, const double* gx, const double* gy, const double* gz,
                                     const gpufftComplex* hx_g, const gpufftComplex* hy_g,
                                     const gpufftComplex* hz_g, gpufftComplex* div_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        // div = i * (gx*hx + gy*hy + gz*hz)
        div_g[i].x = -(gx[i] * hx_g[i].y + gy[i] * hy_g[i].y + gz[i] * hz_g[i].y);
        div_g[i].y = (gx[i] * hx_g[i].x + gy[i] * hy_g[i].x + gz[i] * hz_g[i].x);
    }
}

// Kernel to multiply by i * G components in reciprocal space
__global__ void multiply_ig_kernel(int n, const double* g_comp, const gpufftComplex* rho_g,
                                   gpufftComplex* grad_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        // i * g * (rho.x + i*rho.y) = -g*rho.y + i*g*rho.x
        grad_g[i].x = -g_comp[i] * rho_g[i].y;
        grad_g[i].y = g_comp[i] * rho_g[i].x;
    }
}

__global__ void pbe_kernel(int n, const double* rho, const double* sigma, double* v1, double* v2,
                           double* energy_density, PBE::Parameters params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        double s2 = sigma[i];

        if (r <= params.rho_threshold || s2 <= params.sigma_threshold) {
            v1[i] = 0.0;
            v2[i] = 0.0;
            energy_density[i] = 0.0;
            return;
        }

        // --- Exchange ---
        const double pi = constants::D_PI;
        double kf = pow(3.0 * pi * pi * r, 1.0 / 3.0);

        // s_param2 = |grad rho|^2 / (4 * kf^2 * rho^2)
        double s_param2 = s2 / (4.0 * kf * kf * r * r);
        double ex_lda = constants::EX_LDA_COEFF * pow(r, 1.0 / 3.0);

        double denom_x = 1.0 + params.mu_x * s_param2 / params.kappa;
        double fx = 1.0 + params.kappa - params.kappa / denom_x;

        double dfx_ds2 = params.mu_x / (denom_x * denom_x);
        double dfx_dr = dfx_ds2 * (-8.0 / 3.0) * s_param2 / r;

        double vx = ex_lda * (fx + r * dfx_dr + (1.0 / 3.0) * fx);
        double vxs = ex_lda * r * dfx_ds2 / (4.0 * kf * kf * r * r);

        // --- Correlation ---
        double rs = pow(3.0 / (4.0 * pi * r), 1.0 / 3.0);
        double rs_sqrt = sqrt(rs);
        double zeta_p = 2.0 * params.a *
                        (params.beta1 * rs_sqrt + params.beta2 * rs + params.beta3 * rs * rs_sqrt +
                         params.beta4 * rs * rs);

        // Avoid log(1+1/0)
        if (zeta_p < 1e-20)
            zeta_p = 1e-20;

        double dzeta_drs = 2.0 * params.a *
                           (0.5 * params.beta1 / rs_sqrt + params.beta2 +
                            1.5 * params.beta3 * rs_sqrt + 2.0 * params.beta4 * rs);

        double log_1_zeta = log(1.0 + 1.0 / zeta_p);
        double eta = -2.0 * params.a * (1.0 + params.alpha1 * rs) * log_1_zeta;

        double drs_dr = -1.0 / (3.0 * r) * rs;
        double deta_dr =
            (-2.0 * params.a * params.alpha1 * log_1_zeta +
             2.0 * params.a * (1.0 + params.alpha1 * rs) / (1.0 + zeta_p) / zeta_p * dzeta_drs) *
            drs_dr;

        double t2 = s2 / (r * r * (16.0 * kf / pi));

        // PBE Correlation H term
        double exp_val = exp(-eta / params.pbe_gamma);
        double aa = params.pbe_beta / params.pbe_gamma / (exp_val - 1.0);

        double a_t2 = aa * t2;
        double poly = 1.0 + a_t2 + a_t2 * a_t2;
        double h_pbe = params.pbe_gamma *
                       log(1.0 + params.pbe_beta / params.pbe_gamma * t2 * (1.0 + a_t2) / poly);

        double denom_h = 1.0 + params.pbe_beta / params.pbe_gamma * t2 * (1.0 + a_t2) / poly;
        double dh_dt2 = params.pbe_beta / denom_h *
                        ((1.0 + 2.0 * a_t2) * poly - (1.0 + a_t2) * (aa + 2.0 * aa * a_t2)) /
                        (poly * poly);
        double dh_daa = params.pbe_beta * t2 / denom_h *
                        (t2 * poly - (1.0 + a_t2) * (t2 + 2.0 * t2 * a_t2)) / (poly * poly);

        double daa_deta = aa * aa / params.pbe_gamma * exp_val;
        double dh_dr = dh_daa * daa_deta * deta_dr + dh_dt2 * t2 * (-7.0 / 3.0 / r);

        double vc = h_pbe + eta + r * (deta_dr + dh_dr);
        double vcs = dh_dt2 / (r * (16.0 * kf / pi));

        v1[i] = vx + vc;
        v2[i] = vxs + vcs;
        energy_density[i] = (ex_lda * fx + h_pbe + eta) * r;
    }
}

__global__ void reduce_sum_kernel(const int n, const double* data, double* partial_sums,
                                  const int block_size) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * block_size + tid;
    sdata[tid] = (i < n) ? data[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[tid];
    }
}

}  // namespace

PBE::PBE(std::shared_ptr<Grid> grid) : grid_(grid), fft_(std::make_unique<FFTSolver>(grid)) {}

double PBE::compute(const RealField& rho, RealField& v_xc) {
    size_t n = grid_->nnr();
    double dv = grid_->dv();

    // 1. FFT of density
    ComplexField rho_g(grid_);
    rho_g.fill({0, 0});
    real_to_complex(n, rho.data(), rho_g.data());
    fft_->forward(rho_g);

    // 2. Compute gradient components in real space
    RealField grad_x(grid_), grad_y(grid_), grad_z(grid_);
    ComplexField tmp_g(grid_);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // X component
    multiply_ig_kernel<<<grid_size, block_size>>>(n, grid_->gx(), rho_g.data(), tmp_g.data());
    fft_->backward(tmp_g);
    complex_to_real(n, tmp_g.data(), grad_x.data());

    // Y component
    multiply_ig_kernel<<<grid_size, block_size>>>(n, grid_->gy(), rho_g.data(), tmp_g.data());
    fft_->backward(tmp_g);
    complex_to_real(n, tmp_g.data(), grad_y.data());

    // Z component
    multiply_ig_kernel<<<grid_size, block_size>>>(n, grid_->gz(), rho_g.data(), tmp_g.data());
    fft_->backward(tmp_g);
    complex_to_real(n, tmp_g.data(), grad_z.data());

    // 3. Compute sigma = |grad rho|^2
    GPU_Vector<double> sigma(n);
    v_mul(n, grad_x.data(), grad_x.data(), sigma.data());

    GPU_Vector<double> tmp_v(n);
    v_mul(n, grad_y.data(), grad_y.data(), tmp_v.data());
    v_axpy(n, 1.0, tmp_v.data(), sigma.data());

    v_mul(n, grad_z.data(), grad_z.data(), tmp_v.data());
    v_axpy(n, 1.0, tmp_v.data(), sigma.data());

    // 4. Compute PBE terms
    GPU_Vector<double> v1(n), v2(n), energy_density(n);
    pbe_kernel<<<grid_size, block_size>>>(n, rho.data(), sigma.data(), v1.data(), v2.data(),
                                          energy_density.data(), params_);
    GPU_CHECK_KERNEL

    // 5. Compute divergence term: div(2 * v2 * grad_rho)
    RealField h_x(grid_), h_y(grid_), h_z(grid_);
    compute_h_kernel<<<grid_size, block_size>>>(n, v2.data(), grad_x.data(), grad_y.data(),
                                                grad_z.data(), h_x.data(), h_y.data(), h_z.data());

    ComplexField hx_g(grid_), hy_g(grid_), hz_g(grid_), div_g(grid_);
    hx_g.fill({0, 0});
    hy_g.fill({0, 0});
    hz_g.fill({0, 0});
    real_to_complex(n, h_x.data(), hx_g.data());
    real_to_complex(n, h_y.data(), hy_g.data());
    real_to_complex(n, h_z.data(), hz_g.data());

    fft_->forward(hx_g);
    fft_->forward(hy_g);
    fft_->forward(hz_g);

    compute_div_g_kernel<<<grid_size, block_size>>>(n, grid_->gx(), grid_->gy(), grid_->gz(),
                                                    hx_g.data(), hy_g.data(), hz_g.data(),
                                                    div_g.data());
    fft_->backward(div_g);

    RealField div_r(grid_);
    complex_to_real(n, div_g.data(), div_r.data());

    // 6. Final potential V_xc = v1 - div_r
    v_sub(n, v1.data(), div_r.data(), v_xc.data());

    // 7. Compute total energy
    GPU_Vector<double> partial_sums(grid_size);
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(double)>>>(
        n, energy_density.data(), partial_sums.data(), block_size);
    GPU_CHECK_KERNEL

    std::vector<double> h_partial_sums(grid_size);
    partial_sums.copy_to_host(h_partial_sums.data());
    double total_energy = 0.0;
    for (double s : h_partial_sums)
        total_energy += s;

    return total_energy * dv;
}

}  // namespace dftcu
