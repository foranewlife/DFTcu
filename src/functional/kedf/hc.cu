#include <algorithm>
#include <cmath>

#include "hc.cuh"
#include "hc_kernel_data.h"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace dftcu {

namespace {

const double DFT_ZERO = 1e-16;
const double SAVE_TOL = 1e-16;

__device__ inline double evaluate_spline(double x, const double* y, const double* y2, int n,
                                         double etamax) {
    if (x <= 0.0)
        return y[0];
    if (x >= etamax)
        return y[n - 1];

    double h = etamax / (n - 1);
    double idx_f = x / h;
    int i = (int)idx_f;
    if (i >= n - 1)
        return y[n - 1];

    double t = idx_f - i;
    double A = 1.0 - t;
    double B = t;
    double h2_6 = h * h / 6.0;

    return A * y[i] + B * y[i + 1] + h2_6 * ((A * A * A - A) * y2[i] + (B * B * B - B) * y2[i + 1]);
}

__global__ void compute_s_kernel(size_t n, const double* rho, const double* grad_x,
                                 const double* grad_y, const double* grad_z, double* s) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        if (r < DFT_ZERO) {
            s[i] = 0.0;
        } else {
            double g2 = grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i] + grad_z[i] * grad_z[i];
            s[i] = sqrt(g2) / pow(r, 4.0 / 3.0);
        }
    }
}

__global__ void compute_kf_eff_kernel(size_t n, const double* rho, const double* s, double* kf_eff,
                                      double kappa, double mu) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        if (r < DFT_ZERO) {
            kf_eff[i] = 0.0;
        } else {
            const double pi = 3.14159265358979323846;
            double ckf = pow(3.0 * pi * pi, 1.0 / 3.0);
            double kf_std = ckf * pow(r, 1.0 / 3.0);
            double ss = s[i] / (2.0 * ckf);
            double F = 1.0 + mu * ss * ss / (1.0 + kappa * ss * ss);
            kf_eff[i] = F * kf_std;
        }
    }
}

__global__ void multiply_kernel_spline_kernel(size_t n, gpufftComplex* data_g, const double* gg,
                                              double kf_ref, const double* k_table,
                                              const double* k2_table, const double* d_table,
                                              const double* d2_table, int size, double etamax,
                                              bool is_deriv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        if (kf_ref < 1e-12) {
            data_g[i].x = 0.0;
            data_g[i].y = 0.0;
            return;
        }
        double xi = 2.0 * kf_ref;
        double q = sqrt(gg[i]);
        double eta = q / xi;

        if (is_deriv) {
            double raw_k = evaluate_spline(eta, k_table, k2_table, size, etamax);
            double raw_d = evaluate_spline(eta, d_table, d2_table, size, etamax);
            double val = 2.0 * (raw_d - 3.0 * raw_k) / (xi * xi * xi * xi);
            data_g[i].x *= val;
            data_g[i].y *= val;
        } else {
            double raw_k = evaluate_spline(eta, k_table, k2_table, size, etamax);
            double val = raw_k / (xi * xi * xi);
            data_g[i].x *= val;
            data_g[i].y *= val;
        }
    }
}

__global__ void interpolate_potential_hermite_high_precision_kernel(size_t n, const double* kf_eff,
                                                                    double kf_min, double ratio,
                                                                    int nsp, const double* pots,
                                                                    const double* mders,
                                                                    double* v_out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double kf = kf_eff[i];

        if (kf < kf_min + 1e-18) {
            v_out[i] = pots[i] * (kf / kf_min);
        } else {
            double idx_f = log(kf / kf_min) / log(ratio);
            int idx = (int)idx_f;
            if (idx >= nsp - 1) {
                v_out[i] = pots[(nsp - 1) * n + i];
            } else {
                double k_i = kf_min * pow(ratio, (double)idx);
                double k_next = k_i * ratio;
                double Dkf = k_next - k_i;
                double t = (kf - k_i) / Dkf;
                double t2 = t * t;
                double t3 = t2 * t;
                double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                double h10 = t3 - 2.0 * t2 + t;
                double h01 = 1.0 - h00;
                double h11 = t3 - t2;

                v_out[i] = h00 * pots[idx * n + i] + h01 * pots[(idx + 1) * n + i] +
                           Dkf * (h10 * mders[idx * n + i] + h11 * mders[(idx + 1) * n + i]);
            }
        }
    }
}

__global__ void ldw_kernel(size_t n, const double* rho, double* v, double rhov) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        if (r <= 0.0) {
            v[i] = 0.0;
        } else if (r < 1e-6) {
            double ld = 1.0 / 6.0;
            v[i] *= pow(r, ld) / pow(rhov, ld);
        }
    }
}

__global__ void power_kernel(size_t n, const double* in, double* out, double p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = (in[i] > DFT_ZERO) ? pow(in[i], p) : 0.0;
    }
}

__global__ void multiply_ig_kernel(int n, const double* g_comp, const gpufftComplex* rho_g,
                                   gpufftComplex* grad_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        grad_g[i].x = -g_comp[i] * rho_g[i].y;
        grad_g[i].y = g_comp[i] * rho_g[i].x;
    }
}

__global__ void finalize_potential_kernel(size_t n, const double* rho, const double* pot1,
                                          const double* pot2, double alpha, double beta,
                                          double* v_out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        if (r > DFT_ZERO) {
            v_out[i] = alpha * pow(r, alpha - 1.0) * pot1[i] + beta * pow(r, beta - 1.0) * pot2[i];
        } else {
            v_out[i] = 0.0;
        }
    }
}

}  // namespace

revHC::revHC(std::shared_ptr<Grid> grid, double alpha, double beta)
    : grid_(grid), fft_(std::make_unique<FFTSolver>(grid)), alpha_(alpha), beta_(beta) {
    kappa_ = 0.1;
    mu_ = 0.45;
}

double revHC::compute(const RealField& rho, RealField& v_kedf) {
    size_t n = grid_->nnr();
    double dv = grid_->dv();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // 1. Gradient
    ComplexField rho_g(grid_);
    real_to_complex(n, rho.data(), rho_g.data());
    fft_->forward(rho_g);
    RealField gx(grid_), gy(grid_), gz(grid_);
    ComplexField tmp_g(grid_);
    multiply_ig_kernel<<<grid_size, block_size>>>(n, grid_->gx(), rho_g.data(), tmp_g.data());
    fft_->backward(tmp_g);
    complex_to_real(n, tmp_g.data(), gx.data());
    multiply_ig_kernel<<<grid_size, block_size>>>(n, grid_->gy(), rho_g.data(), tmp_g.data());
    fft_->backward(tmp_g);
    complex_to_real(n, tmp_g.data(), gy.data());
    multiply_ig_kernel<<<grid_size, block_size>>>(n, grid_->gz(), rho_g.data(), tmp_g.data());
    fft_->backward(tmp_g);
    complex_to_real(n, tmp_g.data(), gz.data());

    // 2. s and kf
    GPU_Vector<double> s(n), kf_eff(n);
    compute_s_kernel<<<grid_size, block_size>>>(n, rho.data(), gx.data(), gy.data(), gz.data(),
                                                s.data());
    compute_kf_eff_kernel<<<grid_size, block_size>>>(n, rho.data(), s.data(), kf_eff.data(), kappa_,
                                                     mu_);

    // 3. Range
    thrust::device_ptr<const double> kf_ptr(kf_eff.data());
    double kf_raw_min = *thrust::min_element(kf_ptr, kf_ptr + n);
    double kf_raw_max = *thrust::max_element(kf_ptr, kf_ptr + n);
    double ratio = 1.15;
    double kf0 = 1.0;
    double kf_min_v = std::max(kf_raw_min, 1e-3);
    int n_min = (int)std::floor(std::log(kf_min_v / kf0) / std::log(ratio)) - 1;
    double kf_min = kf0 * std::pow(ratio, (double)n_min);
    double kf_max_v = std::min(kf_raw_max, 100.0);
    int n_max = (int)std::ceil(std::log(kf_max_v / kf0) / std::log(ratio)) + 1;

    kf_min -= SAVE_TOL;
    int current_nsp = n_max - n_min + 1;
    if (current_nsp > 128)
        current_nsp = 128;

    // 4. Convolutions
    GPU_Vector<double> rho_beta(n), rho_alpha(n);
    power_kernel<<<grid_size, block_size>>>(n, rho.data(), rho_beta.data(), beta_);
    power_kernel<<<grid_size, block_size>>>(n, rho.data(), rho_alpha.data(), alpha_);
    ComplexField rb_g(grid_), ra_g(grid_);
    real_to_complex(n, rho_beta.data(), rb_g.data());
    real_to_complex(n, rho_alpha.data(), ra_g.data());
    fft_->forward(rb_g);
    fft_->forward(ra_g);

    GPU_Vector<double> pots1(n * current_nsp), pots2(n * current_nsp);
    GPU_Vector<double> mders1(n * current_nsp), mders2(n * current_nsp);

    GPU_Vector<double> d_k(HC_KERNEL_SIZE), d_k2(HC_KERNEL_SIZE), d_d(HC_KERNEL_SIZE),
        d_d2(HC_KERNEL_SIZE);
    d_k.copy_from_host(HC_K_DATA);
    d_k2.copy_from_host(HC_K2_DATA);
    d_d.copy_from_host(HC_D_DATA);
    d_d2.copy_from_host(HC_D2_DATA);

    for (int i = 0; i < current_nsp; ++i) {
        double kf_ref = (kf_min + SAVE_TOL) * pow(ratio, (double)i);
        GPU_Vector<gpufftComplex> tv_b(n), tv_a(n);
        ComplexField tb_g(grid_), ta_g(grid_);

        // K convolution
        CHECK(cudaMemcpy(tv_b.data(), rb_g.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        multiply_kernel_spline_kernel<<<grid_size, block_size>>>(
            n, tv_b.data(), grid_->gg(), kf_ref, d_k.data(), d_k2.data(), d_d.data(), d_d2.data(),
            HC_KERNEL_SIZE, HC_KERNEL_ETAMAX, false);
        CHECK(cudaMemcpy(tb_g.data(), tv_b.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        fft_->backward(tb_g);
        complex_to_real(n, tb_g.data(), pots1.data() + i * n);

        CHECK(cudaMemcpy(tv_a.data(), ra_g.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        multiply_kernel_spline_kernel<<<grid_size, block_size>>>(
            n, tv_a.data(), grid_->gg(), kf_ref, d_k.data(), d_k2.data(), d_d.data(), d_d2.data(),
            HC_KERNEL_SIZE, HC_KERNEL_ETAMAX, false);
        CHECK(cudaMemcpy(ta_g.data(), tv_a.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        fft_->backward(ta_g);
        complex_to_real(n, ta_g.data(), pots2.data() + i * n);

        // dK/dkf convolution
        CHECK(cudaMemcpy(tv_b.data(), rb_g.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        multiply_kernel_spline_kernel<<<grid_size, block_size>>>(
            n, tv_b.data(), grid_->gg(), kf_ref, d_k.data(), d_k2.data(), d_d.data(), d_d2.data(),
            HC_KERNEL_SIZE, HC_KERNEL_ETAMAX, true);
        CHECK(cudaMemcpy(tb_g.data(), tv_b.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        fft_->backward(tb_g);
        complex_to_real(n, tb_g.data(), mders1.data() + i * n);

        CHECK(cudaMemcpy(tv_a.data(), ra_g.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        multiply_kernel_spline_kernel<<<grid_size, block_size>>>(
            n, tv_a.data(), grid_->gg(), kf_ref, d_k.data(), d_k2.data(), d_d.data(), d_d2.data(),
            HC_KERNEL_SIZE, HC_KERNEL_ETAMAX, true);
        CHECK(cudaMemcpy(ta_g.data(), tv_a.data(), n * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToDevice));
        fft_->backward(ta_g);
        complex_to_real(n, ta_g.data(), mders2.data() + i * n);
    }

    GPU_Vector<double> pot1(n), pot2(n);
    interpolate_potential_hermite_high_precision_kernel<<<grid_size, block_size>>>(
        n, kf_eff.data(), kf_min + SAVE_TOL, ratio, current_nsp, pots1.data(), mders1.data(),
        pot1.data());
    interpolate_potential_hermite_high_precision_kernel<<<grid_size, block_size>>>(
        n, kf_eff.data(), kf_min + SAVE_TOL, ratio, current_nsp, pots2.data(), mders2.data(),
        pot2.data());

    double energy = dot_product(n, rho_alpha.data(), pot1.data()) * dv;
    finalize_potential_kernel<<<grid_size, block_size>>>(n, rho.data(), pot1.data(), pot2.data(),
                                                         alpha_, beta_, v_kedf.data());

    thrust::device_ptr<const double> rho_ptr(rho.data());
    double rhov = *thrust::max_element(rho_ptr, rho_ptr + n);
    ldw_kernel<<<grid_size, block_size>>>(n, rho.data(), v_kedf.data(), rhov);

    return energy;
}

}  // namespace dftcu
