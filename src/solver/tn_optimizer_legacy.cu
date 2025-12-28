#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "math/dcsrch.cuh"
#include "math/linesearch.cuh"
#include "tn_optimizer_legacy.cuh"  // New header for TNOptimizerLegacy
#include "utilities/kernels.cuh"

namespace dftcu {

__global__ void compute_gradient_kernel_legacy(size_t n, const double* phi, const double* v_tot,
                                               double mu, double* g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        // Gradient of E w.r.t phi is 2 * phi * (V_tot - mu)
        g[i] = 2.0 * phi[i] * (v_tot[i] - mu);
    }
}

TNOptimizerLegacy::TNOptimizerLegacy(Grid& grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void TNOptimizerLegacy::solve(RealField& rho, Evaluator& evaluator) {
    size_t n = grid_.nnr();
    RealField phi(grid_);
    RealField v_tot(grid_);
    RealField g(grid_);
    RealField d(grid_);
    RealField Hd(grid_);
    RealField r(grid_);
    RealField p_cg(grid_);
    RealField g_offset(grid_);
    RealField phi_offset(grid_);
    RealField rho_offset(grid_);
    RealField v_offset(grid_);

    v_sqrt(n, rho.data(), phi.data(), grid_.stream());
    double target_ne = rho.integral();
    double energy = evaluator.compute(rho, v_tot);

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Starting Truncated Newton Optimizer (Legacy)" << std::endl;
    std::cout << std::setw(8) << "Step" << std::setw(24) << "Energy(a.u.)" << std::setw(16) << "dE"
              << std::setw(8) << "Nd" << std::setw(8) << "Nls" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        double mu = rho.dot(v_tot) * grid_.dv() / target_ne;
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        compute_gradient_kernel_legacy<<<grid_size, block_size, 0, grid_.stream()>>>(
            n, phi.data(), v_tot.data(), mu, g.data());
        GPU_CHECK_KERNEL;

        // CG Loop
        d.fill(0.0);
        CHECK(cudaMemcpyAsync(r.data(), g.data(), n * sizeof(double), cudaMemcpyDeviceToDevice,
                              grid_.stream()));
        v_scale(n, -1.0, r.data(), p_cg.data(), grid_.stream());

        double r_norm_sq = r.dot(r);
        double r0_norm_sq = r_norm_sq;
        double r_conv = 0.1 * r0_norm_sq;

        RealField best_d(grid_);
        double min_r_norm_sq = r0_norm_sq;

        int nd_steps = 0;
        for (int j = 0; j < 50; ++j) {
            nd_steps++;

            double p_cg_norm = sqrt(p_cg.dot(p_cg) * grid_.dv());
            double current_eps = 1e-7 / std::max(1.0, p_cg_norm);

            v_scale(n, 1.0, phi.data(), phi_offset.data(), grid_.stream());
            v_axpy(n, current_eps, p_cg.data(), phi_offset.data(), grid_.stream());

            v_mul(n, phi_offset.data(), phi_offset.data(), rho_offset.data(), grid_.stream());
            evaluator.compute(rho_offset, v_offset);

            const int block_size = 256;
            const int grid_size = (n + block_size - 1) / block_size;
            compute_gradient_kernel_legacy<<<grid_size, block_size, 0, grid_.stream()>>>(
                n, phi_offset.data(), v_offset.data(), mu, g_offset.data());
            GPU_CHECK_KERNEL;
            v_sub(n, g_offset.data(), g.data(), Hd.data(), grid_.stream());
            v_scale(n, 1.0 / current_eps, Hd.data(), Hd.data(), grid_.stream());

            double pHp = p_cg.dot(Hd);

            if (!isfinite(pHp) || std::abs(pHp) > 1e15) {
                break;
            }

            if (pHp < 0) {
                if (j == 0)
                    v_scale(n, r0_norm_sq / (pHp - 1e-12), p_cg.data(), d.data(), grid_.stream());
                break;
            }

            double alpha_cg = r_norm_sq / pHp;
            v_axpy(n, alpha_cg, p_cg.data(), d.data(), grid_.stream());
            v_axpy(n, alpha_cg, Hd.data(), r.data(), grid_.stream());

            double r_new_norm_sq = r.dot(r);
            if (r_new_norm_sq < min_r_norm_sq) {
                min_r_norm_sq = r_new_norm_sq;
                CHECK(cudaMemcpyAsync(best_d.data(), d.data(), n * sizeof(double),
                                      cudaMemcpyDeviceToDevice, grid_.stream()));
            }
            if (r_new_norm_sq < r_conv)
                break;

            double beta_cg = r_new_norm_sq / r_norm_sq;
            v_scale(n, beta_cg, p_cg.data(), p_cg.data(), grid_.stream());
            v_axpy(n, -1.0, r.data(), p_cg.data(), grid_.stream());
            r_norm_sq = r_new_norm_sq;
        }

        RealField p(grid_);
        CHECK(cudaMemcpyAsync(p.data(), d.data(), n * sizeof(double), cudaMemcpyDeviceToDevice,
                              grid_.stream()));

        // Orthogonalization
        double p_dot_phi = p.dot(phi) * grid_.dv();
        v_axpy(n, -p_dot_phi / target_ne, phi.data(), p.data(), grid_.stream());
        double p_norm = sqrt(p.dot(p) * grid_.dv());
        if (p_norm > 1e-15) {
            v_scale(n, sqrt(target_ne) / p_norm, p.data(), p.data(), grid_.stream());
        }

        double g_dot_p = g.dot(p) * grid_.dv();
        if (g_dot_p > 0) {
            CHECK(cudaMemcpyAsync(p.data(), g.data(), n * sizeof(double), cudaMemcpyDeviceToDevice,
                                  grid_.stream()));
            v_scale(n, -1.0, p.data(), p.data(), grid_.stream());
            p_dot_phi = p.dot(phi) * grid_.dv();
            v_axpy(n, -p_dot_phi / target_ne, phi.data(), p.data(), grid_.stream());
            p_norm = sqrt(p.dot(p) * grid_.dv());
            if (p_norm > 1e-15)
                v_scale(n, sqrt(target_ne) / p_norm, p.data(), p.data(), grid_.stream());
            g_dot_p = g.dot(p) * grid_.dv();
        }

        RealField phi_backup(grid_);
        CHECK(cudaMemcpyAsync(phi_backup.data(), phi.data(), n * sizeof(double),
                              cudaMemcpyDeviceToDevice, grid_.stream()));

        int nls_evals = 0;
        auto phi_func = [&](double theta) -> double {
            nls_evals++;
            RealField phi_t(grid_);
            RealField rho_t(grid_);
            RealField v_t(grid_);
            v_scale(n, cos(theta), phi_backup.data(), phi_t.data(), grid_.stream());
            v_axpy(n, sin(theta), p.data(), phi_t.data(), grid_.stream());
            v_mul(n, phi_t.data(), phi_t.data(), rho_t.data(), grid_.stream());
            return evaluator.compute(rho_t, v_t);
        };

        auto derphi_func = [&](double theta) -> double {
            RealField phi_t(grid_);
            RealField rho_t(grid_);
            RealField v_t(grid_);
            RealField v_phi_t(grid_);
            RealField p_rot_t(grid_);
            v_scale(n, cos(theta), phi_backup.data(), phi_t.data(), grid_.stream());
            v_axpy(n, sin(theta), p.data(), phi_t.data(), grid_.stream());
            v_mul(n, phi_t.data(), phi_t.data(), rho_t.data(), grid_.stream());
            evaluator.compute(rho_t, v_t);
            v_scale(n, cos(theta), p.data(), p_rot_t.data(), grid_.stream());
            v_axpy(n, -sin(theta), phi_backup.data(), p_rot_t.data(), grid_.stream());
            v_mul(n, v_t.data(), phi_t.data(), v_phi_t.data(), grid_.stream());
            grid_.synchronize();
            return 2.0 * v_phi_t.dot(p_rot_t) * grid_.dv();
        };

        double alpha_star, phi_star;
        scalar_search_wolfe1(phi_func, derphi_func, energy, 2.0 * g_dot_p, 1e10, 1e-4, 0.2, M_PI,
                             0.0, 1e-14, alpha_star, phi_star);

        v_scale(n, cos(alpha_star), phi_backup.data(), phi.data(), grid_.stream());
        v_axpy(n, sin(alpha_star), p.data(), phi.data(), grid_.stream());
        v_mul(n, phi.data(), phi.data(), rho.data(), grid_.stream());

        double de = phi_star - energy;
        std::cout << std::setw(8) << iter << std::fixed << std::setprecision(12) << std::setw(24)
                  << energy << std::scientific << std::setprecision(6) << std::setw(16) << de
                  << std::setw(8) << nd_steps << std::setw(8) << nls_evals << std::endl;

        if (iter > 0 && std::abs(de) < options_.econv) {
            std::cout << "#### Density Optimization Converged ####" << std::endl;
            break;
        }

        energy = evaluator.compute(rho, v_tot);
    }
}

}  // namespace dftcu
