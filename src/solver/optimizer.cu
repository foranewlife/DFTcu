#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "math/dcsrch.cuh"
#include "math/linesearch.cuh"
#include "optimizer.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void compute_gradient_kernel(size_t n, const double* phi, const double* v_tot, double mu,
                                        double* g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        // Gradient of E w.r.t phi is 2 * phi * (V_tot - mu)
        g[i] = 2.0 * phi[i] * (v_tot[i] - mu);
    }
}

__global__ void update_phi_kernel(size_t n, double* phi, const double* g, double step) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        phi[i] -= step * g[i];
        if (phi[i] < 0.0)
            phi[i] = 1e-15;  // Maintain positivity
    }
}
}  // namespace

SimpleOptimizer::SimpleOptimizer(std::shared_ptr<Grid> grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void SimpleOptimizer::solve(RealField& rho, Evaluator& evaluator) {
    size_t n = grid_->nnr();
    RealField phi(grid_);
    RealField v_tot(grid_);
    RealField g(grid_);

    // 1. Initial phi = sqrt(rho)
    v_sqrt(n, rho.data(), phi.data());
    double ne = rho.integral();

    double prev_energy = 1e10;

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Starting Simple SCF Optimizer" << std::endl;
    std::cout << std::setw(8) << "Iter" << std::setw(20) << "Energy (Ha)" << std::setw(15) << "dE"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        // Evaluate energy and potential
        double energy = evaluator.compute(rho, v_tot);

        // Compute chemical potential mu = integral(rho * v_tot) / Ne
        double mu = rho.dot(v_tot) * grid_->dv() / ne;

        double de = energy - prev_energy;
        std::cout << std::setw(8) << iter << std::fixed << std::setprecision(10) << std::setw(20)
                  << energy << std::scientific << std::setw(15) << de << std::endl;

        if (std::abs(de) < options_.econv && iter > 0) {
            std::cout << "Converged!" << std::endl;
            break;
        }
        prev_energy = energy;

        // Compute gradient g = 2 * phi * (v_tot - mu)
        compute_gradient_kernel<<<grid_size, block_size>>>(n, phi.data(), v_tot.data(), mu,
                                                           g.data());
        GPU_CHECK_KERNEL

        // Update phi = phi - step * g
        update_phi_kernel<<<grid_size, block_size>>>(n, phi.data(), g.data(), options_.step_size);
        GPU_CHECK_KERNEL

        // Re-normalize phi to maintain Ne: integral(phi^2) = Ne
        double current_ne = phi.dot(phi) * grid_->dv();
        v_scale(n, sqrt(ne / current_ne), phi.data(), phi.data());

        // Update rho = phi^2
        v_mul(n, phi.data(), phi.data(), rho.data());
    }
    std::cout << std::string(60, '-') << std::endl;
}

// -----------------------------------------------------------------------------
// CGOptimizer Implementation
// -----------------------------------------------------------------------------

CGOptimizer::CGOptimizer(std::shared_ptr<Grid> grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void CGOptimizer::solve(RealField& rho, Evaluator& evaluator) {
    size_t n = grid_->nnr();
    RealField phi(grid_);
    RealField v_tot(grid_);
    RealField g(grid_);
    RealField g_prev(grid_);
    RealField p(grid_);
    RealField dg(grid_);
    RealField phi_trial(grid_);
    RealField rho_trial(grid_);
    RealField v_trial(grid_);

    // Initial phi = sqrt(rho)
    v_sqrt(n, rho.data(), phi.data());
    double ne = rho.integral();

    std::vector<double> energy_history;

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Starting Conjugate Gradient (HS) Optimizer" << std::endl;
    std::cout << std::setw(8) << "Iter" << std::setw(20) << "Energy (Ha)" << std::setw(15) << "dE"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        // 1. Evaluate energy and potential
        double energy = evaluator.compute(rho, v_tot);
        energy_history.push_back(energy);

        // 2. Compute chemical potential mu
        double mu = rho.dot(v_tot) * grid_->dv() / ne;

        double de = (iter > 0) ? (energy - energy_history[iter - 1]) : energy;
        std::cout << std::setw(8) << iter << std::fixed << std::setprecision(10) << std::setw(20)
                  << energy << std::scientific << std::setw(15) << de << std::endl;

        // 3. Check Convergence
        if (iter >= options_.ncheck) {
            bool converged = true;
            for (int i = 0; i < options_.ncheck; ++i) {
                if (std::abs(energy_history.back() -
                             energy_history[energy_history.size() - 2 - i]) > options_.econv) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                std::cout << "#### Density Optimization Converged ####" << std::endl;
                break;
            }
        }

        // 4. Compute gradient g_phi = 2 * phi * (v_tot - mu)
        compute_gradient_kernel<<<grid_size, block_size>>>(n, phi.data(), v_tot.data(), mu,
                                                           g.data());
        GPU_CHECK_KERNEL

        // 4. Update search direction p
        if (iter == 0) {
            v_scale(n, -1.0, g.data(), p.data());  // p = -g
        } else {
            // Hestenes-Stiefel beta
            v_sub(n, g.data(), g_prev.data(), dg.data());
            double num = g.dot(dg);
            double den = p.dot(dg);
            double beta = (std::abs(den) > 1e-20) ? num / den : 0.0;
            if (beta < 0)
                beta = 0;

            // p = -g + beta * p
            v_scale(n, -1.0, g.data(), dg.data());
            v_axpy(n, beta, p.data(), dg.data());
            cudaMemcpy(p.data(), dg.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        // Orthogonalization
        double p_dot_phi = p.dot(phi) * grid_->dv();
        v_axpy(n, -p_dot_phi / ne, phi.data(), p.data());
        double p_norm = sqrt(p.dot(p) * grid_->dv());
        if (p_norm > 1e-15) {
            v_scale(n, sqrt(ne) / p_norm, p.data(), p.data());
        }

        // 5. Save g for next iteration
        cudaMemcpy(g_prev.data(), g.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        // 6. Quadratic Line Search
        double e0 = energy;
        double de0 = g.dot(p) * grid_->dv();

        if (de0 > 0) {
            v_scale(n, -1.0, g.data(), p.data());
            double p_dot_phi_sd = p.dot(phi) * grid_->dv();
            v_axpy(n, -p_dot_phi_sd / ne, phi.data(), p.data());
            double p_norm_sd = sqrt(p.dot(p) * grid_->dv());
            v_scale(n, sqrt(ne) / p_norm_sd, p.data(), p.data());
            de0 = g.dot(p) * grid_->dv();
        }

        double theta_trial = 0.05;
        v_scale(n, cos(theta_trial), phi.data(), phi_trial.data());
        v_axpy(n, sin(theta_trial), p.data(), phi_trial.data());

        v_mul(n, phi_trial.data(), phi_trial.data(), rho_trial.data());
        double e1 = evaluator.compute(rho_trial, v_trial);

        double denom = 2.0 * (e1 - e0 - de0 * theta_trial);
        double theta_opt = theta_trial;
        if (std::abs(denom) > 1e-20) {
            theta_opt = -de0 * theta_trial * theta_trial / denom;
        }

        if (theta_opt <= 0)
            theta_opt = theta_trial * 0.1;
        if (theta_opt > 0.5)
            theta_opt = 0.5;

        v_scale(n, cos(theta_opt), phi.data(), dg.data());
        v_axpy(n, sin(theta_opt), p.data(), dg.data());
        cudaMemcpy(phi.data(), dg.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        v_mul(n, phi.data(), phi.data(), rho.data());
    }
    std::cout << std::string(60, '-') << std::endl;
}

// -----------------------------------------------------------------------------
// TNOptimizer Implementation
// -----------------------------------------------------------------------------

TNOptimizer::TNOptimizer(std::shared_ptr<Grid> grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void TNOptimizer::solve(RealField& rho, Evaluator& evaluator) {
    size_t n = grid_->nnr();
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

    v_sqrt(n, rho.data(), phi.data());
    double target_ne = rho.integral();
    double energy = evaluator.compute(rho, v_tot);
    double prev_energy = 1e10;

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Starting Truncated Newton Optimizer" << std::endl;
    std::cout << std::setw(8) << "Step" << std::setw(24) << "Energy(a.u.)" << std::setw(16) << "dE"
              << std::setw(8) << "Nd" << std::setw(8) << "Nls" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        double mu = rho.dot(v_tot) * grid_->dv() / target_ne;
        compute_gradient_kernel<<<grid_size, block_size>>>(n, phi.data(), v_tot.data(), mu,
                                                           g.data());
        GPU_CHECK_KERNEL

        // CG Loop (Inner loop for Truncated Newton)
        d.fill(0.0);
        cudaMemcpy(r.data(), g.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);
        v_scale(n, -1.0, r.data(), p_cg.data());

        double r_norm_sq = r.dot(r);
        double r0_norm_sq = r_norm_sq;
        double r_conv = 0.1 * r0_norm_sq;

        RealField best_d(grid_);
        double min_r_norm_sq = r0_norm_sq;

        int nd_steps = 0;
        for (int j = 0; j < 50; ++j) {
            nd_steps++;

            double p_cg_norm = sqrt(p_cg.dot(p_cg) * grid_->dv());
            double current_eps = 1e-7 / std::max(1.0, p_cg_norm);

            v_scale(n, 1.0, phi.data(), phi_offset.data());
            v_axpy(n, current_eps, p_cg.data(), phi_offset.data());

            v_mul(n, phi_offset.data(), phi_offset.data(), rho_offset.data());
            evaluator.compute(rho_offset, v_offset);

            compute_gradient_kernel<<<grid_size, block_size>>>(
                n, phi_offset.data(), v_offset.data(), mu, g_offset.data());
            GPU_CHECK_KERNEL
            v_sub(n, g_offset.data(), g.data(), Hd.data());
            v_scale(n, 1.0 / current_eps, Hd.data(), Hd.data());

            double pHp = p_cg.dot(Hd);

            if (!isfinite(pHp) || std::abs(pHp) > 1e15) {
                break;
            }

            if (pHp < 0) {
                if (j == 0)
                    v_scale(n, r0_norm_sq / (pHp - 1e-12), p_cg.data(), d.data());
                break;
            }

            double alpha_cg = r_norm_sq / pHp;
            v_axpy(n, alpha_cg, p_cg.data(), d.data());
            v_axpy(n, alpha_cg, Hd.data(), r.data());

            double r_new_norm_sq = r.dot(r);
            if (r_new_norm_sq < min_r_norm_sq) {
                min_r_norm_sq = r_new_norm_sq;
                cudaMemcpy(best_d.data(), d.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);
            }
            if (r_new_norm_sq < r_conv)
                break;

            double beta_cg = r_new_norm_sq / r_norm_sq;
            v_scale(n, beta_cg, p_cg.data(), p_cg.data());
            v_axpy(n, -1.0, r.data(), p_cg.data());
            r_norm_sq = r_new_norm_sq;
        }

        RealField p(grid_);
        cudaMemcpy(p.data(), d.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        // Orthogonalization and normalization to maintain constraint integral(phi^2) = Ne
        double p_dot_phi = p.dot(phi) * grid_->dv();
        v_axpy(n, -p_dot_phi / target_ne, phi.data(), p.data());
        double p_norm = sqrt(p.dot(p) * grid_->dv());
        if (p_norm > 1e-15) {
            v_scale(n, sqrt(target_ne) / p_norm, p.data(), p.data());
        }

        double g_dot_p = g.dot(p) * grid_->dv();
        if (g_dot_p > 0) {
            cudaMemcpy(p.data(), g.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);
            v_scale(n, -1.0, p.data(), p.data());
            p_dot_phi = p.dot(phi) * grid_->dv();
            v_axpy(n, -p_dot_phi / target_ne, phi.data(), p.data());
            p_norm = sqrt(p.dot(p) * grid_->dv());
            if (p_norm > 1e-15)
                v_scale(n, sqrt(target_ne) / p_norm, p.data(), p.data());
            g_dot_p = g.dot(p) * grid_->dv();
        }

        RealField phi_backup(grid_);
        cudaMemcpy(phi_backup.data(), phi.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        int nls_evals = 0;
        auto phi_func = [&](double theta) -> double {
            nls_evals++;
            RealField phi_t(grid_);
            RealField rho_t(grid_);
            RealField v_t(grid_);
            v_scale(n, cos(theta), phi_backup.data(), phi_t.data());
            v_axpy(n, sin(theta), p.data(), phi_t.data());
            v_mul(n, phi_t.data(), phi_t.data(), rho_t.data());
            return evaluator.compute(rho_t, v_t);
        };

        auto derphi_func = [&](double theta) -> double {
            RealField phi_t(grid_);
            RealField rho_t(grid_);
            RealField v_t(grid_);
            RealField v_phi_t(grid_);
            RealField p_rot_t(grid_);
            v_scale(n, cos(theta), phi_backup.data(), phi_t.data());
            v_axpy(n, sin(theta), p.data(), phi_t.data());
            v_mul(n, phi_t.data(), phi_t.data(), rho_t.data());
            evaluator.compute(rho_t, v_t);
            v_scale(n, cos(theta), p.data(), p_rot_t.data());
            v_axpy(n, -sin(theta), phi_backup.data(), p_rot_t.data());
            v_mul(n, v_t.data(), phi_t.data(), v_phi_t.data());
            // Energy derivative w.r.t theta is 2.0 * integral(V * phi * d_phi/d_theta)
            return 2.0 * v_phi_t.dot(p_rot_t) * grid_->dv();
        };

        double alpha_star, phi_star;
        scalar_search_wolfe1(phi_func, derphi_func, energy, 2.0 * g_dot_p, 1e10, 1e-4, 0.2, M_PI,
                             0.0, 1e-14, alpha_star, phi_star);

        v_scale(n, cos(alpha_star), phi_backup.data(), phi.data());
        v_axpy(n, sin(alpha_star), p.data(), phi.data());
        v_mul(n, phi.data(), phi.data(), rho.data());

        double de = phi_star - energy;
        std::cout << std::setw(8) << iter << std::fixed << std::setprecision(12) << std::setw(24)
                  << energy << std::scientific << std::setprecision(6) << std::setw(16) << de
                  << std::setw(8) << nd_steps << std::setw(8) << nls_evals << std::endl;

        if (iter > 0 && std::abs(de) < options_.econv) {
            std::cout << "#### Density Optimization Converged ####" << std::endl;
            break;
        }

        prev_energy = energy;
        energy = evaluator.compute(rho, v_tot);
    }
}

}  // namespace dftcu
