#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "math/dcsrch.cuh"
#include "math/linesearch.cuh"
#include "optimizer.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

SimpleOptimizer::SimpleOptimizer(Grid& grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void SimpleOptimizer::solve(RealField& rho, Evaluator& evaluator) {
    size_t n = grid_.nnr();

    RealField phi(grid_);

    RealField v_tot(grid_);

    RealField g(grid_);

    v_sqrt(n, rho.data(), phi.data(), grid_.stream());

    double ne = rho.integral();

    double prev_energy = 1e10;

    std::cout << std::string(60, '-') << std::endl;

    std::cout << "Starting Simple SCF Optimizer (ET)" << std::endl;

    std::cout << std::setw(8) << "Iter" << std::setw(20) << "Energy (Ha)" << std::setw(15) << "dE"

              << std::endl;

    std::cout << std::string(60, '-') << std::endl;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        double energy = evaluator.compute(rho, v_tot);

        double mu = rho.dot(v_tot) * grid_.dv() / ne;

        double de = energy - prev_energy;

        std::cout << std::setw(8) << iter << std::fixed << std::setprecision(10) << std::setw(20)

                  << energy << std::scientific << std::setw(15) << de << std::endl;

        if (std::abs(de) < options_.econv && iter > 0) {
            std::cout << "Converged!" << std::endl;

            break;
        }

        prev_energy = energy;

        g = 2.0 * phi * (v_tot - mu);

        phi = phi - options_.step_size * g;

        // Note: positivity is not enforced here as in the old kernel. This could be added

        // as a `max(expr, 0.0)` expression if needed.

        double current_ne = phi.dot(phi) * grid_.dv();

        phi = phi * sqrt(ne / current_ne);

        rho = phi * phi;
    }

    std::cout << std::string(60, '-') << std::endl;
}

// -----------------------------------------------------------------------------

// CGOptimizer Implementation

// -----------------------------------------------------------------------------

CGOptimizer::CGOptimizer(Grid& grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void CGOptimizer::solve(RealField& rho, Evaluator& evaluator) {
    size_t n = grid_.nnr();

    RealField phi(grid_);

    RealField v_tot(grid_);

    RealField g(grid_);

    RealField g_prev(grid_);

    RealField p(grid_);

    RealField dg(grid_);

    RealField phi_trial(grid_);

    RealField rho_trial(grid_);

    RealField v_trial(grid_);

    v_sqrt(n, rho.data(), phi.data(), grid_.stream());

    double ne = rho.integral();

    std::vector<double> energy_history;

    std::cout << std::string(60, '-') << std::endl;

    std::cout << "Starting Conjugate Gradient (HS) Optimizer (ET)" << std::endl;

    std::cout << std::setw(8) << "Iter" << std::setw(20) << "Energy (Ha)" << std::setw(15) << "dE"

              << std::endl;

    std::cout << std::string(60, '-') << std::endl;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        double energy = evaluator.compute(rho, v_tot);

        energy_history.push_back(energy);

        double mu = rho.dot(v_tot) * grid_.dv() / ne;

        double de = (iter > 0) ? (energy - energy_history[iter - 1]) : energy;

        std::cout << std::setw(8) << iter << std::fixed << std::setprecision(10) << std::setw(20)

                  << energy << std::scientific << std::setw(15) << de << std::endl;

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

        g = 2.0 * phi * (v_tot - mu);

        if (iter == 0) {
            p = -1.0 * g;

        } else {
            dg = g - g_prev;

            double num = g.dot(dg);

            double den = p.dot(dg);

            double beta = (std::abs(den) > 1e-20) ? num / den : 0.0;

            if (beta < 0)

                beta = 0;

            p = -1.0 * g + beta * p;
        }

        double p_dot_phi = p.dot(phi) * grid_.dv();

        p = p - (p_dot_phi / ne) * phi;

        double p_norm = sqrt(p.dot(p) * grid_.dv());

        if (p_norm > 1e-15) {
            p = p * (sqrt(ne) / p_norm);
        }

        g_prev = g;

        double e0 = energy;

        double de0 = g.dot(p) * grid_.dv();

        if (de0 > 0) {
            p = -1.0 * g;

            double p_dot_phi_sd = p.dot(phi) * grid_.dv();

            p = p - (p_dot_phi_sd / ne) * phi;

            double p_norm_sd = sqrt(p.dot(p) * grid_.dv());

            p = p * (sqrt(ne) / p_norm_sd);

            de0 = g.dot(p) * grid_.dv();
        }

        double theta_trial = 0.05;

        phi_trial = cos(theta_trial) * phi + sin(theta_trial) * p;

        rho_trial = phi_trial * phi_trial;

        double e1 = evaluator.compute(rho_trial, v_trial);

        double denom = 2.0 * (e1 - e0 - de0 * theta_trial);

        double theta_opt = (std::abs(denom) > 1e-20) ? -de0 * theta_trial * theta_trial / denom

                                                     : theta_trial;

        if (theta_opt <= 0)

            theta_opt = theta_trial * 0.1;

        if (theta_opt > 0.5)

            theta_opt = 0.5;

        phi = cos(theta_opt) * phi + sin(theta_opt) * p;

        rho = phi * phi;
    }

    std::cout << std::string(60, '-') << std::endl;
}

// -----------------------------------------------------------------------------
// TNOptimizer Implementation
// -----------------------------------------------------------------------------

TNOptimizer::TNOptimizer(Grid& grid, OptimizationOptions options)
    : grid_(grid), options_(options) {}

void TNOptimizer::solve(RealField& rho, Evaluator& evaluator) {
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

    std::cout << "Starting Truncated Newton Optimizer (Expression Templates)" << std::endl;

    std::cout << std::setw(8) << "Step" << std::setw(24) << "Energy(a.u.)" << std::setw(16) << "dE"

              << std::setw(8) << "Nd" << std::setw(8) << "Nls" << std::endl;

    std::cout << std::string(80, '-') << std::endl;

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        double mu = rho.dot(v_tot) * grid_.dv() / target_ne;

        g = 2.0 * phi * (v_tot - mu);  // Expression Template

        // CG Loop

        d.fill(0.0);

        r = g;

        p_cg = -1.0 * r;

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

            phi_offset = phi + current_eps * p_cg;

            rho_offset = phi_offset * phi_offset;

            evaluator.compute(rho_offset, v_offset);

            g_offset = 2.0 * phi_offset * (v_offset - mu);

            Hd = (g_offset - g) * (1.0 / current_eps);

            double pHp = p_cg.dot(Hd);

            if (!isfinite(pHp) || std::abs(pHp) > 1e15) {
                break;
            }

            if (pHp < 0) {
                if (j == 0)

                    d = (r0_norm_sq / (pHp - 1e-12)) * p_cg;

                break;
            }

            double alpha_cg = r_norm_sq / pHp;

            d = d + alpha_cg * p_cg;

            r = r + alpha_cg * Hd;

            double r_new_norm_sq = r.dot(r);

            if (r_new_norm_sq < min_r_norm_sq) {
                min_r_norm_sq = r_new_norm_sq;

                best_d = d;
            }

            if (r_new_norm_sq < r_conv)

                break;

            double beta_cg = r_new_norm_sq / r_norm_sq;

            p_cg = -1.0 * r + beta_cg * p_cg;

            r_norm_sq = r_new_norm_sq;
        }

        p_cg = d;  // Use p_cg as a temporary for the final direction

        // Orthogonalization

        double p_dot_phi = p_cg.dot(phi) * grid_.dv();

        p_cg = p_cg - (p_dot_phi / target_ne) * phi;

        double p_norm = sqrt(p_cg.dot(p_cg) * grid_.dv());

        if (p_norm > 1e-15) {
            p_cg = (sqrt(target_ne) / p_norm) * p_cg;
        }

        double g_dot_p = g.dot(p_cg) * grid_.dv();

        if (g_dot_p > 0) {
            p_cg = -1.0 * g;

            p_dot_phi = p_cg.dot(phi) * grid_.dv();

            p_cg = p_cg - (p_dot_phi / target_ne) * phi;

            p_norm = sqrt(p_cg.dot(p_cg) * grid_.dv());

            if (p_norm > 1e-15)

                p_cg = (sqrt(target_ne) / p_norm) * p_cg;

            g_dot_p = g.dot(p_cg) * grid_.dv();
        }

        RealField phi_backup(grid_);

        phi_backup = phi;

        int nls_evals = 0;

        auto phi_func = [&](double theta) -> double {
            nls_evals++;

            RealField phi_t(grid_);

            RealField rho_t(grid_);

            RealField v_t(grid_);

            phi_t = cos(theta) * phi_backup + sin(theta) * p_cg;

            rho_t = phi_t * phi_t;

            return evaluator.compute(rho_t, v_t);
        };

        auto derphi_func = [&](double theta) -> double {
            RealField phi_t(grid_);

            RealField rho_t(grid_);

            RealField v_t(grid_);

            RealField p_rot_t(grid_);

            phi_t = cos(theta) * phi_backup + sin(theta) * p_cg;

            rho_t = phi_t * phi_t;

            evaluator.compute(rho_t, v_t);

            p_rot_t = cos(theta) * p_cg - sin(theta) * phi_backup;

            RealField v_phi_t(grid_);

            v_phi_t = v_t * phi_t;

            grid_.synchronize();

            return 2.0 * v_phi_t.dot(p_rot_t) * grid_.dv();
        };

        double alpha_star, phi_star;

        scalar_search_wolfe1(phi_func, derphi_func, energy, 2.0 * g_dot_p, 1e10, 1e-4, 0.2, M_PI,

                             0.0, 1e-14, alpha_star, phi_star);

        phi = cos(alpha_star) * phi_backup + sin(alpha_star) * p_cg;

        rho = phi * phi;

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
