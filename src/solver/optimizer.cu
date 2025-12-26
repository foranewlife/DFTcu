#include <iomanip>
#include <iostream>

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

        // 3. Check Convergence (Align with DFTpy: last ncheck steps < econv)
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

        // Orthogonalization: Project p to be orthogonal to phi and normalize it to sqrt(ne)
        double p_dot_phi = p.dot(phi) * grid_->dv();
        v_axpy(n, -p_dot_phi / ne, phi.data(), p.data());
        double p_norm = sqrt(p.dot(p) * grid_->dv());
        if (p_norm > 1e-15) {
            v_scale(n, sqrt(ne) / p_norm, p.data(), p.data());
        }

        // 5. Save g for next iteration
        cudaMemcpy(g_prev.data(), g.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        // 6. Quadratic Line Search on Theta
        // DFTpy: phi(theta) = phi*cos(theta) + p*sin(theta)
        // E'(0) = grad . (p * cos(0) - phi * sin(0)) = grad . p
        double e0 = energy;
        double de0 = g.dot(p) * grid_->dv();  // Gradient w.r.t theta at theta=0

        if (de0 > 0) {
            // Not a descent direction! Reset to SD
            v_scale(n, -1.0, g.data(), p.data());
            double p_dot_phi_sd = p.dot(phi) * grid_->dv();
            v_axpy(n, -p_dot_phi_sd / ne, phi.data(), p.data());
            double p_norm_sd = sqrt(p.dot(p) * grid_->dv());
            v_scale(n, sqrt(ne) / p_norm_sd, p.data(), p.data());
            de0 = g.dot(p) * grid_->dv();
        }

        double theta_trial = 0.05;  // Small angle trial

        // phi_trial = phi*cos(theta) + p*sin(theta)
        // Since both phi and p are normalized to sqrt(ne), phi_trial is too.
        v_scale(n, cos(theta_trial), phi.data(), phi_trial.data());
        v_axpy(n, sin(theta_trial), p.data(), phi_trial.data());

        v_mul(n, phi_trial.data(), phi_trial.data(), rho_trial.data());
        double e1 = evaluator.compute(rho_trial, v_trial);

        // Parabolic fit for theta
        double denom = 2.0 * (e1 - e0 - de0 * theta_trial);
        double theta_opt = theta_trial;
        if (std::abs(denom) > 1e-20) {
            theta_opt = -de0 * theta_trial * theta_trial / denom;
        }

        if (theta_opt <= 0)
            theta_opt = theta_trial * 0.1;
        if (theta_opt > 0.5)
            theta_opt = 0.5;  // Cap at ~30 degrees

        // 7. Final Update
        // Use dg as temporary to store new phi
        v_scale(n, cos(theta_opt), phi.data(), dg.data());
        v_axpy(n, sin(theta_opt), p.data(), dg.data());
        cudaMemcpy(phi.data(), dg.data(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        // 8. Ensure phi >= 0 (Optional but helpful for stability)
        // update_phi_kernel<<<grid_size, block_size>>>(n, phi.data(), g.data(), 0.0);

        // 9. Update rho = phi^2
        v_mul(n, phi.data(), phi.data(), rho.data());
    }
    std::cout << std::string(60, '-') << std::endl;
}

}  // namespace dftcu
