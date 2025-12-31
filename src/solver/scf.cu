#include <cmath>
#include <iomanip>
#include <iostream>

#include "solver/scf.cuh"
#include "utilities/error.cuh"

namespace dftcu {

// CUDA kernels
__global__ void mix_density_kernel(size_t n, double* d_rho_old, const double* d_rho_new,
                                   double alpha, double beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_rho_old[idx] = alpha * d_rho_old[idx] + beta * d_rho_new[idx];
    }
}

__global__ void density_diff_kernel(size_t n, const double* d_rho1, const double* d_rho2,
                                    double* d_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_diff[idx] = fabs(d_rho1[idx] - d_rho2[idx]);
    }
}

SCFSolver::SCFSolver(Grid& grid, const Options& options)
    : grid_(grid),
      options_(options),
      davidson_(grid, options.davidson_max_iter, options.davidson_tol) {
    history_.reserve(options_.max_iter);

    if (options_.mixing_type == MixingType::Linear) {
        mixer_ = std::make_unique<LinearMixer>(grid, options_.mixing_beta);
    } else {
        mixer_ =
            std::make_unique<BroydenMixer>(grid, options_.mixing_beta, options_.mixing_history);
    }
}

double SCFSolver::solve(Hamiltonian& ham, Wavefunction& psi, const std::vector<double>& occupations,
                        RealField& rho) {
    converged_ = false;
    num_iterations_ = 0;
    history_.clear();
    mixer_->reset();

    if (options_.verbose) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "SCF Iteration Started\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << std::setw(6) << "Iter" << std::setw(20) << "Total Energy (Ha)" << std::setw(15)
                  << "ΔE (Ha)" << std::setw(15) << "Δρ (e⁻)"
                  << "\n";
        std::cout << std::string(70, '-') << "\n";
    }

    double e_total = 0.0;
    double e_total_old = 0.0;

    // Allocate workspace for density mixing
    RealField rho_old(grid_, 1);
    RealField rho_new(grid_, 1);

    // Copy initial density
    rho_old.copy_from(rho);

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        num_iterations_ = iter + 1;

        // Step 1: Update Hamiltonian potentials with current density
        ham.update_potentials(rho);
        grid_.synchronize();

        // Step 2: Solve eigenvalue problem H|ψ⟩ = ε|ψ⟩
        std::vector<double> eigenvalues = davidson_.solve(ham, psi);
        grid_.synchronize();

        // Step 3: Compute new density from wavefunctions
        rho_new.fill(0.0);
        psi.compute_density(occupations, rho_new);
        grid_.synchronize();

        // Step 4: Compute total energy using KS-DFT formula
        e_total = compute_total_energy(eigenvalues, occupations, ham, rho);

        // Step 5: Check convergence
        double delta_e = e_total - e_total_old;
        double delta_rho = density_difference(rho_old, rho_new);

        // Record history
        history_.push_back({static_cast<double>(iter), e_total, delta_e, delta_rho});

        if (options_.verbose) {
            std::cout << std::setw(6) << iter << std::setw(20) << std::fixed
                      << std::setprecision(10) << e_total << std::setw(15) << std::scientific
                      << std::setprecision(2) << delta_e << std::setw(15) << delta_rho << "\n";

            // Component breakdown
            Evaluator& eval = ham.get_evaluator();
            RealField vt(grid_);
            double e_f = eval.compute(rho, vt);
            double e_k = psi.compute_kinetic_energy(occupations) * 2.0;  // To Ry
            double e_nl = 0.0;
            if (ham.has_nonlocal())
                e_nl = ham.get_nonlocal().calculate_energy(psi, occupations) * 2.0;

            printf("      DEBUG Components (Ry): E_kin=%.6f, E_nl=%.6f, E_eval=%.6f\n", e_k, e_nl,
                   e_f * 2.0);
        }

        // Check convergence (skip first iteration for delta_e check)
        if (iter > 0 && std::abs(delta_e) < options_.e_conv && delta_rho < options_.rho_conv) {
            converged_ = true;
            if (options_.verbose) {
                std::cout << std::string(70, '-') << "\n";
                std::cout << "✓ SCF Converged!\n";
                std::cout << "  Final Energy: " << std::fixed << std::setprecision(10) << e_total
                          << " Ha\n";
                std::cout << "  Iterations:   " << num_iterations_ << "\n";
                std::cout << std::string(70, '=') << "\n";
            }
            break;
        }

        // Step 6: Mix densities for next iteration
        mixer_->mix(rho, rho_new, rho);
        rho_old.copy_from(rho);

        e_total_old = e_total;
    }

    if (!converged_ && options_.verbose) {
        std::cout << std::string(70, '-') << "\n";
        std::cout << "⚠ SCF did not converge within " << options_.max_iter << " iterations\n";
        std::cout << "  Last ΔE:   " << std::scientific << std::abs(e_total - e_total_old)
                  << " Ha\n";
        std::cout << "  Last Δρ:   " << density_difference(rho_old, rho_new) << " e⁻\n";
        std::cout << std::string(70, '=') << "\n";
    }

    return e_total;
}

double SCFSolver::compute_total_energy(const std::vector<double>& eigenvalues,
                                       const std::vector<double>& occupations, Hamiltonian& ham,
                                       const RealField& rho) {
    // Exact KS-DFT Total Energy Formula:
    // E_total = E_band - ∫ ρ(r)V_eff(r) dr + E_functional(ρ)
    //
    // where:
    //   E_band = Σ f_i · ε_i is the sum of eigenvalues (in Hartree).
    //   V_eff  = V_loc + V_H + V_XC is the total local potential.
    //   E_functional = E_vloc + E_H + E_XC + E_ewald is the sum of energies from functionals.
    //
    // Note: Since V_eff is unscaled (contains N_nr factor) if it comes from raw IFFT,
    // and dv_bohr contains 1/N_nr factor, the product rho.dot(v_eff) * dv_bohr
    // yields the correct physical energy in Hartree.

    // 1. Calculate E_band
    double eband = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        eband += occupations[i] * eigenvalues[i];
    }

    // 2. Calculate <rho | V_eff> using the total potential from the Hamiltonian
    const RealField& v_eff = ham.v_loc();
    double rho_dot_veff = rho.dot(v_eff) * grid_.dv_bohr();

    // 3. Calculate E_functional from Evaluator
    Evaluator& evaluator = ham.get_evaluator();
    RealField v_eval_tmp(grid_, 1);
    double e_eval = evaluator.compute(rho, v_eval_tmp);

    // Final Total Energy in Hartree
    double e_total = eband - rho_dot_veff + e_eval;

    return e_total;
}

void SCFSolver::mix_density(RealField& rho_old, const RealField& rho_new, double beta) {
    // Legacy method, no longer used as we use Mixer class
    LinearMixer(grid_, beta).mix(rho_old, rho_new, rho_old);
}

double SCFSolver::density_difference(const RealField& rho1, const RealField& rho2) {
    // Compute ∫|ρ₁ - ρ₂|dr = Σ|ρ₁(i) - ρ₂(i)| * dV (in e⁻)

    size_t n = rho1.size();
    const double* d_rho1 = rho1.data();
    const double* d_rho2 = rho2.data();
    double dv = grid_.dv_bohr();

    // Allocate temporary buffer for absolute differences
    GPU_Vector<double> diff(n);

    // Compute |ρ₁ - ρ₂|
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    density_diff_kernel<<<blocks, threads>>>(n, d_rho1, d_rho2, diff.data());
    CHECK(cudaGetLastError());

    // Sum all differences
    double sum = 0.0;
    std::vector<double> h_diff(n);
    CHECK(cudaMemcpy(h_diff.data(), diff.data(), n * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        sum += h_diff[i];
    }

    return sum * dv;
}

}  // namespace dftcu
