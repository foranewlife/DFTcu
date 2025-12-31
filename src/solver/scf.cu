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
}

double SCFSolver::solve(Hamiltonian& ham, Wavefunction& psi, const std::vector<double>& occupations,
                        RealField& rho) {
    converged_ = false;
    num_iterations_ = 0;
    history_.clear();

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
        mix_density(rho, rho_new, options_.mixing_beta);
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
    // KS-DFT Total Energy Formula with Double-Counting Correction:
    // E_total = eband + deband + E_H + E_XC + E_Ewald
    //
    // where:
    //   eband  = Σ f_i · ε_i                   (band energy sum)
    //   deband = -∫ ρ(r) · [V_H(r) + V_XC(r)] dr  (double-counting correction)
    //   E_H    = Hartree energy
    //   E_XC   = Exchange-correlation energy
    //   E_Ewald = Ewald energy (ion-ion repulsion)
    //
    // The deband term corrects for the fact that E_H and E_XC are already
    // included in the eigenvalues through the KS potential.

    // 1. Calculate eband = Σ f_i · ε_i
    double eband = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        eband += occupations[i] * eigenvalues[i];
    }

    // 2. Get the current local potential V_tot = V_H + V_XC + V_loc
    const RealField& v_tot = ham.v_loc();

    // 3. We need to extract V_H and V_XC separately
    // To do this, we need to compute them individually from the evaluator
    // For now, use a simplified approach: compute all energies from evaluator

    // Get evaluator to compute individual energies
    Evaluator& evaluator = ham.get_evaluator();

    // Compute total potential and energy
    RealField v_eval(grid_, 1);
    double e_eval = evaluator.compute(rho, v_eval);

    // e_eval includes: E_loc + E_H + E_XC + E_Ewald
    // But for the KS formula, we need to separate them and apply double-counting correction

    // The correct formula is:
    // E_total = eband + (-∫ρV_H - ∫ρV_XC) + E_H + E_XC + E_Ewald + E_loc
    //         = eband - ∫ρ(V_H + V_XC) + (E_H + E_XC + E_Ewald + E_loc)
    //         = eband + deband + e_eval
    //
    // where deband = -∫ρV_tot + ∫ρV_loc (exclude V_loc from double-counting)
    //
    // But since V_tot = V_H + V_XC + V_loc, we have:
    // deband = -∫ρ·V_tot + ∫ρ·V_loc = -∫ρ(V_H + V_XC)

    // Calculate ∫ρ·V_tot
    double rho_dot_vtot = rho.dot(v_tot) * grid_.dv();

    // For now, use a simplified formula that assumes V_loc contribution
    // is small compared to V_H + V_XC
    // TODO: Separate V_loc to get exact deband = -∫ρ(V_H + V_XC)

    // Approximate: deband ≈ -∫ρ·V_tot (includes V_loc in double-counting)
    double deband = -rho_dot_vtot;

    // Total energy
    double e_total = eband + deband + e_eval;

    return e_total;
}

void SCFSolver::mix_density(RealField& rho_old, const RealField& rho_new, double beta) {
    // Simple linear mixing: ρ = (1-β)ρ_old + βρ_new
    // This is done in-place on rho_old

    size_t n = rho_old.size();
    double* d_rho_old = rho_old.data();
    const double* d_rho_new = rho_new.data();

    double alpha = 1.0 - beta;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    mix_density_kernel<<<blocks, threads>>>(n, d_rho_old, d_rho_new, alpha, beta);
    CHECK(cudaGetLastError());
}

double SCFSolver::density_difference(const RealField& rho1, const RealField& rho2) {
    // Compute ∫|ρ₁ - ρ₂|dr = Σ|ρ₁(i) - ρ₂(i)| * dV

    size_t n = rho1.size();
    const double* d_rho1 = rho1.data();
    const double* d_rho2 = rho2.data();
    double dv = grid_.dv();

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
