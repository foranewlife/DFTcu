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
      subspace_solver_(grid),
      hartree_(),
      lda_(),
      ecutrho_ha_(200.0) {  // Default ecutrho = 400 Ry = 200 Ha
    history_.reserve(options_.max_iter);

    // Configure Hartree functional (no spherical cutoff, matching QE)
    hartree_.set_gcut(-1.0);

    // Configure LDA_PZ with QE's hardcoded threshold
    lda_.set_rho_threshold(1e-10);

    if (options_.mixing_type == MixingType::Linear) {
        mixer_ = std::make_unique<LinearMixer>(grid, options_.mixing_beta);
    } else {
        mixer_ =
            std::make_unique<BroydenMixer>(grid, options_.mixing_beta, options_.mixing_history);
    }
}

double SCFSolver::solve(Hamiltonian& ham, Wavefunction& psi, const std::vector<double>& occupations,
                        RealField& rho, std::shared_ptr<Atoms> atoms, double ecutrho) {
    converged_ = false;
    num_iterations_ = 0;
    history_.clear();
    mixer_->reset();

    // Initialize Ewald functional with atoms and ecutrho
    ecutrho_ha_ = ecutrho * 0.5;  // Convert Ry to Ha
    ewald_ = std::make_unique<Ewald>(grid_, atoms, 1e-10, ecutrho_ha_);

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

    // Save V_ps from initial Hamiltonian (it doesn't change during SCF)
    // We'll reconstruct V_eff = V_H + V_XC + V_ps manually to use abs(rho) for XC
    RealField v_ps_saved(grid_, 1);
    bool v_ps_initialized = false;

    // Allocate workspace for density mixing
    RealField rho_old(grid_, 1);
    RealField rho_new(grid_, 1);

    // Copy initial density
    rho_old.copy_from(rho);

    for (int iter = 0; iter < options_.max_iter; ++iter) {
        num_iterations_ = iter + 1;

        // Step 1: Update Hamiltonian potentials with current density
        if (iter == 0) {
            // Save V_ps from initial Hamiltonian (extract it by computing V_H + V_XC and
            // subtracting)
            RealField vh_init(grid_, 1);
            RealField vxc_init(grid_, 1);
            RealField rho_abs_init(grid_, 1);

            // Compute V_H and V_XC from initial density
            hartree_.compute(rho, vh_init);

            // Use abs(rho) for XC
            size_t n = rho.size();
            std::vector<double> rho_host(n);
            rho.copy_to_host(rho_host.data());
            for (size_t i = 0; i < n; ++i) {
                rho_host[i] = std::abs(rho_host[i]);
            }
            rho_abs_init.copy_from_host(rho_host.data());
            lda_.compute(rho_abs_init, vxc_init);

            // V_ps = V_eff - V_H - V_XC
            v_ps_saved = ham.v_loc() - vh_init - vxc_init;
            v_ps_initialized = true;
        } else {
            // Update V_eff = V_H + V_XC + V_ps manually
            RealField vh(grid_, 1);
            RealField vxc(grid_, 1);
            RealField rho_abs(grid_, 1);

            // Compute V_H from current density
            hartree_.compute(rho, vh);

            // Compute V_XC from abs(current density)
            size_t n = rho.size();
            std::vector<double> rho_host(n);
            rho.copy_to_host(rho_host.data());
            for (size_t i = 0; i < n; ++i) {
                rho_host[i] = std::abs(rho_host[i]);
            }
            rho_abs.copy_from_host(rho_host.data());
            lda_.compute(rho_abs, vxc);

            // Update ham.v_loc() = V_H + V_XC + V_ps
            ham.v_loc() = vh + vxc + v_ps_saved;
            grid_.synchronize();
        }

        // Step 2: Solve eigenvalue problem H|ψ⟩ = ε|ψ⟩ using subspace diagonalization
        int nbands = psi.num_bands();

        // Apply Hamiltonian to get H|ψ⟩
        Wavefunction h_psi(grid_, nbands, 100.0);  // Use same ecut as psi
        ham.apply(psi, h_psi);
        grid_.synchronize();

        // Build subspace matrices H and S
        GPU_Vector<gpufftComplex> h_matrix(nbands * nbands);
        GPU_Vector<gpufftComplex> s_matrix(nbands * nbands);
        GPU_Vector<double> eigenvalues_gpu(nbands);
        GPU_Vector<gpufftComplex> eigenvectors(nbands * nbands);

        // Copy wavefunctions to host for matrix construction
        size_t nnr = grid_.nnr();
        std::vector<std::complex<double>> psi_h(nbands * nnr);
        std::vector<std::complex<double>> hpsi_h(nbands * nnr);
        psi.copy_to_host(psi_h.data());
        h_psi.copy_to_host(hpsi_h.data());

        // Build matrices: H_ij = <ψ_i|H|ψ_j>, S_ij = <ψ_i|ψ_j>
        std::vector<gpufftComplex> h_mat(nbands * nbands);
        std::vector<gpufftComplex> s_mat(nbands * nbands);
        for (int i = 0; i < nbands; ++i) {
            for (int j = 0; j < nbands; ++j) {
                std::complex<double> h_sum = 0.0;
                std::complex<double> s_sum = 0.0;
                for (size_t k = 0; k < nnr; ++k) {
                    std::complex<double> psi_i = psi_h[i * nnr + k];
                    std::complex<double> psi_j = psi_h[j * nnr + k];
                    std::complex<double> hpsi_j = hpsi_h[j * nnr + k];
                    // h_sum += conj(psi_i) * hpsi_j
                    h_sum += std::conj(psi_i) * hpsi_j;
                    // s_sum += conj(psi_i) * psi_j
                    s_sum += std::conj(psi_i) * psi_j;
                }
                // No volume normalization or unit conversion needed
                // Hamiltonian::apply() already returns H|ψ⟩ in Hartree units
                h_mat[i * nbands + j] = {h_sum.real(), h_sum.imag()};
                s_mat[i * nbands + j] = {s_sum.real(), s_sum.imag()};
            }
        }

        h_matrix.copy_from_host(h_mat.data());
        s_matrix.copy_from_host(s_mat.data());

        // Solve generalized eigenvalue problem
        subspace_solver_.solve_generalized(nbands, h_matrix.data(), s_matrix.data(),
                                           eigenvalues_gpu.data(), eigenvectors.data());
        grid_.synchronize();

        // Copy eigenvalues back to host
        std::vector<double> eigenvalues(nbands);
        eigenvalues_gpu.copy_to_host(eigenvalues.data());

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
    // Kohn-Sham DFT Total Energy Formula (verified in align_step0.py):
    // E_total = E_band + E_deband + E_H + E_XC + E_Ewald
    //
    // where:
    //   E_band   = Σ f_i · ε_i
    //   E_deband = -∫ ρ(r)[V_H(r) + V_XC(r)] dr  (NOTE: NO V_ps!)
    //   E_H      = Hartree energy
    //   E_XC     = Exchange-correlation energy
    //   E_Ewald  = Ewald energy (ion-ion)

    // 1. E_band = Σ f_i * ε_i
    double eband = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        eband += occupations[i] * eigenvalues[i];
    }

    // 2. Compute E_H and V_H using Hartree functional
    RealField vh(grid_, 1);
    double ehart = hartree_.compute(rho, vh);

    // 3. Compute E_XC and V_XC using LDA_PZ functional
    RealField vxc(grid_, 1);
    double etxc = lda_.compute(rho, vxc);

    // 4. Compute deband = -∫ ρ * (V_H + V_XC) dr
    // NOTE: We use vh + vxc, NOT ham.v_loc() which includes V_ps
    RealField v_hxc(grid_, 1);
    v_hxc = vh + vxc;  // Expression template addition

    double deband = -rho.dot(v_hxc) * grid_.dv_bohr();

    // 5. Compute Ewald energy (ion-ion interaction)
    double eewld = ewald_->compute(false, ecutrho_ha_);

    // 6. Total energy
    double e_total = eband + deband + ehart + etxc + eewld;

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

EnergyBreakdown SCFSolver::compute_energy_breakdown(const std::vector<double>& eigenvalues,
                                                    const std::vector<double>& occupations,
                                                    Hamiltonian& ham, const Wavefunction& psi,
                                                    const RealField& rho) {
    EnergyBreakdown breakdown;

    // 1. Band energy: Σ f_i * ε_i (Hartree)
    breakdown.eband = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        breakdown.eband += occupations[i] * eigenvalues[i];
    }

    // 2. Compute Hartree energy and potential
    RealField vh(grid_, 1);
    breakdown.ehart = hartree_.compute(rho, vh);

    // 3. Compute XC energy and potential
    RealField vxc(grid_, 1);
    breakdown.etxc = lda_.compute(rho, vxc);

    // 4. Compute deband = -∫ ρ * (V_H + V_XC) dr
    RealField v_hxc(grid_, 1);
    v_hxc = vh + vxc;  // Expression template addition
    breakdown.deband = -rho.dot(v_hxc) * grid_.dv_bohr();

    // 5. Compute Ewald energy
    breakdown.eewld = ewald_->compute(false, ecutrho_ha_);

    // 6. Alpha term (G=0 limit correction) - extract from Ewald
    // For now, set to 0.0 as we don't expose this separately
    breakdown.alpha = 0.0;

    // 7. Total energy
    breakdown.etot =
        breakdown.eband + breakdown.deband + breakdown.ehart + breakdown.etxc + breakdown.eewld;

    return breakdown;
}

}  // namespace dftcu
