#include <cmath>
#include <iomanip>
#include <iostream>

#include "solver/scf.cuh"
#include "utilities/cublas_manager.cuh"
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
                        RealField& rho, std::shared_ptr<Atoms> atoms, double ecutrho,
                        const RealField* rho_core, double alpha_energy) {
    converged_ = false;
    num_iterations_ = 0;
    history_.clear();
    mixer_->reset();
    alpha_energy_ = alpha_energy;
    atoms_ = atoms;

    // Store rho_core if provided
    if (rho_core) {
        rho_core_ = std::make_unique<RealField>(grid_, 1);
        rho_core_->copy_from(*rho_core);
    } else {
        rho_core_.reset();
    }

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
            RealField rho_xc_init(grid_, 1);

            // Compute V_H and V_XC from initial density
            hartree_.compute(rho, vh_init);

            // XC uses rho_val + rho_core
            rho_xc_init.copy_from(rho);
            if (rho_core_) {
                rho_xc_init = rho_xc_init + *rho_core_;
            }

            lda_.compute(rho_xc_init, vxc_init);

            // V_ps = V_eff - V_H - V_XC
            v_ps_saved = ham.v_loc() - vh_init - vxc_init;
            grid_.synchronize();
        } else {
            // Update V_eff = V_H + V_XC + V_ps manually
            RealField vh(grid_, 1);
            RealField vxc(grid_, 1);
            RealField rho_xc(grid_, 1);

            // Compute V_H from current density
            hartree_.compute(rho, vh);

            // Compute V_XC from abs(current density + rho_core)
            rho_xc.copy_from(rho);
            if (rho_core_) {
                rho_xc = rho_xc + *rho_core_;
            }

            // Apply abs and threshold in XC calculation is handled by LDA_PZ kernel
            lda_.compute(rho_xc, vxc);

            // Update ham.v_loc() = V_H + V_XC + V_ps
            ham.v_loc() = vh + vxc + v_ps_saved;
            grid_.synchronize();
        }

        // Step 2: Solve eigenvalue problem H|ψ⟩ = ε|ψ⟩ using subspace diagonalization

        int nbands = psi.num_bands();

        // Apply Hamiltonian to get H|ψ⟩

        Wavefunction h_psi(grid_, nbands, psi.encut());

        ham.apply(psi, h_psi);

        grid_.synchronize();

        // Build subspace matrices H and S on GPU using cublasZgemm

        GPU_Vector<gpufftComplex> h_matrix(nbands * nbands);

        GPU_Vector<gpufftComplex> s_matrix(nbands * nbands);

        GPU_Vector<double> eigenvalues_gpu(nbands);

        GPU_Vector<gpufftComplex> eigenvectors(nbands * nbands);

        size_t nnr = grid_.nnr();

        cublasHandle_t cb_handle = CublasManager::instance().handle();

        CUBLAS_SAFE_CALL(cublasSetStream(cb_handle, grid_.stream()));

        gpufftComplex alpha = {1.0, 0.0};

        gpufftComplex beta = {0.0, 0.0};

        CublasPointerModeGuard guard(cb_handle, CUBLAS_POINTER_MODE_HOST);

        // H_ij = <psi_i | H | psi_j>

        CUBLAS_SAFE_CALL(cublasZgemm(cb_handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, (int)nnr,

                                     (const cuDoubleComplex*)&alpha,

                                     (const cuDoubleComplex*)psi.data(), (int)nnr,

                                     (const cuDoubleComplex*)h_psi.data(), (int)nnr,

                                     (const cuDoubleComplex*)&beta,

                                     (cuDoubleComplex*)h_matrix.data(), nbands));

        // S_ij = <psi_i | psi_j>

        CUBLAS_SAFE_CALL(cublasZgemm(cb_handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, (int)nnr,

                                     (const cuDoubleComplex*)&alpha,

                                     (const cuDoubleComplex*)psi.data(), (int)nnr,

                                     (const cuDoubleComplex*)psi.data(), (int)nnr,

                                     (const cuDoubleComplex*)&beta,

                                     (cuDoubleComplex*)s_matrix.data(), nbands));

        // Solve generalized eigenvalue problem

        subspace_solver_.solve_generalized(nbands, h_matrix.data(), s_matrix.data(),

                                           eigenvalues_gpu.data(), eigenvectors.data());

        grid_.synchronize();

        // Update wavefunctions: psi_new = psi * eigenvectors

        Wavefunction psi_new(grid_, nbands, psi.encut());

        CUBLAS_SAFE_CALL(cublasZgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)nnr, nbands, nbands,

                                     (const cuDoubleComplex*)&alpha,

                                     (const cuDoubleComplex*)psi.data(), (int)nnr,

                                     (const cuDoubleComplex*)eigenvectors.data(), nbands,

                                     (const cuDoubleComplex*)&beta,

                                     (cuDoubleComplex*)psi_new.data(), (int)nnr));

        CHECK(cudaMemcpyAsync(psi.data(), psi_new.data(),

                              nnr * nbands * sizeof(gpufftComplex),

                              cudaMemcpyDeviceToDevice, grid_.stream()));

        grid_.synchronize();

        // Copy eigenvalues back to host

        std::vector<double> eigenvalues(nbands);

        eigenvalues_gpu.copy_to_host(eigenvalues.data());

        grid_.synchronize();

        // Step 3: Compute new density from wavefunctions

        rho_new.fill(0.0);

        psi.compute_density(occupations, rho_new);

        grid_.synchronize();

        // Step 4: Compute total energy BEFORE mixing (following QE's exact order)
        e_total = compute_total_energy(eigenvalues, occupations, ham, rho, rho_new);

        if (iter == 0 && options_.verbose) {
            EnergyBreakdown eb = compute_energy_breakdown(eigenvalues, occupations, ham, psi, rho,
                                                          rho_core_ ? rho_core_.get() : nullptr);
            printf("      DEBUG Iter 0 Breakdown: ehart=%.6f, etxc=%.6f, eeval=%.6f, eewld=%.6f, "
                   "deband=%.6f\n",
                   eb.ehart, eb.etxc, eb.eband, eb.eewld, eb.deband);
        }

        // Step 5: Mix densities for next iteration
        mixer_->mix(rho, rho_new, rho);  // rho becomes ρ_mixed
        grid_.synchronize();

        // Step 6: Check convergence
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

        // Update rho_old for next iteration (rho is already mixed at line 201)
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
                                       const RealField& rho_in, const RealField& rho_out) {
    // 1. E_band = Σ f_i * ε_i
    double eband = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        eband += occupations[i] * eigenvalues[i];
    }

    // 2. Compute E_H and V_H using rho_in (input/mixed density)
    RealField vh(grid_, 1);
    double ehart = hartree_.compute(rho_in, vh);

    // 3. Compute E_XC and V_XC using rho_in + rho_core
    RealField vxc(grid_, 1);
    RealField rho_xc(grid_, 1);
    rho_xc.copy_from(rho_in);
    if (rho_core_) {
        rho_xc = rho_xc + *rho_core_;
    }
    double etxc = lda_.compute(rho_xc, vxc);

    // 4. Compute deband = -∫ ρ_out * (V_H[ρ_in] + V_XC[ρ_in]) dr
    RealField v_hxc(grid_, 1);
    v_hxc = vh + vxc;

    double deband = -rho_out.dot(v_hxc) * grid_.dv_bohr();

    // 5. Compute Ewald energy
    double eewld = ewald_->compute(false, ecutrho_ha_);

    // 6. Total energy
    double e_total = eband + deband + ehart + etxc + eewld - alpha_energy_;

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
    std::vector<double> h_diff(n);
    diff.copy_to_host(h_diff.data());

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += h_diff[i];
    }

    return sum * dv;
}

EnergyBreakdown SCFSolver::compute_energy_breakdown(const std::vector<double>& eigenvalues,
                                                    const std::vector<double>& occupations,
                                                    Hamiltonian& ham, const Wavefunction& psi,
                                                    const RealField& rho_val,
                                                    const RealField* rho_core) {
    EnergyBreakdown breakdown;

    // 1. Band energy: Σ f_i * ε_i (Hartree)
    breakdown.eband = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        breakdown.eband += occupations[i] * eigenvalues[i];
    }

    // 2. Compute Hartree energy and potential
    RealField vh(grid_, 1);
    breakdown.ehart = hartree_.compute(rho_val, vh);

    // 3. Compute XC energy and potential
    RealField vxc(grid_, 1);
    RealField rho_xc(grid_, 1);
    rho_xc.copy_from(rho_val);
    if (rho_core) {
        rho_xc = rho_xc + *rho_core;
    }
    breakdown.etxc = lda_.compute(rho_xc, vxc);

    // 4. Compute deband = -∫ ρ_val * (V_H + V_XC) dr
    RealField v_hxc(grid_, 1);
    v_hxc = vh + vxc;
    breakdown.deband = -rho_val.dot(v_hxc) * grid_.dv_bohr();

    // 5. Compute Ewald energy
    breakdown.eewld = ewald_->compute(false, ecutrho_ha_);

    // 6. Alpha term (G=0 limit correction)
    breakdown.alpha = alpha_energy_;

    // 7. Total energy
    breakdown.etot = breakdown.eband + breakdown.deband + breakdown.ehart + breakdown.etxc +
                     breakdown.eewld - breakdown.alpha;

    return breakdown;
}

}  // namespace dftcu
