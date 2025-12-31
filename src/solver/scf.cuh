#pragma once
#include <vector>

#include "model/field.cuh"
#include "model/wavefunction.cuh"
#include "solver/davidson.cuh"
#include "solver/hamiltonian.cuh"
#include "solver/mixer.cuh"

namespace dftcu {

/**
 * @brief Self-consistent field (SCF) solver for Kohn-Sham DFT
 *
 * Implements the iterative SCF procedure:
 * 1. Start with initial density ρ₀
 * 2. Update Hamiltonian with current density
 * 3. Solve eigenvalue problem H|ψ⟩ = ε|ψ⟩
 * 4. Compute new density from wavefunctions
 * 5. Mix densities and check convergence
 * 6. Repeat until converged
 */
class SCFSolver {
  public:
    enum class MixingType { Linear, Broyden };

    struct Options {
        int max_iter;            ///< Maximum SCF iterations
        double e_conv;           ///< Energy convergence threshold (Ha)
        double rho_conv;         ///< Density convergence threshold (electrons)
        MixingType mixing_type;  ///< Type of density mixing
        double mixing_beta;      ///< Density mixing parameter (0 < β ≤ 1)
        int mixing_history;      ///< History size for Broyden mixing
        int davidson_max_iter;   ///< Davidson solver max iterations
        double davidson_tol;     ///< Davidson solver tolerance
        bool verbose;            ///< Print iteration info

        Options()
            : max_iter(100),
              e_conv(1e-8),
              rho_conv(1e-6),
              mixing_type(MixingType::Broyden),
              mixing_beta(0.5),
              mixing_history(8),
              davidson_max_iter(50),
              davidson_tol(1e-8),
              verbose(true) {}
    };

    /**
     * @brief Construct SCF solver
     * @param grid Reference to simulation grid
     * @param options SCF solver options
     */
    SCFSolver(Grid& grid, const Options& options);
    ~SCFSolver() = default;

    /**
     * @brief Run SCF calculation
     * @param ham Hamiltonian with all functionals configured
     * @param psi Initial wavefunctions (will be updated to converged eigenstates)
     * @param occupations Band occupations
     * @param rho_init Initial density (will be updated to converged density)
     * @return Final total energy (Ha)
     */
    double solve(Hamiltonian& ham, Wavefunction& psi, const std::vector<double>& occupations,
                 RealField& rho_init);

    /**
     * @brief Get convergence history
     * @return Vector of {iteration, total_energy, delta_E, delta_rho}
     */
    const std::vector<std::array<double, 4>>& get_history() const { return history_; }

    /**
     * @brief Check if last solve converged
     */
    bool is_converged() const { return converged_; }

    /**
     * @brief Get number of iterations performed
     */
    int num_iterations() const { return num_iterations_; }

  private:
    Grid& grid_;
    Options options_;
    DavidsonSolver davidson_;
    std::unique_ptr<Mixer> mixer_;

    // Convergence tracking
    bool converged_ = false;
    int num_iterations_ = 0;
    std::vector<std::array<double, 4>> history_;  // {iter, E_tot, dE, dRho}

    /**
     * @brief Compute total energy using KS-DFT formula with double-counting correction
     *
     * Formula: E = eband + deband + E_H + E_XC + E_Ewald
     * where:
     *   eband  = Σ f_i · ε_i
     *   deband = -∫ ρ(V_H + V_XC) dr
     */
    double compute_total_energy(const std::vector<double>& eigenvalues,
                                const std::vector<double>& occupations, Hamiltonian& ham,
                                const RealField& rho);

    /**
     * @brief Simple linear density mixing: ρ_new = (1-β)ρ_old + βρ_new
     */
    void mix_density(RealField& rho_old, const RealField& rho_new, double beta);

    /**
     * @brief Calculate density difference integral
     */
    double density_difference(const RealField& rho1, const RealField& rho2);
};

}  // namespace dftcu
