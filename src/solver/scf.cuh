#pragma once
#include <vector>

#include "functional/ewald.cuh"
#include "functional/hartree.cuh"
#include "functional/xc/lda_pz.cuh"
#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/wavefunction.cuh"
#include "solver/hamiltonian.cuh"
#include "solver/mixer.cuh"
#include "solver/subspace_solver.cuh"

namespace dftcu {

/**
 * @brief Detailed energy breakdown for SCF analysis
 */
struct EnergyBreakdown {
    double etot;                      ///< Total energy (Ha)
    double eband;                     ///< Band energy: Σ f_i * ε_i (Ha)
    double deband;                    ///< Double-counting correction: -∫ ρ * V_eff dr (Ha)
    double ehart;                     ///< Hartree energy (Ha)
    double etxc;                      ///< XC energy (Ha)
    double eewld;                     ///< Ewald energy (Ha)
    double alpha;                     ///< Alpha term (G=0 limit correction) (Ha)
    std::vector<double> eigenvalues;  ///< Eigenvalues from last diagonalization (Ha)
};

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
        bool verbose;            ///< Print iteration info

        Options()
            : max_iter(100),
              e_conv(1e-8),
              rho_conv(1e-6),
              mixing_type(MixingType::Broyden),
              mixing_beta(0.5),
              mixing_history(8),
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
     * @param atoms Atomic positions (needed for Ewald energy)
     * @param ecutrho Energy cutoff in Ry (for Ewald, typically 400.0 Ry = 200.0 Ha)
     * @return Final total energy (Ha)
     */
    double solve(Hamiltonian& ham, Wavefunction& psi, const std::vector<double>& occupations,
                 RealField& rho_init, std::shared_ptr<Atoms> atoms, double ecutrho,
                 const RealField* rho_core = nullptr, double alpha_energy = 0.0);

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

    /**
     * @brief Compute detailed energy breakdown for analysis
     * @param eigenvalues Eigenvalues from diagonalization
     * @param occupations Band occupations
     * @param ham Hamiltonian (must have potentials updated)
     * @param psi Wavefunctions
     * @param rho Current density
     * @return Detailed energy breakdown structure
     */
    EnergyBreakdown compute_energy_breakdown(const std::vector<double>& eigenvalues,
                                             const std::vector<double>& occupations,
                                             Hamiltonian& ham, const Wavefunction& psi,
                                             const RealField& rho_val,
                                             const RealField* rho_core = nullptr);

    void set_alpha_energy(double alpha) { alpha_energy_ = alpha; }
    void set_atoms(std::shared_ptr<Atoms> atoms) { atoms_ = atoms; }
    void set_ecutrho(double ecutrho_ry);

  private:
    Grid& grid_;
    Options options_;
    SubspaceSolver subspace_solver_;
    std::unique_ptr<Mixer> mixer_;

    // Direct functional access (bypassing Evaluator for correct energy calculation)
    Hartree hartree_;
    LDA_PZ lda_;
    std::unique_ptr<Ewald> ewald_;
    std::unique_ptr<RealField> rho_core_;
    double ecutrho_ha_;  // Energy cutoff in Ha (ecutrho in Ry / 2)
    double alpha_energy_ = 0.0;
    std::shared_ptr<Atoms> atoms_;

    // Convergence tracking
    bool converged_ = false;
    int num_iterations_ = 0;
    std::vector<std::array<double, 4>> history_;  // {iter, E_tot, dE, dRho}

    /**
     * @brief Compute total energy following QE's convention
     *
     * Uses TWO densities (following QE's electrons.f90):
     *   - rho_in: input/mixed density, used for computing V_H, V_XC, ehart, etxc
     *   - rho_out: output density from wavefunctions, used for computing deband
     *
     * Formula: E = eband + deband + E_H + E_XC + E_Ewald
     * where:
     *   eband  = Σ f_i · ε_i
     *   deband = -∫ ρ_out · (V_H[ρ_in] + V_XC[ρ_in]) dr
     *   ehart  = E_H[ρ_in]
     *   etxc   = E_XC[ρ_in + rho_core]
     */
    double compute_total_energy(const std::vector<double>& eigenvalues,
                                const std::vector<double>& occupations, Hamiltonian& ham,
                                const RealField& rho_in, const RealField& rho_out);

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
