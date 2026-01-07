#pragma once
#include <vector>

#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "solver/hamiltonian.cuh"
#include "solver/scf.cuh"  // For EnergyBreakdown
#include "solver/subspace_solver.cuh"

namespace dftcu {

/**
 * @brief Non-Self-Consistent Field (NSCF) solver for Kohn-Sham DFT
 *
 * Implements the single-shot diagonalization of a fixed potential.
 */
class NonSCFSolver {
  public:
    NonSCFSolver(Grid& grid);
    ~NonSCFSolver() = default;

    /**
     * @brief Run NSCF calculation
     * @param ham Hamiltonian with potentials already updated
     * @param psi Initial wavefunctions, updated to eigenstates
     * @param nelec Number of electrons
     * @param atoms Atomic positions (needed for Ewald energy)
     * @param ecutrho Energy cutoff in Ry
     * @param rho_core Core charge density
     * @param alpha_energy Alpha term correction
     * @return Energy breakdown structure
     */
    EnergyBreakdown solve(Hamiltonian& ham, Wavefunction& psi, double nelec,
                          std::shared_ptr<Atoms> atoms, double ecutrho,
                          const RealField* rho_core = nullptr, double alpha_energy = 0.0);

  private:
    Grid& grid_;
    SubspaceSolver subspace_solver_;

    /**
     * @brief Compute weights (occupations) for insulator case
     */
    void compute_weights_insulator(int nbands, double nelec, const std::vector<double>& eigenvalues,
                                   std::vector<double>& occupations, double& ef);
};

}  // namespace dftcu
