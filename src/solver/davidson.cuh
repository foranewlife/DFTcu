#pragma once
#include <vector>

#include "solver/hamiltonian.cuh"

#include <cusolverDn.h>

namespace dftcu {

/**
 * @brief Davidson iterative solver for the Kohn-Sham eigenvalue problem.
 *
 * This class implements a simplified Davidson algorithm for finding the lowest
 * eigenvalues and eigenvectors of the Hamiltonian. It matches VASP's
 * ALGO=Normal logic for subspace rotation and refinement.
 */
class DavidsonSolver {
  public:
    /**
     * @brief Construct Davidson solver
     * @param grid Reference to simulation grid
     * @param max_iter Maximum number of Davidson iterations
     * @param tol Convergence tolerance for eigenvalues
     */
    DavidsonSolver(Grid& grid, int max_iter = 10, double tol = 1e-6);
    ~DavidsonSolver();

    /**
     * @brief Solve H|psi> = epsilon|psi>
     * @param ham The Hamiltonian operator
     * @param psi Starting wavefunctions (reciprocal space), updated to
     * eigenvectors
     * @param eigenvalues Output list of computed eigenvalues (energies)
     */
    std::vector<double> solve(Hamiltonian& ham, Wavefunction& psi);

  private:
    Grid& grid_;
    int max_iter_;
    double tol_;

    cusolverDnHandle_t cusolver_handle_ = nullptr;

    // Subspace buffers
    GPU_Vector<gpufftComplex> h_matrix_;  // [nbands][nbands]
    GPU_Vector<double> eval_buffer_;      // Eigenvalues buffer for cuSOLVER

    /** @brief Subspace rotation: psi = psi * U */
    void rotate_subspace(Wavefunction& psi, const GPU_Vector<gpufftComplex>& eigenvectors);

    /** @brief Orthogonalize bands using Cholesky-based Lowdin method */
    void orthogonalize(Wavefunction& psi);
};

}  // namespace dftcu
