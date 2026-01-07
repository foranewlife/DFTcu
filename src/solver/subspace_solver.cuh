#pragma once
#include <vector>

#include "model/grid.cuh"
#include "model/wavefunction.cuh"

#include <cusolverDn.h>

namespace dftcu {

class Hamiltonian;

/**
 * @brief Handles the generalized eigenvalue problem in the electronic subspace.
 */
class SubspaceSolver {
  public:
    SubspaceSolver(Grid& grid);
    ~SubspaceSolver();

    /**
     * @brief Solve generalized eigenvalue problem Hc = epsilon Sc
     * @param h_matrix Hamiltonian matrix in subspace (nbands x nbands)
     * @param s_matrix Overlap matrix in subspace (nbands x nbands)
     * @param eigenvalues Output eigenvalues
     * @param eigenvectors Output eigenvectors (subspace rotation matrix)
     */
    void solve_generalized(int nbands, gpufftComplex* h_matrix, gpufftComplex* s_matrix,
                           double* eigenvalues, gpufftComplex* eigenvectors = nullptr);

    /**
     * @brief Solve generalized eigenvalue problem for Gamma-only (real symmetric matrices)
     *
     * Uses cusolverDnDsygvd (real symmetric) instead of cusolverDnZhegvd (complex Hermitian).
     * This matches QE's regterg.f90:234 which uses DSYGVD for Gamma-only calculations.
     *
     * @param h_matrix Real symmetric Hamiltonian matrix (nbands x nbands)
     * @param s_matrix Real symmetric overlap matrix (nbands x nbands)
     * @param eigenvalues Output eigenvalues
     * @param eigenvectors Output eigenvectors (subspace rotation matrix, column-major)
     */
    void solve_generalized_gamma(int nbands, double* h_matrix, double* s_matrix,
                                 double* eigenvalues, double* eigenvectors = nullptr);

    /**
     * @brief Compute eigenvalues directly from Hamiltonian and Wavefunction on GPU.
     * @param ham Hamiltonian operator
     * @param psi Current wavefunctions
     * @return Vector of eigenvalues (Ha)
     */
    std::vector<double> solve_direct(Hamiltonian& ham, Wavefunction& psi);

  private:
    Grid& grid_;
    cusolverDnHandle_t handle_ = nullptr;
};

}  // namespace dftcu
