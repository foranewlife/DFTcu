#pragma once
#include <complex>
#include <vector>

#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

/**
 * @brief Non-local pseudopotential using Kleinman-Bylander (KB) projectors.
 *
 * This class implements the action of the non-local part of the pseudopotential
 * on wavefunctions in reciprocal space. It follows the VASP logic where
 * projectors are applied to coefficients C_n(G).
 */
class NonLocalPseudo {
  public:
    /**
     * @brief Construct NonLocalPseudo handler
     * @param grid Reference to the simulation grid
     */
    NonLocalPseudo(Grid& grid);
    ~NonLocalPseudo() = default;

    /**
     * @brief Apply non-local operator: psi_out += V_nl * psi_in
     *
     * The operation is performed entirely in reciprocal space:
     * psi_out(G) += \sum_i D_i |\beta_i(G)> <\beta_i(G) | psi_in(G)>
     *
     * @param psi_in Input wavefunction (coefficients)
     * @param h_psi_out Wavefunction to accumulate results into
     */
    void apply(Wavefunction& psi_in, Wavefunction& h_psi_out);

    /**
     * @brief Load a KB projector into GPU memory
     * @param beta_g Reciprocal space projector values (size should match grid.nnr())
     * @param coupling_constant The D_l value associated with this projector
     */
    void add_projector(const std::vector<std::complex<double>>& beta_g, double coupling_constant);

    /** @brief Remove all projectors */
    void clear();

    /** @brief Calculate non-local energy contribution for a wavefunction */
    double calculate_energy(const Wavefunction& psi, const std::vector<double>& occupations);

    int num_projectors() const { return num_projectors_; }

  private:
    Grid& grid_;
    int num_projectors_ = 0;

    // Projectors stored as [num_projectors][grid_nnr]
    GPU_Vector<gpufftComplex> d_projectors_;
    // Coupling constants D_i for each projector
    GPU_Vector<double> d_coupling_;

    // Internal buffers for matrix operations
    // <beta_i | psi_n>: matrix of size [num_projectors][num_bands]
    GPU_Vector<gpufftComplex> d_projections_;
};

}  // namespace dftcu
