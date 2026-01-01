#pragma once
#include <complex>
#include <vector>

#include "model/atoms.cuh"
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

    /**
     * @brief Initialize interpolation table for non-local projectors from radial data.
     *
     * This mimics QE's init_tab_beta. It computes:
     *   tab(q, nb) = (4π/sqrt(Ω)) ∫ r β(r) j_l(qr) r dr
     *
     * @param type Atom type index.
     * @param r_grid Radial grid points (Bohr).
     * @param beta_r Projector functions on radial grid (units vary, usually Ha/Bohr).
     *               beta_r[nb][ir] corresponds to r*beta_l(r).
     * @param rab Radial grid integration weights.
     * @param l_list Angular momentum for each projector.
     * @param omega_angstrom Unit cell volume (Angstrom³).
     */
    void init_tab_beta(int type, const std::vector<double>& r_grid,
                       const std::vector<std::vector<double>>& beta_r,
                       const std::vector<double>& rab, const std::vector<int>& l_list,
                       double omega_angstrom);

    /**
     * @brief Manually set the radial interpolation table for a specific atom type and projector.
     * Useful for importing high-precision tables from external codes like QE.
     */
    void set_tab_beta(int type, int nb, const std::vector<double>& tab);

    /**
     * @brief Initialize the DIJ coupling matrix from radial data.
     * @param type Atom type index.
     * @param dij Coupling matrix (nb x nb) from UPF, provided as a flat vector.
     */
    void init_dij(int type, const std::vector<double>& dij);

    /**
     * @brief Compute projectors on the FFT grid using interpolation and structure factor.
     * This fills the internal d_projectors_ buffer.
     * @param atoms Atom positions and types.
     */
    void update_projectors(const Atoms& atoms);

    int num_projectors() const { return num_projectors_; }

    /**
     * @brief Get the radial interpolation table for a specific projector.
     * @param type Atom type index.
     * @param nb Projector index for this atom type.
     * @return Vector of values at q = i * dq_
     */
    std::vector<double> get_tab_beta(int type, int nb) const {
        if (type < (int)tab_beta_.size() && nb < (int)tab_beta_[type].size()) {
            return tab_beta_[type][nb];
        }
        return {};
    }

    std::vector<std::complex<double>> get_projector(int idx) const {
        size_t nnr = grid_.nnr();
        std::vector<std::complex<double>> host(nnr);
        CHECK(cudaMemcpy(host.data(), d_projectors_.data() + idx * nnr, nnr * sizeof(gpufftComplex),
                         cudaMemcpyDeviceToHost));
        return host;
    }

    std::vector<std::complex<double>> get_projections() const {
        std::vector<std::complex<double>> host(d_projections_.size());
        CHECK(cudaMemcpy(host.data(), d_projections_.data(),
                         d_projections_.size() * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));
        return host;
    }

  private:
    Grid& grid_;
    int num_projectors_ = 0;

    // Interpolation table: tab_beta_[type][beta_idx][iq]
    // Stores (4π/sqrt(Ω)) ∫ r² β(r) j_l(qr) dr
    std::vector<std::vector<std::vector<double>>> tab_beta_;

    // Angular momenta for each projector
    std::vector<std::vector<int>> l_list_;

    // Interpolation parameters (matching QE)
    static constexpr double dq_ = 0.01;  // Bohr^-1
    int nqx_ = 0;

    // Unit cell volume in Bohr³
    double omega_ = 0.0;

    // Coupling matrix from UPF: d_ij_[type][nb1][nb2]
    std::vector<std::vector<std::vector<double>>> d_ij_;

    // Projectors stored as [num_projectors][grid_nnr]
    GPU_Vector<gpufftComplex> d_projectors_;
    // Coupling constants D_i for each projector
    GPU_Vector<double> d_coupling_;

    // Internal buffers for matrix operations
    // <beta_i | psi_n>: matrix of size [num_projectors][num_bands]
    GPU_Vector<gpufftComplex> d_projections_;
};

}  // namespace dftcu
