#pragma once
#include <complex>
#include <vector>

#include "model/atoms.cuh"
#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

// Forward declaration
class PseudopotentialData;

/**
 * @brief Non-local pseudopotential using Kleinman-Bylander (KB) projectors.
 */
class NonLocalPseudo {
  public:
    NonLocalPseudo(Grid& grid);
    ~NonLocalPseudo() = default;

    /**
     * @brief Factory method to create NonLocalPseudo from UPF data
     * @param grid Grid object
     * @param atoms Atoms object containing atomic positions
     * @param upf_data Parsed UPF pseudopotential data
     * @param atom_type Atom type index (default 0)
     * @return Shared pointer to initialized NonLocalPseudo object
     */
    static std::shared_ptr<NonLocalPseudo> from_upf(Grid& grid, const Atoms& atoms,
                                                    const PseudopotentialData& upf_data,
                                                    int atom_type = 0);

    void apply(Wavefunction& psi_in, Wavefunction& h_psi_out);
    void add_projector(const std::vector<std::complex<double>>& beta_g, double coupling_constant);
    void clear();
    double calculate_energy(const Wavefunction& psi, const std::vector<double>& occupations);

    void init_tab_beta(int type, const std::vector<double>& r_grid,
                       const std::vector<std::vector<double>>& beta_r,
                       const std::vector<double>& rab, const std::vector<int>& l_list,
                       const std::vector<int>& kkbeta_list, double omega_angstrom);

    void set_tab_beta(int type, int nb, const std::vector<double>& tab);
    void init_dij(int type, const std::vector<double>& dij);
    void update_projectors(const Atoms& atoms);
    void set_projectors(const std::vector<std::complex<double>>& projectors);

    int num_projectors() const { return num_projectors_; }

    std::vector<double> get_tab_beta(int type, int nb) const {
        if (type < (int)tab_beta_.size() && nb < (int)tab_beta_[type].size()) {
            return tab_beta_[type][nb];
        }
        return {};
    }

    std::vector<std::complex<double>> get_projector(int idx) const;
    std::vector<std::complex<double>> get_projections() const;
    std::vector<double> get_coupling() const;
    std::vector<std::complex<double>> get_d_projections() const;
    void debug_projections(const Wavefunction& psi, const std::vector<double>& qe_becp,
                           const std::vector<std::complex<double>>& qe_vkb,
                           const std::vector<std::complex<double>>& qe_evc,
                           const std::vector<std::vector<int>>& miller);

  private:
    Grid& grid_;
    int num_projectors_ = 0;
    std::vector<std::vector<std::vector<double>>> tab_beta_;
    std::vector<std::vector<int>> l_list_;
    double dq_ = 0.01;
    int nqx_ = 0;
    double omega_ = 0.0;
    std::vector<std::vector<std::vector<double>>> d_ij_;

    GPU_Vector<gpufftComplex> d_projectors_;
    GPU_Vector<double> d_coupling_;
    GPU_Vector<gpufftComplex> d_projections_;
    GPU_Vector<gpufftComplex> d_dps_;
};

}  // namespace dftcu
