#pragma once
#include <complex>
#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

class FFTSolver;
class PseudopotentialData;  // Forward declaration

/**
 * @brief Local pseudopotential implementation following Quantum ESPRESSO's interpolation scheme.
 */
class LocalPseudoOperator {
  public:
    LocalPseudoOperator(Grid& grid, std::shared_ptr<Atoms> atoms);
    ~LocalPseudoOperator() = default;

    /**
     * @brief Initialize V_loc interpolation table from UPF radial data.
     * @param type Atom type index
     * @param r_grid Radial mesh (Bohr)
     * @param vloc_r Local potential on radial mesh (Hartree)
     * @param rab Integration weights dr (Bohr)
     * @param zp Valence charge
     * @param omega_bohr Unit cell volume in Bohr³
     * @param mesh_cutoff Mesh size cutoff (default: -1 = use full mesh)
     *
     * @note FIXME: Mixed units - omega_bohr is in Bohr³, but grid.gg() is in Angstrom⁻²
     *       This will be fixed when Grid units are unified to pure atomic units
     */
    void init_tab_vloc(int type, const std::vector<double>& r_grid,
                       const std::vector<double>& vloc_r, const std::vector<double>& rab, double zp,
                       double omega_bohr, int mesh_cutoff = -1);

    void set_valence_charge(int type, double zp);
    void compute(RealField& vloc_r);
    void set_gcut(double gcut) { gcut_ = gcut; }
    double compute(const RealField& rho, RealField& v_out);

    /** @brief Returns V_loc(G=0) in Hartree. Matches QE v_of_0 * 0.5. */
    double get_v_of_0() const { return v_of_0_; }

    void set_tab_vloc(int type, const std::vector<double>& tab) {
        if (type >= static_cast<int>(tab_vloc_.size()))
            tab_vloc_.resize(type + 1);
        tab_vloc_[type] = tab;
        nqx_ = (int)tab.size() - 1;
        d_tab_.resize(0);
    }

    std::vector<double> get_tab_vloc(int type) const {
        if (type >= static_cast<int>(tab_vloc_.size()))
            return {};
        return tab_vloc_[type];
    }

    double get_alpha(int type) const {
        if (type >= static_cast<int>(tab_vloc_.size()) || tab_vloc_[type].empty())
            return 0.0;
        return tab_vloc_[type][0];
    }

    /**
     * @brief Get V_loc(G) for specified G-shells.
     * @param type Atom type index
     * @param g_shells Vector of |G| moduli in Physical units: 2π/Bohr
     * @return V_loc(G) values in Hartree
     *
     * @note CRITICAL: g_shells must be in Physical units (2π/Bohr), not Crystallographic
     *       because Coulomb correction uses physical |G|: -4πZ/(Ω|G|²)exp(-|G|²/4)
     */
    std::vector<double> get_vloc_g_shells(int type, const std::vector<double>& g_shells) const;

    void set_dq(double dq) { dq_ = dq; }
    double get_dq() const { return dq_; }
    int get_nqx() const { return nqx_; }
    double get_omega() const { return omega_; }

  private:
    void initialize_buffers(Grid& grid);

    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;
    int atom_type_ = -1;  // 添加：记录此算符对应的原子类型
    std::vector<std::vector<double>> tab_vloc_;
    std::vector<double> zp_;
    double omega_ = 0.0;
    double v_of_0_ = 0.0;
    double gcut_ = -1.0;
    double dq_ = 0.01;
    int nqx_ = 0;

    Grid* grid_ptr_ = nullptr;
    std::unique_ptr<ComplexField> v_g_;
    std::unique_ptr<FFTSolver> fft_solver_;
    GPU_Vector<double> d_tab_;
    GPU_Vector<double> d_zp_;
};

}  // namespace dftcu
