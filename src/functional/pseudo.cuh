#pragma once
#include <complex>
#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

class FFTSolver;

/**
 * @brief Local pseudopotential implementation following Quantum ESPRESSO's interpolation scheme.
 */
class LocalPseudo {
  public:
    LocalPseudo(Grid& grid, std::shared_ptr<Atoms> atoms);
    ~LocalPseudo() = default;

    void init_tab_vloc(int type, const std::vector<double>& r_grid,
                       const std::vector<double>& vloc_r, const std::vector<double>& rab, double zp,
                       double omega_angstrom, int mesh_cutoff = -1);

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

    std::vector<double> get_vloc_g_shells(int type, const std::vector<double>& g_shells) const;

    void set_dq(double dq) { dq_ = dq; }
    double get_dq() const { return dq_; }
    int get_nqx() const { return nqx_; }
    double get_omega() const { return omega_; }

  private:
    void initialize_buffers(Grid& grid);

    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;
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
