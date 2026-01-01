#pragma once
#include <complex>
#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

// Forward declaration
class FFTSolver;

/**
 * @brief Local pseudopotential implementation following Quantum ESPRESSO's interpolation scheme.
 *
 * This class implements the three-step V_loc calculation:
 * 1. init_tab_vloc: Generate 1D interpolation table on uniform q-grid (dq = 0.01 Bohr^-1)
 * 2. interp_vloc: Interpolate using 4-point cubic Lagrange for arbitrary G-vectors
 * 3. vloc_of_g: Subtract analytical erf long-range term
 *
 * References: QE upflib/vloc_mod.f90
 */
class LocalPseudo {
  public:
    /**
     * @brief Constructs a local pseudopotential functional.
     * @param grid Reference to the simulation grid.
     * @param atoms Shared pointer to the atom collection.
     */
    LocalPseudo(Grid& grid, std::shared_ptr<Atoms> atoms);

    /** @brief Default destructor. */
    ~LocalPseudo() = default;

    /**
     * @brief Initialize interpolation table from radial pseudopotential data.
     *
     * This function mimics QE's init_tab_vloc. It computes:
     *   V_short(q) = (4π/Ω) ∫ [r*V_loc(r) + Z*e²*erf(r)] * sin(qr)/q dr
     *
     * @param type Atom type index.
     * @param r_grid Radial grid points (Bohr).
     * @param vloc_r Local potential on radial grid (Hartree).
     * @param rab Radial grid integration weights dr/di.
     * @param zp Valence charge Z.
     * @param omega_angstrom Unit cell volume (Angstrom³) - will be converted internally to Bohr³.
     */
    void init_tab_vloc(int type, const std::vector<double>& r_grid,
                       const std::vector<double>& vloc_r, const std::vector<double>& rab, double zp,
                       double omega_angstrom);

    /**
     * @brief Set valence charge for an atom type.
     * @param type Atom type index.
     * @param zp Valence charge (number of valence electrons).
     */
    void set_valence_charge(int type, double zp);

    /**
     * @brief Compute the local pseudopotential in real space.
     *
     * This function:
     * 1. Interpolates V_short(|G|) using 4-point cubic Lagrange
     * 2. Subtracts analytical erf term: -4πZe²/(ΩG²) * exp(-G²/4)
     * 3. Multiplies by structure factor exp(-i G·R)
     * 4. Performs inverse FFT to get real-space potential
     *
     * @param vloc_r Output real space potential field.
     */
    void compute(RealField& vloc_r);

    /**
     * @brief Set G-vector cutoff for the local potential (in Rydberg).
     * @param gcut Cutoff energy in Rydberg. Set to -1 to use full grid.
     */
    void set_gcut(double gcut) { gcut_ = gcut; }

    /**
     * @brief Unified interface for Evaluator: compute energy and add to potential.
     */
    double compute(const RealField& rho, RealField& v_out);

    /**
     * @brief Get interpolation table for testing and comparison with QE.
     * @param type Atom type index.
     * @return Vector of tab_vloc values [V_short(q)] for q = iq * dq.
     */
    std::vector<double> get_tab_vloc(int type) const {
        if (type >= static_cast<int>(tab_vloc_.size())) {
            return std::vector<double>();
        }
        return tab_vloc_[type];
    }

    /**
     * @brief Get the G=0 term (alpha) for an atom type.
     * @param type Atom type index.
     * @return Alpha term in Hartree.
     */
    double get_alpha(int type) const {
        if (type >= static_cast<int>(tab_vloc_.size()) || tab_vloc_[type].empty()) {
            return 0.0;
        }
        return tab_vloc_[type][0];
    }

    /**
     * @brief Compute V_loc(G) for a list of G-shell moduli (Bohr^-1).
     * This is used for testing the interpolation and erf correction.
     * @param type Atom type index.
     * @param g_shells Moduli of G-vectors in Bohr^-1.
     * @return Vector of V_loc(G) in Hartree.
     */
    std::vector<double> get_vloc_g_shells(int type, const std::vector<double>& g_shells) const;

    /**
     * @brief Get interpolation parameters for testing.
     */
    double get_dq() const { return dq_; }
    int get_nqx() const { return nqx_; }
    double get_omega() const { return omega_; }

  private:
    void initialize_buffers(Grid& grid);

    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;

    // Interpolation table: tab_vloc_[type][iq] stores V_short(q) for q = iq * dq
    std::vector<std::vector<double>> tab_vloc_;

    // Valence charges for each atom type
    std::vector<double> zp_;

    // Unit cell volume in Bohr³
    double omega_ = 0.0;

    // G-vector cutoff
    double gcut_ = -1.0;

    // Interpolation parameters (matching QE)
    static constexpr double dq_ = 0.01;  // Bohr^-1
    int nqx_ = 0;                        // Size of interpolation table

    // Persistent buffers (lazy initialization like Hartree)
    Grid* grid_ptr_ = nullptr;
    std::unique_ptr<ComplexField> v_g_;
    std::unique_ptr<FFTSolver> fft_solver_;

    // Device-side caches for interpolation table and charges
    GPU_Vector<double> d_tab_;
    GPU_Vector<double> d_zp_;
};

}  // namespace dftcu
