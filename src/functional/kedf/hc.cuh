#pragma once
#include <memory>
#include <vector>

#include "fft/fft_solver.cuh"
#include "functional/kedf/kedf_base.cuh"

namespace dftcu {

/**
 * @brief Revised Huang-Carter (revHC) non-local kinetic energy density functional
 *
 * This is a density-dependent non-local KEDF where the effective Fermi momentum
 * is modified by a GGA factor.
 */
class revHC : public KEDF_Base {
  public:
    revHC(std::shared_ptr<Grid> grid, double alpha = 2.0, double beta = 2.0 / 3.0);
    virtual ~revHC() = default;

    /**
     * @brief Compute revHC energy and potential
     * @param rho Input density
     * @param v_kedf Output potential
     * @return Energy contribution
     */
    virtual double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "revHC"; }

  private:
    std::shared_ptr<Grid> grid_;
    std::unique_ptr<FFTSolver> fft_;
    double alpha_;
    double beta_;

    // Parameters for the GGA factor F(s) - PBE2 variant
    double kappa_ = 1.245;
    double mu_ = 0.23889;

    // Discretization for multi-kernel interpolation
    int nsp_ = 20;
};

}  // namespace dftcu
