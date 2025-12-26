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

    struct Parameters {
        double kappa = 0.1;
        double mu = 0.45;
        double kf_min_clamp = 1e-3;
        double kf_max_clamp = 100.0;
        int max_nsp = 128;
    };

    void set_parameters(const Parameters& params) { params_ = params; }

  private:
    std::shared_ptr<Grid> grid_;
    std::unique_ptr<FFTSolver> fft_;
    double alpha_;
    double beta_;

    Parameters params_;

    // Pre-computed kernel tables on device
    GPU_Vector<double> d_k_;
    GPU_Vector<double> d_k2_;
    GPU_Vector<double> d_d_;
    GPU_Vector<double> d_d2_;
};

}  // namespace dftcu
