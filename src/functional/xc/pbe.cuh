#pragma once
#include "fft/fft_solver.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Perdew-Burke-Ernzerhof (PBE) GGA exchange-correlation functional
 *
 * Perdew, Burke, and Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
 */
class PBE {
  public:
    struct Parameters {
        double kappa = 0.804;
        double mu_x = 0.2195149727645171;
        double rho_threshold = 1e-16;
        double sigma_threshold = 1e-20;

        // Correlation constants
        double a = 0.0310907;
        double alpha1 = 0.21370;
        double beta1 = 7.5957;
        double beta2 = 3.5876;
        double beta3 = 1.6382;
        double beta4 = 0.49294;
        double pbe_beta = 0.06672455060314922;
        double pbe_gamma = 0.031090690869654894;
    };

    PBE(std::shared_ptr<Grid> grid);
    ~PBE() = default;

    /**
     * @brief Compute PBE XC energy and potential
     * @param rho Input density field
     * @param v_xc Output potential field
     * @return Total XC energy
     */
    double compute(const RealField& rho, RealField& v_xc);

    const char* name() const { return "PBE"; }

    void set_parameters(const Parameters& params) { params_ = params; }

  private:
    std::shared_ptr<Grid> grid_;
    std::unique_ptr<FFTSolver> fft_;
    Parameters params_;
};

}  // namespace dftcu
