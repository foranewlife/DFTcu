#pragma once
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Perdew-Zunger LDA exchange-correlation functional
 *
 * Perdew and Zunger, Phys. Rev. B 23, 5048 (1981)
 */
class LDA_PZ {
  public:
    struct Parameters {
        double rho_threshold = 1e-16;
        // Constants from DFTpy (semilocal_xc.py)
        double a = 0.0311;
        double b = -0.048;
        double c = 0.0020;
        double d = -0.0116;
        double gamma = -0.1423;
        double beta1 = 1.0529;
        double beta2 = 0.3334;
    };

    LDA_PZ() = default;
    ~LDA_PZ() = default;

    /**
     * @brief Compute LDA XC energy and potential (Perdew-Zunger)
     * @param rho Input density field
     * @param v_xc Output potential field
     * @return Total XC energy
     */
    double compute(const RealField& rho, RealField& v_xc);

    const char* name() const { return "LDA_PZ"; }

    void set_parameters(const Parameters& params) { params_ = params; }

  private:
    Parameters params_;
};

}  // namespace dftcu
