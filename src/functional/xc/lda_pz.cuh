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
    LDA_PZ() = default;
    ~LDA_PZ() = default;

    /**
     * @brief Compute LDA XC energy and potential
     * @param rho Input density field
     * @param v_xc Output potential field
     * @return Total XC energy
     */
    double compute(const RealField& rho, RealField& v_xc);

    const char* name() const { return "LDA-PZ"; }
};

}  // namespace dftcu
