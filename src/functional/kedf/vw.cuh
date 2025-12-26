#pragma once
#include "fft/fft_solver.cuh"
#include "kedf_base.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief von Weizsacker kinetic energy density functional
 *
 * Energy: T_vW[ρ] = ∫ |∇ρ|² / (8ρ) dτ = ∫ |∇√ρ|² / 2 dτ
 * Potential: V_vW = -1/2 * ∇²√ρ / √ρ
 */
class vonWeizsacker : public KEDF_Base {
  public:
    vonWeizsacker(double coeff = 1.0);
    ~vonWeizsacker() = default;

    double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "von Weizsacker"; }

  private:
    double coeff_;
};

}  // namespace dftcu
