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
    struct Parameters {
        double rho_threshold = 1e-30;
        double phi_threshold = 1e-15;
    };

    vonWeizsacker(double coeff = 1.0);
    virtual ~vonWeizsacker() = default;

    /**
     * @brief Compute von Weizsäcker energy and potential
     * @param rho Input density
     * @param v_kedf Output potential
     * @return Energy contribution
     */
    virtual double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "vW"; }

    void set_parameters(const Parameters& params) { params_ = params; }

  private:
    double coeff_;
    Parameters params_;
};

}  // namespace dftcu
