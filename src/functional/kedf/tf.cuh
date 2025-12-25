#pragma once
#include "kedf_base.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Thomas-Fermi kinetic energy density functional
 *
 * Energy density: τ_TF[ρ] = C_TF * ρ^(5/3)
 * Potential: V_TF = δE/δρ = (5/3) * C_TF * ρ^(2/3)
 *
 * where C_TF = (3/10) * (3π²)^(2/3) ≈ 2.871234
 */
class ThomasFermi : public KEDF_Base {
  public:
    /**
     * @brief Constructor
     * @param coeff Scaling coefficient (default 1.0)
     */
    ThomasFermi(double coeff = 1.0);

    /**
     * @brief Compute TF kinetic energy and potential
     * @param rho Input density field
     * @param v_kedf Output potential field
     * @return Total kinetic energy
     */
    double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "Thomas-Fermi"; }

  private:
    double coeff_;     // Overall scaling coefficient
    double c_tf_;      // Thomas-Fermi constant: (3/10)*(3π²)^(2/3)
    double c_tf_pot_;  // Potential prefactor: (5/3)*C_TF
};

}  // namespace dftcu
