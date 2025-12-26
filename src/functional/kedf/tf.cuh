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
    struct Parameters {
        double rho_threshold = 1e-30;
    };

    ThomasFermi(double coeff = 1.0);
    virtual ~ThomasFermi() = default;

    /**
     * @brief Compute TF energy and potential
     * @param rho Input density
     * @param v_kedf Output potential
     * @return Energy contribution
     */
    virtual double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "TF"; }

    void set_parameters(const Parameters& params) { params_ = params; }

  private:
    double coeff_;
    double c_tf_;
    double c_tf_pot_;
    Parameters params_;
};

}  // namespace dftcu
