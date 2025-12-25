#pragma once
#include "kedf_base.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "fft/fft_solver.cuh"

namespace dftcu {

/**
 * @brief Wang-Teter non-local kinetic energy density functional (NL part)
 *
 * Energy: E_NL = C_TF * ∫ ρ^α(r) K(r-r') ρ^β(r') dr dr'
 * Potential: V_NL = C_TF * [ α ρ^(α-1) (K * ρ^β) + β ρ^(β-1) (K * ρ^α) ]
 *
 * For Wang-Teter: α = β = 5/6
 */
class WangTeter : public KEDF_Base {
  public:
    WangTeter(double coeff = 1.0, double alpha = 5.0/6.0, double beta = 5.0/6.0);
    ~WangTeter() = default;

    double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "Wang-Teter (NL)"; }

  private:
    double coeff_;
    double alpha_;
    double beta_;
};

}  // namespace dftcu
