#pragma once
#include "fft/fft_solver.cuh"
#include "kedf_base.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

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
    struct Parameters {
        double rho_threshold = 1e-30;
        double rho0_threshold = 1e-12;
    };

    WangTeter(Grid& grid, double coeff = 1.0, double alpha = 5.0 / 6.0, double beta = 5.0 / 6.0);
    virtual ~WangTeter() = default;

    /**
     * @brief Compute Wang-Teter energy and potential
     * @param rho Input density
     * @param v_kedf Output potential
     * @return Energy contribution
     */
    virtual double compute(const RealField& rho, RealField& v_kedf) override;

    const char* name() const override { return "WT"; }

    void set_parameters(const Parameters& params) { params_ = params; }

  private:
    Grid& grid_;
    double coeff_;
    double alpha_;
    double beta_;
    Parameters params_;

    // Persistent buffers
    RealField rho_beta_;
    ComplexField rho_beta_g_;
    RealField v_conv_;
};

}  // namespace dftcu
