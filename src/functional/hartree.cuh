#pragma once
#include "fft/fft_solver.cuh"
#include "model/field.cuh"

namespace dftcu {

class Hartree {
  public:
    /**
     * @brief Constructs a Hartree functional.
     * @param grid Reference to the simulation grid.
     */
    Hartree(Grid& grid);

    /** @brief Default destructor. */
    ~Hartree() = default;

    /**
     * @brief Computes Hartree potential and energy.
     * @param rho Input real-space density field.
     * @param vh Output real-space Hartree potential field.
     * @param energy Output Hartree energy.
     */
    void compute(const RealField& rho, RealField& vh, double& energy);

    /**
     * @brief Unified interface for Evaluator.
     */
    double compute(const RealField& rho, RealField& v_out) {
        double energy = 0.0;
        compute(rho, v_out, energy);
        return energy;
    }

    const char* name() const { return "Hartree"; }

  private:
    Grid& grid_;         /**< Associated simulation grid reference */
    ComplexField rho_g_; /**< Persistent buffer for G-space density */
};

}  // namespace dftcu
