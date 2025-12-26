#pragma once
#include "fft/fft_solver.cuh"
#include "model/field.cuh"

namespace dftcu {

class Hartree {
  public:
    Hartree(std::shared_ptr<Grid> grid);
    ~Hartree() = default;

    /**
     * @brief Compute Hartree potential and energy.
     * @param rho Electron density
     * @param vh Output Hartree potential
     * @param energy Output Hartree energy
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

  private:
    std::shared_ptr<Grid> grid_;
};

}  // namespace dftcu
