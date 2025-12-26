#pragma once
#include "fft/fft_solver.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Perdew-Burke-Ernzerhof (PBE) GGA exchange-correlation functional
 *
 * Perdew, Burke, and Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
 */
class PBE {
  public:
    PBE(std::shared_ptr<Grid> grid);
    ~PBE() = default;

    /**
     * @brief Compute PBE XC energy and potential
     * @param rho Input density field
     * @param v_xc Output potential field
     * @return Total XC energy
     */
    double compute(const RealField& rho, RealField& v_xc);

    const char* name() const { return "PBE"; }

  private:
    std::shared_ptr<Grid> grid_;
    std::unique_ptr<FFTSolver> fft_;
};

}  // namespace dftcu
