#pragma once
#include "fft/fft_solver.cuh"
#include "model/field.cuh"

namespace dftcu {

class Hartree {
  public:
    Hartree(const Grid& grid) : grid_(grid), solver_(grid) {}

    void compute(const RealField& rho, RealField& vh, double& energy);

  private:
    const Grid& grid_;
    FFTSolver solver_;
};

}  // namespace dftcu
