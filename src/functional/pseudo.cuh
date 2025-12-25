#pragma once
#include <map>

#include "fft/fft_solver.cuh"
#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

class LocalPseudo {
  public:
    LocalPseudo(const Grid& grid, const Atoms& atoms) : grid_(grid), atoms_(atoms), solver_(grid) {}

    void set_vloc(int type, const std::vector<double>& vloc_g);
    void compute(RealField& vh);

  private:
    const Grid& grid_;
    const Atoms& atoms_;
    FFTSolver solver_;
    std::map<int, GPU_Vector<double>> vlines_;  // vloc(G) for each atom type
};

}  // namespace dftcu
