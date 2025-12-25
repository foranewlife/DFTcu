#pragma once
#include <map>

#include "fft/fft_solver.cuh"
#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

class LocalPseudo {
  public:
    LocalPseudo(const Grid& grid, const Atoms& atoms);
    ~LocalPseudo() = default;

    void set_vloc(int type, const std::vector<double>& vloc_g);
    void compute(RealField& v);

  private:
    const Grid& grid_;
    const Atoms& atoms_;
    FFTSolver solver_;
    GPU_Vector<double> vloc_types_;  // Contiguous vloc(G) for each atom type
    int num_types_;
};

}  // namespace dftcu
