#pragma once
#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Local pseudopotential implementation using reciprocal space summation.
 */
class LocalPseudo {
  public:
    LocalPseudo(std::shared_ptr<Grid> grid, std::shared_ptr<Atoms> atoms);
    ~LocalPseudo() = default;

    /**
     * @brief Set local pseudopotential for an atom type.
     * @param type Atom type index.
     * @param vloc_g Local pseudopotential in reciprocal space (aligned with grid).
     */
    void set_vloc(int type, const std::vector<double>& vloc_g);

    /**
     * @brief Set local pseudopotential for an atom type using radial data.
     * @param type Atom type index.
     * @param q Radial grid points.
     * @param v_q Pseudopotential values on radial grid.
     */
    void set_vloc_radial(int type, const std::vector<double>& q, const std::vector<double>& v_q);

    /**
     * @brief Compute the local pseudopotential on the real space grid.
     * @param vloc_r Output real space potential field.
     */
    void compute(RealField& vloc_r);

    /**
     * @brief Unified interface for Evaluator.
     */
    double compute(const RealField& rho, RealField& v_out);

  private:
    std::shared_ptr<Grid> grid_;
    std::shared_ptr<Atoms> atoms_;
    GPU_Vector<double> vloc_types_;
    int num_types_ = 0;
};

}  // namespace dftcu
