#pragma once
#include <vector>

#include "functional/functional.cuh"

namespace dftcu {

/**
 * @brief Aggregates different energy and potential terms using a composition-based design.
 */
class Evaluator {
  public:
    Evaluator(std::shared_ptr<Grid> grid);
    ~Evaluator() = default;

    /**
     * @brief Add a functional component to the evaluator.
     * @param f A functional object (TF, vW, Hartree, etc.)
     */
    void add_functional(Functional f) { components_.push_back(std::move(f)); }

    /**
     * @brief Clear all functional components.
     */
    void clear() { components_.clear(); }

    /**
     * @brief Compute total energy and potential
     * @param rho Input density
     * @param v_tot Output total potential field
     * @return Total energy
     */
    double compute(const RealField& rho, RealField& v_tot);

  private:
    std::shared_ptr<Grid> grid_;
    std::vector<Functional> components_;
};

}  // namespace dftcu
