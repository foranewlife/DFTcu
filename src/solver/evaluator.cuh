#pragma once
#include <vector>

#include "functional/functional.cuh"

namespace dftcu {

/**
 * @brief Aggregates different energy and potential terms using a composition-based design.
 *
 * The Evaluator acts as a container for multiple Functional objects. It computes
 * the total energy as the sum of all components and the total potential as the
 * functional derivative of the total energy with respect to the density.
 */
class Evaluator {
  public:
    /**
     * @brief Constructs an Evaluator.
     * @param grid Shared pointer to the simulation grid.
     */
    Evaluator(std::shared_ptr<Grid> grid);

    /** @brief Default destructor. */
    ~Evaluator() = default;

    /**
     * @brief Adds a functional component to the evaluator.
     * @param f A functional object (e.g., TF, vW, Hartree, LDA).
     */
    void add_functional(Functional f) { components_.push_back(std::move(f)); }

    /**
     * @brief Removes all functional components from the evaluator.
     */
    void clear() { components_.clear(); }

    /**
     * @brief Computes total energy and total potential.
     *
     * Iterates through all added functionals, summing their energy contributions
     * and adding their individual potentials to v_tot.
     *
     * @param rho Input real-space density field.
     * @param v_tot Output real-space total potential field (will be reset to zero first).
     * @return Total energy (Sum of all component energies).
     */
    double compute(const RealField& rho, RealField& v_tot);

  private:
    std::shared_ptr<Grid> grid_;         /**< Associated simulation grid */
    std::vector<Functional> components_; /**< List of functional components */
};

}  // namespace dftcu
