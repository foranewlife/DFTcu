#pragma once
#include <vector>

#include "functional/functional.cuh"

namespace dftcu {

/**
 * @brief Computes potentials V[ρ] = δE[ρ]/δρ from density functionals.
 *
 * DensityFunctionalPotential aggregates multiple Functional objects and computes
 * the total potential as the functional derivative of the total energy with
 * respect to the density. It is used for both KS-DFT (Hartree + XC) and OFDFT
 * (Hartree + XC + kinetic functionals).
 *
 * Usage:
 *   - KS-DFT SCF: V[ρ] = V_Hartree[ρ] + V_XC[ρ]
 *   - OFDFT: V[ρ] = V_Hartree[ρ] + V_XC[ρ] + δT_TF[ρ]/δρ + δT_vW[ρ]/δρ
 */
class DensityFunctionalPotential {
  public:
    /**
     * @brief Constructs a DensityFunctionalPotential.
     * @param grid Reference to the simulation grid.
     */
    DensityFunctionalPotential(Grid& grid);

    /** @brief Default destructor. */
    ~DensityFunctionalPotential() = default;

    /**
     * @brief Adds a functional component.
     * @param f A functional object (e.g., TF, vW, Hartree, LDA).
     */
    void add_functional(Functional f) { components_.push_back(std::move(f)); }

    /**
     * @brief Removes all functional components.
     */
    void clear() { components_.clear(); }

    /**
     * @brief Computes total energy and total potential from density.
     *
     * Iterates through all added functionals, summing their energy contributions
     * and adding their individual potentials to v_tot.
     *
     * @param rho Input real-space density field.
     * @param v_tot Output real-space total potential field (will be reset to zero first).
     * @return Total energy (Sum of all component energies).
     */
    double compute(const RealField& rho, RealField& v_tot);

    /** @brief Aggregates G=0 potential components from all functionals. */
    double get_v0() const;

    /** @brief Get access to functional components */
    std::vector<Functional>& get_components() { return components_; }
    const std::vector<Functional>& get_components() const { return components_; }

  private:
    Grid& grid_;                         /**< Associated simulation grid reference */
    std::vector<Functional> components_; /**< List of functional components */
    RealField v_tmp_;                    /**< Persistent buffer for intermediate potentials */
};

}  // namespace dftcu
