#pragma once
#include <memory>
#include <vector>

#include "model/wavefunction.cuh"
#include "solver/evaluator.cuh"

namespace dftcu {

/**
 * @brief Hamiltonian class implementing the H|psi> operation.
 *
 * This class coordinates the action of the kinetic energy operator and the
 * local potential operator on a wavefunction. Non-local contributions will be
 * added in a future step.
 */
class Hamiltonian {
  public:
    /**
     * @brief Construct Hamiltonian
     * @param grid Reference to the simulation grid
     * @param evaluator Evaluator providing the local potential fields
     */
    Hamiltonian(Grid& grid, Evaluator& evaluator);
    ~Hamiltonian() = default;

    /**
     * @brief Apply the Hamiltonian to a wavefunction: H|psi> -> H_psi
     *
     * In reciprocal space: H|psi> = T|psi> + FFT(V_loc * IFFT(psi))
     *
     * @param psi Input wavefunction in reciprocal space
     * @param h_psi Output wavefunction (H*psi) in reciprocal space
     */
    void apply(Wavefunction& psi, Wavefunction& h_psi);

    /** @brief Get the total local potential used by the Hamiltonian */
    const RealField& v_loc() const { return v_loc_tot_; }

    /** @brief Update the local potential from the evaluator */
    void update_potentials(const RealField& rho);

  private:
    Grid& grid_;
    Evaluator& evaluator_;

    // Persistent buffer for the total local potential in real space
    RealField v_loc_tot_;
};

}  // namespace dftcu
