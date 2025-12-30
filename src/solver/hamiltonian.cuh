#pragma once
#include <memory>
#include <vector>

#include "functional/nonlocal_pseudo.cuh"
#include "model/wavefunction.cuh"
#include "solver/evaluator.cuh"

namespace dftcu {

/**
 * @brief Hamiltonian class implementing the H|psi> operation.
 *
 * This class coordinates the action of the kinetic energy operator, the
 * local potential operator, and the non-local pseudopotential operator on a
 * wavefunction.
 */
class Hamiltonian {
  public:
    /**
     * @brief Construct Hamiltonian
     * @param grid Reference to the simulation grid
     * @param evaluator Evaluator providing the local potential fields
     * @param nl_pseudo Optional non-local pseudopotential handler
     */
    Hamiltonian(Grid& grid, Evaluator& evaluator,
                std::shared_ptr<NonLocalPseudo> nl_pseudo = nullptr);
    ~Hamiltonian() = default;

    /**
     * @brief Apply the Hamiltonian to a wavefunction: H|psi> -> H_psi
     *
     * H|psi> = T|psi> + FFT(V_loc * IFFT(psi)) + V_nl|psi>
     *
     * @param psi Input wavefunction in reciprocal space
     * @param h_psi Output wavefunction (H*psi) in reciprocal space
     */
    void apply(Wavefunction& psi, Wavefunction& h_psi);

    /** @brief Set or update the non-local potential handler */
    void set_nonlocal(std::shared_ptr<NonLocalPseudo> nl_pseudo) { nonlocal_ = nl_pseudo; }

    /** @brief Get the total local potential used by the Hamiltonian */
    const RealField& v_loc() const { return v_loc_tot_; }

    /** @brief Update the local potential from the evaluator */
    void update_potentials(const RealField& rho);

    /** @brief Check if non-local pseudopotential is present */
    bool has_nonlocal() const { return nonlocal_ != nullptr; }

    /** @brief Get reference to evaluator */
    Evaluator& get_evaluator() { return evaluator_; }
    const Evaluator& get_evaluator() const { return evaluator_; }

    /** @brief Get reference to non-local pseudopotential */
    NonLocalPseudo& get_nonlocal() { return *nonlocal_; }
    const NonLocalPseudo& get_nonlocal() const { return *nonlocal_; }

  private:
    Grid& grid_;
    Evaluator& evaluator_;
    std::shared_ptr<NonLocalPseudo> nonlocal_;

    // Persistent buffer for the total local potential in real space
    RealField v_loc_tot_;
};

}  // namespace dftcu
