#pragma once
#include <memory>
#include <vector>

#include "functional/density_functional_potential.cuh"
#include "functional/nonlocal_pseudo.cuh"
#include "model/wavefunction.cuh"

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
     * @brief Construct Hamiltonian (base constructor)
     * @param grid Reference to the simulation grid
     */
    explicit Hamiltonian(Grid& grid);

    /**
     * @brief Construct Hamiltonian (deprecated, for backward compatibility)
     * @param grid Reference to the simulation grid
     * @param dfp DensityFunctionalPotential providing the local potential fields
     * @param nl_pseudo Optional non-local pseudopotential handler
     */
    Hamiltonian(Grid& grid, std::shared_ptr<DensityFunctionalPotential> dfp,
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

    /**
     * @brief Apply only the kinetic energy operator: T|psi> -> h_psi
     *
     * T|psi> = ½|G|² |psi(G)>
     *
     * @param psi Input wavefunction
     * @param h_psi Output (will be overwritten with T|psi>)
     */
    void apply_kinetic(Wavefunction& psi, Wavefunction& h_psi);

    /**
     * @brief Apply only the local potential operator: V_loc|psi> -> result
     *
     * V_loc|psi> = FFT(V_loc(r) * IFFT(psi(G)))
     *
     * @param psi Input wavefunction
     * @param h_psi Output (will be ADDED to, not overwritten)
     */
    void apply_local(Wavefunction& psi, Wavefunction& h_psi);

    /**
     * @brief Apply only the nonlocal potential operator: V_NL|psi> -> result
     *
     * V_NL|psi> = Σ_ij D_ij |β_i⟩⟨β_j|psi⟩
     *
     * @param psi Input wavefunction
     * @param h_psi Output (will be ADDED to, not overwritten)
     */
    void apply_nonlocal(Wavefunction& psi, Wavefunction& h_psi);

    /** @brief Set or update the non-local potential handler */
    void set_nonlocal(std::shared_ptr<NonLocalPseudo> nl_pseudo) { nonlocal_ = nl_pseudo; }

    /** @brief Set or update the density functional potential handler */
    void set_density_functional_potential(std::shared_ptr<DensityFunctionalPotential> dfp) {
        dfp_ = dfp;
    }

    /** @brief Get the total local potential used by the Hamiltonian */
    RealField& v_loc() { return v_loc_tot_; }
    const RealField& v_loc() const { return v_loc_tot_; }

    /** @brief Get individual potential components (only available after update_potentials) */
    RealField& v_ps() { return v_ps_; }
    const RealField& v_ps() const { return v_ps_; }
    RealField& v_h() { return v_h_; }
    const RealField& v_h() const { return v_h_; }
    RealField& v_xc() { return v_xc_; }
    const RealField& v_xc() const { return v_xc_; }

    /** @brief Get the aggregate G=0 potential in Hartree. Matches QE v_of_0 * 0.5. */
    double get_v_of_0() const { return v_of_0_; }

    /** @brief Set the aggregate G=0 potential in Hartree. */
    void set_v_of_0(double v0) { v_of_0_ = v0; }

    /** @brief Update the local potential from the density functional potential */
    void update_potentials(const RealField& rho);

    /**
     * @brief Set the G-vector cutoff for all local potential components (matching QE's ecutrho).
     * @param ecutrho Energy cutoff in Ry (internally uses Bohr^-2).
     */
    void set_ecutrho(double ecutrho);

    /** @brief Check if non-local pseudopotential is present */
    bool has_nonlocal() const { return nonlocal_ != nullptr; }

    /** @brief Get reference to density functional potential */
    DensityFunctionalPotential& get_density_functional_potential() { return *dfp_; }
    const DensityFunctionalPotential& get_density_functional_potential() const { return *dfp_; }

    /** @brief Get reference to non-local pseudopotential */
    NonLocalPseudo& get_nonlocal() { return *nonlocal_; }
    const NonLocalPseudo& get_nonlocal() const { return *nonlocal_; }

  private:
    Grid& grid_;
    std::shared_ptr<DensityFunctionalPotential> dfp_;
    std::shared_ptr<NonLocalPseudo> nonlocal_;

    // Persistent buffers for potentials in real space
    RealField v_loc_tot_;  // Total V_loc = V_ps + V_H + V_xc
    RealField v_ps_;       // Pseudopotential local component
    RealField v_h_;        // Hartree potential
    RealField v_xc_;       // Exchange-correlation potential
    double v_of_0_ = 0.0;
};

}  // namespace dftcu
