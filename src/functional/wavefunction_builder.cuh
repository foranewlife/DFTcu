#pragma once
#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

/**
 * @brief Builder for initializing Kohn-Sham wavefunctions.
 *
 * Supports atomic wavefunction superposition matching Quantum ESPRESSO's starting_wfc='atomic'.
 */
class WavefunctionBuilder {
  public:
    WavefunctionBuilder(Grid& grid, std::shared_ptr<Atoms> atoms);
    ~WavefunctionBuilder() = default;

    /**
     * @brief Set atomic radial orbital for an atom type.
     * @param type Atom type index.
     * @param l Angular momentum.
     * @param r Radial grid.
     * @param chi Radial orbital (chi(r)).
     * @param rab Integration weights.
     * @param msh Last grid point with r < 10 Bohr (QE convention, 0 = use all points).
     */
    void add_atomic_orbital(int type, int l, const std::vector<double>& r,
                            const std::vector<double>& chi, const std::vector<double>& rab,
                            int msh = 0);

    /**
     * @brief Build wavefunctions from set orbitals (legacy interface).
     * @param psi Output Wavefunction object.
     * @param randomize_phase If true, add random phase to atomic orbitals (default: false).
     *
     * @deprecated Use build() instead, which returns a Wavefunction object.
     */
    void build_atomic_wavefunctions(Wavefunction& psi, bool randomize_phase = false);

    /**
     * @brief Build and return atomic wavefunctions (recommended interface).
     *
     * This method:
     * 1. Automatically calculates the number of bands (n_starting_wfc)
     * 2. Creates a Wavefunction object
     * 3. Fills it with atomic orbital superposition
     * 4. Returns the ready-to-use Wavefunction
     *
     * @param randomize_phase If true, add random phase to atomic orbitals (default: false).
     * @return unique_ptr to Wavefunction object with atomic orbitals (non-orthogonal)
     *
     * Example:
     *   WavefunctionBuilder factory(grid, atoms);
     *   factory.add_atomic_orbital(0, 0, r, chi_s, rab, msh);  // Si s orbital
     *   factory.add_atomic_orbital(0, 1, r, chi_p, rab, msh);  // Si p orbital
     *   auto psi = factory.build();  // Returns unique_ptr<Wavefunction>
     */
    std::unique_ptr<Wavefunction> build(bool randomize_phase = false);

    /**
     * @brief Get the number of bands that will be created.
     * @return Total number of atomic wavefunctions (n_starting_wfc)
     *
     * Useful for diagnostics before calling build().
     */
    int num_bands() const;

  private:
    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;

    struct OrbitalTable {
        int l;
        std::vector<double> chi_q;
    };

    // Mapping: [type_idx][orbital_idx] -> table
    std::vector<std::vector<OrbitalTable>> orbital_tables_;

    // Persistent GPU storage for build process
    GPU_Vector<double> d_tab_;

    static constexpr double dq_ = 0.01;

    // Internal helper methods
    int calculate_num_bands() const;
    void build_atomic_wavefunctions_internal(Wavefunction& psi, bool randomize_phase);
};

}  // namespace dftcu
