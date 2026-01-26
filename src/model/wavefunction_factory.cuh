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
class WavefunctionFactory {
  public:
    WavefunctionFactory(Grid& grid, std::shared_ptr<Atoms> atoms);
    ~WavefunctionFactory() = default;

    /**
     * @brief Set atomic radial orbital for an atom type.
     * @param type Atom type index.
     * @param l Angular momentum.
     * @param r Radial grid.
     * @param chi Radial orbital (chi(r)).
     * @param rab Integration weights.
     */
    void add_atomic_orbital(int type, int l, const std::vector<double>& r,
                            const std::vector<double>& chi, const std::vector<double>& rab);

    /**
     * @brief Build wavefunctions from set orbitals.
     * @param psi Output Wavefunction object.
     * @param randomize_phase If true, add random phase to atomic orbitals (default: false).
     */
    void build_atomic_wavefunctions(Wavefunction& psi, bool randomize_phase = false);

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
};

}  // namespace dftcu
