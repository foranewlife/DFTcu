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
     * [SIDE_EFFECT] Modifies internal orbital_tables_ state
     *
     * @param type Atom type index.
     * @param l Angular momentum.
     * @param r Radial grid.
     * @param chi Radial orbital (chi(r)).
     * @param rab Integration weights.
     *
     * @note msh (r < 10 Bohr cutoff) is automatically computed following QE convention.
     */
    void add_atomic_orbital(int type, int l, const std::vector<double>& r,
                            const std::vector<double>& chi, const std::vector<double>& rab);

    /**
     * @brief Build wavefunctions from set orbitals (legacy interface).
     * [SIDE_EFFECT] Modifies psi object in-place
     *
     * @param psi Output Wavefunction object.
     * @param randomize_phase If true, add random phase to atomic orbitals (default: false).
     *
     * @deprecated Use build() instead, which returns a Wavefunction object.
     */
    void build_atomic_wavefunctions(Wavefunction& psi, bool randomize_phase = false);

    /**
     * @brief Build and return atomic wavefunctions (recommended interface).
     * [BUILDER] Heavy construction: 1D radial → 3D wavefunction with FFT/integration
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
     *   factory.add_atomic_orbital(0, 0, r, chi_s, rab);  // Si s orbital
     *   factory.add_atomic_orbital(0, 1, r, chi_p, rab);  // Si p orbital
     *   auto psi = factory.build();  // Returns unique_ptr<Wavefunction>
     */
    std::unique_ptr<Wavefunction> build(bool randomize_phase = false);

    /**
     * @brief Get the number of bands that will be created.
     * [CONST] Does not modify object state
     *
     * @return Total number of atomic wavefunctions (n_starting_wfc)
     *
     * Useful for diagnostics before calling build().
     */
    int num_bands() const;

    /**
     * @brief Get chi_q table for diagnostics (testing only).
     * [CONST] Does not modify object state
     *
     * @param type Atom type index.
     * @param orbital_idx Orbital index for this type.
     * @return Reference to chi_q vector.
     *
     * @note This is a diagnostic interface for testing Bessel transform accuracy.
     */
    const std::vector<double>& get_chi_q(int type, int orbital_idx) const;

  private:
    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;

    struct OrbitalTable {
        int l;
        std::vector<double> chi_q;
        // 三次样条插值系数（预计算）
        std::vector<double> spline_M;  // 二阶导数 M[i]
        std::vector<double> spline_h;  // 网格间距 h[i]
    };

    // Mapping: [type_idx][orbital_idx] -> table
    std::vector<std::vector<OrbitalTable>> orbital_tables_;

    // Persistent GPU storage for build process
    GPU_Vector<double> d_tab_;       // chi_q 数据
    GPU_Vector<double> d_spline_M_;  // 三次样条二阶导数
    GPU_Vector<double> d_spline_h_;  // 三次样条网格间距

    static constexpr double dq_ = 0.01;

    // Internal helper methods
    int calculate_num_bands() const;
    void build_atomic_wavefunctions_internal(Wavefunction& psi, bool randomize_phase);
};

}  // namespace dftcu
