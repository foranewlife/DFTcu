#pragma once
#include <memory>
#include <vector>

#include "fft/fft_solver.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

/**
 * @brief Wavefunction class storing Kohn-Sham orbitals in reciprocal space.
 *
 * This class manages multiple electronic bands (NBANDS in VASP).
 * For the initial implementation, we store coefficients on the full FFT grid
 * for simplicity, using an internal mask to enforce ENCUT truncation.
 */
class Wavefunction {
  public:
    /**
     * @brief Construct Wavefunction
     * @param grid Reference to simulation grid
     * @param num_bands Number of bands
     * @param encut Energy cutoff in Hartree
     */
    Wavefunction(Grid& grid, int num_bands, double encut);
    ~Wavefunction() = default;

    // --- Data Access ---
    int num_bands() const { return num_bands_; }
    int num_pw() const { return num_pw_; }
    double encut() const { return encut_; }
    Grid& grid() const { return grid_; }

    gpufftComplex* data() { return data_.data(); }
    const gpufftComplex* data() const { return data_.data(); }

    /** @brief Get device pointer to a specific band's data */
    gpufftComplex* band_data(int n) { return data_.data() + n * grid_.nnr(); }

    // --- Operations ---
    /** @brief Initialize with random coefficients and normalize */
    void randomize(unsigned int seed = 42);

    /** @brief Project bands onto the valid plane-wave sphere (G^2/2 < ENCUT) */
    void apply_mask();

    /**
     * @brief Compute real-space charge density: rho(r) = sum_n f_n |psi_n(r)|^2
     * @param occupations Band occupation numbers (Fermi weights)
     * @param rho Output real-space density field
     */
    void compute_density(const std::vector<double>& occupations, RealField& rho);

    /** @brief Compute norm of each band on GPU */
    void compute_norms(std::vector<double>& norms);

  private:
    Grid& grid_;
    int num_bands_;
    double encut_;
    int num_pw_;

    // Data layout: [band_index][grid_index]
    GPU_Vector<gpufftComplex> data_;
    GPU_Vector<int> pw_mask_;

    void initialize_mask();
};

}  // namespace dftcu
