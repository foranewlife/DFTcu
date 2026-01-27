#pragma once
#include <complex>
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

    /** @brief Get the host vector of valid plane-wave indices */
    std::vector<int> get_pw_indices();

    /** @brief Copy coefficients from host memory (NumPy, std::vector, etc.) */
    void copy_from_host(const std::complex<double>* data);

    /** @brief Copy coefficients to host memory */
    void copy_to_host(std::complex<double>* data) const;

    /** @brief Get coefficients of a specific band on the full grid */
    std::vector<std::complex<double>> get_coefficients(int band) const;

    /** @brief Copy data from another Wavefunction (device to device) */
    void copy_from(const Wavefunction& other);

    /** @brief Set coefficients of a specific band on the full grid */
    void set_coefficients(const std::vector<std::complex<double>>& coeffs, int band);

    /**
     * @brief Set coefficients using Miller indices (QE style).
     * @param h Vector of Miller h indices
     * @param k Vector of Miller k indices
     * @param l Vector of Miller l indices
     * @param values Flat vector of complex coefficients (num_bands * num_miller)
     * @param expand_hermitian If true, applies C(-G) = C*(G) for G != 0
     */
    void set_coefficients_miller(const std::vector<int>& h, const std::vector<int>& k,
                                 const std::vector<int>& l,
                                 const std::vector<std::complex<double>>& values,
                                 bool expand_hermitian = true);

    // --- Operations ---
    /** @brief Initialize with random coefficients and normalize */
    void randomize(unsigned int seed = 42);

    /** @brief Project bands onto the valid plane-wave sphere (G^2/2 < ENCUT)
     * [SIDE_EFFECT] Modifies wavefunction data in-place
     */
    void apply_mask_inplace();

    /** @brief Orthonormalize bands using Gram-Schmidt process on GPU
     * [SIDE_EFFECT] Modifies wavefunction data in-place
     */
    void orthonormalize_inplace();

    /**
     * @brief Compute real-space charge density: rho(r) = sum_n f_n |psi_n(r)|^2
     * @param occupations Band occupation numbers (Fermi weights)
     * @param rho Output real-space density field
     */
    void compute_density(const std::vector<double>& occupations, RealField& rho);

    /** @brief Compute norm of each band on GPU */
    void compute_norms(std::vector<double>& norms);

    /** @brief Compute Hermitian inner product between two bands: <band_a | band_b> */
    std::complex<double> dot(int band_a, int band_b);

    /** @brief Compute the total kinetic energy: Ts = sum_n f_n <psi_n | -0.5 nabla^2 | psi_n> */
    double compute_kinetic_energy(const std::vector<double>& occupations);

    /** @brief Get |G|^2 values in Rydberg for all active plane waves */
    std::vector<double> get_g2kin();

    /**
     * @brief Enforce Gamma-point constraint: Im[ψ(G=0)] = 0
     * [SIDE_EFFECT] Modifies wavefunction data in-place
     *
     * For Gamma-only calculations, wavefunctions must satisfy ψ(-G) = ψ*(G).
     * This implies that ψ(G=0) must be purely real.
     * QE enforces this after every wavefunction update (regterg.f90:172, 375).
     *
     * This function sets Im[ψ(0,n)] = 0 for all bands n.
     */
    void enforce_gamma_constraint_inplace();

    /**
     * @brief Check if G=0 is included in this process's G-vectors
     * @return true if G=0 is present (gstart==2 in QE terminology)
     */
    bool has_g0() const;

    /**
     * @brief Compute band occupations using Fermi-Dirac distribution
     * @param eigenvalues Computed band energies
     * @param nelectrons Target total number of electrons
     * @param sigma Smearing width (Hartree)
     * @param occupations Output occupation numbers
     * @param fermi_energy Output computed Fermi energy
     */
    static void compute_occupations(const std::vector<double>& eigenvalues, double nelectrons,
                                    double sigma, std::vector<double>& occupations,
                                    double& fermi_energy);

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
