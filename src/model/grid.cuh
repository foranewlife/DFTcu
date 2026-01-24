#pragma once
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

/**
 * @brief Represents a 3D simulation grid and its reciprocal space properties.
 *
 * The Grid class manages the real-space lattice, the reciprocal lattice,
 * and pre-computed G-vectors (wave vectors) on the GPU. It handles the
 * mapping between real space and reciprocal space for FFT-based operations.
 */
class Grid {
  public:
    /**
     * @brief Constructs a simulation grid.
     *
     * UNIT CONVENTION (Hartree Atomic Units - Internal Only):
     *   - All inputs must be in atomic units (Bohr, Hartree)
     *   - For user-facing creation, use factory functions in grid_factory.cuh
     *
     * @param lattice_bohr 9-element lattice matrix in BOHR (row-major)
     * @param nr 3-element vector: FFT grid dimensions [nr1, nr2, nr3]
     * @param ecutwfc_ha Wavefunction cutoff energy in HARTREE
     * @param ecutrho_ha Density cutoff energy in HARTREE (default: 4*ecutwfc_ha)
     * @param is_gamma True for Gamma-only calculation
     *
     * @note This constructor is for internal use. Use factory functions:
     *       - create_grid_from_qe() for QE units (Angstrom + Rydberg)
     *       - create_grid_from_atomic_units() for atomic units
     */
    Grid(const std::vector<double>& lattice_bohr, const std::vector<int>& nr, double ecutwfc_ha,
         double ecutrho_ha = -1.0, bool is_gamma = false)
        : is_gamma_(is_gamma) {
        CHECK(cudaStreamCreate(&stream_));

        // Store lattice directly (already in Bohr)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                lattice_[i][j] = lattice_bohr[i * 3 + j];
            }
            nr_[i] = nr[i];
        }
        nnr_ = nr_[0] * nr_[1] * nr_[2];

        // Set cutoff energies (already in Hartree)
        ecutwfc_ = ecutwfc_ha;
        ecutrho_ = (ecutrho_ha > 0) ? ecutrho_ha : 4.0 * ecutwfc_ha;

        compute_reciprocal_lattice();

        gg_.resize(nnr_);
        gx_.resize(nnr_);
        gy_.resize(nnr_);
        gz_.resize(nnr_);
        compute_g_vectors();

        // Generate Smooth and Dense grids
        generate_gvectors();
    }

    ~Grid() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    cudaStream_t stream() const { return stream_; }
    void synchronize() const { CHECK(cudaStreamSynchronize(stream_)); }

    /**
     * @brief Total number of grid points (nr[0] * nr[1] * nr[2]).
     */
    size_t nnr() const { return nnr_; }

    /** @brief Unit cell volume in Bohr³ (Hartree atomic units) */
    double volume() const { return volume_; }
    double volume_bohr() const { return volume_; }

    /** @brief Differential volume element in Bohr³ (Hartree atomic units) */
    double dv() const { return volume_ / (double)nnr_; }
    double dv_bohr() const { return volume_ / (double)nnr_; }
    const int* nr() const { return nr_; }

    /**
     * @brief Pointer to the 3x3 real-space lattice matrix.
     */
    const double (*lattice() const)[3] { return lattice_; }

    /**
     * @brief Pointer to the GPU-resident squared magnitude of G-vectors (|G|^2).
     */
    const double* gg() const { return gg_.data(); }

    /**
     * @brief Pointer to the GPU-resident x-component of G-vectors (Gx).
     */
    const double* gx() const { return gx_.data(); }

    /**
     * @brief Pointer to the GPU-resident y-component of G-vectors (Gy).
     */
    const double* gy() const { return gy_.data(); }

    /**
     * @brief Pointer to the GPU-resident z-component of G-vectors (Gz).
     */
    const double* gz() const { return gz_.data(); }

    /**
     * @brief Pointer to the 3x3 reciprocal-space lattice matrix.
     */
    const double (*rec_lattice() const)[3] { return rec_lattice_; }

    /**
     * @brief Computes the maximum squared magnitude of G-vectors in the grid.
     * @return Maximum |G|^2 value.
     */
    double g2max() const {
        double g2max_val = 0;
        std::vector<double> h_gg(nnr_);
        gg_.copy_to_host(h_gg.data());
        for (double g : h_gg)
            if (g > g2max_val)
                g2max_val = g;
        return g2max_val;
    }

    bool is_gamma() const { return is_gamma_; }

    /**
     * @brief Get gstart index for Gamma-only calculations.
     * @return 2 if G=0 exists (Gamma-only), 1 otherwise.
     * @note QE uses Fortran 1-based indexing, so gstart=2 means G=0 is at index 1.
     *       In DFTcu (C++, 0-based), G=0 is at index 0, but we return 2 to match QE convention.
     */
    int gstart() const { return is_gamma_ ? 2 : 1; }

    // ========================================================================
    // Cutoff energies (双网格设计) - Read-only accessors
    // ========================================================================

    /**
     * @brief Get wavefunction cutoff energy in Hartree.
     * @note Set at construction time via factory functions.
     */
    double ecutwfc() const { return ecutwfc_; }

    /**
     * @brief Get density cutoff energy in Hartree.
     * @note Set at construction time via factory functions.
     */
    double ecutrho() const { return ecutrho_; }

    // ========================================================================
    // G-vector management (Phase 0c.4)
    // ========================================================================

    /**
     * @brief Generate G-vectors natively based on cutoff energy.
     * @param ecutwfc Wavefunction cutoff energy in Hartree.
     * @param is_gamma True for Gamma-only optimization (only half-sphere).
     *
     * This function generates Miller indices (h,k,l) by:
     * 1. Computing reciprocal lattice vectors b1, b2, b3
     * 2. Iterating over Miller indices in a cube: h,k,l ∈ [-hmax, hmax]
     * 3. Filtering by |G|² < 2×ecutwfc (in Hartree units, T = ½|k|²)
     * 4. For Gamma-only: keeping only half-sphere (h>0 or h=0,k>0 or h=k=0,l>=0)
     * 5. Computing g2kin = ½|G|² for each G-vector
     *
     * This is a "side-effect-free" function that only depends on lattice and ecutwfc.
     */
    void generate_gvectors();

    /**
     * @brief Load G-vector data from QE output files (TEST ONLY).
     * @param data_dir Directory containing QE exported data files.
     *
     * This is a TEST FUNCTION for comparing DFTcu's native generation with QE.
     * It loads Miller indices directly from QE's exported file.
     * NOTE: This loads Smooth grid (ecutwfc) G-vectors.
     */
    void load_gvectors_from_qe(const std::string& data_dir);

    /**
     * @brief Load Miller indices (h, k, l) from file (TEST ONLY).
     */
    void load_miller_indices_from_file(const std::string& filename);

    /**
     * @brief Load nl_d and nlm_d mapping from file (TEST ONLY).
     *
     * This loads FFT grid mapping indices for Smooth grid G-vectors.
     * QE exports nl_d[ig] (G → FFT) and nlm_d[ig] (-G → FFT) arrays.
     */
    void load_nl_mapping_from_file(const std::string& filename);

    // ========================================================================
    // Smooth Grid (ecutwfc) - for wavefunctions and beta projectors
    // ========================================================================

    /**
     * @brief Number of G-vectors in Smooth grid (ecutwfc cutoff).
     * @note Alias for ngw() for consistency with QE terminology.
     */
    int ngm() const { return ngw_; }

    /**
     * @brief Number of G-vectors in Smooth grid (ecutwfc cutoff).
     */
    int ngw() const { return ngw_; }

    /**
     * @brief Kinetic energy coefficients (g2kin) for Smooth grid G-vectors (GPU).
     */
    const double* g2kin() const { return g2kin_.data(); }

    /**
     * @brief Squared G-vector magnitudes for Smooth grid (GPU).
     * @note Physical units: (2π/Bohr)² - used for kinetic energy T = ½|G|²
     */
    const double* gg_smooth() const { return gg_wfc_.data(); }

    /**
     * @brief Get Smooth grid G² values as host vector.
     */
    std::vector<double> get_gg_smooth() const {
        std::vector<double> h_gg(ngw_);
        if (ngw_ > 0)
            gg_wfc_.copy_to_host(h_gg.data());
        return h_gg;
    }

    // ========================================================================
    // Dense Grid (ecutrho) - for density and local potential
    // ========================================================================

    /**
     * @brief Number of G-vectors in Dense grid (ecutrho cutoff).
     * @note Currently returns 0, will be implemented in Phase 0c.
     */
    int ngm_dense() const { return ngm_dense_; }

    /**
     * @brief Number of G-shells in Dense grid.
     * @note Currently returns 0, will be implemented in Phase 0c.
     */
    int ngl() const { return ngl_; }

    /**
     * @brief Squared G-vector magnitudes for Dense grid (GPU).
     * @note Crystallographic units: 1/Bohr² - used by Hartree, LocalPseudo
     */
    const double* gg_dense() const { return gg_dense_.data(); }

    /**
     * @brief Get Dense grid G² values as host vector.
     */
    std::vector<double> get_gg_dense() const {
        std::vector<double> h_gg(ngm_dense_);
        if (ngm_dense_ > 0)
            gg_dense_.copy_to_host(h_gg.data());
        return h_gg;
    }

    /**
     * @brief Squared G-shell magnitudes (GPU).
     * @note gl[igl] = |G|² for shell igl.
     * @note Crystallographic units: 1/Bohr² - same as gg_dense
     */
    const double* gl_shells() const { return gl_.data(); }

    /**
     * @brief Get G-shell values as host vector.
     */
    std::vector<double> get_gl_shells() const {
        std::vector<double> h_gl(ngl_);
        if (ngl_ > 0)
            gl_.copy_to_host(h_gl.data());
        return h_gl;
    }

    /**
     * @brief G-vector to G-shell mapping (GPU).
     * @note igtongl[ig] gives the shell index for G-vector ig.
     * @note Currently empty, will be implemented in Phase 0c.
     */
    const int* igtongl() const { return igtongl_.data(); }

    /**
     * @brief Get igtongl mapping as host vector.
     */
    std::vector<int> get_igtongl() const {
        std::vector<int> h_igtongl(ngm_dense_);
        if (ngm_dense_ > 0)
            igtongl_.copy_to_host(h_igtongl.data());
        return h_igtongl;
    }

    /**
     * @brief Dense grid G-vector components (GPU).
     * @note Units: 1/Bohr (crystallographic, NO 2π factor)
     */
    const double* gx_dense() const { return gx_dense_.data(); }
    const double* gy_dense() const { return gy_dense_.data(); }
    const double* gz_dense() const { return gz_dense_.data(); }

    /**
     * @brief Dense grid FFT mapping (GPU).
     */
    const int* nl_dense() const { return nl_dense_.data(); }
    const int* nlm_dense() const { return nlm_dense_.data(); }

    /**
     * @brief Dense grid Miller indices (GPU).
     * @note Available for debugging and verification purposes.
     */
    const int* miller_h_dense() const { return miller_h_dense_.data(); }
    const int* miller_k_dense() const { return miller_k_dense_.data(); }
    const int* miller_l_dense() const { return miller_l_dense_.data(); }

    /**
     * @brief Get Dense grid Miller indices as host vectors.
     */
    std::vector<int> miller_h_dense_host() const {
        std::vector<int> h(ngm_dense_);
        if (ngm_dense_ > 0)
            miller_h_dense_.copy_to_host(h.data());
        return h;
    }
    std::vector<int> miller_k_dense_host() const {
        std::vector<int> k(ngm_dense_);
        if (ngm_dense_ > 0)
            miller_k_dense_.copy_to_host(k.data());
        return k;
    }
    std::vector<int> miller_l_dense_host() const {
        std::vector<int> l(ngm_dense_);
        if (ngm_dense_ > 0)
            miller_l_dense_.copy_to_host(l.data());
        return l;
    }

    // ========================================================================
    // Smooth to Dense grid mapping
    // ========================================================================

    /**
     * @brief Smooth grid G-vector to Dense grid index mapping (GPU).
     * @note igk[ig_smooth] gives the Dense grid index for Smooth grid G-vector ig_smooth.
     * @note Currently empty, will be implemented in Phase 0c.
     */
    const int* igk() const { return igk_.data(); }

    /**
     * @brief Get igk mapping as host vector.
     * @note Returns empty vector until Phase 0c generates Dense grid.
     */
    std::vector<int> get_igk() const {
        std::vector<int> h_igk(igk_.size());
        if (igk_.size() > 0)
            igk_.copy_to_host(h_igk.data());
        return h_igk;
    }

    /**
     * @brief Get Miller indices as host vectors (for testing).
     */
    std::vector<int> get_miller_h() const {
        std::vector<int> h_miller(ngw_);
        if (ngw_ > 0)
            miller_h_.copy_to_host(h_miller.data());
        return h_miller;
    }

    std::vector<int> get_miller_k() const {
        std::vector<int> k_miller(ngw_);
        if (ngw_ > 0)
            miller_k_.copy_to_host(k_miller.data());
        return k_miller;
    }

    std::vector<int> get_miller_l() const {
        std::vector<int> l_miller(ngw_);
        if (ngw_ > 0)
            miller_l_.copy_to_host(l_miller.data());
        return l_miller;
    }

    /**
     * @brief Get nl_d mapping as host vector (for testing).
     */
    std::vector<int> get_nl_d() const {
        std::vector<int> nl(ngw_);
        if (ngw_ > 0)
            nl_d_.copy_to_host(nl.data());
        return nl;
    }

    std::vector<int> get_nlm_d() const {
        std::vector<int> nlm(ngw_);
        if (ngw_ > 0)
            nlm_d_.copy_to_host(nlm.data());
        return nlm;
    }

    // ========================================================================
    // Legacy interfaces (backward compatibility)
    // ========================================================================

    /**
     * @brief FFT grid indices for G-vectors (nl_d) (GPU).
     */
    const int* nl_d() const { return nl_d_.data(); }

    /**
     * @brief FFT grid indices for -G vectors (nlm_d) (GPU).
     */
    const int* nlm_d() const { return nlm_d_.data(); }

    /**
     * @brief Miller indices (h, k, l) for all G-vectors (GPU).
     */
    const int* miller_h() const { return miller_h_.data(); }
    const int* miller_k() const { return miller_k_.data(); }
    const int* miller_l() const { return miller_l_.data(); }

  private:
    /**
     * @brief Computes the reciprocal lattice matrix from the real-space lattice.
     */
    void compute_reciprocal_lattice();

    /**
     * @brief Computes all G-vectors and stores them on the GPU.
     */
    void compute_g_vectors();

    // ========================================================================
    // G-vector management private methods (Phase 0c.4)
    // ========================================================================

    /**
     * @brief Reverse-engineer Miller indices from nl_d.
     */
    void reverse_engineer_miller_indices();

    /**
     * @brief Compute g2kin values on GPU.
     */
    void compute_g2kin_gpu();

    /**
     * @brief Generate G-shell grouping for Dense grid.
     * @param g2_dense Squared G-vector magnitudes for Dense grid.
     *
     * Groups Dense grid G-vectors into shells by |G|² value:
     * - ngl_: number of unique G-shells
     * - gl_: |G|² for each shell (sorted)
     * - igtongl_: mapping from Dense G-vector index to shell index
     */
    void generate_gshell_grouping(const std::vector<double>& g2_dense);

    /**
     * @brief Generate igk mapping from Smooth to Dense grid.
     * @param h_smooth, k_smooth, l_smooth Miller indices for Smooth grid
     * @param h_dense, k_dense, l_dense Miller indices for Dense grid
     *
     * Builds igk_[ig_smooth] = ig_dense mapping by matching Miller indices.
     */
    void generate_igk_mapping(const std::vector<int>& h_smooth, const std::vector<int>& k_smooth,
                              const std::vector<int>& l_smooth, const std::vector<int>& h_dense,
                              const std::vector<int>& k_dense, const std::vector<int>& l_dense);

    cudaStream_t stream_ = nullptr;        /**< CUDA stream for grid operations */
    double lattice_[3][3];                 /**< Real-space lattice matrix */
    double rec_lattice_[3][3];             /**< Reciprocal-space lattice matrix */
    int nr_[3];                            /**< Grid dimensions */
    size_t nnr_;                           /**< Total number of points */
    double volume_;                        /**< Unit cell volume */
    bool is_gamma_ = false;                /**< True if Gamma-point only calculation */
    GPU_Vector<double> gg_, gx_, gy_, gz_; /**< GPU data for全部 G-vectors (FFT grid) */

    // ========================================================================
    // Cutoff energies
    // ========================================================================
    double ecutwfc_ = 0.0; /**< Wavefunction cutoff energy (Ry) */
    double ecutrho_ = 0.0; /**< Density cutoff energy (Ry) */

    // ========================================================================
    // Smooth Grid (ecutwfc) - for wavefunctions and beta projectors
    // ========================================================================
    int ngw_ = 0;               /**< Number of Smooth grid G-vectors */
    GPU_Vector<double> gg_wfc_; /**< |G|² for Smooth grid (GPU, Physical: (2π/Bohr)²) */
    GPU_Vector<int> miller_h_;  /**< Miller index h (GPU) */
    GPU_Vector<int> miller_k_;  /**< Miller index k (GPU) */
    GPU_Vector<int> miller_l_;  /**< Miller index l (GPU) */
    GPU_Vector<double> g2kin_;  /**< Kinetic energy coefficients (GPU, Hartree) */
    GPU_Vector<int> nl_d_;      /**< Smooth G → FFT grid mapping (GPU) */
    GPU_Vector<int> nlm_d_;     /**< Smooth -G → FFT grid mapping (GPU) */

    // ========================================================================
    // Dense Grid (ecutrho) - for density and local potential
    // ========================================================================
    int ngm_dense_ = 0;           /**< Number of Dense grid G-vectors */
    int ngl_ = 0;                 /**< Number of G-shells */
    GPU_Vector<double> gg_dense_; /**< |G|² for Dense grid (GPU, Crystallographic: 1/Bohr²) */
    GPU_Vector<double> gl_;       /**< |G|² for each G-shell (GPU, Crystallographic: 1/Bohr²) */
    GPU_Vector<int> igtongl_;     /**< Dense G → G-shell mapping (GPU) */
    GPU_Vector<int> nl_dense_;    /**< Dense G → FFT grid mapping (GPU) */
    GPU_Vector<int> nlm_dense_;   /**< Dense -G → FFT grid mapping (GPU) */
    GPU_Vector<double> gx_dense_; /**< Dense grid Gx components (GPU, 1/Bohr) */
    GPU_Vector<double> gy_dense_; /**< Dense grid Gy components (GPU, 1/Bohr) */
    GPU_Vector<double> gz_dense_; /**< Dense grid Gz components (GPU, 1/Bohr) */

    // Dense grid Miller indices (for debugging and verification)
    GPU_Vector<int> miller_h_dense_; /**< Dense grid Miller index h (GPU) */
    GPU_Vector<int> miller_k_dense_; /**< Dense grid Miller index k (GPU) */
    GPU_Vector<int> miller_l_dense_; /**< Dense grid Miller index l (GPU) */

    // ========================================================================
    // Smooth to Dense grid mapping
    // ========================================================================
    GPU_Vector<int> igk_; /**< Smooth G → Dense G mapping (GPU) */

    // Prevent copying
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(Grid&&) = delete;
};

}  // namespace dftcu
