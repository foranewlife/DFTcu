#pragma once
#include <array>
#include <cmath>
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
     * @param lattice 9-element vector representing the 3x3 lattice matrix (row-major).
     * @param nr 3-element vector representing the number of grid points in each dimension.
     */
    Grid(const std::vector<double>& lattice, const std::vector<int>& nr) {
        CHECK(cudaStreamCreate(&stream_));
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                lattice_[i][j] = lattice[i * 3 + j];
            }
            nr_[i] = nr[i];
        }
        nnr_ = nr_[0] * nr_[1] * nr_[2];

        compute_reciprocal_lattice();

        gg_.resize(nnr_);
        gx_.resize(nnr_);
        gy_.resize(nnr_);
        gz_.resize(nnr_);
        compute_g_vectors();
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

    double volume() const { return volume_; }
    /** @brief Unit cell volume in Bohr³ */
    double volume_bohr() const {
        const double B = constants::BOHR_TO_ANGSTROM;
        return volume_ / (B * B * B);
    }
    /** @brief Differential volume element dv = Volume / N (in Angstrom³) */
    double dv() const { return volume_ / (double)nnr_; }
    /** @brief Differential volume element in Bohr³ */
    double dv_bohr() const {
        const double B = constants::BOHR_TO_ANGSTROM;
        return dv() / (B * B * B);
    }
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
    void set_is_gamma(bool is_gamma) { is_gamma_ = is_gamma; }

    // ========================================================================
    // G-vector management (Phase 0c.4)
    // ========================================================================

    /**
     * @brief Load G-vector data from QE output files (QE-compatible mode).
     * @param data_dir Directory containing QE exported data files.
     *
     * This function loads nl_d/nlm_d mappings from QE, reverse-engineers
     * Miller indices, and computes g2kin values.
     */
    void load_gvectors_from_qe(const std::string& data_dir);

    /**
     * @brief Number of G-vectors within ecutwfc cutoff.
     */
    int ngm() const { return ngm_; }

    /**
     * @brief Kinetic energy coefficients (g2kin) for all G-vectors (GPU).
     */
    const double* g2kin() const { return g2kin_.data(); }

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
     * @brief Load nl_d and nlm_d from QE output file.
     */
    void load_nl_mapping_from_file(const std::string& filename);

    /**
     * @brief Reverse-engineer Miller indices from nl_d.
     */
    void reverse_engineer_miller_indices();

    /**
     * @brief Compute g2kin values on GPU.
     */
    void compute_g2kin_gpu();

    cudaStream_t stream_ = nullptr;        /**< CUDA stream for grid operations */
    double lattice_[3][3];                 /**< Real-space lattice matrix */
    double rec_lattice_[3][3];             /**< Reciprocal-space lattice matrix */
    int nr_[3];                            /**< Grid dimensions */
    size_t nnr_;                           /**< Total number of points */
    double volume_;                        /**< Unit cell volume */
    bool is_gamma_ = false;                /**< True if Gamma-point only calculation */
    GPU_Vector<double> gg_, gx_, gy_, gz_; /**< GPU data for G-vectors */

    // ========================================================================
    // G-vector management data (Phase 0c.4)
    // ========================================================================
    int ngm_ = 0;              /**< Number of G-vectors */
    GPU_Vector<int> miller_h_; /**< Miller index h (GPU) */
    GPU_Vector<int> miller_k_; /**< Miller index k (GPU) */
    GPU_Vector<int> miller_l_; /**< Miller index l (GPU) */
    GPU_Vector<double> g2kin_; /**< Kinetic energy coefficients (GPU) */
    GPU_Vector<int> nl_d_;     /**< G-vector → FFT grid mapping (GPU) */
    GPU_Vector<int> nlm_d_;    /**< -G vector → FFT grid mapping (GPU) */

    // Prevent copying
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(Grid&&) = delete;
};

}  // namespace dftcu
