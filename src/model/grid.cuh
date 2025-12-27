#pragma once
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
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                lattice_[i][j] = lattice[i * 3 + j];
            }
            nr_[i] = nr[i];
        }
        nnr_ = nr_[0] * nr_[1] * nr_[2];

        // Calculate volume
        volume_ = std::abs(
            lattice_[0][0] * (lattice_[1][1] * lattice_[2][2] - lattice_[1][2] * lattice_[2][1]) -
            lattice_[0][1] * (lattice_[1][0] * lattice_[2][2] - lattice_[1][2] * lattice_[2][0]) +
            lattice_[0][2] * (lattice_[1][0] * lattice_[2][1] - lattice_[1][1] * lattice_[2][0]));
        dv_ = volume_ / nnr_;

        // Reciprocal lattice (physics convention: 2*pi * inv(lattice)^T)
        compute_reciprocal_lattice();

        // Allocate and compute g-vectors on GPU
        gg_.resize(nnr_);
        gx_.resize(nnr_);
        gy_.resize(nnr_);
        gz_.resize(nnr_);
        compute_g_vectors();
    }

    /**
     * @brief Total number of grid points (nr[0] * nr[1] * nr[2]).
     */
    size_t nnr() const { return nnr_; }

    /**
     * @brief Differential volume element (volume / nnr).
     */
    double dv() const { return dv_; }

    /**
     * @brief Total volume of the simulation cell.
     */
    double volume() const { return volume_; }

    /**
     * @brief Array containing the number of grid points in each dimension.
     */
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

  private:
    /**
     * @brief Computes the reciprocal lattice matrix from the real-space lattice.
     */
    void compute_reciprocal_lattice() {
        double a11 = lattice_[0][0], a12 = lattice_[0][1], a13 = lattice_[0][2];
        double a21 = lattice_[1][0], a22 = lattice_[1][1], a23 = lattice_[1][2];
        double a31 = lattice_[2][0], a32 = lattice_[2][1], a33 = lattice_[2][2];

        double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
                     a13 * (a21 * a32 - a22 * a31);

        volume_ = std::abs(det);
        dv_ = volume_ / nnr_;

        double inv[3][3];
        inv[0][0] = (a22 * a33 - a23 * a32) / det;
        inv[0][1] = (a13 * a32 - a12 * a33) / det;
        inv[0][2] = (a12 * a23 - a13 * a22) / det;
        inv[1][0] = (a23 * a31 - a21 * a33) / det;
        inv[1][1] = (a11 * a33 - a13 * a31) / det;
        inv[1][2] = (a13 * a21 - a11 * a23) / det;
        inv[2][0] = (a21 * a32 - a22 * a31) / det;
        inv[2][1] = (a12 * a31 - a11 * a32) / det;
        inv[2][2] = (a11 * a22 - a12 * a21) / det;

        // Physics convention: G = 2*pi * inv(L)^T
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rec_lattice_[i][j] = 2.0 * constants::D_PI * inv[j][i];
            }
        }
    }

    /**
     * @brief Computes all G-vectors and stores them on the GPU.
     */
    void compute_g_vectors();

    double lattice_[3][3];                 /**< Real-space lattice matrix */
    double rec_lattice_[3][3];             /**< Reciprocal-space lattice matrix */
    int nr_[3];                            /**< Grid dimensions */
    size_t nnr_;                           /**< Total number of points */
    double volume_;                        /**< Unit cell volume */
    double dv_;                            /**< Volume element */
    GPU_Vector<double> gg_, gx_, gy_, gz_; /**< GPU data for G-vectors */

    // Prevent copying
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(Grid&&) = delete;
};

}  // namespace dftcu
