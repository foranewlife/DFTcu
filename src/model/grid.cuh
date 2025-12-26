#pragma once
#include <cmath>
#include <vector>

#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

class Grid {
  public:
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

    size_t nnr() const { return nnr_; }
    double dv() const { return dv_; }
    double volume() const { return volume_; }
    const int* nr() const { return nr_; }
    const double (*lattice() const)[3] { return lattice_; }
    const double* gg() const { return gg_.data(); }
    const double* gx() const { return gx_.data(); }
    const double* gy() const { return gy_.data(); }
    const double* gz() const { return gz_.data(); }
    const double (*rec_lattice() const)[3] { return rec_lattice_; }

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
    void compute_reciprocal_lattice() {
        double det =
            (lattice_[0][0] * (lattice_[1][1] * lattice_[2][2] - lattice_[1][2] * lattice_[2][1]) -
             lattice_[0][1] * (lattice_[1][0] * lattice_[2][2] - lattice_[1][2] * lattice_[2][0]) +
             lattice_[0][2] * (lattice_[1][0] * lattice_[2][1] - lattice_[1][1] * lattice_[2][0]));

        double inv[3][3];
        inv[0][0] = (lattice_[1][1] * lattice_[2][2] - lattice_[1][2] * lattice_[2][1]) / det;
        inv[0][1] = (lattice_[0][2] * lattice_[2][1] - lattice_[0][1] * lattice_[2][2]) / det;
        inv[0][2] = (lattice_[0][1] * lattice_[1][2] - lattice_[0][2] * lattice_[1][1]) / det;
        inv[1][0] = (lattice_[1][2] * lattice_[2][0] - lattice_[1][0] * lattice_[2][2]) / det;
        inv[1][1] = (lattice_[0][0] * lattice_[2][2] - lattice_[0][2] * lattice_[2][0]) / det;
        inv[1][2] = (lattice_[1][0] * lattice_[0][2] - lattice_[0][0] * lattice_[1][2]) / det;
        inv[2][0] = (lattice_[1][0] * lattice_[2][1] - lattice_[1][1] * lattice_[2][0]) / det;
        inv[2][1] = (lattice_[2][0] * lattice_[0][1] - lattice_[0][0] * lattice_[2][1]) / det;
        inv[2][2] = (lattice_[0][0] * lattice_[1][1] - lattice_[1][0] * lattice_[0][1]) / det;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rec_lattice_[i][j] = 2.0 * PI * inv[j][i];
            }
        }
    }

    void compute_g_vectors();

    double lattice_[3][3];
    double rec_lattice_[3][3];
    int nr_[3];
    size_t nnr_;
    double volume_;
    double dv_;
    GPU_Vector<double> gg_, gx_, gy_, gz_;

    // Prevent copying
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(Grid&&) = delete;
};

}  // namespace dftcu
