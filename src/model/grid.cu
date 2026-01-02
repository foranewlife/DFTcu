#include "grid.cuh"
#include "utilities/error.cuh"

namespace dftcu {

void __global__ compute_g_vectors_kernel(int nr0, int nr1, int nr2, double b00, double b01,
                                         double b02, double b10, double b11, double b12, double b20,
                                         double b21, double b22, double* gg, double* gx, double* gy,
                                         double* gz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nnr = nr0 * nr1 * nr2;
    if (i < nnr) {
        int n0 = i / (nr1 * nr2);
        int n1 = (i % (nr1 * nr2)) / nr2;
        int n2 = i % nr2;

        auto get_freq = [](int n, int nr) {
            if (n < (nr + 1) / 2)
                return (double)n;
            else
                return (double)(n - nr);
        };

        double f0 = get_freq(n0, nr0);
        double f1 = get_freq(n1, nr1);
        double f2 = get_freq(n2, nr2);

        double cur_gx = f0 * b00 + f1 * b10 + f2 * b20;
        double cur_gy = f0 * b01 + f1 * b11 + f2 * b21;
        double cur_gz = f0 * b02 + f1 * b12 + f2 * b22;

        gx[i] = cur_gx;
        gy[i] = cur_gy;
        gz[i] = cur_gz;
        gg[i] = cur_gx * cur_gx + cur_gy * cur_gy + cur_gz * cur_gz;
    }
}

void Grid::compute_reciprocal_lattice() {
    double a11 = lattice_[0][0], a12 = lattice_[0][1], a13 = lattice_[0][2];
    double a21 = lattice_[1][0], a22 = lattice_[1][1], a23 = lattice_[1][2];
    double a31 = lattice_[2][0], a32 = lattice_[2][1], a33 = lattice_[2][2];

    double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
                 a13 * (a21 * a32 - a22 * a31);

    volume_ = std::abs(det);

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

void Grid::compute_g_vectors() {
    const int block_size = 256;
    const int grid_size = (static_cast<int>(nnr_) + block_size - 1) / block_size;

    compute_g_vectors_kernel<<<grid_size, block_size, 0, stream_>>>(
        nr_[0], nr_[1], nr_[2], rec_lattice_[0][0], rec_lattice_[0][1], rec_lattice_[0][2],
        rec_lattice_[1][0], rec_lattice_[1][1], rec_lattice_[1][2], rec_lattice_[2][0],
        rec_lattice_[2][1], rec_lattice_[2][2], gg_.data(), gx_.data(), gy_.data(), gz_.data());

    GPU_CHECK_KERNEL;
}

}  // namespace dftcu
