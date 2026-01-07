#include <fstream>
#include <sstream>

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

// ============================================================================
// G-vector management implementation (Phase 0c.4)
// ============================================================================

void Grid::load_gvectors_from_qe(const std::string& data_dir) {
    // 1. Load nl_d and nlm_d from QE output
    std::string nl_file = data_dir + "/dftcu_debug_nl_mapping.txt";
    load_nl_mapping_from_file(nl_file);

    // 2. Reverse-engineer Miller indices from nl_d
    reverse_engineer_miller_indices();

    // 3. Compute g2kin on GPU
    compute_g2kin_gpu();
}

void Grid::load_nl_mapping_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read header lines (skip lines starting with '#')
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '#') {
            break;  // Found first data line or empty line
        }
    }

    // Read data: ig, nl_d, nlm_d
    // NOTE: QE uses 1-based indexing (Fortran), need to convert to 0-based (C++)
    std::vector<int> nl_host, nlm_host;
    int ig, nl, nlm;

    // Process the first line we already read (if it's not empty)
    if (!line.empty() && line[0] != '#') {
        std::istringstream iss(line);
        if (iss >> ig >> nl >> nlm) {
            nl_host.push_back(nl - 1);  // Convert to 0-based
            nlm_host.push_back(nlm - 1);
        }
    }

    // Read remaining lines
    while (file >> ig >> nl >> nlm) {
        nl_host.push_back(nl - 1);  // Convert to 0-based
        nlm_host.push_back(nlm - 1);
    }

    ngm_ = nl_host.size();

    if (ngm_ == 0) {
        throw std::runtime_error("load_nl_mapping_from_file: No data loaded from " + filename);
    }

    // Allocate GPU memory
    nl_d_.resize(ngm_);
    nlm_d_.resize(ngm_);

    // Copy to GPU
    CHECK(cudaMemcpy(nl_d_.data(), nl_host.data(), ngm_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nlm_d_.data(), nlm_host.data(), ngm_ * sizeof(int), cudaMemcpyHostToDevice));

    file.close();
}

void Grid::reverse_engineer_miller_indices() {
    // Allocate host memory for Miller indices
    std::vector<int> h_host(ngm_);
    std::vector<int> k_host(ngm_);
    std::vector<int> l_host(ngm_);

    // Copy nl_d from GPU to host for processing
    std::vector<int> nl_host(ngm_);
    CHECK(cudaMemcpy(nl_host.data(), nl_d_.data(), ngm_ * sizeof(int), cudaMemcpyDeviceToHost));

    // Reverse-engineer Miller indices from nl_d
    for (int ig = 0; ig < ngm_; ++ig) {
        int nl = nl_host[ig];

        // nl → (n1, n2, n3) (C-style, 0-based)
        int n3 = nl / (nr_[0] * nr_[1]);
        int remainder = nl % (nr_[0] * nr_[1]);
        int n2 = remainder / nr_[0];
        int n1 = remainder % nr_[0];

        // (n1, n2, n3) → (h, k, l) (periodic boundary)
        // CRITICAL: Handle negative indices explicitly (matches Python implementation)
        h_host[ig] = (n1 < nr_[0] / 2) ? n1 : n1 - nr_[0];
        k_host[ig] = (n2 < nr_[1] / 2) ? n2 : n2 - nr_[1];
        l_host[ig] = (n3 < nr_[2] / 2) ? n3 : n3 - nr_[2];
    }

    // Allocate GPU memory
    miller_h_.resize(ngm_);
    miller_k_.resize(ngm_);
    miller_l_.resize(ngm_);

    // Copy to GPU
    CHECK(cudaMemcpy(miller_h_.data(), h_host.data(), ngm_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(miller_k_.data(), k_host.data(), ngm_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(miller_l_.data(), l_host.data(), ngm_ * sizeof(int), cudaMemcpyHostToDevice));
}

__global__ void compute_g2kin_kernel(
    const int* miller_h, const int* miller_k, const int* miller_l,
    const double* bg,  // reciprocal lattice vectors (9 elements, row-major)
    double tpiba2, double* g2kin, int ngm) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= ngm)
        return;

    int h = miller_h[ig];
    int k = miller_k[ig];
    int l = miller_l[ig];

    // G = h*b1 + k*b2 + l*b3
    // bg is stored row-major: [b1x, b1y, b1z, b2x, b2y, b2z, b3x, b3y, b3z]
    double gx = h * bg[0] + k * bg[3] + l * bg[6];
    double gy = h * bg[1] + k * bg[4] + l * bg[7];
    double gz = h * bg[2] + k * bg[5] + l * bg[8];

    double g2 = gx * gx + gy * gy + gz * gz;
    g2kin[ig] = g2 * tpiba2;
}

void Grid::compute_g2kin_gpu() {
    // Check if we have any G-vectors
    if (ngm_ == 0) {
        throw std::runtime_error("compute_g2kin_gpu: ngm_ is 0. No G-vectors loaded.");
    }

    // Allocate g2kin
    g2kin_.resize(ngm_);

    // Note: rec_lattice_ is in units of 2π/Angstrom
    // For g2kin, we want |G|² in Rydberg units
    // G = h*b1 + k*b2 + l*b3, where b vectors are in 2π/Angstrom
    // |G|² is already in (2π/Angstrom)² = (2π)² / Angstrom²
    // So tpiba2 = 1 (no additional scaling needed)
    double tpiba2 = 1.0;

    // Prepare reciprocal lattice on GPU (9 elements, row-major)
    GPU_Vector<double> bg_gpu(9);
    double bg_host[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            bg_host[i * 3 + j] = rec_lattice_[i][j];
        }
    }
    CHECK(cudaMemcpy(bg_gpu.data(), bg_host, 9 * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256;
    int grid_size = (ngm_ + block_size - 1) / block_size;

    compute_g2kin_kernel<<<grid_size, block_size, 0, stream_>>>(miller_h_.data(), miller_k_.data(),
                                                                miller_l_.data(), bg_gpu.data(),
                                                                tpiba2, g2kin_.data(), ngm_);

    GPU_CHECK_KERNEL;
}

}  // namespace dftcu
