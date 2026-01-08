#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <tuple>

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
    // lattice_ is in Bohr (atomic units)
    double a11 = lattice_[0][0], a12 = lattice_[0][1], a13 = lattice_[0][2];
    double a21 = lattice_[1][0], a22 = lattice_[1][1], a23 = lattice_[1][2];
    double a31 = lattice_[2][0], a32 = lattice_[2][1], a33 = lattice_[2][2];

    double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
                 a13 * (a21 * a32 - a22 * a31);

    volume_ = std::abs(det);  // Bohr³

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

    // Reciprocal lattice: rec_lattice_ = 2π * inv(lattice)^T
    // Units: 2π/Bohr (atomic units)
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
// G-vector management implementation (Phase 0c)
// ============================================================================

void Grid::generate_gvectors() {
    // This function generates G-vectors natively based on:
    // - Reciprocal lattice vectors (already computed in constructor)
    // - Cutoff energies ecutwfc_ and ecutrho_ (must be set before calling this)
    // - Gamma-only flag is_gamma_
    //
    // UNIT CONVENTION (Hartree Atomic Units):
    //   ecutwfc_, ecutrho_ are stored in Hartree
    //   rec_lattice_ is in 2π/Bohr
    //   G-vectors are in 2π/Bohr
    //   |G|² is in (2π/Bohr)²
    //
    // Kinetic energy in Hartree atomic units:
    //   T = ½|k|² [Ha] when k is in 2π/Bohr
    //
    // Therefore, the cutoff condition is:
    //   ½|G|² ≤ ecutwfc_ha
    //   |G|² ≤ 2 × ecutwfc_ha
    //
    // This is equivalent to QE's Rydberg convention:
    //   |G|² ≤ ecutwfc_ry (since 1 Ry = 0.5 Ha)
    //
    // This function generates BOTH Smooth and Dense grids:
    // - Smooth grid: based on ecutwfc (for wavefunctions)
    // - Dense grid: based on ecutrho (for density/potentials)

    if (ecutwfc_ <= 0.0) {
        throw std::runtime_error("generate_gvectors: ecutwfc not set.");
    }
    if (ecutrho_ <= 0.0) {
        throw std::runtime_error("generate_gvectors: ecutrho not set.");
    }

    // Use the larger cutoff (Dense grid) for iteration
    double gcut2_smooth = 2.0 * ecutwfc_;  // Smooth grid cutoff
    double gcut2_dense = 2.0 * ecutrho_;   // Dense grid cutoff
    double gcut2_max = std::max(gcut2_smooth, gcut2_dense);

    // Estimate hmax for iteration (using Dense grid cutoff)
    double b_min = 1e10;
    for (int i = 0; i < 3; ++i) {
        double b_len = std::sqrt(rec_lattice_[i][0] * rec_lattice_[i][0] +
                                 rec_lattice_[i][1] * rec_lattice_[i][1] +
                                 rec_lattice_[i][2] * rec_lattice_[i][2]);
        if (b_len < b_min)
            b_min = b_len;
    }
    int hmax = static_cast<int>(std::sqrt(gcut2_max) / b_min) + 2;  // +2 for safety

    // Generate G-vectors for BOTH grids in a single pass
    std::vector<int> h_smooth, k_smooth, l_smooth;
    std::vector<double> g2_smooth;

    std::vector<int> h_dense, k_dense, l_dense;
    std::vector<double> g2_dense;

    for (int h = -hmax; h <= hmax; ++h) {
        for (int k = -hmax; k <= hmax; ++k) {
            for (int l = -hmax; l <= hmax; ++l) {
                // Compute |G|² = |h·b1 + k·b2 + l·b3|²
                double gx =
                    h * rec_lattice_[0][0] + k * rec_lattice_[1][0] + l * rec_lattice_[2][0];
                double gy =
                    h * rec_lattice_[0][1] + k * rec_lattice_[1][1] + l * rec_lattice_[2][1];
                double gz =
                    h * rec_lattice_[0][2] + k * rec_lattice_[1][2] + l * rec_lattice_[2][2];
                double g2 = gx * gx + gy * gy + gz * gz;

                // Gamma-only: keep only half-sphere using h-priority convention
                // This matches QE's convention: h>0, or (h=0 and k>0), or (h=k=0 and l>=0)
                if (is_gamma_) {
                    if (h < 0)
                        continue;
                    if (h == 0 && k < 0)
                        continue;
                    if (h == 0 && k == 0 && l < 0)
                        continue;
                }

                // Check if G-vector belongs to Smooth grid
                if (g2 <= gcut2_smooth) {
                    h_smooth.push_back(h);
                    k_smooth.push_back(k);
                    l_smooth.push_back(l);
                    g2_smooth.push_back(g2);
                }

                // Check if G-vector belongs to Dense grid
                if (g2 <= gcut2_dense) {
                    h_dense.push_back(h);
                    k_dense.push_back(k);
                    l_dense.push_back(l);
                    g2_dense.push_back(g2);
                }
            }
        }
    }

    ngw_ = h_smooth.size();
    ngm_dense_ = h_dense.size();

    if (ngw_ == 0) {
        throw std::runtime_error("generate_gvectors: No Smooth grid G-vectors generated.");
    }
    if (ngm_dense_ == 0) {
        throw std::runtime_error("generate_gvectors: No Dense grid G-vectors generated.");
    }

    // ========================================================================
    // Smooth Grid Allocation and Copy
    // ========================================================================
    miller_h_.resize(ngw_);
    miller_k_.resize(ngw_);
    miller_l_.resize(ngw_);
    g2kin_.resize(ngw_);
    gg_wfc_.resize(ngw_);

    CHECK(
        cudaMemcpy(miller_h_.data(), h_smooth.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(
        cudaMemcpy(miller_k_.data(), k_smooth.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(
        cudaMemcpy(miller_l_.data(), l_smooth.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));

    // g2kin in Hartree: T = ½|G|²
    std::vector<double> g2kin_ha(ngw_);
    for (int i = 0; i < ngw_; ++i) {
        g2kin_ha[i] = 0.5 * g2_smooth[i];
    }
    CHECK(
        cudaMemcpy(g2kin_.data(), g2kin_ha.data(), ngw_ * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gg_wfc_.data(), g2_smooth.data(), ngw_ * sizeof(double),
                     cudaMemcpyHostToDevice));

    // ========================================================================
    // Dense Grid Allocation and Copy
    // ========================================================================
    gg_dense_.resize(ngm_dense_);
    CHECK(cudaMemcpy(gg_dense_.data(), g2_dense.data(), ngm_dense_ * sizeof(double),
                     cudaMemcpyHostToDevice));

    // ========================================================================
    // Generate G-shell grouping for Dense grid
    // ========================================================================
    generate_gshell_grouping(g2_dense);

    // ========================================================================
    // Generate igk mapping (Smooth -> Dense)
    // ========================================================================
    generate_igk_mapping(h_smooth, k_smooth, l_smooth, h_dense, k_dense, l_dense);
}

void Grid::load_gvectors_from_qe(const std::string& data_dir) {
    // TEST FUNCTION: Load Miller indices directly from QE export
    std::string miller_file = data_dir + "/dftcu_debug_miller.txt";

    std::ifstream file(miller_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + miller_file);
    }

    // Skip header
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '#')
            break;
    }

    // Read Miller indices
    std::vector<int> h_list, k_list, l_list;
    int ig, h, k, l;

    // Process first line if not header
    if (!line.empty() && line[0] != '#') {
        std::istringstream iss(line);
        if (iss >> ig >> h >> k >> l) {
            h_list.push_back(h);
            k_list.push_back(k);
            l_list.push_back(l);
        }
    }

    // Read remaining lines
    while (file >> ig >> h >> k >> l) {
        h_list.push_back(h);
        k_list.push_back(k);
        l_list.push_back(l);
    }

    ngw_ = h_list.size();

    if (ngw_ == 0) {
        throw std::runtime_error("load_gvectors_from_qe: No Miller indices loaded from " +
                                 miller_file);
    }

    // Allocate GPU memory
    miller_h_.resize(ngw_);
    miller_k_.resize(ngw_);
    miller_l_.resize(ngw_);

    // Copy to GPU
    CHECK(cudaMemcpy(miller_h_.data(), h_list.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(miller_k_.data(), k_list.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(miller_l_.data(), l_list.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));

    file.close();

    // Compute g2kin from Miller indices
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

    ngw_ = nl_host.size();

    if (ngw_ == 0) {
        throw std::runtime_error("load_nl_mapping_from_file: No data loaded from " + filename);
    }

    // Allocate GPU memory
    nl_d_.resize(ngw_);
    nlm_d_.resize(ngw_);

    // Copy to GPU
    CHECK(cudaMemcpy(nl_d_.data(), nl_host.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nlm_d_.data(), nlm_host.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));

    file.close();
}

void Grid::reverse_engineer_miller_indices() {
    // Allocate host memory for Miller indices
    std::vector<int> h_host(ngw_);
    std::vector<int> k_host(ngw_);
    std::vector<int> l_host(ngw_);

    // Copy nl_d from GPU to host for processing
    std::vector<int> nl_host(ngw_);
    CHECK(cudaMemcpy(nl_host.data(), nl_d_.data(), ngw_ * sizeof(int), cudaMemcpyDeviceToHost));

    // Reverse-engineer Miller indices from nl_d
    for (int ig = 0; ig < ngw_; ++ig) {
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
    miller_h_.resize(ngw_);
    miller_k_.resize(ngw_);
    miller_l_.resize(ngw_);

    // Copy to GPU
    CHECK(cudaMemcpy(miller_h_.data(), h_host.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(miller_k_.data(), k_host.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(miller_l_.data(), l_host.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
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

void Grid::generate_gshell_grouping(const std::vector<double>& g2_dense) {
    // This function groups Dense grid G-vectors into shells by |G|² value.
    // QE convention:
    // - gl[igl] = |G|² for shell igl (sorted, unique values)
    // - igtongl[ig] = shell index for Dense G-vector ig
    // - ngl = number of unique G-shells

    if (g2_dense.size() != static_cast<size_t>(ngm_dense_)) {
        throw std::runtime_error("generate_gshell_grouping: size mismatch");
    }

    // Create a set of unique |G|² values (using epsilon for floating-point comparison)
    const double eps = 1e-14;  // Tolerance for |G|² equality
    std::vector<double> unique_g2;

    for (double g2 : g2_dense) {
        // Check if this |G|² is already in the list
        bool found = false;
        for (double ug2 : unique_g2) {
            if (std::abs(g2 - ug2) < eps) {
                found = true;
                break;
            }
        }
        if (!found) {
            unique_g2.push_back(g2);
        }
    }

    // Sort unique |G|² values (ascending order)
    std::sort(unique_g2.begin(), unique_g2.end());

    ngl_ = unique_g2.size();

    // Build igtongl mapping
    std::vector<int> igtongl_host(ngm_dense_);
    for (int ig = 0; ig < ngm_dense_; ++ig) {
        double g2 = g2_dense[ig];
        // Find which shell this G-vector belongs to
        int igl = -1;
        for (int i = 0; i < ngl_; ++i) {
            if (std::abs(g2 - unique_g2[i]) < eps) {
                igl = i;
                break;
            }
        }
        if (igl < 0) {
            throw std::runtime_error("generate_gshell_grouping: failed to find shell for G-vector");
        }
        igtongl_host[ig] = igl;
    }

    // Copy to GPU
    gl_.resize(ngl_);
    igtongl_.resize(ngm_dense_);

    CHECK(cudaMemcpy(gl_.data(), unique_g2.data(), ngl_ * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(igtongl_.data(), igtongl_host.data(), ngm_dense_ * sizeof(int),
                     cudaMemcpyHostToDevice));
}

void Grid::generate_igk_mapping(const std::vector<int>& h_smooth, const std::vector<int>& k_smooth,
                                const std::vector<int>& l_smooth, const std::vector<int>& h_dense,
                                const std::vector<int>& k_dense, const std::vector<int>& l_dense) {
    // This function builds igk_[ig_smooth] = ig_dense mapping
    // by matching Miller indices (h,k,l) between Smooth and Dense grids.
    //
    // QE convention:
    // - igk[ig_smooth] gives the Dense grid index for Smooth G-vector ig_smooth
    // - Every Smooth G-vector must exist in Dense grid (since ecutrho >= ecutwfc)

    if (h_smooth.size() != static_cast<size_t>(ngw_) ||
        k_smooth.size() != static_cast<size_t>(ngw_) ||
        l_smooth.size() != static_cast<size_t>(ngw_)) {
        throw std::runtime_error("generate_igk_mapping: Smooth grid size mismatch");
    }

    if (h_dense.size() != static_cast<size_t>(ngm_dense_) ||
        k_dense.size() != static_cast<size_t>(ngm_dense_) ||
        l_dense.size() != static_cast<size_t>(ngm_dense_)) {
        throw std::runtime_error("generate_igk_mapping: Dense grid size mismatch");
    }

    // Build a hash map for Dense grid (h,k,l) -> ig_dense
    // Simple hash: use tuple<int,int,int> as key
    std::map<std::tuple<int, int, int>, int> dense_map;
    for (int ig = 0; ig < ngm_dense_; ++ig) {
        dense_map[std::make_tuple(h_dense[ig], k_dense[ig], l_dense[ig])] = ig;
    }

    // Build igk mapping
    std::vector<int> igk_host(ngw_);
    for (int ig_smooth = 0; ig_smooth < ngw_; ++ig_smooth) {
        auto key = std::make_tuple(h_smooth[ig_smooth], k_smooth[ig_smooth], l_smooth[ig_smooth]);
        auto it = dense_map.find(key);
        if (it == dense_map.end()) {
            // This should never happen if ecutrho >= ecutwfc
            throw std::runtime_error(
                "generate_igk_mapping: Smooth G-vector not found in Dense grid");
        }
        igk_host[ig_smooth] = it->second;
    }

    // Copy to GPU
    igk_.resize(ngw_);
    CHECK(cudaMemcpy(igk_.data(), igk_host.data(), ngw_ * sizeof(int), cudaMemcpyHostToDevice));
}

void Grid::compute_g2kin_gpu() {
    // Check if we have any G-vectors
    if (ngw_ == 0) {
        throw std::runtime_error("compute_g2kin_gpu: ngw_ is 0. No G-vectors loaded.");
    }

    // Allocate g2kin and gg_wfc
    g2kin_.resize(ngw_);
    gg_wfc_.resize(ngw_);

    // UNIT CONVENTION (Hartree Atomic Units):
    //   rec_lattice_ is in 2π/Bohr
    //   G = h·b1 + k·b2 + l·b3, where b vectors are in 2π/Bohr
    //   |G|² is in (2π/Bohr)²
    //   g2kin = ½|G|² [Ha] (kinetic energy in Hartree)
    //
    // Note: tpiba2 = 0.5 to convert |G|² → g2kin in Hartree
    double tpiba2 = 0.5;  // Hartree: T = ½|k|²

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
    int grid_size = (ngw_ + block_size - 1) / block_size;

    compute_g2kin_kernel<<<grid_size, block_size, 0, stream_>>>(miller_h_.data(), miller_k_.data(),
                                                                miller_l_.data(), bg_gpu.data(),
                                                                tpiba2, g2kin_.data(), ngw_);

    GPU_CHECK_KERNEL;

    // Copy |G|² (without ½ factor) to gg_wfc
    // Re-compute |G|² on GPU using tpiba2 = 1.0
    GPU_Vector<double> g2_temp(ngw_);
    compute_g2kin_kernel<<<grid_size, block_size, 0, stream_>>>(miller_h_.data(), miller_k_.data(),
                                                                miller_l_.data(), bg_gpu.data(),
                                                                1.0, g2_temp.data(), ngw_);
    GPU_CHECK_KERNEL;
    CHECK(cudaMemcpy(gg_wfc_.data(), g2_temp.data(), ngw_ * sizeof(double),
                     cudaMemcpyDeviceToDevice));
}

}  // namespace dftcu
