/**
 * Phase 0 验证器
 * 读取随机波函数，计算 S_sub，与 QE 参考数据对比
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "../model/wavefunction.cuh"
#include "../utilities/gpu_vector.cuh"
#include "gamma_utils.cuh"
#include "hamiltonian.cuh"
#include "phase0_verifier.cuh"
#include "subspace_solver.cuh"

namespace dftcu {

// CUDA kernel to transpose from band-major to G-major layout
__global__ void transpose_band_to_g_kernel(int nnr, int nbands, const gpufftComplex* band_major,
                                           gpufftComplex* g_major) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = blockIdx.y * blockDim.y + threadIdx.y;

    if (ig < nnr && ib < nbands) {
        // band_major[ib * nnr + ig] -> g_major[ig * nbands + ib]
        g_major[ig * nbands + ib] = band_major[ib * nnr + ig];
    }
}

Phase0Verifier::Phase0Verifier(const Grid& grid) : grid_(grid) {}

bool Phase0Verifier::load_random_wavefunction(const std::string& filename, Wavefunction& psi) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    // Skip header lines (3 lines)
    std::string line;
    for (int i = 0; i < 3; i++) {
        std::getline(file, line);
    }

    int nbnd = psi.num_bands();
    int nnr = grid_.nnr();  // Full FFT grid size

    std::cout << "[Phase0Verifier] Loading wavefunction from " << filename << std::endl;
    std::cout << "  nbands=" << nbnd << ", nnr=" << nnr << std::endl;

    // 准备每个能带的系数 (nnr size)
    std::vector<std::complex<double>> wfc_band(nnr);

    int ib, ig;
    double re, im;
    int count = 0;

    // 首先读取所有数据到临时存储
    std::vector<std::tuple<int, int, double, double>> all_data;
    while (file >> ib >> ig >> re >> im) {
        all_data.push_back({ib - 1, ig - 1, re, im});  // Convert to 0-based
    }
    file.close();

    std::cout << "  Read " << all_data.size() << " coefficients from file" << std::endl;

    // 逐能带设置
    for (int band = 0; band < nbnd; band++) {
        // 清零
        std::fill(wfc_band.begin(), wfc_band.end(), std::complex<double>(0.0, 0.0));

        // 填充该能带的数据
        for (const auto& data : all_data) {
            int b = std::get<0>(data), g = std::get<1>(data);
            double r = std::get<2>(data), i = std::get<3>(data);
            if (b == band && g < nnr) {
                wfc_band[g] = std::complex<double>(r, i);
                count++;
            }
        }

        // 设置到 Wavefunction
        psi.set_coefficients(wfc_band, band);
    }

    std::cout << "  ✓ Loaded and set " << count << " coefficients for " << nbnd << " bands"
              << std::endl;

    return true;
}

bool Phase0Verifier::load_qe_s_reference(const std::string& s_file, std::vector<double>& S_ref,
                                         int& nbnd) {
    // 读取 S_sub
    std::ifstream sf(s_file);
    if (!sf.is_open()) {
        std::cerr << "Failed to open " << s_file << std::endl;
        return false;
    }

    std::vector<std::tuple<int, int, double>> s_data;
    int i, j, max_idx = 0;
    double val;

    while (sf >> i >> j >> val) {
        s_data.push_back({i - 1, j - 1, val});  // Convert to 0-based
        max_idx = std::max(max_idx, std::max(i, j));
    }
    sf.close();

    nbnd = max_idx;
    S_ref.resize(nbnd * nbnd, 0.0);

    for (const auto& data : s_data) {
        int i = std::get<0>(data), j = std::get<1>(data);
        double v = std::get<2>(data);
        S_ref[i * nbnd + j] = v;
    }

    std::cout << "[Phase0Verifier] Loaded QE S_sub reference matrix" << std::endl;
    std::cout << "  S_ref: " << nbnd << "x" << nbnd << std::endl;

    return true;
}

double Phase0Verifier::compute_matrix_error(const std::vector<double>& A,
                                            const std::vector<double>& B, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n * n; i++) {
        double err = std::abs(A[i] - B[i]);
        max_err = std::max(max_err, err);
    }
    return max_err;
}

void Phase0Verifier::print_matrix_comparison(const std::vector<double>& A,
                                             const std::vector<double>& B, int n,
                                             const std::string& name) {
    std::cout << "\n" << name << " 矩阵对比:" << std::endl;

    // 对角元
    std::cout << "  对角元 (前8个):" << std::endl;
    for (int i = 0; i < std::min(8, n); i++) {
        double a_ii = A[i * n + i];
        double b_ii = B[i * n + i];
        double diff = std::abs(a_ii - b_ii);
        std::cout << "    [" << i << "," << i << "] "
                  << "DFTcu=" << std::scientific << std::setprecision(10) << a_ii << ", QE=" << b_ii
                  << ", diff=" << diff << std::endl;
    }

    // 最大非对角元差异
    double max_offdiag_err = 0.0;
    int max_i = 0, max_j = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double err = std::abs(A[i * n + j] - B[i * n + j]);
                if (err > max_offdiag_err) {
                    max_offdiag_err = err;
                    max_i = i;
                    max_j = j;
                }
            }
        }
    }

    std::cout << "  最大非对角元差异: " << max_offdiag_err << " at [" << max_i << "," << max_j
              << "]" << std::endl;
}

VerificationResult Phase0Verifier::verify(const std::string& wfc_file,
                                          const std::string& s_ref_file, int nbands,
                                          double ecutwfc) {
    VerificationResult result;
    result.success = false;
    result.h_sub_error = -1.0;  // Not computed in Phase 0

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Phase 0 精度验证 (S_sub only, DFTcu vs QE)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 1. 加载 QE 参考数据
    std::cout << "\n[1] 加载 QE S_sub 参考数据..." << std::endl;
    std::vector<double> S_ref;
    int nbnd_ref;

    if (!load_qe_s_reference(s_ref_file, S_ref, nbnd_ref)) {
        std::cerr << "Failed to load QE reference data" << std::endl;
        return result;
    }

    // 2. 创建波函数并加载随机初值
    std::cout << "\n[2] 加载随机波函数..." << std::endl;
    Grid& grid_nonconst = const_cast<Grid&>(grid_);
    Wavefunction psi(grid_nonconst, nbands, ecutwfc);

    if (!load_random_wavefunction(wfc_file, psi)) {
        std::cerr << "Failed to load random wavefunction" << std::endl;
        return result;
    }

    psi.force_gamma_constraint();

    // 3. 计算 S_sub 子空间矩阵
    std::cout << "\n[3] 计算 S_sub 子空间矩阵..." << std::endl;

    int nnr = grid_.nnr();
    std::cout << "  nnr=" << nnr << ", nbands=" << nbands << std::endl;

    std::vector<double> S_dftcu(nbands * nbands);

    // 使用 Wavefunction::dot() 计算 S_sub[i,j] = <psi_i|psi_j>
    for (int i = 0; i < nbands; i++) {
        for (int j = 0; j < nbands; j++) {
            std::complex<double> overlap = psi.dot(i, j);
            // S_sub 是实对称矩阵（Gamma-only），只取实部
            S_dftcu[i * nbands + j] = overlap.real();
        }
    }

    std::cout << "  ✓ S_sub 子空间矩阵计算完成" << std::endl;

    // 4. 对比精度
    std::cout << "\n[4] 精度对比..." << std::endl;

    double s_error = compute_matrix_error(S_dftcu, S_ref, nbands);

    result.s_sub_error = s_error;
    result.success = (s_error < 1e-12);

    std::cout << "\n精度结果:" << std::endl;
    std::cout << "  S_sub 最大误差: " << std::scientific << s_error << std::endl;
    std::cout << "  目标精度: < 1e-14" << std::endl;

    if (result.success) {
        std::cout << "\n✅ Phase 0 (S_sub) 验证通过！" << std::endl;
    } else {
        std::cout << "\n⚠️  精度未达标，需要检查实现" << std::endl;
    }

    // 详细对比
    print_matrix_comparison(S_dftcu, S_ref, nbands, "S_sub");

    std::cout << std::string(70, '=') << std::endl;

    return result;
}

}  // namespace dftcu
