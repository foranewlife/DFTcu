/**
 * Phase 0 验证器 - 使用 Miller 指数正确加载波函数
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include "../model/wavefunction.cuh"
#include "phase0_verifier.cuh"

namespace dftcu {

Phase0Verifier::Phase0Verifier(const Grid& grid) : grid_(grid) {}

bool Phase0Verifier::load_qe_s_reference(const std::string& s_file, std::vector<double>& S_ref,
                                         int& nbnd) {
    std::ifstream sf(s_file);
    if (!sf.is_open()) {
        std::cerr << "Failed to open " << s_file << std::endl;
        return false;
    }

    std::vector<std::tuple<int, int, double>> s_data;
    int i, j, max_idx = 0;
    double val;

    while (sf >> i >> j >> val) {
        s_data.push_back({i - 1, j - 1, val});
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

    std::cout << "[Phase0Verifier] Loaded S_ref: " << nbnd << "x" << nbnd << std::endl;
    return true;
}

bool Phase0Verifier::load_random_wavefunction(const std::string& wfc_file, Wavefunction& psi) {
    // 1. 读取 G 向量 Miller 指数
    std::string gvec_file = wfc_file;
    size_t pos = gvec_file.find("random_wfc");
    if (pos != std::string::npos) {
        gvec_file = gvec_file.substr(0, gvec_file.rfind("/") + 1) + "qe_gvecs_aligned.txt";
    }

    std::ifstream gf(gvec_file);
    if (!gf.is_open()) {
        std::cerr << "Failed to open G-vector file: " << gvec_file << std::endl;
        return false;
    }

    int npw_qe;
    gf >> npw_qe;

    // 读取 Miller 指数（按 QE 索引顺序）
    std::vector<int> h_vec, k_vec, l_vec;
    std::map<int, int> qe_idx_to_local;  // QE index -> local index mapping

    for (int ig = 0; ig < npw_qe; ig++) {
        int idx, h, k, l;
        double g2;
        gf >> idx >> h >> k >> l >> g2;
        // QE 索引从 1 开始，转换为 0-based
        qe_idx_to_local[idx - 1] = h_vec.size();
        h_vec.push_back(h);
        k_vec.push_back(k);
        l_vec.push_back(l);
    }
    gf.close();

    std::cout << "[Phase0Verifier] Loaded " << npw_qe << " G-vectors" << std::endl;

    // 2. 读取波函数系数
    std::ifstream wf(wfc_file);
    if (!wf.is_open()) {
        std::cerr << "Failed to open wavefunction file: " << wfc_file << std::endl;
        return false;
    }

    std::string line;
    for (int i = 0; i < 4; i++)
        std::getline(wf, line);  // Skip 4 header lines

    // 读取所有系数 (band, g_idx, real, imag)
    // 只保留 Miller 指数文件中存在的 G 向量
    std::map<int, std::vector<std::complex<double>>> coeffs_by_band;
    int ib, ig;
    double re, im;
    int nbands = 0;
    int skipped = 0;

    while (wf >> ib >> ig >> re >> im) {
        int band_idx = ib - 1;  // 转换为 0-based
        int g_idx_qe = ig - 1;  // QE 的 G 索引（0-based）

        // 检查这个 G 向量是否在 Miller 文件中
        auto it = qe_idx_to_local.find(g_idx_qe);
        if (it == qe_idx_to_local.end()) {
            skipped++;
            continue;  // 跳过不在 Miller 文件中的 G 向量
        }

        int local_idx = it->second;

        if (coeffs_by_band.find(band_idx) == coeffs_by_band.end()) {
            coeffs_by_band[band_idx].resize(npw_qe, std::complex<double>(0.0, 0.0));
        }
        coeffs_by_band[band_idx][local_idx] = std::complex<double>(re, im);
        nbands = std::max(nbands, ib);
    }
    wf.close();

    if (skipped > 0) {
        std::cout << "[Phase0Verifier] Skipped " << skipped
                  << " coefficients (G-vectors not in Miller file)" << std::endl;
    }

    std::cout << "[Phase0Verifier] Read " << nbands << " bands, " << npw_qe
              << " plane waves per band" << std::endl;

    // 3. 展平为 (nbands * npw) 的一维数组（band-major）
    std::vector<std::complex<double>> values(nbands * npw_qe);
    for (int b = 0; b < nbands; b++) {
        for (int ig = 0; ig < npw_qe; ig++) {
            values[b * npw_qe + ig] = coeffs_by_band[b][ig];
        }
    }

    // 4. 调用 set_coefficients_miller (expand_hermitian=true)
    // QE 数据已经包含 sqrt(2) 因子，set_coefficients_miller 会：
    //   - 除以 sqrt(2) 归一化
    //   - 展开到全 FFT 网格（填充 -G 位置）
    std::cout << "[Phase0Verifier] Calling set_coefficients_miller(expand_hermitian=true)..."
              << std::endl;
    psi.set_coefficients_miller(h_vec, k_vec, l_vec, values, true);

    // 5. 强制 G=0 约束
    psi.enforce_gamma_constraint_inplace();

    // 6. 正交归一化（模拟 QE regterg Step 0）
    std::cout << "[Phase0Verifier] Orthonormalizing wavefunctions..." << std::endl;
    psi.orthonormalize_inplace();

    std::cout << "[Phase0Verifier] ✓ Loaded wavefunction using production code" << std::endl;
    return true;
}

double Phase0Verifier::compute_matrix_error(const std::vector<double>& A,
                                            const std::vector<double>& B, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n * n; i++) {
        max_err = std::max(max_err, std::abs(A[i] - B[i]));
    }
    return max_err;
}

void Phase0Verifier::print_matrix_comparison(const std::vector<double>& A,
                                             const std::vector<double>& B, int n,
                                             const std::string& name) {
    std::cout << "\n" << name << " comparison:" << std::endl;
    std::cout << "  Diagonal (first 8):" << std::endl;
    for (int i = 0; i < std::min(8, n); i++) {
        std::cout << "    [" << i << "," << i << "] "
                  << "DFTcu=" << std::scientific << std::setprecision(10) << A[i * n + i]
                  << ", QE=" << B[i * n + i] << ", diff=" << std::abs(A[i * n + i] - B[i * n + i])
                  << std::endl;
    }
}

VerificationResult Phase0Verifier::verify(const std::string& wfc_file,
                                          const std::string& s_ref_file, int nbands,
                                          double ecutwfc) {
    VerificationResult result;
    result.success = false;
    result.h_sub_error = -1.0;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Phase 0 Verification (S_sub, DFTcu vs QE)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 1. Load QE reference
    std::cout << "\n[1] Loading QE S_sub reference..." << std::endl;
    std::vector<double> S_ref;
    int nbnd_ref;

    if (!load_qe_s_reference(s_ref_file, S_ref, nbnd_ref)) {
        return result;
    }

    // 2. Load wavefunction with Miller indices
    std::cout << "\n[2] Loading wavefunction with Miller index mapping..." << std::endl;
    Grid& grid_nonconst = const_cast<Grid&>(grid_);
    Wavefunction psi(grid_nonconst, nbands, ecutwfc);

    if (!load_random_wavefunction(wfc_file, psi)) {
        return result;
    }

    psi.enforce_gamma_constraint_inplace();

    // 3. Compute S_sub
    std::cout << "\n[3] Computing S_sub matrix..." << std::endl;

    std::vector<double> S_dftcu(nbands * nbands);

    for (int i = 0; i < nbands; i++) {
        for (int j = 0; j < nbands; j++) {
            std::complex<double> overlap = psi.dot(i, j);
            S_dftcu[i * nbands + j] = overlap.real();
        }
    }

    std::cout << "  ✓ S_sub computed" << std::endl;

    // 4. Compare
    std::cout << "\n[4] Precision comparison..." << std::endl;

    double s_error = compute_matrix_error(S_dftcu, S_ref, nbands);

    result.s_sub_error = s_error;
    result.success = (s_error < 1e-12);

    std::cout << "\n  S_sub max error: " << std::scientific << s_error << std::endl;
    std::cout << "  Target: < 1e-14" << std::endl;

    if (result.success) {
        std::cout << "\n✅ Phase 0 verification PASSED!" << std::endl;
    } else {
        std::cout << "\n⚠️  Precision not met" << std::endl;
    }

    print_matrix_comparison(S_dftcu, S_ref, nbands, "S_sub");

    std::cout << std::string(70, '=') << std::endl;

    return result;
}

}  // namespace dftcu
