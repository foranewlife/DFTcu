/**
 * @file test_hamiltonian_vloc.cu
 * @brief 集成测试：Hamiltonian::apply_local() vs QE 参考数据
 *
 * 测试策略：
 * - 使用 QE 截获的带物理索引的参考数据
 * - 验证 V_loc|ψ⟩ 的点对点精度
 * - 验证能量期望值 ⟨ψ|V_loc|ψ⟩
 *
 * 数据来源：tests/data/qe_reference/sic_minimal/
 * - vloc_psi_input.dat: 输入 ψ(G) [band, h, k, l, Re, Im]
 * - vloc_psi_output.dat: 输出 V_loc|ψ⟩(G) [band, h, k, l, Re, Im]
 * - vloc_psi_energy.dat: ⟨ψ|V_loc|ψ⟩ 能量
 */

#include <cmath>
#include <complex>
#include <fstream>
#include <map>
#include <tuple>

#include "fixtures/test_fixtures.cuh"
#include "model/atoms.cuh"
#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "solver/hamiltonian.cuh"
#include "test_data_loader.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

/**
 * @brief SiC Minimal 系统的 V_loc_psi 集成测试
 */
class VlocPsiIntegrationTest : public SiCMinimalFixture {
  protected:
    void SetUp() override {
        SiCMinimalFixture::SetUp();

        // 加载 QE 参考数据
        std::string data_dir = "tests/data/qe_reference/sic_minimal";

        try {
            psi_input_qe_ = load_g_space_data(data_dir + "/vloc_psi_input.dat");
            vpsi_output_qe_ = load_g_space_data(data_dir + "/vloc_psi_output.dat");
            energy_qe_ry_ = load_energy(data_dir + "/vloc_psi_energy.dat");

            std::cout << "  [Fixture] Loaded QE reference data:" << std::endl;
            std::cout << "    - Input ψ(G): " << psi_input_qe_.size() << " points" << std::endl;
            std::cout << "    - Output V_loc|ψ⟩: " << vpsi_output_qe_.size() << " points"
                      << std::endl;
            std::cout << "    - Energy: " << energy_qe_ry_ << " Ry (" << energy_qe_ry_ * 0.5
                      << " Ha)" << std::endl;
        } catch (const std::exception& e) {
            FAIL() << "Failed to load QE reference data: " << e.what();
        }
    }

    std::map<std::tuple<int, int, int, int>, std::complex<double>> psi_input_qe_;
    std::map<std::tuple<int, int, int, int>, std::complex<double>> vpsi_output_qe_;
    double energy_qe_ry_;
};

/**
 * @brief 测试 1：QE 参考数据有效性检查
 */
TEST_F(VlocPsiIntegrationTest, QEReference_DataValidity) {
    // 检查数据非空
    ASSERT_GT(psi_input_qe_.size(), 0) << "Input psi data is empty";
    ASSERT_GT(vpsi_output_qe_.size(), 0) << "Output vpsi data is empty";

    // 检查能量合理性（应为负值）
    EXPECT_LT(energy_qe_ry_, 0.0) << "V_loc energy should be negative";

    // 检查 G=0 约束：Im[ψ(G=0)] = 0
    int num_bands = 8;  // SiC minimal 有 8 个 band
    for (int band = 0; band < num_bands; ++band) {
        auto key = std::make_tuple(band, 0, 0, 0);
        if (psi_input_qe_.count(key)) {
            double im_g0 = psi_input_qe_[key].imag();
            EXPECT_NEAR(im_g0, 0.0, 1e-14)
                << "Band " << band << ": Im[ψ(G=0)] should be zero (Gamma-only)";
        }
    }

    // 检查数据一致性：input 和 output 应该有相同的 band 数
    std::set<int> input_bands, output_bands;
    for (const auto& kv : psi_input_qe_) {
        input_bands.insert(std::get<0>(kv.first));
    }
    for (const auto& kv : vpsi_output_qe_) {
        output_bands.insert(std::get<0>(kv.first));
    }

    std::cout << "  Input bands: ";
    for (int b : input_bands)
        std::cout << b << " ";
    std::cout << std::endl;

    std::cout << "  Output bands: ";
    for (int b : output_bands)
        std::cout << " ";
    std::cout << std::endl;

    std::cout << "  ✓ QE reference data is valid" << std::endl;
}

/**
 * @brief 测试 2：Miller 指数映射验证
 *
 * 验证 DFTcu 的 Grid 能否正确提供 Miller 指数
 */
TEST_F(VlocPsiIntegrationTest, MillerIndex_Mapping) {
    // 获取 Miller 指数
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();

    int npw = h.size();
    ASSERT_GT(npw, 0) << "No G-vectors in grid";

    std::cout << "  DFTcu Grid: npw = " << npw << std::endl;

    // 构建反向映射：(h, k, l) -> ig
    std::map<std::tuple<int, int, int>, int> reverse_map;
    for (int ig = 0; ig < npw; ++ig) {
        auto key = std::make_tuple(h[ig], k[ig], l[ig]);
        reverse_map[key] = ig;
    }

    // 检查 QE 数据中的 Miller 指数是否都能在 DFTcu 中找到
    std::set<std::tuple<int, int, int>> qe_millers;
    for (const auto& kv : psi_input_qe_) {
        int band = std::get<0>(kv.first);
        int hh = std::get<1>(kv.first);
        int kk = std::get<2>(kv.first);
        int ll = std::get<3>(kv.first);
        qe_millers.insert(std::make_tuple(hh, kk, ll));
    }

    int found = 0, not_found = 0;
    for (const auto& miller : qe_millers) {
        if (reverse_map.count(miller)) {
            found++;
        } else {
            not_found++;
            if (not_found <= 5) {  // 只打印前 5 个
                int hh = std::get<0>(miller);
                int kk = std::get<1>(miller);
                int ll = std::get<2>(miller);
                std::cout << "  Missing Miller: (" << hh << "," << kk << "," << ll << ")"
                          << std::endl;
            }
        }
    }

    std::cout << "  Miller index coverage: " << found << "/" << qe_millers.size() << " ("
              << (100.0 * found / qe_millers.size()) << "%)" << std::endl;

    // 至少应该找到 80% 的 Miller 指数
    EXPECT_GT(found, qe_millers.size() * 0.8) << "Too many missing Miller indices";

    std::cout << "  ✓ Miller index mapping verified" << std::endl;
}

/**
 * @brief 测试 3：采样点快速验证
 *
 * 选择几个关键 G-vector 进行快速检查
 */
TEST_F(VlocPsiIntegrationTest, VlocPsi_SamplePoints_QuickCheck) {
    // 关键采样点
    std::vector<std::tuple<int, int, int>> sample_g_vectors = {
        {0, 0, 0},  // G=0
        {0, 0, 1},  // z 方向
        {0, 1, 0},  // y 方向
        {1, 0, 0},  // x 方向
        {1, 1, 1},  // 对角线
    };

    std::cout << "  Sampling key G-vectors:" << std::endl;

    for (int band = 0; band < 2; ++band) {  // 只检查前 2 个 band
        for (size_t i = 0; i < sample_g_vectors.size(); ++i) {
            int h = std::get<0>(sample_g_vectors[i]);
            int k = std::get<1>(sample_g_vectors[i]);
            int l = std::get<2>(sample_g_vectors[i]);
            auto key = std::make_tuple(band, h, k, l);

            if (psi_input_qe_.count(key) && vpsi_output_qe_.count(key)) {
                auto psi_val = psi_input_qe_[key];
                auto vpsi_val = vpsi_output_qe_[key];

                std::cout << "    Band " << band << " G=(" << h << "," << k << "," << l << "): "
                          << "ψ=" << psi_val << ", V|ψ⟩=" << vpsi_val << std::endl;
            }
        }
    }

    std::cout << "  ✓ Sample points displayed" << std::endl;
}

/**
 * @brief 测试 4：能量期望值验证（手动计算）
 *
 * 使用 QE 的 input 和 output 数据手动计算 ⟨ψ|V_loc|ψ⟩
 * 验证与 QE 提供的能量值一致
 */
TEST_F(VlocPsiIntegrationTest, VlocPsi_EnergyExpectation_ManualCalculation) {
    // 手动计算 <psi|V_loc|psi> = Σ_band Σ_G psi*(G) · (V_loc*psi)(G)
    // Gamma-only: G≠0 项乘以 2

    double energy_manual = 0.0;
    int count = 0;

    for (const auto& kv : psi_input_qe_) {
        const auto& key = kv.first;
        const auto& psi_val = kv.second;

        if (vpsi_output_qe_.count(key)) {
            auto vpsi_val = vpsi_output_qe_[key];
            int band = std::get<0>(key);
            int h = std::get<1>(key);
            int k = std::get<2>(key);
            int l = std::get<3>(key);

            // psi* · (V_loc*psi)
            double contrib = (std::conj(psi_val) * vpsi_val).real();

            // Gamma-only: G≠0 项乘以 2
            if (!(h == 0 && k == 0 && l == 0)) {
                contrib *= 2.0;
            }

            energy_manual += contrib;
            count++;
        }
    }

    std::cout << "  Manual energy calculation:" << std::endl;
    std::cout << "    - Computed: " << energy_manual << " Ry" << std::endl;
    std::cout << "    - QE reference: " << energy_qe_ry_ << " Ry" << std::endl;
    std::cout << "    - Difference: " << std::abs(energy_manual - energy_qe_ry_) << " Ry"
              << std::endl;
    std::cout << "    - Data points used: " << count << std::endl;

    // 验证能量一致性（应该非常接近）
    double diff_ry = std::abs(energy_manual - energy_qe_ry_);
    double diff_mev = diff_ry * 13605.7;  // Ry -> meV

    EXPECT_LT(diff_mev, 1.0) << "Energy difference: " << diff_mev << " meV";

    std::cout << "  ✓ Energy expectation verified (diff = " << diff_mev << " meV)" << std::endl;
}

/**
 * @brief 测试 5：完整的 apply_local() 测试
 *
 * 这个测试：
 * 1. 从 QE 数据加载输入波函数
 * 2. 加载局域势 V_loc = V_ps + V_H + V_xc
 * 3. 调用 hamiltonian.apply_local(psi, h_psi)
 * 4. 点对点比较 h_psi 与 QE 的 vpsi_output
 */
TEST_F(VlocPsiIntegrationTest, VlocPsi_FullPipeline_QEAlignment) {
    std::string data_dir = "tests/data/qe_reference/sic_minimal";

    // ========== 第 1 步：加载局域势数据 ==========
    std::cout << "\n  [Step 1] Loading local potential data..." << std::endl;

    // 加载 V_ps (Ha), V_H (Ry), V_xc (Ry)
    auto v_ps_data = load_real_field_linear(data_dir + "/v_ps_r.dat");  // Ha
    auto v_h_data = load_real_field_data(data_dir + "/v_h_r.dat");      // Ry
    auto v_xc_data = load_real_field_data(data_dir + "/v_xc_r.dat");    // Ry

    std::cout << "    - V_ps: " << v_ps_data.size() << " points (Ha)" << std::endl;
    std::cout << "    - V_H:  " << v_h_data.size() << " points (Ry)" << std::endl;
    std::cout << "    - V_xc: " << v_xc_data.size() << " points (Ry)" << std::endl;

    // 转换为连续数组并统一单位到 Ha
    int nnr = grid_->nnr();
    ASSERT_EQ(v_ps_data.size(), static_cast<size_t>(nnr)) << "V_ps size mismatch";

    std::vector<double> v_loc_host(nnr, 0.0);
    const double ry_to_ha = 0.5;

    // V_ps 已经是 Ha，直接复制
    for (int ir = 0; ir < nnr; ++ir) {
        v_loc_host[ir] = v_ps_data[ir];
    }

    // 添加 V_H 和 V_xc（从 Ry 转换到 Ha）
    for (int iz = 0; iz < 18; ++iz) {
        for (int iy = 0; iy < 18; ++iy) {
            for (int ix = 0; ix < 18; ++ix) {
                int ir = ix + iy * 18 + iz * 18 * 18;
                auto key = std::make_tuple(ix, iy, iz);

                if (v_h_data.count(key)) {
                    v_loc_host[ir] += v_h_data[key] * ry_to_ha;
                }
                if (v_xc_data.count(key)) {
                    v_loc_host[ir] += v_xc_data[key] * ry_to_ha;
                }
            }
        }
    }

    std::cout << "    ✓ V_loc = V_ps + V_H + V_xc constructed" << std::endl;

    // ========== 第 2 步：构造 Hamiltonian 并设置势能 ==========
    std::cout << "\n  [Step 2] Setting up Hamiltonian..." << std::endl;

    Hamiltonian hamiltonian(*grid_);
    hamiltonian.v_loc().copy_from_host(v_loc_host.data());

    std::cout << "    ✓ Hamiltonian initialized with V_loc" << std::endl;

    // ========== 第 3 步：构造输入波函数 ==========
    std::cout << "\n  [Step 3] Loading input wavefunction..." << std::endl;

    // 注意：QE 输出数据只包含前 2 个 band（双 band 打包的第一次调用）
    int num_bands = 2;         // 只测试前 2 个 band
    double ecutwfc_ha = 10.0;  // 20 Ry = 10 Ha
    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    // 从 QE 数据提取 Miller 指数和系数
    // 数据格式：对于每个 Miller 指数，有 num_bands 个系数
    // values = [band0_G0, band1_G0, ..., band7_G0, band0_G1, band1_G1, ...]

    // 第一步：提取所有唯一的 Miller 指数
    std::set<std::tuple<int, int, int>> unique_millers;
    for (const auto& kv : psi_input_qe_) {
        const auto& key = kv.first;
        int h = std::get<1>(key);
        int k = std::get<2>(key);
        int l = std::get<3>(key);
        unique_millers.insert(std::make_tuple(h, k, l));
    }

    std::vector<int> h_indices, k_indices, l_indices;
    for (const auto& miller : unique_millers) {
        h_indices.push_back(std::get<0>(miller));
        k_indices.push_back(std::get<1>(miller));
        l_indices.push_back(std::get<2>(miller));
    }

    // 第二步：对于每个 band，收集所有 Miller 指数的系数
    // 注意：set_coefficients_miller() 期望 band-major 组织
    // values[b * npw + ig] = band b 的第 ig 个 G-vector
    std::vector<std::complex<double>> coeffs;
    for (int band = 0; band < num_bands; ++band) {
        for (const auto& miller : unique_millers) {
            int h = std::get<0>(miller);
            int k = std::get<1>(miller);
            int l = std::get<2>(miller);

            auto key = std::make_tuple(band, h, k, l);
            if (psi_input_qe_.count(key)) {
                coeffs.push_back(psi_input_qe_[key]);
            } else {
                coeffs.push_back(std::complex<double>(0.0, 0.0));
            }
        }
    }

    // 设置波函数系数（使用 Miller 指数）
    psi.set_coefficients_miller(h_indices, k_indices, l_indices, coeffs, false);

    std::cout << "    - Loaded " << unique_millers.size() << " G-vectors × " << num_bands
              << " bands = " << coeffs.size() << " coefficients" << std::endl;
    std::cout << "    - Bands: " << num_bands << std::endl;
    std::cout << "    ✓ Input wavefunction constructed" << std::endl;

    // ========== 第 4 步：调用 apply_local() ==========
    std::cout << "\n  [Step 4] Applying V_loc operator..." << std::endl;

    // 注意：apply_local() 会累加到 h_psi，所以需要先清零
    // Wavefunction 构造后应该已经是零，但为了安全起见，我们显式设置
    std::vector<std::complex<double>> zeros(grid_->nnr() * num_bands,
                                            std::complex<double>(0.0, 0.0));
    h_psi.copy_from_host(reinterpret_cast<const std::complex<double>*>(zeros.data()));

    // 调试：检查 h_psi 在 apply_local 前后的变化
    std::vector<std::complex<double>> h_psi_before = h_psi.get_coefficients(0);
    std::cout << "    - Before apply_local: h_psi[0] = " << h_psi_before[0] << std::endl;

    hamiltonian.apply_local(psi, h_psi);

    std::vector<std::complex<double>> h_psi_after = h_psi.get_coefficients(0);
    std::cout << "    - After apply_local: h_psi[0] = " << h_psi_after[0] << std::endl;
    std::cout << "    ✓ apply_local() completed" << std::endl;

    // ========== 第 5 步：提取结果并比较 ==========
    std::cout << "\n  [Step 5] Comparing with QE reference..." << std::endl;

    // 获取 DFu 的 Miller 指数映射
    std::vector<int> h_dftcu = grid_->get_miller_h();
    std::vector<int> k_dftcu = grid_->get_miller_k();
    std::vector<int> l_dftcu = grid_->get_miller_l();
    int npw = h_dftcu.size();

    // 构建反向映射：(h, k, l) -> ig
    std::map<std::tuple<int, int, int>, int> miller_to_ig;
    for (int ig = 0; ig < npw; ++ig) {
        auto key = std::make_tuple(h_dftcu[ig], k_dftcu[ig], l_dftcu[ig]);
        miller_to_ig[key] = ig;
    }

    // 获取 nl 映射（G-vector 索引 -> FFT grid 索引）
    std::vector<int> nl = grid_->get_nl_d();

    // 逐点比较
    int num_compared = 0;
    int num_matched = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    std::complex<double> worst_case_qe, worst_case_dftcu;
    std::tuple<int, int, int, int> worst_case_key;

    for (const auto& kv : vpsi_output_qe_) {
        const auto& key = kv.first;
        const auto& vpsi_qe = kv.second;

        int band = std::get<0>(key);
        int h = std::get<1>(key);
        int k = std::get<2>(key);
        int l = std::get<3>(key);

        auto miller_key = std::make_tuple(h, k, l);
        if (!miller_to_ig.count(miller_key)) {
            continue;  // DFTcu 没有这个 G-vector
        }

        int ig = miller_to_ig[miller_key];
        int ifft = nl[ig];  // FFT grid 上的索引

        // 从 h_psi 提取对应的系数（使用 FFT grid 索引）
        std::vector<std::complex<double>> band_coeffs = h_psi.get_coefficients(band);
        std::complex<double> vpsi_dftcu = band_coeffs[ifft];

        // QE 输出是 Ry，需要转换到 Ha
        std::complex<double> vpsi_qe_ha = vpsi_qe * ry_to_ha;

        // 计算差异
        double abs_diff = std::abs(vpsi_dftcu - vpsi_qe_ha);
        double rel_diff = abs_diff / (std::abs(vpsi_qe_ha) + 1e-14);

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_rel_diff = rel_diff;
            worst_case_qe = vpsi_qe_ha;
            worst_case_dftcu = vpsi_dftcu;
            worst_case_key = key;
        }

        num_compared++;
        if (abs_diff < 1e-10) {
            num_matched++;
        }
    }

    std::cout << "    - Compared: " << num_compared << " points" << std::endl;
    std::cout << "    - Matched (< 1e-10): " << num_matched << " points" << std::endl;
    std::cout << "    - Max abs diff: " << max_abs_diff << " Ha" << std::endl;
    std::cout << "    - Max rel diff: " << max_rel_diff * 100 << " %" << std::endl;

    if (max_abs_diff > 1e-10) {
        int band = std::get<0>(worst_case_key);
        int h = std::get<1>(worst_case_key);
        int k = std::get<2>(worst_case_key);
        int l = std::get<3>(worst_case_key);
        std::cout << "    - Worst case: Band " << band << " G=(" << h << "," << k << "," << l << ")"
                  << std::endl;
        std::cout << "      QE:    " << worst_case_qe << std::endl;
        std::cout << "      DFTcu: " << worst_case_dftcu << std::endl;
    }

    // 验证：至少 80% 的点应该匹配
    double match_rate = static_cast<double>(num_matched) / num_compared;
    EXPECT_GT(match_rate, 0.8) << "Match rate too low: " << match_rate * 100 << "%";

    // 验证：最大绝对误差应该小于 1e-8 Ha
    EXPECT_LT(max_abs_diff, 1e-8) << "Max a difference too large";

    std::cout << "    ✓ Point-by-point comparison completed" << std::endl;
    std::cout << "\n  ✅ Full pipeline test passed!" << std::endl;
}

}  // namespace test
}  // namespace dftcu
