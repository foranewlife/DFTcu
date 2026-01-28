/**
 * @file test_vloc_index_mapping.cu
 * @brief 诊断测试：验证 nl_d 索引映射的正确性
 *
 * 测试策略：
 * - 验证 Miller 指数 (h,k,l) → ig → ifft 的映射链
 * - 检查 QE 和 DFTcu 的 Miller 指数是否完全一致
 * - 验证 nl_d 映射是否正确
 */

#include <complex>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "fixtures/test_fixtures.cuh"
#include "model/wavefunction.cuh"
#include "test_data_loader.cuh"
#include "utilities/error.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

/**
 * @brief 索引映射诊断测试
 */
class VlocIndexMappingTest : public SiCMinimalFixture {
  protected:
    void SetUp() override {
        SiCMinimalFixture::SetUp();

        // 加载 QE 参考数据
        std::string data_dir = "tests/data/qe_reference/sic_minimal";
        psi_input_qe_ = load_g_space_data(data_dir + "/vloc_psi_input.dat");
        vpsi_output_qe_ = load_g_space_data(data_dir + "/vloc_psi_output.dat");
    }

    std::map<std::tuple<int, int, int, int>, std::complex<double>> psi_input_qe_;
    std::map<std::tuple<int, int, int, int>, std::complex<double>> vpsi_output_qe_;
};

/**
 * @brief 测试 1：Miller 指数完整性检查
 *
 * 验证 DFTcu 的 Miller 指数集合是否包含 QE 的所有 Miller 指数
 */
TEST_F(VlocIndexMappingTest, MillerIndices_Completeness) {
    // 获取 DFTcu 的 Miller 指数
    std::vector<int> h_dftcu = grid_->get_miller_h();
    std::vector<int> k_dftcu = grid_->get_miller_k();
    std::vector<int> l_dftcu = grid_->get_miller_l();
    int npw = h_dftcu.size();

    std::cout << "  DFTcu npw = " << npw << std::endl;

    // 构建 DFTcu Miller 指数集合
    std::set<std::tuple<int, int, int>> dftcu_millers;
    for (int ig = 0; ig < npw; ++ig) {
        dftcu_millers.insert(std::make_tuple(h_dftcu[ig], k_dftcu[ig], l_dftcu[ig]));
    }

    // 提取 QE 的 Miller 指数
    std::set<std::tuple<int, int, int>> qe_millers;
    for (const auto& kv : psi_input_qe_) {
        int h = std::get<1>(kv.first);
        int k = std::get<2>(kv.first);
        int l = std::get<3>(kv.first);
        qe_millers.insert(std::make_tuple(h, k, l));
    }

    std::cout << "  QE unique G-vectors: " << qe_millers.size() << std::endl;

    // 检查覆盖率
    int found = 0, missing = 0;
    std::vector<std::tuple<int, int, int>> missing_millers;

    for (const auto& miller : qe_millers) {
        if (dftcu_millers.count(miller)) {
            found++;
        } else {
            missing++;
            if (missing <= 10) {  // 记录前 10 个缺失的
                missing_millers.push_back(miller);
            }
        }
    }

    std::cout << "  Coverage: " << found << "/" << qe_millers.size() << " ("
              << (100.0 * found / qe_millers.size()) << "%)" << std::endl;

    if (missing > 0) {
        std::cout << "  Missing Miller indices (first 10):" << std::endl;
        for (const auto& m : missing_millers) {
            std::cout << "    (" << std::get<0>(m) << ", " << std::get<1>(m) << ", "
                      << std::get<2>(m) << ")" << std::endl;
        }
    }

    // 验证：应该 100% 覆盖
    EXPECT_EQ(missing, 0) << "DFTcu missing " << missing << " G-vectors from QE";
    EXPECT_EQ(found, qe_millers.size()) << "Coverage should be 100%";
}

/**
 * @brief 测试 2：nl_d 映射一致性检查
 *
 * 验证 nl_d[ig] 映射到的 FFT grid 位置是否与 Miller 指数一致
 */
TEST_F(VlocIndexMappingTest, NlMapping_Consistency) {
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    int npw = h.size();

    int nr1 = 18, nr2 = 18, nr3 = 18;  // SiC minimal grid

    std::cout << "  Checking nl_d mapping consistency..." << std::endl;

    int errors = 0;
    for (int ig = 0; ig < npw && errors < 10; ++ig) {
        int ifft = nl[ig];

        // 从 ifft 反推 (i0, i1, i2)
        // DFTcu 约定：ifft = i0 * (nr1*nr2) + i1 * nr1 + i2
        int i0 = ifft / (nr1 * nr2);
        int i1 = (ifft % (nr1 * nr2)) / nr1;
        int i2 = ifft % nr1;

        // Miller 指数到 FFT grid 的映射
        // 根据 Grid::reverse_engineer_miller_indices():
        // i0 = h, i1 = k, i2 = l
        // 正 Miller 指数：直接映射
        // 负 Miller 指数：映射到 nr - |h|

        int expected_i0 = (h[ig] >= 0) ? h[ig] : (nr1 + h[ig]);
        int expected_i1 = (k[ig] >= 0) ? k[ig] : (nr2 + k[ig]);
        int expected_i2 = (l[ig] >= 0) ? l[ig] : (nr3 + l[ig]);

        if (i0 != expected_i0 || i1 != expected_i1 || i2 != expected_i2) {
            std::cout << "  ERROR at ig=" << ig << ": Miller=(" << h[ig] << "," << k[ig] << ","
                      << l[ig] << ")"
                      << " ifft=" << ifft << " -> (i0=" << i0 << ",i1=" << i1 << ",i2=" << i2 << ")"
                      << " expected (i0=" << expected_i0 << ",i1=" << expected_i1
                      << ",i2=" << expected_i2 << ")" << std::endl;
            errors++;
        }
    }

    EXPECT_EQ(errors, 0) << "Found " << errors << " nl_d mapping errors";

    if (errors == 0) {
        std::cout << "  ✓ nl_d mapping is consistent" << std::endl;
    }
}

/**
 * @brief 测试 3：G=0 位置验证
 *
 * 验证 G=0 在 DFTcu 和 QE 中的位置是否一致
 */
TEST_F(VlocIndexMappingTest, GZero_Position) {
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    int npw = h.size();

    // 查找 G=0
    int ig_g0 = -1;
    for (int ig = 0; ig < npw; ++ig) {
        if (h[ig] == 0 && k[ig] == 0 && l[ig] == 0) {
            ig_g0 = ig;
            break;
        }
    }

    ASSERT_GE(ig_g0, 0) << "G=0 not found in DFTcu Miller indices";

    int ifft_g0 = nl[ig_g0];

    std::cout << "  G=0 position:" << std::endl;
    std::cout << "    - ig = " << ig_g0 << std::endl;
    std::cout << "    - ifft = " << ifft_g0 << std::endl;

    // G=0 应该映射到 FFT grid 的 (0,0,0)
    EXPECT_EQ(ifft_g0, 0) << "G=0 should map to ifft=0";

    // 检查 QE 数据中 G=0 的值
    for (int band = 0; band < 2; ++band) {
        auto key = std::make_tuple(band, 0, 0, 0);
        if (psi_input_qe_.count(key)) {
            auto psi_g0 = psi_input_qe_[key];
            std::cout << "    - Band " << band << " ψ(G=0) = " << psi_g0 << std::endl;

            // Gamma-only 约束：Im[ψ(G=0)] = 0
            EXPECT_NEAR(psi_g0.imag(), 0.0, 1e-14)
                << "Band " << band << ": Im[ψ(G=0)] should be zero";
        }
    }

    std::cout << "  ✓ G=0 position verified" << std::endl;
}

/**
 * @brief 测试 4：双向映射验证
 *
 * 验证 Miller → ig → ifft → Miller 的双向映射是否一致
 */
TEST_F(VlocIndexMappingTest, BidirectionalMapping_Consistency) {
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    int npw = h.size();

    // 构建 Miller → ig 映射
    std::map<std::tuple<int, int, int>, int> miller_to_ig;
    for (int ig = 0; ig < npw; ++ig) {
        auto key = std::make_tuple(h[ig], k[ig], l[ig]);
        miller_to_ig[key] = ig;
    }

    // 构建 ifft → ig 映射
    std::map<int, int> ifft_to_ig;
    for (int ig = 0; ig < npw; ++ig) {
        int ifft = nl[ig];
        ifft_to_ig[ifft] = ig;
    }

    std::cout << "  Checking bidirectional mapping..." << std::endl;

    // 验证：Miller → ig → ifft → ig → Miller
    int errors = 0;
    for (int ig = 0; ig < npw && errors < 10; ++ig) {
        auto miller = std::make_tuple(h[ig], k[ig], l[ig]);
        int ifft = nl[ig];

        // 反向查找
        int ig_from_miller = miller_to_ig[miller];
        int ig_from_ifft = ifft_to_ig[ifft];

        if (ig != ig_from_miller) {
            std::cout << "  ERROR: ig=" << ig << " Miller→ig=" << ig_from_miller << std::endl;
            errors++;
        }

        if (ig != ig_from_ifft) {
            std::cout << "  ERROR: ig=" << ig << " ifft→ig=" << ig_from_ifft << std::endl;
            errors++;
        }
    }

    EXPECT_EQ(errors, 0) << "Found " << errors << " bidirectional mapping errors";

    if (errors == 0) {
        std::cout << "  ✓ Bidirectional mapping is consistent" << std::endl;
    }
}

/**
 * @brief 测试 5：数据提取方法验证
 *
 * 验证从 Wavefunction 提取系数的方法是否正确
 */
TEST_F(VlocIndexMappingTest, DataExtraction_Method) {
    std::string data_dir = "tests/data/qe_reference/sic_minimal";

    // 构造测试波函数
    int num_bands = 2;
    double ecutwfc_ha = 10.0;
    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);

    // 从 QE 数据加载
    std::set<std::tuple<int, int, int>> unique_millers;
    for (const auto& kv : psi_input_qe_) {
        int h = std::get<1>(kv.first);
        int k = std::get<2>(kv.first);
        int l = std::get<3>(kv.first);
        unique_millers.insert(std::make_tuple(h, k, l));
    }

    std::vector<int> h_indices, k_indices, l_indices;
    for (const auto& miller : unique_millers) {
        h_indices.push_back(std::get<0>(miller));
        k_indices.push_back(std::get<1>(miller));
        l_indices.push_back(std::get<2>(miller));
    }

    std::vector<std::complex<double>> coeffs;
    // 注意：set_coefficients_miller() 期望 band-major 组织
    // values[b * npw + ig] = band b 的第 ig 个 G-vector
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

    std::cout << "  Setting coefficients..." << std::endl;
    std::cout << "    - num_bands: " << num_bands << std::endl;
    std::cout << "    - unique_millers: " << unique_millers.size() << std::endl;
    std::cout << "    - coeffs.size(): " << coeffs.size() << std::endl;
    std::cout << "    - expected: " << num_bands * unique_millers.size() << std::endl;

    psi.set_coefficients_miller(h_indices, k_indices, l_indices, coeffs, false);

    std::cout << "  Verifying data extraction method..." << std::endl;

    // 获取 DFTcu 的 Miller 指数映射
    std::vector<int> h_dftcu = grid_->get_miller_h();
    std::vector<int> k_dftcu = grid_->get_miller_k();
    std::vector<int> l_dftcu = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    int npw = h_dftcu.size();

    std::map<std::tuple<int, int, int>, int> miller_to_ig;
    for (int ig = 0; ig < npw; ++ig) {
        auto key = std::make_tuple(h_dftcu[ig], k_dftcu[ig], l_dftcu[ig]);
        miller_to_ig[key] = ig;
    }

    // 先检查几个关键点
    std::cout << "  Checking key G-vectors:" << std::endl;
    std::vector<std::tuple<int, int, int>> test_millers = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    for (const auto& miller : test_millers) {
        int h = std::get<0>(miller);
        int k = std::get<1>(miller);
        int l = std::get<2>(miller);

        if (!miller_to_ig.count(miller))
            continue;

        int ig = miller_to_ig[miller];
        int ifft = nl[ig];

        // 从 QE 数据获取
        auto qe_key = std::make_tuple(0, h, k, l);  // band 0
        if (!psi_input_qe_.count(qe_key))
            continue;

        auto psi_qe = psi_input_qe_[qe_key];

        // 从 DFTcu 提取
        std::vector<std::complex<double>> band_coeffs = psi.get_coefficients(0);
        auto psi_dftcu = band_coeffs[ifft];

        std::cout << "    G=(" << h << "," << k << "," << l << "): "
                  << "ig=" << ig << ", ifft=" << ifft << std::endl;
        std::cout << "      QE:    " << psi_qe << std::endl;
        std::cout << "      DFTcu: " << psi_dftcu << std::endl;
        std::cout << "      Diff:  " << std::abs(psi_dftcu - psi_qe) << std::endl;
    }

    // 验证：提取的数据是否与输入一致
    int num_checked = 0;
    int num_matched = 0;
    double max_diff = 0.0;

    for (const auto& kv : psi_input_qe_) {
        int band = std::get<0>(kv.first);
        int h = std::get<1>(kv.first);
        int k = std::get<2>(kv.first);
        int l = std::get<3>(kv.first);
        auto psi_qe = kv.second;

        // 只检查我们加载的 band
        if (band >= num_bands) {
            continue;
        }

        auto miller_key = std::make_tuple(h, k, l);
        if (!miller_to_ig.count(miller_key)) {
            continue;
        }

        int ig = miller_to_ig[miller_key];
        int ifft = nl[ig];

        // 提取 DFTcu 的系数
        std::vector<std::complex<double>> band_coeffs = psi.get_coefficients(band);
        std::complex<double> psi_dftcu = band_coeffs[ifft];

        double diff = std::abs(psi_dftcu - psi_qe);
        max_diff = std::max(max_diff, diff);

        num_checked++;
        if (diff < 1e-14) {
            num_matched++;
        }
    }

    std::cout << "    - Checked: " << num_checked << " points" << std::endl;
    std::cout << "    - Matched (< 1e-14): " << num_matched << " points" << std::endl;
    std::cout << "    - Max diff: " << max_diff << std::endl;

    double match_rate = static_cast<double>(num_matched) / num_checked;
    EXPECT_GT(match_rate, 0.99) << "Match rate too low: " << match_rate * 100 << "%";
    EXPECT_LT(max_diff, 1e-12) << "Max difference too large";

    std::cout << "  ✓ Data extraction method verified" << std::endl;
}

}  // namespace test
}  // namespace dftcu
