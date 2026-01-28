/**
 * @file test_nl_d_convention.cu
 * @brief 诊断测试：验证 nl_d 的索引约定
 *
 * 测试策略：
 * - 使用简单的测试数据，手动验证索引映射
 * - 检查 set_coefficients_miller() 和 nl_d 是否使用相同的索引约定
 */

#include <algorithm>
#include <complex>
#include <iostream>

#include "fixtures/test_fixtures.cuh"
#include "model/wavefunction.cuh"
#include "utilities/error.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

/**
 * @brief nl_d 索引约定诊断测试
 */
class NlDConventionTest : public SiCMinimalFixture {};

/**
 * @brief 测试：手动验证 nl_d 的索引约定
 *
 * 策略：
 * 1. 选择几个已知的 Miller 指数
 * 2. 手动计算它们在 FFT grid 上的位置（使用两种索引约定）
 * 3. 检查 nl_d 使用的是哪种约定
 */
TEST_F(NlDConventionTest, ManualVerification) {
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    int npw = h.size();

    int nr0 = 18, nr1 = 18, nr2 = 18;  // SiC minimal grid

    std::cout << "\n  Testing nl_d indexing convention..." << std::endl;
    std::cout << "  Grid size: " << nr0 << " × " << nr1 << " × " << nr2 << std::endl;

    // 测试几个关键的 G-vector
    struct TestCase {
        int h, k, l;
        std::string name;
    };

    std::vector<TestCase> test_cases = {
        {0, 0, 0, "G=0"},       {1, 0, 0, "G=(1,0,0)"}, {0, 1, 0, "G=(0,1,0)"},
        {0, 0, 1, "G=(0,0,1)"}, {1, 1, 0, "G=(1,1,0)"}, {1, 0, 1, "G=(1,0,1)"},
        {0, 1, 1, "G=(0,1,1)"}, {1, 1, 1, "G=(1,1,1)"},
    };

    int row_major_matches = 0;
    int col_major_matches = 0;

    for (const auto& tc : test_cases) {
        // 查找这个 Miller 指数在 DFTcu 中的位置
        int ig = -1;
        for (int i = 0; i < npw; ++i) {
            if (h[i] == tc.h && k[i] == tc.k && l[i] == tc.l) {
                ig = i;
                break;
            }
        }

        if (ig < 0) {
            std::cout << "  " << tc.name << ": NOT FOUND in DFTcu" << std::endl;
            continue;
        }

        int ifft_actual = nl[ig];

        // 计算周期性索引
        int n0 = (tc.h % nr0 + nr0) % nr0;
        int n1 = (tc.k % nr1 + nr1) % nr1;
        int n2 = (tc.l % nr2 + nr2) % nr2;

        // 方案 A：Row-major (C-style)
        // ifft = n0 * (nr1*nr2) + n1 * nr1 + n2
        // 假设 n0=h, n1=k, n2=l
        int ifft_row_major = n0 * (nr1 * nr2) + n1 * nr1 + n2;

        // 方案 B：Column-major (Fortran-style)
        // ifft = n0 + n1 * nr0 + n2 * (nr0*nr1)
        // 假设 n0=h, n1=k, n2=l
        int ifft_col_major = n0 + n1 * nr0 + n2 * (nr0 * nr1);

        bool row_match = (ifft_actual == ifft_row_major);
        bool col_match = (ifft_actual == ifft_col_major);

        if (row_match)
            row_major_matches++;
        if (col_match)
            col_major_matches++;

        std::cout << "  " << tc.name << " (" << tc.h << "," << tc.k << "," << tc.l
                  << "):" << std::endl;
        std::cout << "    - Actual ifft:      " << ifft_actual << std::endl;
        std::cout << "    - Row-major ifft:   " << ifft_row_major << (row_match ? " ✓" : " ✗")
                  << std::endl;
        std::cout << "    - Column-major ifft: " << ifft_col_major << (col_match ? " ✓" : " ✗")
                  << std::endl;
    }

    std::cout << "\n  Summary:" << std::endl;
    std::cout << "    - Row-major matches:    " << row_major_matches << "/" << test_cases.size()
              << std::endl;
    std::cout << "    - Column-major matches: " << col_major_matches << "/" << test_cases.size()
              << std::endl;

    if (row_major_matches == test_cases.size()) {
        std::cout << "    ✓ nl_d uses ROW-MAJOR (C-style) indexing" << std::endl;
    } else if (col_major_matches == test_cases.size()) {
        std::cout << "    ✓ nl_d uses COLUMN-MAJOR (Fortran-style) indexing" << std::endl;
    } else {
        std::cout << "    ✗ nl_d indexing convention is UNCLEAR" << std::endl;
    }

    // 验证：应该有一种约定完全匹配
    EXPECT_TRUE(row_major_matches == test_cases.size() || col_major_matches == test_cases.size())
        << "nl_d should consistently use either row-major or column-major indexing";
}

/**
 * @brief 测试：set_coefficients_miller() 的索引约定
 *
 * 策略：
 * 1. 设置一个已知的 G-vector 系数
 * 2. 读取整个 FFT grid
 * 3. 检查非零值出现在哪个位置
 * 4. 与两种索引约定比较
 */
TEST_F(NlDConventionTest, SetCoefficientsMillerConvention) {
    int num_bands = 1;
    double ecutwfc_ha = 10.0;
    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);

    int nr0 = 18, nr1 = 18, nr2 = 18;
    size_t nnr = nr0 * nr1 * nr2;

    std::cout << "\n  Testing set_coefficients_miller() indexing convention..." << std::endl;

    // 测试 G=(1,0,0)
    std::vector<int> h_test = {1};
    std::vector<int> k_test = {0};
    std::vector<int> l_test = {0};
    std::vector<std::complex<double>> values = {std::complex<double>(1.0, 2.0)};

    psi.set_coefficients_miller(h_test, k_test, l_test, values, false);

    // 读取整个 FFT grid
    std::vector<std::complex<double>> grid_data = psi.get_coefficients(0);

    // 查找非零值的位置
    std::vector<int> nonzero_indices;
    for (size_t i = 0; i < grid_data.size(); ++i) {
        if (std::abs(grid_data[i]) > 1e-10) {
            nonzero_indices.push_back(i);
        }
    }

    std::cout << "  Set G=(1,0,0) with value (1.0, 2.0)" << std::endl;
    std::cout << "  Found " << nonzero_indices.size() << " non-zero positions:" << std::endl;

    for (int idx : nonzero_indices) {
        auto val = grid_data[idx];
        std::cout << "    - Index " << idx << ": (" << val.real() << ", " << val.imag() << ")"
                  << std::endl;

        // 反推这个索引对应的 (i0, i1, i2)
        // Row-major: idx = i0*(nr1*nr2) + i1*nr1 + i2
        int i0_row = idx / (nr1 * nr2);
        int i1_row = (idx % (nr1 * nr2)) / nr1;
        int i2_row = idx % nr1;

        // Column-major: idx = i0 + i1*nr0 + i2*(nr0*nr1)
        int i2_col = idx / (nr0 * nr1);
        int i1_col = (idx % (nr0 * nr1)) / nr0;
        int i0_col = idx % nr0;

        std::cout << "      Row-major interpretation:    (i0=" << i0_row << ", i1=" << i1_row
                  << ", i2=" << i2_row << ")" << std::endl;
        std::cout << "      Column-major interpretation: (i0=" << i0_col << ", i1=" << i1_col
                  << ", i2=" << i2_col << ")" << std::endl;
    }

    // 计算期望的索引
    int n0 = 1, n1 = 0, n2 = 0;  // G=(1,0,0)

    int expected_row_major = n0 * (nr1 * nr2) + n1 * nr1 + n2;
    int expected_col_major = n0 + n1 * nr0 + n2 * (nr0 * nr1);

    std::cout << "\n  Expected indices:" << std::endl;
    std::cout << "    - Row-major: " << expected_row_major << std::endl;
    std::cout << "    - Column-major: " << expected_col_major << std::endl;

    bool found_row = std::find(nonzero_indices.begin(), nonzero_indices.end(),
                               expected_row_major) != nonzero_indices.end();
    bool found_col = std::find(nonzero_indices.begin(), nonzero_indices.end(),
                               expected_col_major) != nonzero_indices.end();

    if (found_row) {
        std::cout << "    ✓ set_coefficients_miller() uses ROW-MAJOR indexing" << std::endl;
    }
    if (found_col) {
        std::cout << "    ✓ set_coefficients_miller() uses COLUMN-MAJOR indexing" << std::endl;
    }

    EXPECT_TRUE(found_row || found_col)
        << "set_coefficients_miller() should use a consistent indexing convention";
}

}  // namespace test
}  // namespace dftcu
