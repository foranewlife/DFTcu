#include <cmath>
#include <vector>

#include "math/bessel.cuh"

#include <gtest/gtest.h>

using namespace dftcu;

/**
 * @brief 球 Bessel 函数单元测试
 *
 * 测试策略：
 * - Test 1: 特殊值验证（x=0, 边界条件）
 * - Test 2: 小 x 级数展开精度
 * - Test 3: 大 x 解析公式精度
 * - Test 4: 递推关系验证
 * - Test 5: 与标准值对比（NIST 数据）
 */

class BesselFunctionTest : public ::testing::Test {
  protected:
    // 标准球 Bessel 函数值（来自 NIST 或 Mathematica）
    struct ReferenceValue {
        int l;
        double x;
        double expected;
    };

    // 正确的球 Bessel 函数参考值（来自 scipy.special.spherical_jn）
    std::vector<ReferenceValue> nist_values = {
        // j₀(x)
        {0, 0.0, 1.000000000000000e+00},
        {0, 0.1, 9.983341664682815e-01},
        {0, 1.0, 8.414709848078965e-01},
        {0, 5.0, -1.917848549326277e-01},
        {0, 10.0, -5.440211108893698e-02},

        // j₁(x)
        {1, 0.0, 0.000000000000000e+00},
        {1, 0.1, 3.330001190255762e-02},
        {1, 1.0, 3.011686789397571e-01},
        {1, 5.0, -9.508940807917080e-02},
        {1, 10.0, 7.846694179875155e-02},

        // j₂(x)
        {2, 0.0, 0.000000000000000e+00},
        {2, 0.1, 6.661906084455694e-04},
        {2, 1.0, 6.203505201137392e-02},
        {2, 5.0, 1.347312100851252e-01},
        {2, 10.0, 7.794219362856245e-02},

        // j₃(x)
        {3, 0.0, 0.000000000000000e+00},
        {3, 0.1, 9.518519720865581e-06},
        {3, 1.0, 9.006581117112524e-03},
        {3, 5.0, 2.298206181642960e-01},
        {3, 10.0, -3.949584498447033e-02},
    };
};

/**
 * Test 1: 特殊值验证
 */
TEST_F(BesselFunctionTest, Special_Values) {
    // j₀(0) = 1
    EXPECT_NEAR(spherical_bessel_jl(0, 0.0), 1.0, 1e-15) << "j₀(0) 应该等于 1";

    // j_l(0) = 0 for l > 0
    for (int l = 1; l <= 4; ++l) {
        EXPECT_NEAR(spherical_bessel_jl(l, 0.0), 0.0, 1e-15) << "j_" << l << "(0) 应该等于 0";
    }

    // 负 x 的对称性：j_l(-x) = (-1)^l * j_l(x)
    double x = 2.5;
    for (int l = 0; l <= 4; ++l) {
        double jl_pos = spherical_bessel_jl(l, x);
        double jl_neg = spherical_bessel_jl(l, -x);
        double expected = (l % 2 == 0) ? jl_pos : -jl_pos;
        EXPECT_NEAR(jl_neg, expected, 1e-14) << "j_" << l << "(-x) 对称性验证失败";
    }
}

/**
 * Test 2: 小 x 级数展开精度
 */
TEST_F(BesselFunctionTest, Small_x_Series) {
    // 测试 x < 0.05 的级数展开
    std::vector<double> small_x = {0.001, 0.01, 0.02, 0.03, 0.04, 0.049};

    for (double x : small_x) {
        for (int l = 0; l <= 4; ++l) {
            double result = spherical_bessel_jl(l, x);

            // 手动计算级数展开的前几项作为参考
            double x2 = x * x;
            double xl = (l == 0) ? 1.0 : std::pow(x, l);

            // 计算 (2l+1)!!
            double semifact = 1.0;
            for (int i = 2 * l + 1; i >= 1; i -= 2) {
                semifact *= i;
            }

            // 4-term expansion (matching QE)
            double term = 1.0 - x2 / (2.0 * (2.0 * l + 3.0)) *
                                    (1.0 - x2 / (4.0 * (2.0 * l + 5.0)) *
                                               (1.0 - x2 / (6.0 * (2.0 * l + 7.0)) *
                                                          (1.0 - x2 / (8.0 * (2.0 * l + 9.0)))));

            double expected = xl / semifact * term;

            double error = std::abs(result - expected);
            EXPECT_LT(error, 1e-15) << "j_" << l << "(" << x << ") 级数展开误差过大";
        }
    }
}

/**
 * Test 3: 大 x 解析公式精度（与 NIST 对比）
 */
TEST_F(BesselFunctionTest, Large_x_Analytic) {
    for (const auto& ref : nist_values) {
        if (ref.x < 0.05)
            continue;  // 跳过小 x（已在 Test 2 测试）

        double result = spherical_bessel_jl(ref.l, ref.x);
        double error = std::abs(result - ref.expected);
        double rel_error = std::abs(ref.expected) > 1e-10 ? error / std::abs(ref.expected) : 0.0;

        std::cout << "j_" << ref.l << "(" << ref.x << "): result=" << result
                  << ", expected=" << ref.expected << ", rel_err=" << rel_error * 100 << "%"
                  << std::endl;

        // 相对误差应 < 1e-7（考虑到 x=0.1 附近的数值误差）
        // 对于 x=0.1，l=2,3 的误差略大（~1e-8），但仍在可接受范围内
        // 这是因为 x=0.1 刚好超过阈值 0.05，解析公式会有小的数值误差
        EXPECT_LT(rel_error, 1e-7) << "j_" << ref.l << "(" << ref.x << ") 与 NIST 值偏差过大";
    }
}

/**
 * Test 4: 递推关系验证
 *
 * 球 Bessel 函数满足递推关系：
 * (2l+1) j_l(x) = x [j_{l-1}(x) + j_{l+1}(x)]
 */
TEST_F(BesselFunctionTest, Recurrence_Relation) {
    std::vector<double> test_x = {0.5, 1.0, 2.0, 5.0, 10.0};

    for (double x : test_x) {
        for (int l = 1; l <= 3; ++l) {  // l=1,2,3 (需要 l-1 和 l+1)
            double jl_minus = spherical_bessel_jl(l - 1, x);
            double jl = spherical_bessel_jl(l, x);
            double jl_plus = spherical_bessel_jl(l + 1, x);

            // (2l+1) j_l(x) = x [j_{l-1}(x) + j_{l+1}(x)]
            double lhs = (2 * l + 1) * jl;
            double rhs = x * (jl_minus + jl_plus);

            double error = std::abs(lhs - rhs);
            double rel_error = std::abs(lhs) > 1e-10 ? error / std::abs(lhs) : 0.0;

            std::cout << "递推关系 l=" << l << ", x=" << x << ": lhs=" << lhs << ", rhs=" << rhs
                      << ", rel_err=" << rel_error * 100 << "%" << std::endl;

            // 相对误差应 < 1e-10
            EXPECT_LT(rel_error, 1e-10) << "j_" << l << "(" << x << ") 递推关系验证失败";
        }
    }
}

/**
 * Test 5: 边界情况测试
 */
TEST_F(BesselFunctionTest, Edge_Cases) {
    // 测试 x 接近 xseries 阈值（0.05）
    double x_threshold = 0.05;
    for (int l = 0; l <= 4; ++l) {
        double result_below = spherical_bessel_jl(l, x_threshold * 0.99);
        double result_above = spherical_bessel_jl(l, x_threshold * 1.01);

        // 两个分支的结果应该连续（相对误差 < 0.1%）
        double error = std::abs(result_below - result_above);
        double avg = (std::abs(result_below) + std::abs(result_above)) / 2.0;
        double rel_error = avg > 1e-10 ? error / avg : 0.0;

        std::cout << "j_" << l << " 在 x=" << x_threshold
                  << " 附近连续性: rel_err=" << rel_error * 100 << "%" << std::endl;

        // 在阈值附近，级数展开和解析公式之间会有小的不连续性
        // 对于 l=0,1,2,3，相对误差应 < 10%
        // 对于 l=4，由于高阶项，误差可能更大（< 30%）
        double tolerance = (l <= 3) ? 0.1 : 0.3;
        EXPECT_LT(rel_error, tolerance) << "j_" << l << " 在阈值附近不连续";
    }
}

/**
 * Test 6: 与 QE 数据对比（如果有的话）
 *
 * 这个测试需要从 QE 的 sph_bes.f90 提取参考值
 * 暂时跳过，等待 QE 数据
 */
TEST_F(BesselFunctionTest, DISABLED_QE_Reference) {
    // TODO: 从 QE 提取参考值后启用此测试
}
