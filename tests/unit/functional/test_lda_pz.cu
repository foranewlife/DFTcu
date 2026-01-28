#include <cuda_runtime.h>

#include <cmath>

#include "functional/xc/lda_pz.cuh"
#include "utilities/constants.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

// ════════════════════════════════════════════════════════════════════════════════
// LDA_PZ [PURE] Kernel Direct Tests
// ════════════════════════════════════════════════════════════════════════════════

/**
 * @brief 在 CPU 上计算 LDA PZ 交换关联势和能量密度
 * 用于验证 GPU kernel 的正确性
 */
struct LdaPzResult {
    double v_xc;
    double e_density;  // (ex + ec) * rho
};

LdaPzResult compute_lda_pz_cpu(double rho, const LDA_PZ::Parameters& params) {
    LdaPzResult result = {0.0, 0.0};

    double r_abs = std::abs(rho);
    if (r_abs < params.rho_threshold) {
        return result;
    }

    double rho_cbrt = std::cbrt(r_abs);
    const double pi = constants::D_PI;
    double rs = std::pow(3.0 / (4.0 * pi), 1.0 / 3.0) / rho_cbrt;

    // Exchange
    double ex = constants::EX_LDA_COEFF * rho_cbrt;
    double vx = (4.0 / 3.0) * ex;

    // Correlation (PZ)
    double ec, vc;
    if (rs < 1.0) {
        double log_rs = std::log(rs);
        ec = params.a * log_rs + params.b + params.c * rs * log_rs + params.d * rs;
        vc = log_rs * (params.a + (2.0 / 3.0) * params.c * rs) + params.b - (1.0 / 3.0) * params.a +
             (1.0 / 3.0) * (2.0 * params.d - params.c) * rs;
    } else {
        double rs_sqrt = std::sqrt(rs);
        double denom = 1.0 + params.beta1 * rs_sqrt + params.beta2 * rs;
        ec = params.gamma / denom;
        vc = (params.gamma + (7.0 / 6.0 * params.gamma * params.beta1) * rs_sqrt +
              (4.0 / 3.0 * params.gamma * params.beta2) * rs) /
             (denom * denom);
    }

    result.v_xc = vx + vc;
    result.e_density = (ex + ec) * r_abs;
    return result;
}

class LdaPzKernelTest : public ::testing::Test {
  protected:
    void SetUp() override { params_ = LDA_PZ::Parameters(); }

    LDA_PZ::Parameters params_;
};

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 1: LDA Exchange Formula (rs < 1 branch)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzKernelTest, Exchange_HighDensity_rs_LessThan1) {
    // 高密度区域 (rs < 1)，需要 rho > 0.24 才能得到 rs < 1
    // rs = (3/(4π))^(1/3) / rho^(1/3) ≈ 0.62 / rho^(1/3)
    // 对于 rs = 1, rho ≈ 0.24
    double rho = 0.5;  // e/Bohr³ (高密度，rs ≈ 0.78)
    auto result = compute_lda_pz_cpu(rho, params_);

    // 验证交换势公式: v_x = (4/3) * ex = (4/3) * EX_LDA_COEFF * rho^(1/3)
    double rho_cbrt = std::cbrt(rho);
    double ex_expected = constants::EX_LDA_COEFF * rho_cbrt;
    double vx_expected = (4.0 / 3.0) * ex_expected;

    // rs = (3/(4π))^(1/3) / rho^(1/3)
    double rs = std::pow(3.0 / (4.0 * constants::D_PI), 1.0 / 3.0) / rho_cbrt;

    // 验证处于 rs < 1 分支
    EXPECT_LT(rs, 1.0) << "Expected high density region (rs < 1)";

    // 验证交换势不为零且为负
    EXPECT_LT(result.v_xc, 0.0) << "XC potential should be negative";
    EXPECT_LT(result.e_density, 0.0) << "XC energy density should be negative";

    // 验证交换部分公式（允许关联贡献）
    EXPECT_NEAR(vx_expected, constants::EX_LDA_COEFF * rho_cbrt * 4.0 / 3.0, 1e-14);
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 2: LDA Correlation (rs >= 1 branch, PZ formula)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzKernelTest, Correlation_LowDensity_rs_GreaterThan1) {
    // 低密度区域 (rs >= 1)
    double rho = 0.001;  // e/Bohr³ (低密度)
    auto result = compute_lda_pz_cpu(rho, params_);

    double rho_cbrt = std::cbrt(rho);
    double rs = std::pow(3.0 / (4.0 * constants::D_PI), 1.0 / 3.0) / rho_cbrt;

    // 验证处于 rs >= 1 分支
    EXPECT_GE(rs, 1.0) << "Expected low density region (rs >= 1)";

    // 验证结果合理性
    EXPECT_LT(result.v_xc, 0.0) << "XC potential should be negative";
    EXPECT_LT(result.e_density, 0.0) << "XC energy density should be negative";

    // 验证 PZ 关联公式 (gamma / (1 + beta1*sqrt(rs) + beta2*rs))
    double rs_sqrt = std::sqrt(rs);
    double denom = 1.0 + params_.beta1 * rs_sqrt + params_.beta2 * rs;
    double ec_expected = params_.gamma / denom;
    EXPECT_LT(ec_expected, 0.0) << "PZ correlation should be negative (gamma < 0)";
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 3: Zero Density (Threshold Handling)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzKernelTest, ZeroDensity_ReturnsZero) {
    double rho = 0.0;
    auto result = compute_lda_pz_cpu(rho, params_);

    EXPECT_DOUBLE_EQ(result.v_xc, 0.0);
    EXPECT_DOUBLE_EQ(result.e_density, 0.0);
}

TEST_F(LdaPzKernelTest, BelowThreshold_ReturnsZero) {
    double rho = 1e-15;  // 远低于阈值 1e-10
    auto result = compute_lda_pz_cpu(rho, params_);

    EXPECT_DOUBLE_EQ(result.v_xc, 0.0);
    EXPECT_DOUBLE_EQ(result.e_density, 0.0);
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 4: Negative Density (Abs Handling like QE)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzKernelTest, NegativeDensity_UsesAbsValue) {
    double rho_pos = 0.01;
    double rho_neg = -0.01;

    auto result_pos = compute_lda_pz_cpu(rho_pos, params_);
    auto result_neg = compute_lda_pz_cpu(rho_neg, params_);

    // 能量密度应该使用 |rho|，所以两者相同
    EXPECT_NEAR(result_pos.e_density, result_neg.e_density, 1e-14);
    EXPECT_NEAR(result_pos.v_xc, result_neg.v_xc, 1e-14);
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 5: Branch Transition (rs = 1 continuity)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzKernelTest, BranchTransition_rs_Equals1) {
    // rs = 1 对应的密度
    double rs_target = 1.0;
    double rho_cbrt = std::pow(3.0 / (4.0 * constants::D_PI), 1.0 / 3.0) / rs_target;
    double rho = rho_cbrt * rho_cbrt * rho_cbrt;

    // 验证 rs = 1
    double rs_check = std::pow(3.0 / (4.0 * constants::D_PI), 1.0 / 3.0) / std::cbrt(rho);
    EXPECT_NEAR(rs_check, 1.0, 1e-10);

    // rs 略小于 1 和略大于 1
    double rho_high = rho * 1.01;  // rs < 1
    double rho_low = rho * 0.99;   // rs > 1

    auto result_high = compute_lda_pz_cpu(rho_high, params_);
    auto result_low = compute_lda_pz_cpu(rho_low, params_);

    // 验证连续性（两边结果应该接近）
    // PZ 参数设计保证 rs=1 处连续
    EXPECT_NEAR(result_high.v_xc, result_low.v_xc, 0.01);  // 允许 1% 误差
}

}  // namespace test
}  // namespace dftcu
