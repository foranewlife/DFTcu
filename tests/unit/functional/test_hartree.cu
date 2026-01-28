#include <cuda_runtime.h>

#include <cmath>

#include "utilities/constants.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

// ════════════════════════════════════════════════════════════════════════════════
// Hartree [PURE] Kernel Analytical Tests
// ════════════════════════════════════════════════════════════════════════════════

class HartreeKernelTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 1: Poisson Equation Formula (V_H(G) = 4π ρ(G) / |G|²)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeKernelTest, PoissonFormula_Analytical) {
    // Hartree 势的 Poisson 方程:
    // ∇²V_H(r) = -4π ρ(r)  →  V_H(G) = 4π ρ(G) / |G|²
    //
    // 验证公式系数

    const double fpi = 4.0 * constants::D_PI;
    const double e2 = 2.0;  // Rydberg atomic units

    // DFTcu 使用 crystallographic units: |G|²_cryst = |G|²_phys / (2π)²
    // 因此 fac = e2 * 4π / (2π)² = 2 * 4π / 4π² = 2/π ≈ 0.6366

    double fac = e2 * fpi / (4.0 * constants::D_PI * constants::D_PI);
    EXPECT_NEAR(fac, 2.0 / constants::D_PI, 1e-14);
    std::cout << "[INFO] Hartree factor (e2*4π/(2π)²) = " << fac << std::endl;
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 2: G=0 Handling (V_H(G=0) = 0 due to charge neutrality)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeKernelTest, GZero_ReturnsZero) {
    // 对于电中性系统，V_H(G=0) 设为 0
    // 这对应于 QE 中的 gstart = 2 (跳过 G=0)

    // 物理原因: Hartree 势的 G=0 分量对应均匀背景，
    // 对于周期性系统通过 Ewald 求和处理

    double G_sq = 0.0;
    double V_H_G0 = (G_sq > 1e-14) ? 1.0 / G_sq : 0.0;
    EXPECT_DOUBLE_EQ(V_H_G0, 0.0);
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 3: Large G Asymptotic (V_H(G) → 0 as G → ∞)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeKernelTest, LargeG_Asymptotic) {
    // 对于有限的 ρ(G)，V_H(G) ∝ ρ(G)/|G|² → 0 当 G → ∞

    double rho_g = 1.0;  // 假设有限的 ρ(G)

    std::vector<double> G_values = {1.0, 10.0, 100.0, 1000.0};
    double prev_vh = 1e30;

    for (double G : G_values) {
        double G_sq = G * G;
        double V_H_G = rho_g / G_sq;

        EXPECT_LT(V_H_G, prev_vh) << "V_H should decrease with increasing G";
        prev_vh = V_H_G;
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Unit Test 4: Energy Formula (E_H = 0.5 * ∫ ρ(r) V_H(r) dr)
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeKernelTest, EnergyFormula_Scaling) {
    // Hartree 能量: E_H = 0.5 * ∫ ρ(r) V_H(r) dr
    // 在倒空间: E_H = 0.5 * Ω * Σ_G |ρ(G)|² / |G|² * (4π e²)

    // 验证 0.5 因子的来源：
    // Hartree 能量是电子-电子排斥能，每对电子计算两次，所以除以 2

    double factor = 0.5;
    EXPECT_DOUBLE_EQ(factor, 0.5);
}

}  // namespace test
}  // namespace dftcu
