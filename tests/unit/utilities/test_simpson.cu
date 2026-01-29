#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "utilities/math_utils.cuh"

using namespace dftcu;

/**
 * @brief Simpson 积分单元测试
 *
 * 测试策略：
 * - Test 1: 多项式积分（解析解验证）
 * - Test 2: 指数函数积分（解析解验证）
 * - Test 3: 对数网格上的积分（模拟 UPF 径向网格）
 * - Test 4: 奇数/偶数网格行为验证
 */

class SimpsonIntegrationTest : public ::testing::Test {
  protected:
    // 生成均匀网格
    void generate_uniform_grid(int n, double a, double b, std::vector<double>& x,
                               std::vector<double>& dx) {
        x.resize(n);
        dx.resize(n);
        double h = (b - a) / (n - 1);
        for (int i = 0; i < n; ++i) {
            x[i] = a + i * h;
            dx[i] = h;
        }
    }

    // 生成对数网格（模拟 UPF）
    void generate_log_grid(int n, double r0, double r_max, std::vector<double>& r,
                           std::vector<double>& rab) {
        r.resize(n);
        rab.resize(n);
        double dx = std::log(r_max / r0) / (n - 1);
        for (int i = 0; i < n; ++i) {
            r[i] = r0 * std::exp(i * dx);
            rab[i] = r[i] * dx;  // dr = r * dx for log grid
        }
    }
};

/**
 * Test 1: 多项式积分 ∫₀¹ x^n dx = 1/(n+1)
 */
TEST_F(SimpsonIntegrationTest, Polynomial_Integration) {
    std::vector<double> x, dx;
    generate_uniform_grid(1001, 0.0, 1.0, x, dx);  // 奇数网格

    // 测试 x^0, x^1, x^2, x^3, x^4
    for (int n = 0; n <= 4; ++n) {
        std::vector<double> f(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            f[i] = std::pow(x[i], n);
        }

        double result = simpson_integrate(f, dx);
        double expected = 1.0 / (n + 1);
        double error = std::abs(result - expected);

        std::cout << "∫₀¹ x^" << n << " dx: result=" << result << ", expected=" << expected
                  << ", error=" << error << std::endl;

        // Simpson 积分对多项式应该非常精确（< 1e-10）
        EXPECT_LT(error, 1e-10) << "多项式 x^" << n << " 积分误差应 < 1e-10";
    }
}

/**
 * Test 2: 指数函数积分 ∫₀^∞ e^(-ax) dx = 1/a
 */
TEST_F(SimpsonIntegrationTest, Exponential_Integration) {
    std::vector<double> x, dx;
    generate_uniform_grid(10001, 0.0, 10.0, x, dx);  // 奇数网格，积分到 10 近似无穷

    double a = 2.0;
    std::vector<double> f(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        f[i] = std::exp(-a * x[i]);
    }

    double result = simpson_integrate(f, dx);
    double expected = 1.0 / a;  // ∫₀^∞ e^(-ax) dx = 1/a
    double error = std::abs(result - expected);

    std::cout << "∫₀^∞ e^(-" << a << "x) dx: result=" << result << ", expected=" << expected
              << ", error=" << error << std::endl;

    // 截断误差 + Simpson 误差应 < 1e-6
    EXPECT_LT(error, 1e-6) << "指数函数积分误差应 < 1e-6";
}

/**
 * Test 3: 对数网格上的积分 ∫₀^∞ r² e^(-ar) dr = 2/a³
 */
TEST_F(SimpsonIntegrationTest, LogGrid_Integration) {
    std::vector<double> r, rab;
    generate_log_grid(1141, 6.5e-5, 10.0, r, rab);  // 模拟 Si 的网格

    double a = 1.5;
    std::vector<double> f(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        f[i] = r[i] * r[i] * std::exp(-a * r[i]);
    }

    double result = simpson_integrate(f, rab);
    double expected = 2.0 / (a * a * a);  // ∫₀^∞ r² e^(-ar) dr = 2/a³
    double error = std::abs(result - expected);
    double rel_error = error / expected;

    std::cout << "∫₀^∞ r² e^(-" << a << "r) dr (log grid):" << std::endl;
    std::cout << "  result=" << result << ", expected=" << expected << std::endl;
    std::cout << "  abs_error=" << error << ", rel_error=" << rel_error * 100 << "%" << std::endl;

    // 对数网格 + 截断误差，相对误差应 < 0.1%
    EXPECT_LT(rel_error, 1e-3) << "对数网格积分相对误差应 < 0.1%";
}

/**
 * Test 4: 奇数/偶数网格行为验证
 */
TEST_F(SimpsonIntegrationTest, OddEven_Grid_Behavior) {
    // 奇数网格：包含所有点
    {
        std::vector<double> x, dx;
        generate_uniform_grid(1001, 0.0, 1.0, x, dx);  // 奇数

        std::vector<double> f(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            f[i] = x[i] * x[i];  // x²
        }

        double result = simpson_integrate(f, dx);
        double expected = 1.0 / 3.0;  // ∫₀¹ x² dx = 1/3
        double error = std::abs(result - expected);

        std::cout << "奇数网格 (1001 点): result=" << result << ", error=" << error << std::endl;
        EXPECT_LT(error, 1e-10) << "奇数网格误差应 < 1e-10";
    }

    // 偶数网格：忽略最后一点（QE 约定）
    {
        std::vector<double> x, dx;
        generate_uniform_grid(1000, 0.0, 1.0, x, dx);  // 偶数

        std::vector<double> f(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            f[i] = x[i] * x[i];  // x²
        }

        double result = simpson_integrate(f, dx);
        // 偶数网格忽略最后一点，实际积分到 x[998]
        double x_end = x[x.size() - 2];
        double expected = x_end * x_end * x_end / 3.0;  // ∫₀^x_end x² dx

        double error = std::abs(result - expected);

        std::cout << "偶数网格 (1000 点): result=" << result << ", expected=" << expected
                  << ", error=" << error << std::endl;
        EXPECT_LT(error, 1e-10) << "偶数网格误差应 < 1e-10";
    }
}

/**
 * Test 5: Bessel 积分测试（模拟真实场景）
 *
 * 测试 ∫₀^∞ r² e^(-ar) j₀(qr) dr 的精度
 * 解析解：对于 j₀(qr) = sin(qr)/(qr)，积分可以解析计算
 */
TEST_F(SimpsonIntegrationTest, Bessel_Like_Integration) {
    std::vector<double> r, rab;
    generate_log_grid(1141, 6.5e-5, 10.0, r, rab);

    double a = 1.5;  // 衰减系数
    double q = 0.5;  // 波矢

    std::vector<double> f(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        double qr = q * r[i];
        double j0 = (qr < 1e-10) ? 1.0 : std::sin(qr) / qr;  // j₀(qr)
        f[i] = r[i] * r[i] * std::exp(-a * r[i]) * j0;
    }

    double result = simpson_integrate(f, rab);

    // 解析解（通过 Mathematica 计算）：
    // ∫₀^∞ r² e^(-ar) j₀(qr) dr = ∫₀^∞ r² e^(-ar) sin(qr)/(qr) dr
    // = (1/q) ∫₀^∞ r e^(-ar) sin(qr) dr
    // = (1/q) * 2aq/(a²+q²)² = 2a/(a²+q²)²
    double expected = 2.0 * a / std::pow(a * a + q * q, 2);
    double error = std::abs(result - expected);
    double rel_error = error / expected;

    std::cout << "∫₀^∞ r² e^(-ar) j₀(qr) dr:" << std::endl;
    std::cout << "  a=" << a << ", q=" << q << std::endl;
    std::cout << "  result=" << result << ", expected=" << expected << std::endl;
    std::cout << "  abs_error=" << error << ", rel_error=" << rel_error * 100 << "%" << std::endl;

    // 这个积分应该非常精确（< 0.01%）
    EXPECT_LT(rel_error, 1e-4) << "Bessel 积分相对误差应 < 0.01%";
}
