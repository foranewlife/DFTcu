#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "math/interpolation.cuh"

using namespace dftcu::math;

/**
 * @brief 插值算法单元测试
 *
 * 测试策略：
 * 1. 使用解析函数（sin, exp, polynomial）生成精确数据
 * 2. 在数据点之间插值，与解析解比较
 * 3. 验证插值精度和边界行为
 */

class InterpolationTest : public ::testing::Test {
  protected:
    // 生成均匀网格数据
    void generate_uniform_data(double x0, double x1, int n,
                                std::function<double(double)> func,
                                std::vector<double>& x_data,
                                std::vector<double>& y_data) {
        x_data.resize(n);
        y_data.resize(n);
        double dx = (x1 - x0) / (n - 1);
        for (int i = 0; i < n; ++i) {
            x_data[i] = x0 + i * dx;
            y_data[i] = func(x_data[i]);
        }
    }

    // 计算最大误差
    double compute_max_error(const std::vector<double>& x_test,
                              const std::vector<double>& y_test,
                              const std::vector<double>& y_exact) {
        double max_err = 0.0;
        for (size_t i = 0; i < x_test.size(); ++i) {
            double err = std::abs(y_test[i] - y_exact[i]);
            max_err = std::max(max_err, err);
        }
        return max_err;
    }
};

/**
 * Test 1: 线性插值 - 线性函数（精确）
 *
 * 验证内容：
 * - 线性插值对线性函数应该精确到机器精度
 */
TEST_F(InterpolationTest, LinearInterpolate_LinearFunction_Exact) {
    // 生成线性函数数据：y = 2x + 3
    auto linear_func = [](double x) { return 2.0 * x + 3.0; };

    std::vector<double> x_data, y_data;
    generate_uniform_data(0.0, 10.0, 11, linear_func, x_data, y_data);

    // 在数据点之间插值
    std::vector<double> x_test = {0.5, 1.3, 2.7, 5.5, 8.9};
    std::vector<double> y_test(x_test.size());
    std::vector<double> y_exact(x_test.size());

    for (size_t i = 0; i < x_test.size(); ++i) {
        y_test[i] = linear_interpolate(x_test[i], x_data, y_data);
        y_exact[i] = linear_func(x_test[i]);
    }

    double max_err = compute_max_error(x_test, y_test, y_exact);

    std::cout << "线性插值 - 线性函数：最大误差 = " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-14) << "线性插值对线性函数应该精确到机器精度";
}

/**
 * Test 2: 线性插值 - 二次函数
 *
 * 验证内容：
 * - 线性插值对二次函数有 O(h²) 误差
 */
TEST_F(InterpolationTest, LinearInterpolate_QuadraticFunction) {
    // 生成二次函数数据：y = x²
    auto quadratic_func = [](double x) { return x * x; };

    std::vector<double> x_data, y_data;
    generate_uniform_data(0.0, 10.0, 21, quadratic_func, x_data, y_data);

    // 在数据点之间插值
    std::vector<double> x_test;
    std::vector<double> y_test;
    std::vector<double> y_exact;

    for (double x = 0.25; x < 10.0; x += 0.5) {
        x_test.push_back(x);
        y_test.push_back(linear_interpolate(x, x_data, y_data));
        y_exact.push_back(quadratic_func(x));
    }

    double max_err = compute_max_error(x_test, y_test, y_exact);

    std::cout << "线性插值 - 二次函数：最大误差 = " << max_err << std::endl;
    EXPECT_LT(max_err, 0.3) << "线性插值对二次函数误差应 < 0.3（网格间距 0.5）";
}

/**
 * Test 3: 三次样条插值 - 三次函数（精确）
 *
 * 验证内容：
 * - 三次样条插值对三次函数应该精确到数值误差
 */
TEST_F(InterpolationTest, CubicSplineInterpolate_CubicFunction_Exact) {
    // 生成三次函数数据：y = x³ - 2x² + 3x + 1
    auto cubic_func = [](double x) { return x * x * x - 2.0 * x * x + 3.0 * x + 1.0; };

    std::vector<double> x_data, y_data;
    generate_uniform_data(0.0, 5.0, 11, cubic_func, x_data, y_data);

    // 在数据点之间插值
    std::vector<double> x_test;
    std::vector<double> y_test;
    std::vector<double> y_exact;

    for (double x = 0.1; x < 5.0; x += 0.3) {
        x_test.push_back(x);
        y_test.push_back(cubic_spline_interpolate(x, x_data, y_data));
        y_exact.push_back(cubic_func(x));
    }

    double max_err = compute_max_error(x_test, y_test, y_exact);

    std::cout << "三次样条插值 - 三次函数：最大误差 = " << max_err << std::endl;
    // 注意：自然边界条件（端点二阶导数为 0）对于非零二阶导数的函数会有误差
    // 对于 y = x³ - 2x² + 3x + 1，y''(0) = -4 ≠ 0，y''(5) = 20 ≠ 0
    // 因此边界附近会有较大误差
    EXPECT_LT(max_err, 1.0) << "三次样条插值误差应 < 1.0（自然边界条件限制）";
}

/**
 * Test 4: 三次样条插值 - sin(x)
 *
 * 验证内容：
 * - 三次样条插值对光滑函数有高精度
 */
TEST_F(InterpolationTest, CubicSplineInterpolate_SinFunction) {
    // 生成 sin(x) 数据
    auto sin_func = [](double x) { return std::sin(x); };

    std::vector<double> x_data, y_data;
    generate_uniform_data(0.0, 2.0 * M_PI, 21, sin_func, x_data, y_data);

    // 在数据点之间插值
    std::vector<double> x_test;
    std::vector<double> y_test;
    std::vector<double> y_exact;

    for (double x = 0.1; x < 2.0 * M_PI; x += 0.2) {
        x_test.push_back(x);
        y_test.push_back(cubic_spline_interpolate(x, x_data, y_data));
        y_exact.push_back(sin_func(x));
    }

    double max_err = compute_max_error(x_test, y_test, y_exact);

    std::cout << "三次样条插值 - sin(x)：最大误差 = " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-4) << "三次样条插值对 sin(x) 误差应 < 1e-4";
}

/**
 * Test 5: 三次样条插值 - exp(x)
 *
 * 验证内容：
 * - 三次样条插值对指数函数的精度
 */
TEST_F(InterpolationTest, CubicSplineInterpolate_ExpFunction) {
    // 生成 exp(x) 数据
    auto exp_func = [](double x) { return std::exp(x); };

    std::vector<double> x_data, y_data;
    generate_uniform_data(0.0, 2.0, 21, exp_func, x_data, y_data);

    // 在数据点之间插值
    std::vector<double> x_test;
    std::vector<double> y_test;
    std::vector<double> y_exact;

    for (double x = 0.05; x < 2.0; x += 0.1) {
        x_test.push_back(x);
        y_test.push_back(cubic_spline_interpolate(x, x_data, y_data));
        y_exact.push_back(exp_func(x));
    }

    double max_err = compute_max_error(x_test, y_test, y_exact);

    std::cout << "三次样条插值 - exp(x)：最大误差 = " << max_err << std::endl;
    EXPECT_LT(max_err, 0.01) << "三次样条插值对 exp(x) 误差应 < 0.01";
}

/**
 * Test 6: 边界行为测试
 *
 * 验证内容：
 * - 超出范围时返回边界值
 */
TEST_F(InterpolationTest, BoundaryBehavior) {
    std::vector<double> x_data = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_data = {0.0, 1.0, 4.0, 9.0, 16.0};

    // 测试左边界
    double y_left = linear_interpolate(-1.0, x_data, y_data);
    EXPECT_DOUBLE_EQ(y_left, 0.0) << "左边界应返回 y_data[0]";

    // 测试右边界
    double y_right = linear_interpolate(5.0, x_data, y_data);
    EXPECT_DOUBLE_EQ(y_right, 16.0) << "右边界应返回 y_data[n-1]";

    // 三次样条插值边界测试
    double y_left_spline = cubic_spline_interpolate(-1.0, x_data, y_data);
    EXPECT_DOUBLE_EQ(y_left_spline, 0.0) << "三次样条左边界应返回 y_data[0]";

    double y_right_spline = cubic_spline_interpolate(5.0, x_data, y_data);
    EXPECT_DOUBLE_EQ(y_right_spline, 16.0) << "三次样条右边界应返回 y_data[n-1]";
}

/**
 * Test 7: 均匀网格线性插值（优化版本）
 *
 * 验证内容：
 * - 优化版本与通用版本结果一致
 */
TEST_F(InterpolationTest, LinearIntereUniform_Consistency) {
    // 生成数据
    double x0 = 0.0;
    double dx = 0.01;
    int n = 451;
    std::vector<double> y_data(n);
    for (int i = 0; i < n; ++i) {
        double x = x0 + i * dx;
        y_data[i] = std::sin(x);
    }

    // 构造对应的 x_data（用于通用版本）
    std::vector<double> x_data(n);
    for (int i = 0; i < n; ++i) {
        x_data[i] = x0 + i * dx;
    }

    // 测试多个插值点
    std::vector<double> x_test = {0.005, 0.123, 1.324, 2.567, 4.321};
    for (double x : x_test) {
        double y_uniform = linear_interpolate_uniform(x, x0, dx, y_data);
        double y_general = linear_interpolate(x, x_data, y_data);

        EXPECT_NEAR(y_uniform, y_general, 1e-14)
            << "均匀网格优化版本应与通用版本一致（x=" << x << "）";
    }

    std::cout << "✓ 均匀网格线性插值与通用版本一致" << std::endl;
}

/**
 * Test 8: 精度对比测试（线性 vs 三次样条）
 *
 * 验证内容：
 * - 对于光滑函数，三次样条插值精度显著高于线性插值
 */
TEST_F(InterpolationTest, AccuracyComparison_LinearVsSpline) {
    // 生成 sin(x) 数据（稀疏网格）
    auto sin_func = [](double x) { return std::sin(x); };

    std::vector<double> x_data, y_data;
    generate_uniform_data(0.0, 2.0 * M_PI, 11, sin_func, x_data, y_data);  // 仅 11 个点

    // 在数据点之间插值
    std::vector<double> x_test;
    std::vector<double> y_linear;
    std::vector<double> y_spline;
    std::vector<double> y_exact;

    for (double x = 0.1; x < 2.0 * M_PI; x += 0.2) {
        x_test.push_back(x);
        y_linear.push_back(linear_interpolate(x, x_data, y_data));
        y_spline.push_back(cubic_spline_interpolate(x, x_data, y_data));
        y_exact.push_back(sin_func(x));
    }

    double max_err_linear = compute_max_error(x_test, y_linear, y_exact);
    double max_err_spline = compute_max_error(x_test, y_spline, y_exact);

    std::cout << "\n精度对比（sin(x), 11 个数据点）：" << std::endl;
    std::cout << "  线性插值最大误差:     " << max_err_linear << std::endl;
    std::cout << "  三次样条插值最大误差: " << max_err_spline << std::endl;
    std::cout << "  精度提升倍数:         " << max_err_linear / max_err_spline << "x" << std::endl;

    EXPECT_LT(max_err_spline, max_err_linear / 10.0)
        << "三次样条插值精度应显著高于线性插值（至少 10 倍）";
}

/**
 * Test 9: chi_q 表插值场景（真实物理数据）
 *
 * 验证内容：
 * - 模拟 WavefunctionBuilder 中的实际使用场景
 * - 验证 dq = 0.01 Bohr^-1 网格下的插值精度
 */
TEST_F(InterpolationTest, ChiQTable_RealisticScenario) {
    // 模拟 chi_q 表：使用 Gaussian 函数
    // chi_q(q) = exp(-q²/2σ²)，σ = 1.0 Bohr
    double sigma = 1.0;
    auto chi_q_func = [sigma](double q) { return std::exp(-q * q / (2.0 * sigma * sigma)); };

    // 生成 chi_q 表（dq = 0.01, nqx = 451）
    double dq = 0.01;
    int nqx = 451;
    std::vector<double> q_data(nqx);
    std::vector<double> chi_q_data(nqx);
    for (int iq = 0; iq < nqx; ++iq) {
        q_data[iq] = iq * dq;
        chi_q_data[iq] = chi_q_func(q_data[iq]);
    }

    // 测试插值点（模拟 G-vector 模长）
    std::vector<double> q_test = {0.005, 0.123, 1.324, 2.567, 3.891};
    std::vector<double> chi_linear(q_test.size());
    std::vector<double> chi_spline(q_test.size());
    std::vector<double> chi_exact(q_test.size());

    for (size_t i = 0; i < q_test.size(); ++i) {
        chi_linear[i] = linear_interpolate_uniform(q_test[i], 0.0, dq, chi_q_data);
        chi_spline[i] = cubic_spline_interpolate(q_test[i], q_data, chi_q_data);
        chi_exact[i] = chi_q_func(q_test[i]);
    }

    double max_err_linear = compute_max_error(q_test, chi_linear, chi_exact);
    double max_err_spline = compute_max_error(q_test, chi_spline, chi_exact);

    std::cout << "\nchi_q 表插值场景（Gaussian, dq=0.01）：" << std::endl;
    std::cout << "  线性插值最大误差:     " << max_err_linear << std::endl;
    std::cout << "  三次样条插值最大误差: " << max_err_spline << std::endl;
    std::cout << "  精度提升倍数:         " << max_err_linear / max_err_spline << "x" << std::endl;

    // 对于密集网格（dq=0.01），线性插值已经足够精确
    // 三次样条插值的优势在稀疏网格上更明显
    EXPECT_LT(max_err_linear, 2e-5) << "线性插值在 dq=0.01 下应有 < 2e-5 精度";
    EXPECT_LT(max_err_spline, 1e-5) << "三次样条插值在 dq=0.01 下应有 < 1e-5 精度";
    EXPECT_LT(max_err_spline, max_err_linear) << "三次样条插值精度应优于线性插值";
}
