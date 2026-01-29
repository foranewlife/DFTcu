#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

namespace dftcu {
namespace math {

/**
 * @brief 三次样条插值系数（预计算）
 *
 * 存储三次样条插值所需的二阶导数 M[i]
 */
struct CubicSplineCoefficients {
    std::vector<double> M;  // 二阶导数 M[i] = y''[i]
    std::vector<double> h;  // 网格间距 h[i] = x[i+1] - x[i]
    std::vector<double> x;  // x 坐标
    std::vector<double> y;  // y 值
    int n;                  // 数据点数

    CubicSplineCoefficients() : n(0) {}
};

/**
 * @brief 预计算三次样条插值系数（自然边界条件）
 * [PURE] 无副作用，相同输入 → 相同输出
 *
 * @param x_data 已知数据点的 x 坐标（必须单调递增）
 * @param y_data 已知数据点的 y 值
 * @return 三次样条插值系数
 *
 * @note 使用自然边界条件（端点二阶导数为 0）
 */
inline CubicSplineCoefficients precompute_cubic_spline_coefficients(
    const std::vector<double>& x_data, const std::vector<double>& y_data) {
    if (x_data.size() != y_data.size() || x_data.size() < 3) {
        throw std::invalid_argument("precompute_cubic_spline_coefficients: need at least 3 points");
    }

    CubicSplineCoefficients coeff;
    coeff.n = x_data.size();
    coeff.x = x_data;
    coeff.y = y_data;
    coeff.h.resize(coeff.n - 1);
    coeff.M.resize(coeff.n);

    // 计算 h[i] = x[i+1] - x[i]
    for (int i = 0; i < coeff.n - 1; ++i) {
        coeff.h[i] = x_data[i + 1] - x_data[i];
        if (coeff.h[i] <= 0.0) {
            throw std::invalid_argument(
                "precompute_cubic_spline_coefficients: x_data must be strictly increasing");
        }
    }

    // 构建三对角方程组求解二阶导数 M[i]
    std::vector<double> a(coeff.n), b(coeff.n), c(coeff.n), d(coeff.n);

    // 第一行：M[0] = 0（自然边界条件）
    a[0] = 0.0;
    b[0] = 1.0;
    c[0] = 0.0;
    d[0] = 0.0;

    // 中间行
    for (int i = 1; i < coeff.n - 1; ++i) {
        a[i] = coeff.h[i - 1];
        b[i] = 2.0 * (coeff.h[i - 1] + coeff.h[i]);
        c[i] = coeff.h[i];
        d[i] = 6.0 * ((y_data[i + 1] - y_data[i]) / coeff.h[i] -
                      (y_data[i] - y_data[i - 1]) / coeff.h[i - 1]);
    }

    // 最后一行：M[n-1] = 0（自然边界条件）
    a[coeff.n - 1] = 0.0;
    b[coeff.n - 1] = 1.0;
    c[coeff.n - 1] = 0.0;
    d[coeff.n - 1] = 0.0;

    // Thomas 算法求解三对角方程组
    std::vector<double> c_prime(coeff.n);
    std::vector<double> d_prime(coeff.n);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < coeff.n; ++i) {
        double m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }

    coeff.M[coeff.n - 1] = d_prime[coeff.n - 1];
    for (int i = coeff.n - 2; i >= 0; --i) {
        coeff.M[i] = d_prime[i] - c_prime[i] * coeff.M[i + 1];
    }

    return coeff;
}

/**
 * @brief GPU 设备端三次样条插值（使用预计算的系数）
 * [PURE] 无副作用，相同输入 → 相同输出
 *
 * @param x 插值点
 * @param x_data 已知数据点的 x 坐标
 * @param y_data 已知数据点的 y 值
 * @param M 预计算的二阶导数
 * @param h 预计算的网格间距
 * @param n 数据点数
 * @return 插值结果 y(x)
 *
 * @note 如果 x 超出范围，返回边界值
 */
__host__ __device__ inline double cubic_spline_interpolate_device(double x, const double* x_data,
                                                                  const double* y_data,
                                                                  const double* M, const double* h,
                                                                  int n) {
    // 边界处理
    if (x <= x_data[0])
        return y_data[0];
    if (x >= x_data[n - 1])
        return y_data[n - 1];

    // 找到插值区间 [i, i+1]
    int i = 0;
    for (int j = 0; j < n - 1; ++j) {
        if (x >= x_data[j] && x <= x_data[j + 1]) {
            i = j;
            break;
        }
    }

    // 三次样条插值公式
    double dx = x - x_data[i];
    double a_i = y_data[i];
    double b_i = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
    double c_i = M[i] / 2.0;
    double d_i = (M[i + 1] - M[i]) / (6.0 * h[i]);

    return a_i + b_i * dx + c_i * dx * dx + d_i * dx * dx * dx;
}

/**
 * @brief 线性插值
 * [PURE] 无副作用，相同输入 → 相同输出
 *
 * @param x 插值点
 * @param x_data 已知数据点的 x 坐标（必须单调递增）
 * @param y_data 已知数据点的 y 值
 * @return 插值结果 y(x)
 *
 * @note 如果 x 超出范围，返回边界值
 */
inline double linear_interpolate(double x, const std::vector<double>& x_data,
                                 const std::vector<double>& y_data) {
    if (x_data.size() != y_data.size() || x_data.size() < 2) {
        throw std::invalid_argument("linear_interpolate: invalid input data");
    }

    int n = x_data.size();

    // 边界处理
    if (x <= x_data[0])
        return y_data[0];
    if (x >= x_data[n - 1])
        return y_data[n - 1];

    // 二分查找区间 [i, i+1]
    int i = 0;
    int j = n - 1;
    while (j - i > 1) {
        int mid = (i + j) / 2;
        if (x < x_data[mid]) {
            j = mid;
        } else {
            i = mid;
        }
    }

    // 线性插值
    double dx = x_data[i + 1] - x_data[i];
    double t = (x - x_data[i]) / dx;
    return y_data[i] * (1.0 - t) + y_data[i + 1] * t;
}

/**
 * @brief 三次样条插值（自然边界条件）
 * [PURE] 无副作用，相同输入 → 相同输出
 *
 * @param x 插值点
 * @param x_data 已知数据点的 x 坐标（必须单调递增）
 * @param y_data 已知数据点的 y 值
 * @return 插值结果 y(x)
 *
 * @note 使用自然边界条件（端点二阶导数为 0）
 * @note 如果 x 超出范围，返回边界值
 *
 * 算法（标准三次样条插值）：
 * 在区间 [x[i], x[i+1]] 上，样条函数为：
 *   S_i(x) = a_i + b_i*(x-x[i]) + c_i*(x-x[i])² + d_i*(x-x[i])³
 * 其中：
 *   a_i = y[i]
 *   b_i = (y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
 *   c_i = M[i]/2  (M[i] 是二阶导数)
 *   d_i = (M[i+1]-M[i])/(6*h[i])
 */
inline double cubic_spline_interpolate(double x, const std::vector<double>& x_data,
                                       const std::vector<double>& y_data) {
    if (x_data.size() != y_data.size() || x_data.size() < 3) {
        throw std::invalid_argument("cubic_spline_interpolate: need at least 3 points");
    }

    int n = x_data.size();

    // 边界处理
    if (x <= x_data[0])
        return y_data[0];
    if (x >= x_data[n - 1])
        return y_data[n - 1];

    // 计算 h[i] = x[i+1] - x[i]
    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i] = x_data[i + 1] - x_data[i];
        if (h[i] <= 0.0) {
            throw std::invalid_argument(
                "cubic_spline_interpolate: x_data must be strictly increasing");
        }
    }

    // 构建三对角方程组求解二阶导数 M[i]
    // 方程：h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*((y[i+1]-y[i])/h[i] -
    // (y[i]-y[i-1])/h[i-1]) 自然边界条件：M[0] = M[n-1] = 0

    std::vector<double> a(n), b(n), c(n), d(n);

    // 第一行：M[0] = 0
    a[0] = 0.0;
    b[0] = 1.0;
    c[0] = 0.0;
    d[0] = 0.0;

    // 中间行
    for (int i = 1; i < n - 1; ++i) {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * ((y_data[i + 1] - y_data[i]) / h[i] - (y_data[i] - y_data[i - 1]) / h[i - 1]);
    }

    // 最后一行：M[n-1] = 0
    a[n - 1] = 0.0;
    b[n - 1] = 1.0;
    c[n - 1] = 0.0;
    d[n - 1] = 0.0;

    // Thomas 算法求解三对角方程组
    std::vector<double> c_prime(n);
    std::vector<double> d_prime(n);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }

    std::vector<double> M(n);
    M[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        M[i] = d_prime[i] - c_prime[i] * M[i + 1];
    }

    // 找到插值区间 [i, i+1]
    int i = 0;
    for (int j = 0; j < n - 1; ++j) {
        if (x >= x_data[j] && x <= x_data[j + 1]) {
            i = j;
            break;
        }
    }

    // 三次样条插值公式
    double dx = x - x_data[i];
    double a_i = y_data[i];
    double b_i = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
    double c_i = M[i] / 2.0;
    double d_i = (M[i + 1] - M[i]) / (6.0 * h[i]);

    return a_i + b_i * dx + c_i * dx * dx + d_i * dx * dx * dx;
}

/**
 * @brief 均匀网格线性插值（优化版本）
 * [PURE] 无副作用，相同输入 → 相同输出
 *
 * @param x 插值点
 * @param x0 起始点
 * @param dx 网格间距
 * @param y_data 已知数据点的 y 值
 * @return 插值结果 y(x)
 *
 * @note 假设 x_data[i] = x0 + i * dx
 * @note 如果 x 超出范围，返回边界值
 */
inline double linear_interpolate_uniform(double x, double x0, double dx,
                                         const std::vector<double>& y_data) {
    if (y_data.size() < 2) {
        throw std::invalid_argument("linear_interpolate_uniform: need at least 2 points");
    }

    int n = y_data.size();

    // 边界处理
    if (x <= x0)
        return y_data[0];
    if (x >= x0 + (n - 1) * dx)
        return y_data[n - 1];

    // 直接计算索引
    double idx = (x - x0) / dx;
    int i = static_cast<int>(idx);
    if (i >= n - 1)
        return y_data[n - 1];

    // 线性插值
    double t = idx - i;
    return y_data[i] * (1.0 - t) + y_data[i + 1] * t;
}

}  // namespace math
}  // namespace dftcu
