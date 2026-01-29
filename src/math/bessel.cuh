#pragma once
#include <cmath>

#include "utilities/constants.cuh"

namespace dftcu {

/**
 * @brief Spherical Bessel functions j_l(x) matching QE implementation.
 *
 * 实现策略：
 * - 小 x (< 0.05): 使用级数展开（避免数值不稳定）
 * - 大 x: 使用 QE 的解析公式 (l=0,1,2,3,4)
 *
 * 参考：QE upflib/sph_bes.f90
 */
__host__ __device__ inline double spherical_bessel_jl(int l, double x) {
    const double xseries = 0.05;
    double abs_x = std::abs(x);

    // 小 x: 使用级数展开
    if (abs_x < xseries) {
        double x2 = x * x;
        double xl = (l == 0) ? 1.0 : std::pow(abs_x, l);

        double semifact = 1.0;
        for (int i = 2 * l + 1; i >= 1; i -= 2) {
            semifact *= i;
        }

        // 4-term expansion (matching QE)
        double term = 1.0 - x2 / (2.0 * (2.0 * l + 3.0)) *
                                (1.0 - x2 / (4.0 * (2.0 * l + 5.0)) *
                                           (1.0 - x2 / (6.0 * (2.0 * l + 7.0)) *
                                                      (1.0 - x2 / (8.0 * (2.0 * l + 9.0)))));

        double res = xl / semifact * term;
        if (l % 2 != 0 && x < 0)
            return -res;
        return res;
    }

    // 大 x: 使用 QE 的解析公式
    double qr = x;  // 在 QE 中是 q*r，这里 x = q*r
    double sin_qr = std::sin(qr);
    double cos_qr = std::cos(qr);
    double qr2 = qr * qr;
    double qr3 = qr2 * qr;
    double qr4 = qr2 * qr2;
    double qr5 = qr4 * qr;

    if (l == 0) {
        // QE line 132: jl(ir) = sin(q*r(ir)) / (q*r(ir))
        return sin_qr / qr;
    } else if (l == 1) {
        // QE line 149-150: jl(ir) = (sin(q*r(ir))/(q*r(ir)) - cos(q*r(ir))) / (q*r(ir))
        return (sin_qr / qr - cos_qr) / qr;
    } else if (l == 2) {
        // QE line 167-168: jl(ir) = ((3/qr - qr)*sin(qr) - 3*cos(qr)) / qr^2
        return ((3.0 / qr - qr) * sin_qr - 3.0 * cos_qr) / qr2;
    } else if (l == 3) {
        // QE line 187-190: jl(ir) = (sin(qr)*(15/qr - 6*qr) + cos(qr)*(qr^2 - 15)) / qr^3
        return (sin_qr * (15.0 / qr - 6.0 * qr) + cos_qr * (qr2 - 15.0)) / qr3;
    } else if (l == 4) {
        // QE line 210-214: jl(ir) = (sin(qr)*(105 - 45*qr^2 + qr^4) + cos(qr)*(10*qr^3 - 105*qr)) / qr^5
        return (sin_qr * (105.0 - 45.0 * qr2 + qr4) + cos_qr * (10.0 * qr3 - 105.0 * qr)) / qr5;
    }

    return 0.0;
}

/**
 * @brief Computes j_l(x) / x^l stably for small x.
 */
__host__ __device__ inline double spherical_bessel_jl_scaled(int l, double x) {
    const double xseries = 0.5;
    double abs_x = std::abs(x);

    if (abs_x < xseries) {
        double x2 = x * x;
        double semifact = 1.0;
        for (int i = 2 * l + 1; i >= 1; i -= 2)
            semifact *= i;

        double term =
            1.0 - x2 / (2.0 * (2.0 * l + 3.0)) *
                      (1.0 - x2 / (4.0 * (2.0 * l + 5.0)) *
                                 (1.0 - x2 / (6.0 * (2.0 * l + 7.0)) *
                                            (1.0 - x2 / (8.0 * (2.0 * l + 9.0)) *
                                                       (1.0 - x2 / (10.0 * (2.0 * l + 11.0))))));
        return term / semifact;
    }

    return spherical_bessel_jl(l, x) / std::pow(x, (double)l);
}

}  // namespace dftcu
