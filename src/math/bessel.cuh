#pragma once
#include <cmath>

#include "utilities/constants.cuh"

namespace dftcu {

/**
 * @brief Spherical Bessel functions j_l(x) for small l, mimicking QE implementation.
 */
__host__ __device__ inline double spherical_bessel_jl(int l, double x) {
    const double xseries = 0.05;
    double abs_x = std::abs(x);

    if (abs_x < xseries) {
        double x2 = x * x;
        double xl = (l == 0) ? 1.0 : std::pow(abs_x, l);

        double semifact = 1.0;
        for (int i = 2 * l + 1; i >= 1; i -= 2) {
            semifact *= i;
        }

        // Match QE's 4-term expansion
        double term = 1.0 - x2 / (2.0 * (2.0 * l + 3.0)) *
                                (1.0 - x2 / (4.0 * (2.0 * l + 5.0)) *
                                           (1.0 - x2 / (6.0 * (2.0 * l + 7.0)) *
                                                      (1.0 - x2 / (8.0 * (2.0 * l + 9.0)))));

        double res = xl / semifact * term;
        if (l % 2 != 0 && x < 0)
            return -res;
        return res;
    }

    // Analytic forms for larger x
    if (l == 0) {
        return std::sin(x) / x;
    } else if (l == 1) {
        return (std::sin(x) / x - std::cos(x)) / x;
    } else if (l == 2) {
        double x2 = x * x;
        return ((3.0 / x2 - 1.0) * std::sin(x) - 3.0 * std::cos(x) / x) / x;
    } else if (l == 3) {
        double x2 = x * x;
        double x3 = x2 * x;
        return (std::sin(x) * (15.0 / x - 6.0 * x) + std::cos(x) * (x2 - 15.0)) / x3;
    } else if (l == 4) {
        double x2 = x * x;
        double x4 = x2 * x2;
        return (std::sin(x) * (105.0 - 45.0 * x2 + x4) +
                std::cos(x) * (10.0 * x2 * x - 105.0 * x)) /
               (x4 * x);
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
