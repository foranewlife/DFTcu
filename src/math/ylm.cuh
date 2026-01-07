#pragma once
#include <cmath>

#include "utilities/constants.cuh"

namespace dftcu {

/**
 * @brief Real Spherical Harmonics Y_lm(G) up to l=3.
 * Matches Quantum ESPRESSO's ylmr2.f90 convention and ordering.
 *
 * Ordering for each l:
 * m=0, m=1(cos), m=-1(sin), m=2(cos), m=-2(sin), ...
 *
 * Total index lm = l^2 + 1 + offset_m
 */
__host__ __device__ inline double get_ylm(int l, int m_idx, double gx, double gy, double gz,
                                          double gmod) {
    const double fpi = 4.0 * constants::D_PI;
    const double sqrt_fpi_inv = 1.0 / std::sqrt(fpi);

    if (l == 0)
        return sqrt_fpi_inv;

    if (gmod < 1e-12)
        return 0.0;

    double cost = gz / gmod;
    double sent = std::sqrt(fmax(0.0, 1.0 - cost * cost));

    // Normalize Gx, Gy for phi
    double phi = 0.0;
    if (std::abs(gx) > 1e-12) {
        phi = std::atan2(gy, gx);
    } else {
        phi = (gy > 0) ? (constants::D_PI / 2.0) : (-constants::D_PI / 2.0);
    }

    double c = std::sqrt((2.0 * l + 1.0) / fpi);

    if (l == 1) {
        // m_idx=0: m=0 (z)
        // m_idx=1: m=1 (x) -> QE: -c*sent*cos(phi)
        // m_idx=2: m=-1 (y) -> QE: -c*sent*sin(phi)
        if (m_idx == 0)
            return c * cost;
        if (m_idx == 1)
            return -c * sent * std::cos(phi);
        if (m_idx == 2)
            return -c * sent * std::sin(phi);
    }

    if (l == 2) {
        // m_idx=0: m=0 (3z^2-r^2)
        // m_idx=1: m=1 (xz)
        // m_idx=2: m=-1 (yz)
        // m_idx=3: m=2 (x^2-y^2)
        // m_idx=4: m=-2 (xy)
        double c2 = c * std::sqrt(2.0);
        if (m_idx == 0)
            return c * 0.5 * (3.0 * cost * cost - 1.0);
        if (m_idx == 1)
            return -c2 * cost * sent * std::cos(phi);
        if (m_idx == 2)
            return -c2 * cost * sent * std::sin(phi);
        if (m_idx == 3)
            return 0.5 * c2 * sent * sent * std::cos(2.0 * phi);
        if (m_idx == 4)
            return 0.5 * c2 * sent * sent * std::sin(2.0 * phi);
    }

    // l=3 and higher can be added if needed using recursion
    return 0.0;
}

}  // namespace dftcu
