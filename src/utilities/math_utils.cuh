#pragma once
#include <cmath>
#include <vector>

namespace dftcu {

/**
 * @brief Simpson integration (QE style)
 *
 * From QE upflib/simpsn.f90.
 * For even mesh, the last point is NOT included (weight=0).
 */
inline double simpson_integrate(const std::vector<double>& f, const std::vector<double>& rab) {
    int mesh = f.size();
    if (mesh < 2)
        return 0.0;
    if (mesh == 2)
        return 0.5 * (f[0] * rab[0] + f[1] * rab[1]);

    double asum = 0.0;
    const double r12 = 1.0 / 3.0;

    for (int i = 1; i < mesh - 1; ++i) {
        int i_fortran = i + 1;  // Convert to 1-based
        double fct = static_cast<double>(std::abs((i_fortran % 2) - 2) * 2);
        asum += fct * f[i] * rab[i];
    }

    if (mesh % 2 == 1) {
        // Odd mesh
        asum = (asum + f[0] * rab[0] + f[mesh - 1] * rab[mesh - 1]) * r12;
    } else {
        // Even mesh: last point (mesh-1 in Fortran, mesh-2 in C++) is subtracted
        asum = (asum + f[0] * rab[0] - f[mesh - 2] * rab[mesh - 2]) * r12;
    }

    return asum;
}

}  // namespace dftcu
