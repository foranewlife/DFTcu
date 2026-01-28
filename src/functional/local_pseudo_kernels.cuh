#pragma once
#include <cuda_runtime.h>

#include "utilities/constants.cuh"

namespace dftcu {

/**
 * @brief Pure physical logic for local pseudopotential interpolation.
 * [PURE] - No side effects.
 *
 * @param gmod Physical |G| in 2π/Bohr
 * @param table_short Pointer to interpolation table for specific atom type
 * @param stride Leading dimension of flat_tab
 * @param zp Valence charge
 * @param omega Unit cell volume in Bohr³
 * @return V_loc(G) in Hartree
 */
__device__ inline double interpolate_vloc_phys(double gmod, const double* table_short, int stride,
                                               double zp, double omega) {
    const double fpi = 4.0 * constants::D_PI;
    double g2_phys = gmod * gmod;
    double vlocg = 0;

    if (g2_phys < 1e-12) {
        vlocg = table_short[0];
    } else {
        const double dq = 0.01;  // Match QE dq
        int i0 = (int)(gmod / dq) + 1;
        i0 = min(max(i0, 1), stride - 4);
        double px = gmod / dq - (double)(i0 - 1);
        double ux = 1.0 - px;
        double vx = 2.0 - px;
        double wx = 3.0 - px;

        vlocg = table_short[i0] * ux * vx * wx / 6.0 + table_short[i0 + 1] * px * vx * wx / 2.0 -
                table_short[i0 + 2] * px * ux * wx / 2.0 + table_short[i0 + 3] * px * ux * vx / 6.0;

        // Apply Coulomb correction (Physical units)
        vlocg -= (fpi * zp / (omega * g2_phys)) * exp(-0.25 * g2_phys);
    }
    return vlocg;
}

/**
 * @brief Pure logic for structure factor phase.
 * [PURE] - No side effects.
 *
 * @param gx_cryst, gy_cryst, gz_cryst Crystallographic G-vector components (1/Bohr)
 * @param tx, ty, tz Atomic position in Bohr
 * @return Phase factor exp(-i * 2π * G·τ)
 */
__device__ inline gpufftComplex compute_structure_factor_phase(double gx_cryst, double gy_cryst,
                                                               double gz_cryst, double tx,
                                                               double ty, double tz) {
    // phase = -2π * G_cryst · τ_bohr
    // dimensionless, as (1/Bohr * Bohr)
    double phase = -(gx_cryst * tx + gy_cryst * ty + gz_cryst * tz) * 2.0 * constants::D_PI;
    gpufftComplex res;
    sincos(phase, &res.y, &res.x);  // res.x = cos, res.y = sin
    return res;
}

}  // namespace dftcu
