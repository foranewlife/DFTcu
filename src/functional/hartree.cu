#include <cmath>

#include "hartree.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
/**
 * @brief Compute Hartree potential and energy contributions using nl_dense mapping.
 *
 * QE Algorithm (v_of_rho.f90:738-780):
 * 1. V_H(G) = ρ(G) / |G|²
 * 2. V_H(G) *= e² × 4π / tpiba²
 * 3. E_H = Σ |ρ(G)|² / |G|² × fac
 *
 * UNIT CONVENTION (DFTcu - No alat needed!):
 * - gg_dense: [1/Bohr²] (crystallographic, NO 2π)
 * - Physical |G|²: |G|²_phys = gg_dense × (2π)²
 * - No normalization by alat - use physical units directly!
 *
 * @param ngm Number of Dense grid G-vectors
 * @param rho_g_fft Density in reciprocal space on full FFT grid
 * @param gg_dense |G|² for Dense grid [1/Bohr²] (crystallographic units)
 * @param nl_dense Dense G-vector → FFT grid index mapping
 * @param vh_g Output: Hartree potential for Dense grid G-vectors
 * @param energy_contrib Output: Energy contribution per G-vector
 */
__global__ void hartree_potential_energy_kernel(int ngm, const gpufftComplex* rho_g_fft,
                                                const double* gg_dense, const int* nl_dense,
                                                gpufftComplex* vh_g, double* energy_contrib) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;

    if (ig < ngm) {
        if (ig == 0) {
            // G=0: V_H(G=0) = 0 (QE Line 738: gstart=2, skip G=0)
            vh_g[0].x = 0.0;
            vh_g[0].y = 0.0;
            energy_contrib[0] = 0.0;
        } else {
            // Use crystallographic gg_dense [1/Bohr²] directly - no alat conversion!
            double g2 = gg_dense[ig];

            if (g2 > 1e-14) {
                // Extract ρ(G) from FFT grid using nl_dense mapping
                int fft_idx = nl_dense[ig];
                double rho_re = rho_g_fft[fft_idx].x;
                double rho_im = rho_g_fft[fft_idx].y;

                double inv_g2 = 1.0 / g2;

                // V_H(G) = ρ(G) / |G|²
                vh_g[ig].x = rho_re * inv_g2;
                vh_g[ig].y = rho_im * inv_g2;

                // Energy contribution: |ρ(G)|² / |G|²
                double rho_mag2 = rho_re * rho_re + rho_im * rho_im;
                energy_contrib[ig] = rho_mag2 * inv_g2;
            } else {
                vh_g[ig].x = 0.0;
                vh_g[ig].y = 0.0;
                energy_contrib[ig] = 0.0;
            }
        }
    }
}

/**
 * @brief Scale Hartree potential by unit conversion factor.
 *
 * @param ngm Number of Dense grid G-vectors
 * @param vh_g Hartree potential in reciprocal space
 * @param fac Unit conversion factor
 */
__global__ void scale_vh_kernel(int ngm, gpufftComplex* vh_g, double fac) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < ngm) {
        vh_g[ig].x *= fac;
        vh_g[ig].y *= fac;
    }
}

}  // namespace

Hartree::Hartree() {}

void Hartree::initialize_buffers(Grid& grid) {
    if (grid_ == &grid)
        return;
    grid_ = &grid;
    if (!fft_)
        fft_ = std::make_unique<FFTSolver>(grid);
    if (!rho_g_)
        rho_g_ = std::make_unique<ComplexField>(grid);
    if (!v_tmp_)
        v_tmp_ = std::make_unique<RealField>(grid);
}

void Hartree::compute(const RealField& rho, RealField& vh, double& energy) {
    Grid& grid = rho.grid();
    initialize_buffers(grid);

    size_t nnr = grid.nnr();
    int ngm_dense = grid.ngm_dense();

    // DEBUG: Print lattice and volume
    const double(*lat)[3] = grid.lattice();
    printf("DEBUG Hartree: lattice (Bohr):\n");
    for (int i = 0; i < 3; ++i) {
        printf("  [%12.6f, %12.6f, %12.6f]\n", lat[i][0], lat[i][1], lat[i][2]);
    }
    printf("DEBUG Hartree: volume_bohr = %.6f Bohr³\n", grid.volume_bohr());
    printf("DEBUG Hartree: dv_bohr = %.6f Bohr³\n", grid.dv_bohr());

    // Check if Dense grid is available
    if (ngm_dense == 0) {
        throw std::runtime_error("Hartree::compute: Dense grid not initialized. "
                                 "Call grid.generate_gvectors() first.");
    }

    // ========================================================================
    // Step 1: ρ(r) → ρ(G) [Dense grid FFT]
    // ========================================================================
    real_to_complex(nnr, rho.data(), rho_g_->data());
    fft_->forward(*rho_g_);

    // DEBUG: Check rho_g[0] and first few G-vectors via nl_dense
    std::vector<gpufftComplex> rho_g_h(std::min(100, (int)nnr));
    CHECK(cudaMemcpy(rho_g_h.data(), rho_g_->data(),
                     std::min(100, (int)nnr) * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));
    printf("DEBUG Hartree: rho_g[0] = (%.16e, %.16e)\n", rho_g_h[0].x, rho_g_h[0].y);

    // DEBUG: Check nl_dense mapping for first few G-vectors
    std::vector<int> nl_dense_h(std::min(10, ngm_dense));
    CHECK(cudaMemcpy(nl_dense_h.data(), grid.nl_dense(), std::min(10, ngm_dense) * sizeof(int),
                     cudaMemcpyDeviceToHost));

    // DEBUG: Check gg_dense for first few G-vectors
    std::vector<double> gg_dense_h(std::min(10, ngm_dense));
    CHECK(cudaMemcpy(gg_dense_h.data(), grid.gg_dense(), std::min(10, ngm_dense) * sizeof(double),
                     cudaMemcpyDeviceToHost));

    printf("DEBUG Hartree: First few Dense grid G-vectors:\n");
    for (int i = 0; i < std::min(10, ngm_dense); ++i) {
        int fft_idx = nl_dense_h[i];
        if (fft_idx < 100) {
            printf("  ig=%d: nl=%4d, |G|²=%.6e, rho_g=(%.6e, %.6e)\n", i, fft_idx, gg_dense_h[i],
                   rho_g_h[fft_idx].x, rho_g_h[fft_idx].y);
        }
    }

    // ========================================================================
    // Step 2: Compute unit conversion factor
    // ========================================================================
    // Physical formula: E_H = (e² × 4π) × Σ |ρ(G)|² / |G|²_phys × Ω
    //
    // With crystallographic units (NO 2π in rec_lattice):
    //   |G|²_phys = |G|²_cryst × (2π)²
    //
    // Therefore:
    //   E_H = (e² × 4π / (2π)²) × Σ |ρ(G)|² / |G|²_cryst × Ω
    //
    // fac = e² × 4π / (2π)² = 2.0 × 4π / 39.478 ≈ 0.6366
    //
    // NO alat NEEDED! fac is a pure constant!

    const double e2 = 2.0;  // Rydberg units
    const double fpi = 4.0 * constants::D_PI;
    const double TWO_PI_SQ = 4.0 * constants::D_PI * constants::D_PI;  // (2π)²

    double fac = e2 * fpi / TWO_PI_SQ;  // Pure constant, no lattice dependence!

    printf("DEBUG Hartree: fac = e2*fpi/(2π)² = %.6f (constant)\n", fac);

    // ========================================================================
    // Step 3: Compute V_H(G) and energy contributions on Dense grid
    // ========================================================================
    // Use nl_dense mapping to extract rho_g values for Dense grid G-vectors

    // Get Dense grid data
    const double* gg_dense = grid.gg_dense();  // |G|² for Dense grid [730 values]
    const int* nl_dense = grid.nl_dense();     // Dense G → FFT index mapping

    GPU_Vector<gpufftComplex> vh_g_dense(ngm_dense);
    GPU_Vector<double> energy_contrib_dense(ngm_dense);

    const int block_size = 256;
    const int grid_size_dense = (ngm_dense + block_size - 1) / block_size;

    printf("DEBUG Hartree: Using Dense grid via nl_dense mapping (ngm_dense=%d)\n", ngm_dense);

    // Call kernel to compute V_H(G) and energy for Dense grid
    // No alat conversion needed - use crystallographic units directly!
    hartree_potential_energy_kernel<<<grid_size_dense, block_size, 0, grid.stream()>>>(
        ngm_dense, rho_g_->data(), gg_dense, nl_dense, vh_g_dense.data(),
        energy_contrib_dense.data());
    GPU_CHECK_KERNEL;

    // DEBUG: Check energy_contrib for first few G-vectors
    std::vector<double> energy_contrib_h(std::min(10, ngm_dense));
    CHECK(cudaMemcpy(energy_contrib_h.data(), energy_contrib_dense.data(),
                     std::min(10, ngm_dense) * sizeof(double), cudaMemcpyDeviceToHost));
    printf("DEBUG Hartree: First few energy contributions:\n");
    for (int i = 0; i < std::min(10, ngm_dense); ++i) {
        printf("  ig=%d: E_contrib = %.16e\n", i, energy_contrib_h[i]);
    }

    // IMPORTANT: Dense grid already stores only half-sphere (Gamma-only)!
    // The 730 G-vectors include only h>0, or (h=0,k>0), or (h=0,k=0,l>=0)
    // Due to Hermitian symmetry of real ρ(r): ρ(G) = ρ*(-G), so |ρ(G)|² = |ρ(-G)|²
    // Therefore, each G-vector's contribution already accounts for both G and -G
    // NO factor-of-2 correction needed!

    // ========================================================================
    // Step 4-5: V_H(G) → V_H(r) [TODO: implement with proper mapping]
    // ========================================================================
    // For now, skip V_H calculation and only compute energy
    // TODO: Implement Dense→FFT grid mapping for V_H
    // Set vh to zero temporarily
    CHECK(cudaMemset(vh.data(), 0, nnr * sizeof(double)));

    // ========================================================================
    // Step 6: Compute Hartree energy
    // ========================================================================
    // E_H = Σ |ρ(G)|² / |G|² × fac
    // For Gamma-only: multiply by Ω (cell volume)

    double sum_energy = v_sum(ngm_dense, energy_contrib_dense.data(), grid.stream());
    double omega = grid.volume_bohr();  // [Bohr³]

    printf("DEBUG Hartree: sum_energy (before fac*omega) = %.16e\n", sum_energy);
    printf("DEBUG Hartree: omega = %.6f Bohr³\n", omega);

    // QE: ehart = fac * SUM( |rhog|^2 / gg ) * omega (for Gamma-only)
    energy = fac * sum_energy * omega;  // [Ry]

    printf("DEBUG Hartree: energy (after fac*omega) = %.16e Ry\n", energy);

    // Convert to Hartree for internal consistency
    energy *= 0.5;  // [Ry → Ha]

    printf("DEBUG Hartree: energy (final) = %.16e Ha\n", energy);
}

}  // namespace dftcu
