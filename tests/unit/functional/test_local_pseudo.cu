#include <cuda_runtime.h>

#include <cmath>

#include "fixtures/test_data_loader.cuh"
#include "fixtures/test_fixtures.cuh"
#include "functional/local_pseudo_kernels.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

// ════════════════════════════════════════════════════════════════════════════════
// Test Wrappers (Kernels to call __device__ pure functions)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void test_interpolate_kernel(double gmod, const double* tab, int stride, double zp,
                                        double omega, double* result) {
    *result = interpolate_vloc_phys(gmod, tab, stride, zp, omega);
}

__global__ void test_phase_kernel(double3 G, double3 tau, gpufftComplex* result) {
    *result = compute_structure_factor_phase(G.x, G.y, G.z, tau.x, tau.y, tau.z);
}

// ════════════════════════════════════════════════════════════════════════════════
// Test Fixture with QE Reference Data
// ════════════════════════════════════════════════════════════════════════════════

class LocalPseudoKernelsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Load real QE vloc_tab data
        si_data_ = StandardDataLoader::load_vloc_tab(
            "tests/data/qe_reference/sic_minimal/vloc_tab_Si.dat");
        c_data_ =
            StandardDataLoader::load_vloc_tab("tests/data/qe_reference/sic_minimal/vloc_tab_C.dat");

        // Verify data loaded correctly
        ASSERT_GT(si_data_.tab.size(), 0) << "Failed to load Si vloc_tab";
        ASSERT_GT(c_data_.tab.size(), 0) << "Failed to load C vloc_tab";

        // Convert QE Ry to Hartree (DFTcu internal unit)
        for (auto& v : si_data_.tab)
            v *= 0.5;
        for (auto& v : c_data_.tab)
            v *= 0.5;
    }

    VlocTabData si_data_;
    VlocTabData c_data_;
};

// ════════════════════════════════════════════════════════════════════════════════
// Unit Tests: compute_structure_factor_phase [PURE]
// ════════════════════════════════════════════════════════════════════════════════

TEST_F(LocalPseudoKernelsTest, StructureFactor_G0) {
    // G = 0 should give phase = 1 + 0i
    double3 G = {0.0, 0.0, 0.0};
    double3 tau = {0.25, 0.25, 0.25};

    gpufftComplex* d_result;
    cudaMalloc(&d_result, sizeof(gpufftComplex));
    test_phase_kernel<<<1, 1>>>(G, tau, d_result);

    gpufftComplex h_result;
    cudaMemcpy(&h_result, d_result, sizeof(gpufftComplex), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_result.x, 1.0, 1e-14);  // Re = cos(0) = 1
    EXPECT_NEAR(h_result.y, 0.0, 1e-14);  // Im = sin(0) = 0

    cudaFree(d_result);
}

TEST_F(LocalPseudoKernelsTest, StructureFactor_QuarterPhase) {
    // G·τ = 0.25 → phase = -π/2 → exp(-iπ/2) = (0, -1)
    double3 G = {1.0, 0.0, 0.0};
    double3 tau = {0.25, 0.0, 0.0};

    gpufftComplex* d_result;
    cudaMalloc(&d_result, sizeof(gpufftComplex));
    test_phase_kernel<<<1, 1>>>(G, tau, d_result);

    gpufftComplex h_result;
    cudaMemcpy(&h_result, d_result, sizeof(gpufftComplex), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_result.x, 0.0, 1e-14);   // Re = cos(-π/2) = 0
    EXPECT_NEAR(h_result.y, -1.0, 1e-14);  // Im = sin(-π/2) = -1

    cudaFree(d_result);
}

TEST_F(LocalPseudoKernelsTest, StructureFactor_HalfPhase) {
    // G·τ = 0.5 → phase = -π → exp(-iπ) = (-1, 0)
    double3 G = {2.0, 0.0, 0.0};
    double3 tau = {0.25, 0.0, 0.0};

    gpufftComplex* d_result;
    cudaMalloc(&d_result, sizeof(gpufftComplex));
    test_phase_kernel<<<1, 1>>>(G, tau, d_result);

    gpufftComplex h_result;
    cudaMemcpy(&h_result, d_result, sizeof(gpufftComplex), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_result.x, -1.0, 1e-14);  // Re = cos(-π) = -1
    EXPECT_NEAR(h_result.y, 0.0, 1e-14);   // Im = sin(-π) = 0

    cudaFree(d_result);
}

TEST_F(LocalPseudoKernelsTest, StructureFactor_ThreeQuarterPhase) {
    // G·τ = 0.75 → phase = -3π/2 → exp(-i3π/2) = (0, 1)
    double3 G = {3.0, 0.0, 0.0};
    double3 tau = {0.25, 0.0, 0.0};

    gpufftComplex* d_result;
    cudaMalloc(&d_result, sizeof(gpufftComplex));
    test_phase_kernel<<<1, 1>>>(G, tau, d_result);

    gpufftComplex h_result;
    cudaMemcpy(&h_result, d_result, sizeof(gpufftComplex), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_result.x, 0.0, 1e-14);  // Re = cos(-3π/2) = 0
    EXPECT_NEAR(h_result.y, 1.0, 1e-14);  // Im = sin(-3π/2) = 1

    cudaFree(d_result);
}

// ════════════════════════════════════════════════════════════════════════════════
// Unit Tests: interpolate_vloc_phys [PURE] - Boundary Conditions
// ════════════════════════════════════════════════════════════════════════════════

TEST_F(LocalPseudoKernelsTest, InterpolateVloc_G0_ReturnTab0) {
    // G = 0 should return tab[0] directly (no Coulomb correction)
    int stride = static_cast<int>(si_data_.tab.size());

    double* d_tab;
    cudaMalloc(&d_tab, stride * sizeof(double));
    cudaMemcpy(d_tab, si_data_.tab.data(), stride * sizeof(double), cudaMemcpyHostToDevice);

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    // G = 0
    test_interpolate_kernel<<<1, 1>>>(0.0, d_tab, stride, si_data_.zp, si_data_.omega * 0.5,
                                      d_result);

    double h_result;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Should exactly equal tab[0] (already converted to Hartree)
    EXPECT_DOUBLE_EQ(h_result, si_data_.tab[0]);

    cudaFree(d_tab);
    cudaFree(d_result);
}

TEST_F(LocalPseudoKernelsTest, InterpolateVloc_VerySmallG_ReturnTab0) {
    // G < 1e-12 should also return tab[0]
    int stride = static_cast<int>(si_data_.tab.size());

    double* d_tab;
    cudaMalloc(&d_tab, stride * sizeof(double));
    cudaMemcpy(d_tab, si_data_.tab.data(), stride * sizeof(double), cudaMemcpyHostToDevice);

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    // G = 1e-15 (very small)
    test_interpolate_kernel<<<1, 1>>>(1e-15, d_tab, stride, si_data_.zp, si_data_.omega * 0.5,
                                      d_result);

    double h_result;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    EXPECT_DOUBLE_EQ(h_result, si_data_.tab[0]);

    cudaFree(d_tab);
    cudaFree(d_result);
}

// ════════════════════════════════════════════════════════════════════════════════
// Unit Tests: interpolate_vloc_phys [PURE] - QE Alignment
// ════════════════════════════════════════════════════════════════════════════════

TEST_F(LocalPseudoKernelsTest, InterpolateVloc_QEAlignment_Si) {
    // Test interpolation at grid points - should match QE tab_vloc exactly
    // Note: QE stores short-range part, DFTcu adds Coulomb correction
    // Important: QE uses 1-based indexing, so gmod = iq * dq corresponds to tab[iq+1]
    int stride = static_cast<int>(si_data_.tab.size());
    double omega_ha = si_data_.omega * 0.5;  // Convert Bohr^3 (QE uses Ry units for omega too)

    double* d_tab;
    cudaMalloc(&d_tab, stride * sizeof(double));
    cudaMemcpy(d_tab, si_data_.tab.data(), stride * sizeof(double), cudaMemcpyHostToDevice);

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    // Test at several grid points
    // Note: For gmod = iq * dq, DFTcu accesses tab[iq+1] due to QE's 1-based indexing
    std::vector<int> test_iqs = {1, 10, 50, 100, 200};

    for (int iq : test_iqs) {
        if (iq + 1 >= stride - 4)
            continue;  // Skip if too close to boundary

        double gmod = iq * si_data_.dq;  // |G| in 2π/Bohr

        test_interpolate_kernel<<<1, 1>>>(gmod, d_tab, stride, si_data_.zp, omega_ha, d_result);
        cudaDeviceSynchronize();

        double h_result;
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        // At grid points with px=0, interpolation gives tab[iq+1] (QE 1-based indexing)
        // Coulomb correction: -(4π * Z / (Ω * G²)) * exp(-G²/4)
        double g2 = gmod * gmod;
        double fpi = 4.0 * M_PI;
        double coulomb_corr = -(fpi * si_data_.zp / (omega_ha * g2)) * std::exp(-0.25 * g2);

        // Use tab[iq+1] for the expected short-range part
        double expected = si_data_.tab[iq + 1] + coulomb_corr;

        // Allow small interpolation error at grid points
        EXPECT_NEAR(h_result, expected, std::abs(expected) * 1e-10)
            << "Mismatch at iq=" << iq << ", gmod=" << gmod;
    }

    cudaFree(d_tab);
    cudaFree(d_result);
}

TEST_F(LocalPseudoKernelsTest, InterpolateVloc_QEAlignment_C) {
    // Same test for Carbon
    // Important: QE uses 1-based indexing, so gmod = iq * dq corresponds to tab[iq+1]
    int stride = static_cast<int>(c_data_.tab.size());
    double omega_ha = c_data_.omega * 0.5;

    double* d_tab;
    cudaMalloc(&d_tab, stride * sizeof(double));
    cudaMemcpy(d_tab, c_data_.tab.data(), stride * sizeof(double), cudaMemcpyHostToDevice);

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    std::vector<int> test_iqs = {1, 10, 50, 100, 200};

    for (int iq : test_iqs) {
        if (iq + 1 >= stride - 4)
            continue;

        double gmod = iq * c_data_.dq;

        test_interpolate_kernel<<<1, 1>>>(gmod, d_tab, stride, c_data_.zp, omega_ha, d_result);
        cudaDeviceSynchronize();

        double h_result;
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        double g2 = gmod * gmod;
        double fpi = 4.0 * M_PI;
        double coulomb_corr = -(fpi * c_data_.zp / (omega_ha * g2)) * std::exp(-0.25 * g2);

        // Use tab[iq+1] for the expected short-range part
        double expected = c_data_.tab[iq + 1] + coulomb_corr;

        EXPECT_NEAR(h_result, expected, std::abs(expected) * 1e-10)
            << "Mismatch at iq=" << iq << ", gmod=" << gmod;
    }

    cudaFree(d_tab);
    cudaFree(d_result);
}

}  // namespace test
}  // namespace dftcu
