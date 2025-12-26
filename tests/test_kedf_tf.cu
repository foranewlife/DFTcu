#include <cmath>
#include <memory>
#include <vector>

#include "functional/kedf/tf.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "test_utils.h"

#include <gtest/gtest.h>

namespace dftcu {
namespace testing {

class ThomasFermiTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple cubic grid
        std::vector<double> lattice = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
        std::vector<int> nr = {16, 16, 16};
        grid_ = std::make_shared<Grid>(lattice, nr);
    }

    std::shared_ptr<Grid> grid_;
};

TEST_F(ThomasFermiTest, UniformDensity) {
    // Test with uniform density
    RealField rho(grid_);
    RealField v_kedf(grid_);

    const double rho0 = 0.01;  // uniform density
    rho.fill(rho0);

    ThomasFermi tf(1.0);
    double energy = tf.compute(rho, v_kedf);

    // For uniform density: E_TF = V * C_TF * rho^(5/3)
    const double pi = 3.14159265358979323846;
    const double c_tf = (3.0 / 10.0) * std::pow(3.0 * pi * pi, 2.0 / 3.0);
    double expected_energy = grid_->volume() * c_tf * std::pow(rho0, 5.0 / 3.0);

    // V_TF should be constant: (5/3) * C_TF * rho^(2/3)
    double expected_potential = (5.0 / 3.0) * c_tf * std::pow(rho0, 2.0 / 3.0);

    // Check energy
    EXPECT_NEAR(energy, expected_energy, 1e-6 * std::abs(expected_energy));

    // Check potential (sample a few points)
    std::vector<double> v_host(grid_->nnr());
    v_kedf.copy_to_host(v_host.data());

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_NEAR(v_host[i], expected_potential, 1e-6 * expected_potential);
    }
}

TEST_F(ThomasFermiTest, ScalingCoefficient) {
    // Test coefficient scaling
    RealField rho(grid_);
    RealField v_kedf1(grid_);
    RealField v_kedf2(grid_);

    const double rho0 = 0.02;
    rho.fill(rho0);

    // Compute with coeff=1.0
    ThomasFermi tf1(1.0);
    double energy1 = tf1.compute(rho, v_kedf1);

    // Compute with coeff=2.0
    ThomasFermi tf2(2.0);
    double energy2 = tf2.compute(rho, v_kedf2);

    // Energy should scale linearly
    EXPECT_NEAR(energy2, 2.0 * energy1, 1e-10 * std::abs(energy1));

    // Potential should also scale
    std::vector<double> v1(grid_->nnr());
    std::vector<double> v2(grid_->nnr());
    v_kedf1.copy_to_host(v1.data());
    v_kedf2.copy_to_host(v2.data());

    for (size_t i = 0; i < grid_->nnr(); ++i) {
        EXPECT_NEAR(v2[i], 2.0 * v1[i], 1e-10 * std::abs(v1[i]));
    }
}

TEST_F(ThomasFermiTest, NonUniformDensity) {
    // Test with non-uniform density
    std::vector<double> rho_host(grid_->nnr());
    create_test_density(rho_host.data(), 16, 16, 16, 0.01);

    RealField rho(grid_);
    RealField v_kedf(grid_);
    rho.copy_from_host(rho_host.data());

    ThomasFermi tf(1.0);
    double energy = tf.compute(rho, v_kedf);

    // Energy should be positive
    EXPECT_GT(energy, 0.0);

    // Potential should vary spatially
    std::vector<double> v_host(grid_->nnr());
    v_kedf.copy_to_host(v_host.data());

    double v_min = v_host[0];
    double v_max = v_host[0];
    for (const auto& v : v_host) {
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
    }

    // Potential should have spatial variation
    EXPECT_GT(v_max - v_min, 1e-6);
}

}  // namespace testing
}  // namespace dftcu
