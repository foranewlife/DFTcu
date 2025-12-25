#pragma once

#include <cstddef>

namespace dftcu {
namespace testing {

/**
 * @brief Compare two arrays with relative and absolute tolerance
 */
bool compare_arrays(const double* a, const double* b, size_t n, double rtol = 1e-5,
                    double atol = 1e-8);

/**
 * @brief Create a test density field with Gaussian perturbation
 */
void create_test_density(double* rho, int nx, int ny, int nz, double rho0 = 0.01);

}  // namespace testing
}  // namespace dftcu
