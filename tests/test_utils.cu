#include <cmath>
#include <iostream>

#include "test_utils.h"

namespace dftcu {
namespace testing {

bool compare_arrays(const double* a, const double* b, size_t n, double rtol, double atol) {
    double max_diff = 0.0;
    double max_rel_diff = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(a[i] - b[i]);
        double rel_diff = diff / (std::abs(a[i]) + 1e-10);

        max_diff = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (diff > atol && rel_diff > rtol) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                      << " (diff=" << diff << ", rel=" << rel_diff << ")\n";
            return false;
        }
    }

    std::cout << "Arrays match (max_diff=" << max_diff << ", max_rel_diff=" << max_rel_diff
              << ")\n";
    return true;
}

void create_test_density(double* rho, int nx, int ny, int nz, double rho0) {
    int center_x = nx / 2;
    int center_y = ny / 2;
    int center_z = nz / 2;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int idx = i * ny * nz + j * nz + k;
                double dx = i - center_x;
                double dy = j - center_y;
                double dz = k - center_z;
                double r2 = dx * dx + dy * dy + dz * dz;
                rho[idx] = rho0 * (1.0 + 0.5 * std::exp(-r2 / 50.0));
            }
        }
    }
}

}  // namespace testing
}  // namespace dftcu
