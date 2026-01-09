#include "grid_factory.cuh"
#include "utilities/constants.cuh"

namespace dftcu {

std::unique_ptr<Grid> create_grid_from_qe(const std::vector<std::vector<double>>& lattice_ang,
                                          const std::vector<int>& nr, double ecutwfc_ry,
                                          double ecutrho_ry, bool is_gamma) {
    // Validate input: lattice must be 3×3
    if (lattice_ang.size() != 3) {
        throw std::runtime_error("lattice_ang must be 3×3 matrix (got " +
                                 std::to_string(lattice_ang.size()) + " rows)");
    }
    for (size_t i = 0; i < 3; ++i) {
        if (lattice_ang[i].size() != 3) {
            throw std::runtime_error("lattice_ang row " + std::to_string(i) +
                                     " must have 3 elements");
        }
    }

    // Convert lattice from Angstrom to Bohr and flatten to row-major
    const double ANG_TO_BOHR = 1.0 / constants::BOHR_TO_ANGSTROM;
    std::vector<double> lattice_bohr(9);
    printf("DEBUG grid_factory: Input lattice (Ang):\n");
    for (int i = 0; i < 3; ++i) {
        printf("  [%12.6f, %12.6f, %12.6f]\n", lattice_ang[i][0], lattice_ang[i][1],
               lattice_ang[i][2]);
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            lattice_bohr[i * 3 + j] = lattice_ang[i][j] * ANG_TO_BOHR;
        }
    }
    printf("DEBUG grid_factory: Flattened lattice (Bohr): [");
    for (int k = 0; k < 9; ++k) {
        printf("%.2f%s", lattice_bohr[k], (k < 8 ? ", " : "]\n"));
    }

    // Convert cutoff energies from Rydberg to Hartree
    const double RY_TO_HA = 0.5;
    double ecutwfc_ha = ecutwfc_ry * RY_TO_HA;
    double ecutrho_ha = (ecutrho_ry > 0) ? ecutrho_ry * RY_TO_HA : -1.0;

    // Create Grid with atomic units
    return std::make_unique<Grid>(lattice_bohr, nr, ecutwfc_ha, ecutrho_ha, is_gamma);
}

std::unique_ptr<Grid> create_grid_from_atomic_units(
    const std::vector<std::vector<double>>& lattice_bohr_2d, const std::vector<int>& nr,
    double ecutwfc_ha, double ecutrho_ha, bool is_gamma) {
    // Validate input: lattice must be 3×3
    if (lattice_bohr_2d.size() != 3) {
        throw std::runtime_error("lattice_bohr must be 3×3 matrix (got " +
                                 std::to_string(lattice_bohr_2d.size()) + " rows)");
    }
    for (size_t i = 0; i < 3; ++i) {
        if (lattice_bohr_2d[i].size() != 3) {
            throw std::runtime_error("lattice_bohr row " + std::to_string(i) +
                                     " must have 3 elements");
        }
    }

    // Flatten to row-major
    std::vector<double> lattice_bohr(9);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            lattice_bohr[i * 3 + j] = lattice_bohr_2d[i][j];
        }
    }

    // Direct pass-through - no conversions
    return std::make_unique<Grid>(lattice_bohr, nr, ecutwfc_ha, ecutrho_ha, is_gamma);
}

}  // namespace dftcu
