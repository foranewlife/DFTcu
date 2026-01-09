#pragma once
#include <memory>
#include <vector>

#include "grid.cuh"

namespace dftcu {

/**
 * @file grid_factory.cuh
 * @brief Factory functions for creating Grid objects with explicit unit conversions.
 *
 * These factory functions provide a clear, type-safe interface for creating Grid objects
 * from different unit systems. All unit conversions happen at the factory boundary,
 * ensuring the Grid class remains pure with only atomic units internally.
 */

/**
 * @brief Create Grid from QE-compatible units (Angstrom + Rydberg).
 *
 * This is the recommended factory for users working with Quantum ESPRESSO data.
 * All parameters use QE's native units for convenience.
 *
 * @param lattice_ang 3×3 lattice matrix in ANGSTROM
 *                    Each row is a lattice vector: [[a1x,a1y,a1z], [a2x,a2y,a2z], [a3x,a3y,a3z]]
 * @param nr FFT grid dimensions [nr1, nr2, nr3]
 * @param ecutwfc_ry Wavefunction cutoff energy in RYDBERG
 * @param ecutrho_ry Density cutoff energy in RYDBERG (default: 4*ecutwfc_ry)
 * @param is_gamma True for Gamma-only calculation
 * @return Unique pointer to Grid object (internal: Bohr + Hartree)
 *
 * @note Conversions applied:
 *       - lattice: Angstrom → Bohr (× 1/0.529177)
 *       - ecutwfc: Rydberg → Hartree (× 0.5)
 *       - ecutrho: Rydberg → Hartree (× 0.5)
 *
 * @example
 * ```cpp
 * // QE input: ecutwfc = 12.0 Ry, lattice in Angstrom
 * std::vector<std::vector<double>> lattice = {{10,0,0}, {0,10,0}, {0,0,10}};
 * auto grid = create_grid_from_qe(lattice, {18,18,18}, 12.0, 48.0, true);
 * // Internal: ecutwfc = 6.0 Ha, lattice in Bohr
 * ```
 */
std::unique_ptr<Grid> create_grid_from_qe(const std::vector<std::vector<double>>& lattice_ang,
                                          const std::vector<int>& nr, double ecutwfc_ry,
                                          double ecutrho_ry = -1.0, bool is_gamma = false);

/**
 * @brief Create Grid from atomic units (Bohr + Hartree).
 *
 * This factory is for advanced users who work directly in atomic units.
 * No unit conversions are performed.
 *
 * @param lattice_bohr 3×3 lattice matrix in BOHR
 *                     Each row is a lattice vector: [[a1x,a1y,a1z], [a2x,a2y,a2z], [a3x,a3y,a3z]]
 * @param nr FFT grid dimensions [nr1, nr2, nr3]
 * @param ecutwfc_ha Wavefunction cutoff energy in HARTREE
 * @param ecutrho_ha Density cutoff energy in HARTREE (default: 4*ecutwfc_ha)
 * @param is_gamma True for Gamma-only calculation
 * @return Unique pointer to Grid object
 *
 * @note No conversions applied - direct pass-through to Grid constructor.
 *
 * @example
 * ```cpp
 * // Direct atomic units
 * std::vector<std::vector<double>> lattice = {{10,0,0}, {0,10,0}, {0,0,10}};
 * auto grid = create_grid_from_atomic_units(lattice, {18,18,18}, 6.0, 24.0, true);
 * ```
 */
std::unique_ptr<Grid> create_grid_from_atomic_units(
    const std::vector<std::vector<double>>& lattice_bohr, const std::vector<int>& nr,
    double ecutwfc_ha, double ecutrho_ha = -1.0, bool is_gamma = false);

}  // namespace dftcu
