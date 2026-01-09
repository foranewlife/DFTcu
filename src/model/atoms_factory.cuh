#pragma once

#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "utilities/constants.cuh"

namespace dftcu {

/**
 * @brief Factory functions for creating Atoms objects with unit conversion
 *
 * DESIGN PHILOSOPHY:
 * - Atoms class is PURE: only accepts Bohr (atomic units)
 * - All unit conversions happen in these factory functions (boundary layer)
 * - Function names explicitly indicate input units
 *
 * This follows the same pattern as Grid factory functions (grid_factory.cuh)
 */

/**
 * @brief Create Atoms from positions in Angstrom
 * @param atoms Vector of Atom structs with positions in ANGSTROM
 * @return shared_ptr to Atoms object (internal positions in Bohr)
 *
 * This is the recommended way to create Atoms from user-friendly Angstrom input.
 *
 * Example:
 *   std::vector<Atom> atoms_ang = {
 *       {0.0, 0.0, 0.0, 4.0, 0},  // Si at origin, Angstrom
 *       {1.35, 1.35, 1.35, 4.0, 0}
 *   };
 *   auto atoms = create_atoms_from_angstrom(atoms_ang);
 */
std::shared_ptr<Atoms> create_atoms_from_angstrom(const std::vector<Atom>& atoms_ang);

/**
 * @brief Create Atoms from positions in Bohr (atomic units)
 * @param atoms_bohr Vector of Atom structs with positions in BOHR
 * @return shared_ptr to Atoms object (internal positions in Bohr)
 *
 * Use this when positions are already in atomic units.
 * No unit conversion is performed.
 *
 * Example:
 *   std::vector<Atom> atoms_bohr = {
 *       {0.0, 0.0, 0.0, 4.0, 0},  // Si at origin, Bohr
 *       {2.55, 2.55, 2.55, 4.0, 0}
 *   };
 *   auto atoms = create_atoms_from_bohr(atoms_bohr);
 */
std::shared_ptr<Atoms> create_atoms_from_bohr(const std::vector<Atom>& atoms_bohr);

}  // namespace dftcu
