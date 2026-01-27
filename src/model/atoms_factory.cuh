#pragma once

#include <map>
#include <memory>
#include <string>
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

/**
 * @brief Create Atoms from structure data (supports fractional coordinates)
 *
 * This factory handles:
 * 1. Element symbols → type index mapping
 * 2. Fractional coordinates → Cartesian coordinates conversion
 * 3. Angstrom → Bohr conversion
 * 4. Atoms object construction
 *
 * @param elements Element symbols list, e.g., ["Si", "Si", "O"]
 * @param positions Atomic positions (Angstrom or fractional)
 * @param lattice_vectors Lattice vectors (Angstrom), 3x3 matrix
 * @param cartesian true = Cartesian coordinates (Angstrom), false = fractional
 * @param unique_elements Unique element list, e.g., ["Si", "O"]
 * @param valence_electrons Valence electrons per element, e.g., {"Si": 4.0, "O": 6.0}
 * @return shared_ptr to Atoms object (internal positions in Bohr)
 *
 * Example:
 *   auto atoms = create_atoms_from_structure(
 *       {"Si", "Si"},                                    // elements
 *       {{0.0, 0.0, 0.0}, {0.25, 0.25, 0.25}},          // positions (fractional)
 *       {{5.43, 0, 0}, {0, 5.43, 0}, {0, 0, 5.43}},     // lattice (Angstrom)
 *       false,                                           // fractional coordinates
 *       {"Si"},                                          // unique elements
 *       {{"Si", 4.0}}                                    // valence electrons
 *   );
 */
std::shared_ptr<Atoms> create_atoms_from_structure(
    const std::vector<std::string>& elements, const std::vector<std::vector<double>>& positions,
    const std::vector<std::vector<double>>& lattice_vectors, bool cartesian,
    const std::vector<std::string>& unique_elements,
    const std::map<std::string, double>& valence_electrons);

}  // namespace dftcu
