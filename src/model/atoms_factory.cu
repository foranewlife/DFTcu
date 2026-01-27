#include <algorithm>
#include <stdexcept>

#include "model/atoms_factory.cuh"

namespace dftcu {

// ═══════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════

/**
 * @brief Convert fractional coordinates to Cartesian coordinates
 * @param frac Fractional coordinates [x, y, z]
 * @param lattice Lattice vectors (Angstrom), 3x3 matrix
 * @return Cartesian coordinates (Angstrom)
 */
static std::vector<double> fractional_to_cartesian(
    const std::vector<double>& frac, const std::vector<std::vector<double>>& lattice) {
    std::vector<double> cart(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cart[i] += frac[j] * lattice[j][i];
        }
    }
    return cart;
}

// ═══════════════════════════════════════════════════════════
// Factory implementations
// ═══════════════════════════════════════════════════════════

std::shared_ptr<Atoms> create_atoms_from_angstrom(const std::vector<Atom>& atoms_ang) {
    // Convert positions from Angstrom to Bohr
    std::vector<Atom> atoms_bohr;
    atoms_bohr.reserve(atoms_ang.size());

    for (const auto& atom : atoms_ang) {
        Atom atom_bohr;
        atom_bohr.x = atom.x * constants::ANGSTROM_TO_BOHR;
        atom_bohr.y = atom.y * constants::ANGSTROM_TO_BOHR;
        atom_bohr.z = atom.z * constants::ANGSTROM_TO_BOHR;
        atom_bohr.charge = atom.charge;  // No conversion for charge
        atom_bohr.type = atom.type;      // No conversion for type
        atoms_bohr.push_back(atom_bohr);
    }

    // Create Atoms object with Bohr positions
    return std::make_shared<Atoms>(atoms_bohr);
}

std::shared_ptr<Atoms> create_atoms_from_bohr(const std::vector<Atom>& atoms_bohr) {
    // No unit conversion - positions already in Bohr
    return std::make_shared<Atoms>(atoms_bohr);
}

std::shared_ptr<Atoms> create_atoms_from_structure(
    const std::vector<std::string>& elements, const std::vector<std::vector<double>>& positions,
    const std::vector<std::vector<double>>& lattice_vectors, bool cartesian,
    const std::vector<std::string>& unique_elements,
    const std::map<std::string, double>& valence_electrons) {
    // Validate input
    if (elements.size() != positions.size()) {
        throw std::invalid_argument(
            "create_atoms_from_structure: elements.size() != positions.size()");
    }
    if (lattice_vectors.size() != 3) {
        throw std::invalid_argument("create_atoms_from_structure: lattice_vectors must be 3x3");
    }
    for (const auto& vec : lattice_vectors) {
        if (vec.size() != 3) {
            throw std::invalid_argument(
                "create_atoms_from_structure: each lattice vector must have 3 components");
        }
    }
    if (unique_elements.empty()) {
        throw std::invalid_argument("create_atoms_from_structure: unique_elements is empty");
    }

    // Build Atom list
    std::vector<Atom> atom_list;
    atom_list.reserve(elements.size());

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        const auto& pos = positions[i];

        if (pos.size() != 3) {
            throw std::invalid_argument(
                "create_atoms_from_structure: position must have 3 components");
        }

        // Find type index
        auto it = std::find(unique_elements.begin(), unique_elements.end(), elem);
        if (it == unique_elements.end()) {
            throw std::invalid_argument("create_atoms_from_structure: element '" + elem +
                                        "' not in unique_elements");
        }
        int type = std::distance(unique_elements.begin(), it);

        // Get valence electrons (charge)
        double charge = 0.0;
        auto charge_it = valence_electrons.find(elem);
        if (charge_it != valence_electrons.end()) {
            charge = charge_it->second;
        }

        // Coordinate conversion
        std::vector<double> cart_pos = pos;
        if (!cartesian) {
            // Fractional → Cartesian (Angstrom)
            cart_pos = fractional_to_cartesian(pos, lattice_vectors);
        }

        // Add atom (positions in Angstrom)
        atom_list.push_back({cart_pos[0], cart_pos[1], cart_pos[2], charge, type});
    }

    // Call existing factory function (Angstrom → Bohr)
    return create_atoms_from_angstrom(atom_list);
}

}  // namespace dftcu
