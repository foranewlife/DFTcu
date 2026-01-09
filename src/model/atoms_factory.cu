#include "model/atoms_factory.cuh"

namespace dftcu {

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

}  // namespace dftcu
