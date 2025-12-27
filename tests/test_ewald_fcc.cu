#include <iostream>
#include <memory>
#include <vector>

#include "functional/ewald.cuh"
#include "model/atoms.cuh"
#include "model/grid.cuh"

using namespace dftcu;

int main() {
    double a0 = 7.6;
    std::vector<double> lattice = {0, a0 / 2, a0 / 2, a0 / 2, 0, a0 / 2, a0 / 2, a0 / 2, 0};
    std::vector<int> nr = {32, 32, 32};
    auto grid = std::make_shared<Grid>(lattice, nr);

    std::vector<Atom> atoms_vec;
    Atom a;
    a.x = 0;
    a.y = 0;
    a.z = 0;
    a.charge = 3.0;
    a.type = 0;
    atoms_vec.push_back(a);
    auto atoms = std::make_shared<Atoms>(atoms_vec);

    Ewald ewald(grid, atoms);
    double energy = ewald.compute(false);
    std::cout << "Final Ewald Energy: " << energy << std::endl;

    return 0;
}
