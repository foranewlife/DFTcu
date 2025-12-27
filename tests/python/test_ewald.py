import dftcu
import numpy as np
from dftpy.ewald import ewald as EwaldPy
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_ewald_fcc():
    """Compare Ewald energy with DFTpy in FCC cell"""
    a0 = 7.6
    nr = [32, 32, 32]
    lattice = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    # 1. DFTpy Calculation
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    ew_py = EwaldPy(ions=ions, grid=grid_py, PME=False)
    energy_py = ew_py.energy

    # 2. DFTcu Calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])

    ew_cu = dftcu.Ewald(grid_cu, atoms_cu)
    # Match eta
    ew_cu.set_eta(ew_py.eta)
    energy_cu = ew_cu.compute(False)

    # 3. Comparison
    print(f"Ewald Energy Py: {energy_py:.12f}, Cu: {energy_cu:.12f}")
    assert abs(energy_py - energy_cu) < 1e-12


if __name__ == "__main__":
    test_ewald_fcc()
