import dftcu
from dftpy.ewald import ewald as EwaldPy
from dftpy.grid import DirectGrid
from test_utils import get_system, to_dftcu_atoms


def test_ewald_fcc():
    """Compare Ewald energy with DFTpy in FCC cell"""
    nr = [32, 32, 32]
    ions = get_system("Al_primitive", a=7.6)
    lattice = ions.cell.array

    # 1. DFTpy Calculation
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    ew_py = EwaldPy(ions=ions, grid=grid_py, PME=False)
    energy_py = ew_py.energy

    # 2. DFTcu Calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = to_dftcu_atoms(ions)

    ew_cu = dftcu.Ewald(grid_cu, atoms_cu)
    # Match eta
    ew_cu.set_eta(ew_py.eta)
    energy_cu = ew_cu.compute(False)

    # 3. Comparison
    print(f"Ewald Energy Py: {energy_py:.12f}, Cu: {energy_cu:.12f}")
    assert abs(energy_py - energy_cu) < 1e-12


if __name__ == "__main__":
    test_ewald_fcc()
