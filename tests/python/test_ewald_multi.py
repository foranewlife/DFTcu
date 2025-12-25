#!/usr/bin/env python3
import dftcu
import numpy as np
from dftpy.ewald import ewald as DFTpy_Ewald
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_ewald_multi_atoms():
    """Verify Ewald energy for a 32-atom Aluminum supercell (2x2x2)"""
    print("\n" + "=" * 80)
    print("Multi-Atom Ewald Verification (32 Atoms)")
    print("=" * 80)

    # 1. Setup 2x2x2 Al Supercell
    a0 = 7.65  # Bohr
    lattice = np.eye(3) * a0 * 2.0

    # FCC positions in primitive cell
    base_pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])

    # Generate 2x2x2 supercell (4 * 8 = 32 atoms)
    all_pos = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                offset = np.array([ix, iy, iz]) * 0.5
                all_pos.extend((base_pos * 0.5 + offset))

    all_pos = np.array(all_pos) * lattice[0, 0]  # Convert to Bohr
    ions = Ions(symbols=["Al"] * 32, positions=all_pos, cell=lattice)
    ions.set_charges(3.0)

    nr = [64, 64, 64]
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. DFTpy Ewald (Exact & PME)
    print("1. Running DFTpy Ewald (Exact & PME)...")
    ewald_py = DFTpy_Ewald(grid=dftpy_grid, ions=ions, PME=False)
    e_ewald_py_exact = ewald_py.energy

    ewald_py_pme = DFTpy_Ewald(grid=dftpy_grid, ions=ions, PME=True)
    e_ewald_py_pme = ewald_py_pme.energy
    print(f"   DFTpy Energy (Exact): {e_ewald_py_exact:.10f} Ha")
    print(f"   DFTpy Energy (PME):   {e_ewald_py_pme:.10f} Ha")

    # 3. DFTcu Ewald (Exact & PME)
    print("2. Running DFTcu Ewald (Exact & PME)...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)

    atoms_list = []
    for i in range(32):
        atoms_list.append(dftcu.Atom(all_pos[i, 0], all_pos[i, 1], all_pos[i, 2], 3.0, 0))
    atoms_cu = dftcu.Atoms(atoms_list)

    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)
    e_ewald_cu_exact = ewald_cu.compute(use_pme=False)
    e_ewald_cu_pme = ewald_cu.compute(use_pme=True)
    print(f"   DFTcu Energy (Exact): {e_ewald_cu_exact:.10f} Ha")
    print(f"   DFTcu Energy (PME):   {e_ewald_cu_pme:.10f} Ha")

    # 4. Final Comparison
    print("\n3. Final Comparison (Cross-Validation):")
    print(f"   Exact Difference (DFTcu - DFTpy): {abs(e_ewald_cu_exact - e_ewald_py_exact):.2e} Ha")
    print(f"   PME Difference   (DFTcu - DFTpy): {abs(e_ewald_cu_pme - e_ewald_py_pme):.2e} Ha")

    # Check if PME implementations match each other
    assert abs(e_ewald_cu_exact - e_ewald_py_exact) < 1e-10
    assert abs(e_ewald_cu_pme - e_ewald_py_pme) < 2e-5
    print("\n✓ Multi-Atom PME Cross-Verification Passed")


if __name__ == "__main__":
    test_ewald_multi_atoms()
