#!/usr/bin/env python3
import dftcu
import numpy as np
from dftpy.ewald import ewald as DFTpy_Ewald
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_ewald_energy():
    """Verify Ewald energy against DFTpy for Aluminum bulk"""
    # 1. Setup System
    lattice = np.eye(3) * 7.65
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. DFTpy Ewald
    ewald_py = DFTpy_Ewald(grid=dftpy_grid, ions=ions)
    e_ewald_py = ewald_py.energy
    print(f"\nDFTpy Ewald Energy: {e_ewald_py:.10f} Ha")
    print(f"   Real:  {ewald_py.Energy_real():.10f}")
    print(f"   Recip: {ewald_py.Energy_rec():.10f}")
    print(f"   Corr:  {ewald_py.Energy_corr():.10f}")
    print(f"   Eta:   {ewald_py.eta:.6f}")

    # 3. DFTcu Ewald
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])
    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)
    e_ewald_cu = ewald_cu.compute()
    print(f"DFTcu Ewald Energy: {e_ewald_cu:.10f} Ha")

    # 4. Comparison
    abs_diff = abs(e_ewald_py - e_ewald_cu)
    print(f"Absolute Difference: {abs_diff:.2e} Ha")

    # Check if they match
    assert abs_diff < 1e-6


if __name__ == "__main__":
    test_ewald_energy()
