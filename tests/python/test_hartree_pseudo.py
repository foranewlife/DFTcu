#!/usr/bin/env python3
import os

import dftcu
import numpy as np
import pytest
from dftpy.field import DirectField
from dftpy.functional.hartree import Hartree as DFTpy_Hartree
from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


@pytest.mark.comparison
def test_hartree():
    """Compare Hartree implementation with DFTpy"""
    lattice = np.eye(3) * 10.0
    nr = [32, 32, 32]
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # Create test density (Gaussian)
    center = np.array([5.0, 5.0, 5.0])
    r = dftpy_grid.r
    dr = r - center[:, None, None, None]
    r2 = np.sum(dr**2, axis=0)
    rho_np = np.exp(-r2)

    rho_dftpy = DirectField(grid=dftpy_grid, data=rho_np.copy())

    # DFTpy calculation
    result_dftpy = DFTpy_Hartree.compute(rho_dftpy, calcType={"E", "V"})
    energy_dftpy = result_dftpy.energy
    potential_dftpy = np.array(result_dftpy.potential)

    # DFTcu calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_np.flatten(order="C"))
    vh_cu = dftcu.RealField(grid_cu, 1)

    hartree_cu = dftcu.Hartree(grid_cu)
    energy_cu = hartree_cu.compute(rho_cu, vh_cu)

    potential_cu = np.zeros(grid_cu.nnr())
    vh_cu.copy_to_host(potential_cu)
    potential_cu = potential_cu.reshape(nr, order="C")

    # Comparison
    energy_rel_error = abs(energy_cu - energy_dftpy) / max(abs(energy_dftpy), 1e-10)
    potential_max_diff = np.abs(potential_cu - potential_dftpy).max()

    assert energy_rel_error < 1e-6
    assert potential_max_diff < 1e-5


@pytest.mark.comparison
def test_pseudo():
    """Compare Local Pseudopotential implementation with DFTpy"""
    lattice = np.eye(3) * 10.0
    nr = [32, 32, 32]
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # Use real pseudopotential file from DFTpy examples
    # Relative to project root
    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.recpot")
    if not os.path.exists(pp_file):
        pytest.skip(f"Pseudopotential file {pp_file} not found")

    # Setup ions
    pos = np.array([[2.0, 2.0, 2.0], [7.0, 7.0, 7.0]])
    symbols = ["Al", "Al"]
    ions = Ions(symbols=symbols, positions=pos, cell=lattice)

    # DFTpy calculation
    print("\n1. Running DFTpy LocalPseudo...")
    PP_list = {"Al": pp_file}
    pseudo_dftpy = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list=PP_list)

    # Construct density from atomic charges
    from dftpy.density import DensityGenerator

    generator = DensityGenerator(pseudo=pseudo_dftpy)
    rho_dftpy = generator.guess_rho(ions, grid=dftpy_grid)

    result_dftpy = pseudo_dftpy.compute(rho_dftpy, calcType={"E", "V"})
    energy_dftpy = result_dftpy.energy
    potential_dftpy = np.array(result_dftpy.potential)

    # Extract vloc(G) from DFTpy for Aluminum
    vloc_Al_g = pseudo_dftpy.vlines["Al"]

    # DFTcu calculation
    print("\n2. Running DFTcu LocalPseudo...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_list = [
        dftcu.Atom(pos[0, 0], pos[0, 1], pos[0, 2], 3.0, 0),  # Al
        dftcu.Atom(pos[1, 0], pos[1, 1], pos[1, 2], 3.0, 0),  # Al
    ]
    atoms_cu = dftcu.Atoms(atoms_list)

    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc(0, vloc_Al_g.flatten(order="C").tolist())

    v_cu_field = dftcu.RealField(grid_cu, 1)
    pseudo_cu.compute(v_cu_field)

    potential_cu = np.zeros(grid_cu.nnr())
    v_cu_field.copy_to_host(potential_cu)
    potential_cu = potential_cu.reshape(nr, order="C")

    # Energy calculation in DFTcu: E = integral( rho * v_loc ) dV
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_dftpy.flatten(order="C"))
    energy_cu = rho_cu.dot(v_cu_field) * grid_cu.dv()

    # Comparison
    print(f"   DFTpy Energy: {energy_dftpy:.10f} Ha")
    print(f"   DFTcu Energy: {energy_cu:.10f} Ha")

    energy_rel_error = abs(energy_cu - energy_dftpy) / max(abs(energy_dftpy), 1e-10)
    potential_rel_error = (
        np.abs(potential_cu - potential_dftpy).max() / np.abs(potential_dftpy).max()
    )

    print(f"   Energy rel error: {energy_rel_error:.2e}")
    print(f"   Potential rel error: {potential_rel_error:.2e}")

    assert energy_rel_error < 1e-6
    assert potential_rel_error < 1e-4
    print("\n✓ LocalPseudo tests passed")


if __name__ == "__main__":
    test_hartree()
    test_pseudo()
