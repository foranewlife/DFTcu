#!/usr/bin/env python3
import dftcu
import numpy as np
from dftpy.functional import Hartree as DFTpy_Hartree
from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_hartree_and_pseudo():
    """Verify Hartree and Pseudopotential energies against DFTpy (Compatible version)"""
    # 1. Setup Grid and Ions (Al bulk)
    lattice = np.eye(3) * 7.65
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    rho_val = 3.0 / lattice[0, 0] ** 3
    from dftpy.field import DirectField

    rho_dftpy = DirectField(grid=dftpy_grid, data=np.full(nr, rho_val))

    # 2. DFTpy Reference
    print("\n1. Running DFTpy Hartree/Pseudo Reference...")
    # Hartree
    result_dftpy = DFTpy_Hartree().compute(rho_dftpy, calcType={"E", "V"})
    e_h_py = result_dftpy.energy

    # Pseudo
    import os

    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.upf")
    pseudo_dftpy = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})
    result_dftpy = pseudo_dftpy.compute(rho_dftpy, calcType={"E", "V"})
    e_ps_py = result_dftpy.energy

    # 3. DFTcu Implementation
    print("2. Running DFTcu (CUDA) Hartree/Pseudo...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])

    hartree_cu = dftcu.Hartree(grid_cu)
    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)

    # Use internal vlines if present
    vloc_g = pseudo_dftpy.vlines["Al"].flatten(order="C").tolist()
    pseudo_cu.set_vloc(0, vloc_g)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_dftpy.flatten(order="C"))
    vh_cu = dftcu.RealField(grid_cu, 1)
    vps_cu = dftcu.RealField(grid_cu, 1)

    # Use basic compute if unified one is missing in binary
    try:
        e_h_cu = hartree_cu.compute(rho_cu, vh_cu)
    except TypeError:
        # Fallback for some versions
        hartree_cu.compute(rho_cu, vh_cu)
        e_h_cu = 0.5 * rho_cu.dot(vh_cu) * grid_cu.dv()

    try:
        # Try unified first
        e_ps_cu = pseudo_cu.compute(rho_cu, vps_cu)
    except TypeError:
        # Fallback to manual energy calculation
        pseudo_cu.compute(vps_cu)
        e_ps_cu = rho_cu.dot(vps_cu) * grid_cu.dv()

    print(f"Hartree: DFTpy={e_h_py:.10f}, DFTcu={e_h_cu:.10f}")
    print(f"Pseudo:  DFTpy={e_ps_py:.10f}, DFTcu={e_ps_cu:.10f}")

    assert abs(e_h_py - e_h_cu) < 1e-8
    assert abs(e_ps_py - e_ps_cu) < 1e-8
    print("✓ Hartree and Pseudo Cross-Verification Passed")


if __name__ == "__main__":
    test_hartree_and_pseudo()
