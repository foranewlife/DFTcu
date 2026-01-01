import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Hartree as DFTpy_Hartree
from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo
from dftpy.grid import DirectGrid
from test_utils import get_pp_path, get_system, setup_pseudo, to_dftcu_atoms

import dftcu


def test_hartree_and_pseudo():
    """Verify Hartree and Pseudopotential energies against DFTpy (Compatible version)"""
    # 1. Setup Grid and Ions (Al bulk)
    nr = [32, 32, 32]
    ions = get_system("Al_single", a=7.65)
    lattice = ions.cell.array

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    rho_val = 3.0 / ions.cell.volume
    rho_dftpy = DirectField(grid=dftpy_grid, data=np.full(nr, rho_val))

    # 2. DFTpy Reference
    print("\n1. Running DFTpy Hartree/Pseudo Reference...")
    # Hartree
    result_dftpy = DFTpy_Hartree().compute(rho_dftpy, calcType={"E", "V"})
    e_h_py = result_dftpy.energy

    # Pseudo
    pp_file = get_pp_path("al.lda.upf")
    pseudo_dftpy = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})
    result_dftpy = pseudo_dftpy.compute(rho_dftpy, calcType={"E", "V"})
    e_ps_py = result_dftpy.energy

    # 3. DFTcu Implementation
    print("2. Running DFTcu (CUDA) Hartree/Pseudo...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = to_dftcu_atoms(ions)

    hartree_cu = dftcu.Hartree(grid_cu)
    pseudo_cu, _ = setup_pseudo(grid_cu, atoms_cu, "al.lda.upf", ions)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_dftpy.flatten(order="C"))
    vh_cu = dftcu.RealField(grid_cu, 1)
    vps_cu = dftcu.RealField(grid_cu, 1)

    e_h_cu = hartree_cu.compute(rho_cu, vh_cu)
    e_ps_cu = pseudo_cu.compute(rho_cu, vps_cu)

    print(f"Hartree: DFTpy={e_h_py:.10f}, DFTcu={e_h_cu:.10f}")
    print(f"Pseudo:  DFTpy={e_ps_py:.10f}, DFTcu={e_ps_cu:.10f}")

    assert abs(e_h_py - e_h_cu) < 1e-8
    assert abs(e_ps_py - e_ps_cu) < 1e-8
    print("✓ Hartree and Pseudo Cross-Verification Passed")


if __name__ == "__main__":
    test_hartree_and_pseudo()
