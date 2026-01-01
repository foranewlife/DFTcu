import numpy as np
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from test_utils import get_pp_path, get_system, setup_pseudo, to_dftcu_atoms

import dftcu


def test_pseudo_fcc():
    """Compare LocalPseudo energy and potential with DFTpy in FCC cell"""
    nr = [32, 32, 32]
    ions = get_system("Al_primitive", a=7.6)
    lattice = ions.cell.array

    # 1. DFTpy Calculation
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    pp_file = get_pp_path("al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as LP

    pseudo_py = LP(grid=grid_py, ions=ions, PP_list={"Al": pp_file}, PME=False)

    rho_py = DirectField(grid=grid_py, data=np.full(nr, 1.0))
    out_py = pseudo_py.compute(rho_py)

    # 2. DFTcu Calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = to_dftcu_atoms(ions)

    ps_cu, _ = setup_pseudo(grid_cu, atoms_cu, "al.lda.upf", ions)

    rho_cu = dftcu.RealField(grid_cu)
    rho_cu.fill(1.0)
    v_cu_obj = dftcu.RealField(grid_cu)
    v_cu_obj.fill(0.0)  # Clear first
    e_cu = ps_cu.compute(rho_cu, v_cu_obj)

    v_cu = np.zeros(grid_cu.nnr())
    v_cu_obj.copy_to_host(v_cu)
    v_cu = v_cu.reshape(nr)

    # 3. Comparison
    print(f"Pseudo Energy Py: {out_py.energy:.12f}, Cu: {e_cu:.12f}")
    assert abs(out_py.energy - e_cu) < 1e-10

    diff_v = np.abs(out_py.potential - v_cu)
    print(f"Pseudo Potential Max Diff: {diff_v.max():.2e}")
    # Small difference expected due to interpolation and cutoff handling
    assert diff_v.max() < 1e-3


if __name__ == "__main__":
    test_pseudo_fcc()
