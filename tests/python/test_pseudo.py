import dftcu
import numpy as np
from dftpy.field import DirectField
from dftpy.functional.pseudo import LocalPseudo as LP
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_pseudo_fcc():
    """Compare LocalPseudo energy and potential with DFTpy in FCC cell"""
    a0 = 7.6
    nr = [32, 32, 32]
    lattice = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    # 1. DFTpy Calculation
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    pp_file = "external/DFTpy/examples/DATA/al.lda.upf"
    pseudo_py = LP(grid=grid_py, ions=ions, PP_list={"Al": pp_file}, PME=False)

    rho_py = DirectField(grid=grid_py, data=np.full(nr, 1.0))
    out_py = pseudo_py.compute(rho_py)

    # 2. DFTcu Calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])

    ps_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    ps_cu.set_vloc_radial(
        0, pseudo_py.readpp._gp["Al"].tolist(), pseudo_py.readpp._vp["Al"].tolist()
    )

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
