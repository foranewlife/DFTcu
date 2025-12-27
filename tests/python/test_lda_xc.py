import dftcu
import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.grid import DirectGrid


def test_lda_fcc():
    """Compare LDA XC energy and potential with DFTpy in FCC cell"""
    a0 = 7.6
    nr = [32, 32, 32]
    # FCC primitive
    lattice = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])

    # 1. DFTpy Calculation
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    rho_py = DirectField(grid=grid_py)
    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, nr[0], endpoint=False),
        np.linspace(0, 1, nr[1], endpoint=False),
        np.linspace(0, 1, nr[2], endpoint=False),
        indexing="ij",
    )
    rho_py[:] = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / 0.1)
    rho_py *= 3.0 / rho_py.integral()

    xc_py = Functional(type="XC", name="LDA", grid=grid_py)
    out_py = xc_py.compute(rho_py)

    # 2. DFTcu Calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu)
    rho_cu.copy_from_host(rho_py.flatten())

    lda_cu = dftcu.LDA_PZ()
    v_cu_obj = dftcu.RealField(grid_cu)
    e_cu = lda_cu.compute(rho_cu, v_cu_obj)

    v_cu = np.zeros(grid_cu.nnr())
    v_cu_obj.copy_to_host(v_cu)
    v_cu = v_cu.reshape(nr)

    # 3. Comparison
    print(f"LDA Energy Py: {out_py.energy:.12f}, Cu: {e_cu:.12f}")
    assert abs(out_py.energy - e_cu) < 1e-12

    diff_v = np.abs(out_py.potential - v_cu)
    print(f"LDA Potential Max Diff: {diff_v.max():.2e}")
    assert diff_v.max() < 1e-12


if __name__ == "__main__":
    test_lda_fcc()
