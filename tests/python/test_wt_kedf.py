import dftcu
import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.grid import DirectGrid
from test_utils import get_system


def test_wt_kedf():
    """Verify Wang-Teter non-local KEDF energy against DFTpy"""
    nr = [32, 32, 32]
    ions = get_system("Al_single", a=10.0)
    lattice = ions.cell.array

    # 1. Setup Grid
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Gaussian density: rho(r) = exp(-r^2) + offset
    r2 = np.sum(dftpy_grid.r**2, axis=0)
    rho_data = np.exp(-r2) + 0.02
    rho_dftpy = DirectField(grid=dftpy_grid, data=rho_data)

    # 3. DFTpy WT-NL
    wt_py = Functional(type="KEDF", name="WT-NL")
    e_wt_py = wt_py(rho_dftpy).energy

    # 4. DFTcu WT (NL part)
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_data.flatten(order="C"))

    wt_cu = dftcu.WangTeter(coeff=1.0)
    v_wt_cu = dftcu.RealField(grid_cu, 1)
    e_wt_cu = wt_cu.compute(rho_cu, v_wt_cu)

    print(f"DFTpy WT Energy: {e_wt_py:.10f}")
    print(f"DFTcu WT Energy: {e_wt_cu:.10f}")

    assert abs(e_wt_py - e_wt_cu) < 1e-7
    print("✓ WT KEDF Verification Passed")


if __name__ == "__main__":
    test_wt_kedf()
