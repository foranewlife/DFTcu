import dftcu
import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.grid import DirectGrid
from test_utils import get_system


def test_vw_kedf():
    """Verify vW energy against DFTpy for a Gaussian density"""
    nr = [32, 32, 32]
    ions = get_system("Al_single", a=10.0)
    lattice = ions.cell.array

    # 1. Setup Grid
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Gaussian density: rho(r) = exp(-r^2)
    # Using grid.r for coordinates
    r2 = np.sum(dftpy_grid.r**2, axis=0)
    rho_data = np.exp(-r2)
    rho_dftpy = DirectField(grid=dftpy_grid, data=rho_data)

    # 3. DFTpy vW
    vw_py = Functional(type="KEDF", name="vW")
    e_vw_py = vw_py(rho_dftpy).energy

    # 4. DFTcu vW
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_data.flatten(order="C"))

    vw_cu = dftcu.vonWeizsacker(coeff=1.0)
    v_vw_cu = dftcu.RealField(grid_cu, 1)
    e_vw_cu = vw_cu.compute(rho_cu, v_vw_cu)

    print(f"DFTpy vW Energy: {e_vw_py:.10f}")
    print(f"DFTcu vW Energy: {e_vw_cu:.10f}")

    assert abs(e_vw_py - e_vw_cu) < 1e-7
    print("✓ vW KEDF Verification Passed")


if __name__ == "__main__":
    test_vw_kedf()
