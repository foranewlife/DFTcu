#!/usr/bin/env python3
import dftcu
import numpy as np
from dftpy.functional import Functional
from dftpy.grid import DirectGrid


def test_vw_kedf():
    """Verify vW energy against DFTpy for a Gaussian density"""
    # 1. Setup Grid
    lattice = np.eye(3) * 10.0
    nr = [32, 32, 32]
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Gaussian density: rho(r) = exp(-r^2)
    # Use grid.r for coordinates in newer DFTpy
    r2 = np.sum(dftpy_grid.r**2, axis=0)
    rho_data = np.exp(-r2)
    from dftpy.field import DirectField

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
