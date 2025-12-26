#!/usr/bin/env python3
import dftcu
import numpy as np
from dftpy.functional import Functional
from dftpy.grid import DirectGrid


def test_tf_energy_density():
    """Compare TF energy with DFTpy for a uniform density"""
    # 1. Setup Grid
    lattice = np.eye(3) * 10.0
    nr = [32, 32, 32]
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Set uniform density
    rho_val = 0.05
    from dftpy.field import DirectField

    rho_dftpy = DirectField(grid=dftpy_grid, data=np.full(nr, rho_val))

    # 3. DFTpy TF
    tf_py = Functional(type="KEDF", name="TF")
    e_tf_py = tf_py(rho_dftpy).energy

    # 4. DFTcu TF
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.fill(rho_val)

    tf_cu = dftcu.ThomasFermi(coeff=1.0)
    v_tf_cu = dftcu.RealField(grid_cu, 1)
    e_tf_cu = tf_cu.compute(rho_cu, v_tf_cu)

    print(f"DFTpy TF Energy: {e_tf_py:.10f}")
    print(f"DFTcu TF Energy: {e_tf_cu:.10f}")
    assert abs(e_tf_py - e_tf_cu) < 1e-10

    # 5. Check potential consistency
    # V_tf = (5/3) * C_tf * rho^(2/3)
    # E_tf = C_tf * integral(rho^(5/3))
    # For uniform rho: E_tf = V_tf * rho * Vol / (5/3) = V_tf * Ne / (5/3)
    v_host = np.zeros(rho_cu.size())
    v_tf_cu.copy_to_host(v_host)
    v_expected = e_tf_cu * (5 / 3) / (rho_val * 1000.0)
    assert abs(v_host[0] - v_expected) < 1e-10
    print("✓ TF KEDF Verification Passed")


if __name__ == "__main__":
    test_tf_energy_density()
