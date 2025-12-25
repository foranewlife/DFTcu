#!/usr/bin/env python3
"""
Test Wang-Teter KEDF implementation against DFTpy
"""
import sys
import os
import numpy as np

# Add DFTpy to path
sys.path.insert(0, os.path.join(os.getcwd(), "external/DFTpy/src"))

import dftcu
from dftpy.field import DirectField
from dftpy.functional.kedf.wt import WT as DFTpy_WT
from dftpy.grid import DirectGrid

def test_wt_kedf():
    """Compare DFTcu Wang-Teter implementation with DFTpy"""

    print("=" * 70)
    print("Testing Wang-Teter KEDF: DFTcu vs DFTpy")
    print("=" * 70)

    # Setup
    lattice = np.eye(3) * 10.0
    nr = [32, 32, 32]

    # DFTpy grid
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # Create test density
    center = np.array(nr) / 2.0
    x = np.arange(nr[0])
    y = np.arange(nr[1])
    z = np.arange(nr[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R2 = ((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) / 10.0
    rho_np = 0.1 * np.exp(-R2) + 0.01

    # DFTpy calculation
    print("\n1. Running DFTpy Wang-Teter...")
    rho_dftpy = DirectField(grid=dftpy_grid, data=rho_np.copy())
    # Note: Default WT in DFTpy uses alpha=beta=5/6, rho0=rho.amean()
    result_dftpy = DFTpy_WT(rho_dftpy, calcType={"E", "V"})
    energy_dftpy = result_dftpy.energy
    potential_dftpy = np.array(result_dftpy.potential)

    # DFTcu calculation
    print("\n2. Running DFTcu Wang-Teter...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_np.flatten(order="C"))
    v_kedf_cu = dftcu.RealField(grid_cu, 1)

    wt_cu = dftcu.WangTeter(coeff=1.0)
    energy_cu = wt_cu.compute(rho_cu, v_kedf_cu)

    potential_cu = np.zeros_like(rho_np.flatten())
    v_kedf_cu.copy_to_host(potential_cu)
    potential_cu = potential_cu.reshape(nr, order="C")

    # Comparison
    print(f"   DFTpy Energy: {energy_dftpy:.10f} Ha")
    print(f"   DFTcu Energy: {energy_cu:.10f} Ha")

    energy_diff = abs(energy_cu - energy_dftpy)
    energy_rel_error = energy_diff / max(abs(energy_dftpy), 1e-10)
    potential_max_diff = np.abs(potential_cu - potential_dftpy).max()

    print(f"   Energy rel error: {energy_rel_error:.2e}")
    print(f"   Potential max diff: {potential_max_diff:.2e}")

    # Thresholds
    # WT involves many steps and a complex kernel, tolerance might need to be relaxed
    assert energy_rel_error < 1e-5
    assert potential_max_diff < 1e-4
    print("\n✓ ALL TESTS PASSED")

if __name__ == "__main__":
    test_wt_kedf()
