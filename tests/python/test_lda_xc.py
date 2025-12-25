#!/usr/bin/env python3
"""
Test LDA XC implementation against DFTpy
"""
import sys
import os
import numpy as np

# Add DFTpy to path
sys.path.insert(0, os.path.join(os.getcwd(), "external/DFTpy/src"))

import dftcu
from dftpy.field import DirectField
from dftpy.functional.xc.semilocal_xc import LDA as DFTpy_LDA
from dftpy.grid import DirectGrid

def test_lda_xc():
    """Compare DFTcu LDA implementation with DFTpy"""

    print("=" * 70)
    print("Testing LDA XC: DFTcu vs DFTpy")
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
    print("\n1. Running DFTpy LDA...")
    rho_dftpy = DirectField(grid=dftpy_grid, data=rho_np.copy())
    # Note: DFTpy LDA uses PZ by default
    result_dftpy = DFTpy_LDA(rho_dftpy, calcType={"E", "V"})
    energy_dftpy = result_dftpy.energy
    potential_dftpy = np.array(result_dftpy.potential)

    # DFTcu calculation
    print("\n2. Running DFTcu LDA...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_np.flatten(order="C"))
    v_xc_cu = dftcu.RealField(grid_cu, 1)

    lda_cu = dftcu.LDA_PZ()
    energy_cu = lda_cu.compute(rho_cu, v_xc_cu)

    potential_cu = np.zeros_like(rho_np.flatten())
    v_xc_cu.copy_to_host(potential_cu)
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
    assert energy_rel_error < 1e-10
    assert potential_max_diff < 1e-10
    print("\n✓ ALL TESTS PASSED")

if __name__ == "__main__":
    test_lda_xc()
