#!/usr/bin/env python3
"""
Test Thomas-Fermi KEDF implementation against DFTpy
"""
import sys

import numpy as np

sys.path.insert(0, "/workplace/chenys/project/DFTcu/external/DFTpy/src")

import dftcu  # noqa: E402
from dftpy.field import DirectField  # noqa: E402
from dftpy.functional.kedf.tf import TF as DFTpy_TF  # noqa: E402
from dftpy.grid import DirectGrid  # noqa: E402


def test_tf_kedf():
    """Compare DFTcu TF implementation with DFTpy"""

    print("=" * 70)
    print("Testing Thomas-Fermi KEDF: DFTcu vs DFTpy")
    print("=" * 70)

    # Setup: Create a simple uniform density system
    lattice = np.eye(3) * 10.0  # 10 Bohr cubic cell
    nr = [32, 32, 32]

    # DFTpy grid
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # Create test density (slightly non-uniform to avoid trivial case)
    rho_np = np.ones(nr, order="C") * 0.01  # ~0.01 electrons/Bohr³
    # Add a Gaussian perturbation
    center = np.array(nr) // 2
    for i in range(nr[0]):
        for j in range(nr[1]):
            for k in range(nr[2]):
                r2 = (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
                rho_np[i, j, k] += 0.02 * np.exp(-r2 / 50.0)

    # DFTpy calculation
    print("\n1. Running DFTpy Thomas-Fermi...")
    rho_dftpy = DirectField(grid=dftpy_grid, data=rho_np.copy())
    result_dftpy = DFTpy_TF(rho_dftpy, calcType={"E", "V"})
    energy_dftpy = result_dftpy.energy
    potential_dftpy = np.array(result_dftpy.potential)

    print(f"   DFTpy Energy: {energy_dftpy:.10f} Ha")
    print(
        f"   DFTpy Potential (min, max): ({potential_dftpy.min():.6f}, {potential_dftpy.max():.6f})"
    )

    # DFTcu calculation
    print("\n2. Running DFTcu Thomas-Fermi...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)

    # Create density field (rank=1 for scalar field)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_np.flatten(order="C"))

    # Create potential field
    v_kedf_cu = dftcu.RealField(grid_cu, 1)

    # Compute TF energy and potential
    tf_cu = dftcu.ThomasFermi(coeff=1.0)
    energy_cu = tf_cu.compute(rho_cu, v_kedf_cu)

    # Copy potential back to host
    potential_cu = np.zeros_like(rho_np.flatten())
    v_kedf_cu.copy_to_host(potential_cu)
    potential_cu = potential_cu.reshape(nr, order="C")

    print(f"   DFTcu Energy: {energy_cu:.10f} Ha")
    print(f"   DFTcu Potential (min, max): ({potential_cu.min():.6f}, {potential_cu.max():.6f})")

    # Compare results
    print("\n3. Comparison:")
    energy_diff = abs(energy_cu - energy_dftpy)
    energy_rel_error = energy_diff / abs(energy_dftpy)

    potential_diff = np.abs(potential_cu - potential_dftpy)
    potential_max_diff = potential_diff.max()
    potential_mean_diff = potential_diff.mean()
    potential_rel_error = potential_max_diff / np.abs(potential_dftpy).max()

    print(f"   Energy difference:     {energy_diff:.2e} Ha")
    print(f"   Energy relative error: {energy_rel_error:.2e}")
    print(f"   Potential max diff:    {potential_max_diff:.2e}")
    print(f"   Potential mean diff:   {potential_mean_diff:.2e}")
    print(f"   Potential rel error:   {potential_rel_error:.2e}")

    # Validation thresholds
    energy_tolerance = 1e-5
    potential_tolerance = 1e-4

    print("\n4. Validation:")
    energy_pass = energy_rel_error < energy_tolerance
    potential_pass = potential_rel_error < potential_tolerance

    print(f"   Energy test:    {'PASS ✓' if energy_pass else 'FAIL ✗'} (< {energy_tolerance})")
    print(
        f"   Potential test: {'PASS ✓' if potential_pass else 'FAIL ✗'} (< {potential_tolerance})"
    )

    # Additional diagnostics
    print("\n5. Detailed Statistics:")
    print(f"   Total electrons (DFTpy): {rho_dftpy.integral():.6f}")
    print(f"   Total electrons (DFTcu): {rho_cu.integral():.6f}")
    print(f"   Grid volume: {grid_cu.volume():.6f} Bohr³")
    print(f"   Grid points: {grid_cu.nnr()}")
    print(f"   dV: {grid_cu.dv():.6e} Bohr³")

    # Summary
    print("\n" + "=" * 70)
    if energy_pass and potential_pass:
        print("✓ ALL TESTS PASSED")
        print("Thomas-Fermi KEDF implementation is validated!")
        assert True
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the implementation.")
        assert False


if __name__ == "__main__":
    exit_code = test_tf_kedf()
    sys.exit(exit_code)
