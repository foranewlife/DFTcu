#!/usr/bin/env python3
"""
Comprehensive Thomas-Fermi KEDF benchmark: DFTcu vs DFTpy
Tests accuracy and performance across multiple grid sizes
"""
import sys
import time
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, "/workplace/chenys/project/DFTcu/external/DFTpy/src")

try:
    import dftcu  # noqa: E402
    from dftpy.field import DirectField  # noqa: E402
    from dftpy.functional.kedf.tf import TF as DFTpy_TF  # noqa: E402
    from dftpy.grid import DirectGrid  # noqa: E402
    from dftpy.ions import Ions  # noqa: E402
    from dftpy.density import DensityGenerator  # noqa: E402

    DFTPY_AVAILABLE = True
except ImportError:
    DFTPY_AVAILABLE = False


@pytest.mark.skipif(not DFTPY_AVAILABLE, reason="DFTpy not installed")
class TestTFKEDFBenchmark:
    """Benchmark Thomas-Fermi KEDF: DFTcu vs DFTpy"""

    @pytest.fixture
    def grid_sizes(self):
        """Test grid sizes: small, medium, large"""
        return [
            (16, 16, 16),  # Small: 4K points
            (32, 32, 32),  # Medium: 32K points
            (64, 64, 64),  # Large: 262K points
        ]

    def create_test_density(self, nr: Tuple[int, int, int], lattice: np.ndarray) -> np.ndarray:
        """Create a realistic test density using DensityGenerator"""
        dftpy_grid = DirectGrid(lattice, nr=list(nr), full=True)
        # Add atoms to generate non-uniform density
        pos = np.array([[5.0, 5.0, 5.0]])
        ions = Ions(symbols=['Al'], positions=pos, cell=lattice)
        ions.set_charges(3.0)
        generator = DensityGenerator()
        rho = generator.guess_rho(ions, grid=dftpy_grid)
        return np.array(rho)

    def run_dftpy(self, lattice: np.ndarray, nr: Tuple[int, int, int], rho_np: np.ndarray):
        """Run DFTpy calculation with timing"""
        grid = DirectGrid(lattice, nr=list(nr), full=True)
        rho = DirectField(grid=grid, data=rho_np.copy())

        start = time.perf_counter()
        result = DFTpy_TF(rho, calcType={"E", "V"})
        elapsed = time.perf_counter() - start

        return {
            "energy": result.energy,
            "potential": np.array(result.potential),
            "time": elapsed,
        }

    def run_dftcu(self, lattice: np.ndarray, nr: Tuple[int, int, int], rho_np: np.ndarray):
        """Run DFTcu calculation with timing"""
        grid = dftcu.Grid(lattice.flatten().tolist(), list(nr))
        rho = dftcu.RealField(grid, 1)
        rho.copy_from_host(rho_np.flatten(order="C"))
        v_kedf = dftcu.RealField(grid, 1)
        tf = dftcu.ThomasFermi(coeff=1.0)

        # Warmup GPU
        _ = tf.compute(rho, v_kedf)

        # Timed run
        start = time.perf_counter()
        energy = tf.compute(rho, v_kedf)
        elapsed = time.perf_counter() - start

        potential = np.zeros(grid.nnr())
        v_kedf.copy_to_host(potential)
        potential = potential.reshape(nr, order="C")

        return {"energy": energy, "potential": potential, "time": elapsed}

    @pytest.mark.parametrize("nr", [(16, 16, 16), (32, 32, 32), (64, 64, 64)])
    def test_accuracy_vs_dftpy(self, nr):
        """Test accuracy against DFTpy for different grid sizes"""
        # Setup
        lattice = np.eye(3) * 10.0
        rho_np = self.create_test_density(nr, lattice)

        # Run both implementations
        result_dftpy = self.run_dftpy(lattice, nr, rho_np)
        result_dftcu = self.run_dftcu(lattice, nr, rho_np)

        # Compare accuracy
        energy_diff = abs(result_dftcu["energy"] - result_dftpy["energy"])
        energy_rel_error = energy_diff / max(abs(result_dftpy["energy"]), 1e-12)

        potential_diff = np.abs(result_dftcu["potential"] - result_dftpy["potential"])
        potential_max_diff = potential_diff.max()
        potential_rel_error = potential_max_diff / max(np.abs(result_dftpy["potential"]).max(), 1e-12)

        # Assertions
        assert energy_rel_error < 1e-10, f"Energy error too large: {energy_rel_error}"
        assert potential_rel_error < 1e-6, f"Potential error too large: {potential_rel_error}"

    def test_performance_summary(self, grid_sizes):
        """Generate performance summary across all grid sizes"""
        lattice = np.eye(3) * 10.0
        results = []

        for nr in grid_sizes:
            rho_np = self.create_test_density(nr, lattice)

            # Run multiple times for better statistics
            dftpy_times = []
            dftcu_times = []

            for _ in range(3):
                result_dftpy = self.run_dftpy(lattice, nr, rho_np)
                result_dftcu = self.run_dftcu(lattice, nr, rho_np)
                dftpy_times.append(result_dftpy["time"])
                dftcu_times.append(result_dftcu["time"])

            dftpy_avg = np.mean(dftpy_times)
            dftcu_avg = np.mean(dftcu_times)
            speedup = dftpy_avg / dftcu_avg

            results.append(
                {
                    "grid": f"{nr[0]}×{nr[1]}×{nr[2]}",
                    "points": np.prod(nr),
                    "dftpy_ms": dftpy_avg * 1000,
                    "dftcu_ms": dftcu_avg * 1000,
                    "speedup": speedup,
                }
            )

        # Print table
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY: DFTcu vs DFTpy")
        print("=" * 70)
        header = (
            f"{'Grid Size':<15} {'Points':<10} {'DFTpy (ms)':<12} "
            f"{'DFTcu (ms)':<12} {'Speedup':<10}"
        )
        print(header)
        print("-" * 70)
        for r in results:
            row = (
                f"{r['grid']:<15} {r['points']:<10,} {r['dftpy_ms']:<12.2f} "
                f"{r['dftcu_ms']:<12.2f} {r['speedup']:<10.2f}x"
            )
            print(row)
        print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
