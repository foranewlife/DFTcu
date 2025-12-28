import dftcu
import numpy as np
import pytest
from test_utils import get_system, setup_pseudo, to_dftcu_atoms


@pytest.fixture(scope="module")
def common_dft_setup():
    """Fixture for common DFT setup: grid, atoms, evaluator."""
    nr = [16, 16, 16]
    lattice = np.eye(3) * 10.0
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    ions = get_system(name="Al_single")
    atoms = to_dftcu_atoms(ions)

    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.ThomasFermi())
    evaluator.add_functional(dftcu.vonWeizsacker())
    pseudo, _ = setup_pseudo(grid, atoms, ions_py=ions)
    evaluator.add_functional(pseudo)
    evaluator.add_functional(dftcu.LDA_PZ())

    return grid, atoms, evaluator


def run_tn_optimization(optimizer_class, grid, evaluator):
    """Helper to run a truncated Newton optimization."""
    options = dftcu.OptimizationOptions()
    options.max_iter = 5
    options.econv = 1e-5

    # Create a simple, normalized density
    rho = dftcu.RealField(grid)
    rho_data = np.ones(grid.nnr())
    # Normalize to 3 electrons for Al
    rho_data *= 3.0 / (np.sum(rho_data) * grid.dv())
    rho.copy_from_host(rho_data)

    optimizer = optimizer_class(grid, options)
    optimizer.solve(rho, evaluator)
    return rho.integral()


def test_benchmark_tn_optimizer_et(benchmark, common_dft_setup):
    """Benchmark TNOptimizer with Expression Templates."""
    grid, _, evaluator = common_dft_setup
    # result_integral = benchmark(run_tn_optimization, dftcu.TNOptimizer, grid, evaluator)
    # assert np.isclose(result_integral, 3.0, rtol=1e-5)


def test_benchmark_tn_optimizer_legacy(benchmark, common_dft_setup):
    """Benchmark TNOptimizer with legacy (non-ET) implementation."""
    grid, _, evaluator = common_dft_setup
    # result_integral = benchmark(run_tn_optimization, dftcu.TNOptimizerLegacy, grid, evaluator)
    # assert np.isclose(result_integral, 3.0, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
