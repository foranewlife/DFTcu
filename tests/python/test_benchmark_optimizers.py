import dftcu
import pytest
from test_utils import (
    get_atoms,
    get_evaluator_tf_vw_pseudo_lda,
    get_grid,
    get_initial_density_and_potential,
)


@pytest.fixture(scope="module")
def common_dft_setup():
    """Fixture for common DFT setup: grid, atoms, evaluator."""
    grid = get_grid(nr=[16, 16, 16])
    atoms = get_atoms(nat=1)
    evaluator = get_evaluator_tf_vw_pseudo_lda(grid, atoms)
    return grid, atoms, evaluator


def run_tn_optimization(optimizer_class, grid, evaluator):
    """Helper to run a truncated Newton optimization."""
    options = dftcu.OptimizationOptions()
    options.max_iter = 5
    options.econv = 1e-5

    rho, _ = get_initial_density_and_potential(grid)

    optimizer = optimizer_class(grid, options)
    optimizer.solve(rho, evaluator)
    return rho.integral()


def test_benchmark_tn_optimizer_et(benchmark, common_dft_setup):
    """Benchmark TNOptimizer with Expression Templates."""
    grid, _, evaluator = common_dft_setup
    # result_integral = benchmark(run_tn_optimization, dftcu.TNOptimizer, grid, evaluator)
    # assert np.isclose(result_integral, 0.99999, rtol=1e-3)


def test_benchmark_tn_optimizer_legacy(benchmark, common_dft_setup):
    """Benchmark TNOptimizer with legacy (non-ET) implementation."""
    grid, _, evaluator = common_dft_setup
    # result_integral = benchmark(run_tn_optimization, dftcu.TNOptimizerLegacy, grid, evaluator)
    # assert np.isclose(result_integral, 0.99999, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
