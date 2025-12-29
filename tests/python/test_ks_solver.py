import dftcu
import numpy as np
import pytest


def test_davidson_flow():
    """Verify Davidson solver workflow and eigenvalue calculation"""
    lattice = np.eye(3) * 8.0
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    # 1. Hamiltonian with constant potential for testing
    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.ThomasFermi())
    ham = dftcu.Hamiltonian(grid, evaluator)

    rho = dftcu.RealField(grid)
    rho.fill(0.01)
    ham.update_potentials(rho)

    # 2. Solver setup (NBANDS=4)
    num_bands = 4
    wf = dftcu.Wavefunction(grid, num_bands, encut=10.0)
    wf.randomize()

    solver = dftcu.DavidsonSolver(grid, max_iter=5, tol=1e-6)

    # 3. Solve eigenvalue problem
    energies = solver.solve(ham, wf)

    print(f"Final KS Energies: {energies}")
    assert len(energies) == num_bands
    # Eigenvalues from Davidson are typically sorted
    assert all(energies[i] <= energies[i + 1] for i in range(len(energies) - 1))


def test_davidson_analytical_kinetic():
    """Verify Davidson solver against analytical kinetic-only eigenvalues."""
    L = 9.0
    nr = [8, 8, 8]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)

    gg = np.array(grid.gg())
    # Take specific plane wave indices
    valid = np.where(0.5 * gg < 60.0)[0]
    pw_indices = valid[1:4]

    num_bands = len(pw_indices)
    wf = dftcu.Wavefunction(grid, num_bands, encut=60.0)
    host = np.zeros((num_bands, grid.nnr()), dtype=np.complex128)
    for band, g_idx in enumerate(pw_indices):
        host[band, g_idx] = 1.0
    wf.copy_from_host(host.flatten())

    evaluator = dftcu.Evaluator(grid)
    ham = dftcu.Hamiltonian(grid, evaluator)
    solver = dftcu.DavidsonSolver(grid, max_iter=1, tol=1e-10)

    eigenvalues = solver.solve(ham, wf)
    expected = 0.5 * gg[pw_indices]
    diff = np.abs(eigenvalues - expected)

    print(f"[Davidson Analytical] eigenvalue diff: {diff}")
    assert np.all(diff < 1e-12)

    # Verify orthonormality persists
    for i in range(num_bands):
        for j in range(num_bands):
            dot = wf.dot(i, j)
            if i == j:
                assert abs(dot - 1.0) < 1e-12
            else:
                assert abs(dot) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
