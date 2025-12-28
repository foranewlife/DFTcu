import dftcu
import numpy as np
import pytest


def test_wavefunction_init():
    """Test Wavefunction allocation and PW counting logic (VASP-style)"""
    lattice = np.eye(3) * 10.0
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    # VASP logic: ENCUT determines the sphere in G-space
    encut = 12.5
    num_bands = 8
    wf = dftcu.Wavefunction(grid, num_bands, encut)

    assert wf.num_bands() == num_bands
    assert wf.encut() == encut
    assert 0 < wf.num_pw() < grid.nnr()
    print(f"Bands: {wf.num_bands()}, Active PWs: {wf.num_pw()} / {grid.nnr()}")


def test_density_construction():
    """Verify charge density construction from orbitals"""
    lattice = np.eye(3) * 8.0
    nr = [24, 24, 24]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    num_bands = 4
    wf = dftcu.Wavefunction(grid, num_bands, encut=20.0)
    wf.randomize(seed=123)

    rho = dftcu.RealField(grid)
    occupations = [2.0, 2.0, 0.0, 0.0]  # 2 fully occupied bands (spin-degenerate)

    wf.compute_density(occupations, rho)

    integral = rho.integral()
    print(f"Integrated density: {integral:.6f}")
    assert integral > 0


if __name__ == "__main__":
    pytest.main([__file__])
