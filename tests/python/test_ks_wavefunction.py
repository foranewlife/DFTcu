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


def test_wavefunction_dot_precision():
    """Verify that Wavefunction::dot matches NumPy high-precision complex inner product (10^-14)"""
    lattice = np.eye(3) * 10.0
    nr = [8, 8, 8]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    # Allocate wavefunction with 2 bands
    wf = dftcu.Wavefunction(grid, 2, encut=100.0)

    # Generate fixed random data
    rng = np.random.default_rng(42)
    band0 = rng.standard_normal(grid.nnr()) + 1j * rng.standard_normal(grid.nnr())
    band1 = rng.standard_normal(grid.nnr()) + 1j * rng.standard_normal(grid.nnr())

    # Copy to device
    full_data = np.ascontiguousarray(np.concatenate([band0, band1]))
    wf.copy_from_host(full_data)

    # Reference calculation: sum(conj(band0) * band1)
    ref_val = np.vdot(band0, band1)

    # DFTcu calculation
    computed_val = wf.dot(0, 1)

    print("\n[Wavefunction Dot Alignment]")
    print(f"NumPy Ref: {ref_val}")
    print(f"DFTcu Val: {computed_val}")
    print(f"Abs Diff:  {abs(ref_val - computed_val):.2e}")

    assert abs(ref_val - computed_val) < 2e-14


def test_ks_density_integral_precision():
    """
    Final high-precision alignment test for charge density construction.
    Matches Quantum ESPRESSO's normalization convention.
    """
    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)

    num_bands = 3
    wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)

    # Construct a perfectly normalized test wavefunction
    # Let each band be a single plane wave at G=0.
    data = np.zeros((num_bands, grid.nnr()), dtype=np.complex128)
    for b in range(num_bands):
        data[b, 0] = 1.0
    wf.copy_from_host(data.flatten())

    occupations = [2.0, 2.0, 2.0]
    rho = dftcu.RealField(grid)
    wf.compute_density(occupations, rho)

    computed_total_charge = rho.integral()
    expected_total_charge = 6.0

    print("\n[Density Integral Alignment]")
    print(f"Computed Charge: {computed_total_charge:.18f}")
    assert abs(computed_total_charge - expected_total_charge) < 1e-11


if __name__ == "__main__":
    pytest.main([__file__])
