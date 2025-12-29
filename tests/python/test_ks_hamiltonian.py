import dftcu
import numpy as np
import pytest


def test_hamiltonian_basic():
    """Test Hamiltonian construction and basic apply operation"""
    lattice = np.eye(3) * 10.0
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    # 1. Setup Evaluator with TF functional
    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.ThomasFermi())

    # 2. Initialize Hamiltonian
    ham = dftcu.Hamiltonian(grid, evaluator)

    # 3. Create wavefunction and density
    wf = dftcu.Wavefunction(grid, num_bands=2, encut=10.0)
    wf.randomize()

    rho = dftcu.RealField(grid)
    wf.compute_density([2.0, 0.0], rho)  # 2 electrons in the first band

    # 4. Update potential and apply H
    ham.update_potentials(rho)

    h_wf = dftcu.Wavefunction(grid, num_bands=2, encut=10.0)
    ham.apply(wf, h_wf)

    assert h_wf.num_bands() == 2
    print("Hamiltonian successfully applied to wavefunction")


def test_kinetic_operator_precision():
    """
    Verify that the kinetic energy operator T|psi> = 0.5 * G^2 * psi
    matches analytical results to 10^-14.
    """
    L = 10.0
    lattice = np.eye(3) * L
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    # Analytical G^2 for index 1
    idx = 1

    # Use the same PI as dftcu
    PI = 3.14159265358979323846
    expected_g2 = (2.0 * PI / L) ** 2

    # Test operator action: <psi | T | psi>
    wf = dftcu.Wavefunction(grid, 1, 100.0)
    data = np.zeros(grid.nnr(), dtype=np.complex128)
    data[idx] = 1.0  # Pure plane wave
    wf.copy_from_host(data)

    evaluator = dftcu.Evaluator(grid)  # Zero potential
    ham = dftcu.Hamiltonian(grid, evaluator)

    h_wf = dftcu.Wavefunction(grid, 1, 100.0)
    ham.apply(wf, h_wf)

    # Trick: Copy h_wf back to host and use np.vdot
    h_data = np.zeros(grid.nnr(), dtype=np.complex128)
    h_wf.copy_to_host(h_data)
    computed_t = np.vdot(data, h_data).real

    print("\n[Kinetic Operator Alignment]")
    print(f"T (Computed):   {computed_t:.18f}")
    print(f"T (Expected):   {0.5 * expected_g2:.18f}")

    assert abs(computed_t - 0.5 * expected_g2) < 1e-14


if __name__ == "__main__":
    pytest.main([__file__])
