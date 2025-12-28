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


if __name__ == "__main__":
    pytest.main([__file__])
