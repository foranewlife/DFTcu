import dftcu
import numpy as np
import pytest


def test_ks_scf_minimal():
    """Run a minimal but fully-coupled KS-DFT Self-Consistent Field loop."""
    lattice = np.eye(3) * 8.0
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    # Hartree + LDA for true KS-DFT potential updates (kinetic handled by Hamiltonian).
    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.Hartree())
    evaluator.add_functional(dftcu.LDA_PZ())

    num_bands = 4
    nelectrons = 4.0
    wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    wf.randomize()

    solver = dftcu.DavidsonSolver(grid, max_iter=2, tol=1e-5)
    ham = dftcu.Hamiltonian(grid, evaluator)

    rho = dftcu.RealField(grid)
    wf.compute_density([1.0] * num_bands, rho)

    print("\nStarting Minimal KS-DFT SCF Cycle")
    print(f"{'Step':>6} | {'Total Energy':>16} | {'dE':>12}")

    prev_energy = 0.0
    mixing = 0.3
    energy_history = []

    for step in range(12):
        ham.update_potentials(rho)
        energies = solver.solve(ham, wf)

        occ_list = [2.0, 2.0, 0.0, 0.0]
        rho_new = dftcu.RealField(grid)
        wf.compute_density(occ_list, rho_new)

        rho_data = np.zeros(grid.nnr())
        rho.copy_to_host(rho_data)
        rho_new_data = np.zeros(grid.nnr())
        rho_new.copy_to_host(rho_new_data)
        rho_mixed_data = (1.0 - mixing) * rho_data + mixing * rho_new_data
        total_charge = np.sum(rho_mixed_data) * grid.dv()
        if total_charge > 1e-12:
            rho_mixed_data *= nelectrons / total_charge
        rho.copy_from_host(rho_mixed_data)

        current_energy = float(np.sum(energies))
        energy_history.append(current_energy)
        de = current_energy - prev_energy
        print(f"{step:6d} | {current_energy:16.8f} | {de:12.4e}")

        if step > 0 and abs(de) < 1e-6:
            print("SCF Converged!")
            break
        prev_energy = current_energy

    assert len(energies) == num_bands
    initial_delta = abs(energy_history[1] - energy_history[0])
    final_delta = abs(energy_history[-1] - energy_history[-2])
    assert final_delta < initial_delta * 1e-3
    assert abs(rho.integral() - nelectrons) < 1e-8
    assert step < 12


if __name__ == "__main__":
    pytest.main([__file__])
