import numpy as np
import pytest

import dftcu

HARTREE_TO_EV = 27.211386245988
ANGSTROM_TO_BOHR = 1.0 / 0.529177249


def test_pw_count_alignment():
    """Verify that DFTcu plane wave count exactly matches reference codes (QE/VASP)"""
    # QE Case:
    # L = 10.0 Bohr, Ecut = 15.0 Hartree -> NPW = 2777
    L = 10.0
    grid_qe = dftcu.Grid((np.eye(3) * L).flatten().tolist(), [36, 36, 36])
    wf_qe = dftcu.Wavefunction(grid_qe, 1, encut=15.0)
    assert wf_qe.num_pw() == 2777

    # VASP Case:
    # Lattice: 8.0 A cube, ENCUT: 400.0 eV -> NPW = 9315
    lattice_a_bohr = 8.0 * ANGSTROM_TO_BOHR
    grid_vasp = dftcu.Grid((np.eye(3) * lattice_a_bohr).flatten().tolist(), [48, 48, 48])
    encut_hartree = 400.0 / HARTREE_TO_EV
    wf_vasp = dftcu.Wavefunction(grid_vasp, 1, encut=encut_hartree)
    assert wf_vasp.num_pw() == 9315


def test_physical_constant_alignment():
    """Verify physical constants (PI) match industry standard to 10^-14"""
    PI_REF = 3.14159265358979323846
    L = 10.0
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), [16, 16, 16])
    gg = np.sort(grid.gg())

    g2_computed = gg[1]
    g2_expected = (2.0 * PI_REF / L) ** 2
    assert abs(g2_computed - g2_expected) < 1e-14


def test_density_normalization_scaling():
    """Verify that FFT and density construction follow standard normalization"""
    lattice_a = 10.0
    grid = dftcu.Grid((np.eye(3) * lattice_a).flatten().tolist(), [16, 16, 16])
    wf = dftcu.Wavefunction(grid, num_bands=1, encut=50.0)

    # Set psi(G=0) = 1.0. rho(r) = 1/Omega
    data = np.zeros(grid.nnr(), dtype=np.complex128)
    data[0] = 1.0
    wf.copy_from_host(data)

    rho = dftcu.RealField(grid)
    wf.compute_density([1.0], rho)

    integral = rho.integral()
    assert np.allclose(integral, 1.0, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
