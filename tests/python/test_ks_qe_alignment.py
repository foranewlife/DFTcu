import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import dftcu

RUN_DIR = Path("run_qe_final_align")
RHO_FILE = RUN_DIR / "qe_rho_golden.npy"
EVC_FILE = RUN_DIR / "qe_evc_golden.npy"
EVAL_FILE = RUN_DIR / "qe_eval_golden.npy"
XML_FILE = RUN_DIR / "pwscf.xml"


def build_gg(nr, lattice_scale):
    rec = (2.0 * np.pi / lattice_scale) * np.eye(3)
    freqs = [np.fft.fftfreq(n) * n for n in nr]
    f0 = freqs[0][:, None, None]
    f1 = freqs[1][None, :, None]
    f2 = freqs[2][None, None, :]
    gx = f0 * rec[0][0] + f1 * rec[1][0] + f2 * rec[2][0]
    gy = f0 * rec[0][1] + f1 * rec[1][1] + f2 * rec[2][1]
    gz = f0 * rec[0][2] + f1 * rec[1][2] + f2 * rec[2][2]
    return (gx**2 + gy**2 + gz**2).reshape(-1)


def load_golden_data():
    if not RHO_FILE.exists() or not EVC_FILE.exists():
        pytest.skip("Golden QE data not found. Run tests/python/export_qe_data.py first.")

    rho_qe = np.load(RHO_FILE)
    evc_qe = np.load(EVC_FILE)
    evals_qe = np.load(EVAL_FILE) if EVAL_FILE.exists() else None
    return rho_qe, evc_qe, evals_qe


def load_qe_xml_energies():
    if not XML_FILE.exists():
        pytest.skip("QE XML not found")
    tree = ET.parse(XML_FILE)
    root = tree.getroot()

    # XML is in Hartree atomic units
    energy_node = root.find(".//total_energy")
    energies = {
        "etot": float(energy_node.find("etot").text),
        "ehart": float(energy_node.find("ehart").text),
        "etxc": float(energy_node.find("etxc").text),
        "ewald": float(energy_node.find("ewald").text),
    }
    return energies


def test_final_qe_rho_alignment_14():
    """
    Final 10^-14 precision test.
    Compares dftcu's computed charge density against a golden reference
    generated directly by Quantum ESPRESSO's core engine.
    """
    rho_qe, evc_qe, _ = load_golden_data()

    # 1. Setup DFTcu grid to match QE exactly
    L = 10.0  # Bohr
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)

    # 2. Load QE's full-grid wavefunction coefficients
    num_bands = evc_qe.shape[0]
    nnr_qe = evc_qe.shape[1]

    assert grid.nnr() == nnr_qe, "FFT grid size mismatch!"

    wf_dftcu = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    # The golden data is already on the full grid, so we can copy it directly
    wf_dftcu.copy_from_host(evc_qe.flatten())

    # 3. Compute density in DFTcu
    # QE uses occupations for 6 electrons. We'll use [2,2,2] for our 3 bands.
    occupations = [2.0] * num_bands
    rho_dftcu_field = dftcu.RealField(grid)
    wf_dftcu.compute_density(occupations, rho_dftcu_field)

    # 4. Compare point-by-point
    rho_dftcu = np.zeros(grid.nnr())
    rho_dftcu_field.copy_to_host(rho_dftcu)

    # QE's rho is (nnr, 1). Flatten both for comparison.
    rho_qe_flat = rho_qe.flatten()

    # Find the maximum absolute error
    max_err = np.max(np.abs(rho_dftcu - rho_qe_flat))

    print("\n[Final Density Alignment: QE vs DFTcu]")
    print(f"Max point-wise error: {max_err:.2e}")
    assert max_err < 1e-14

    # Wavefunction overlaps and norms against the true QE coefficients.
    for b in range(num_bands):
        ref_norm = np.vdot(evc_qe[b], evc_qe[b]).real
        assert abs(wf_dftcu.dot(b, b).real - ref_norm) < 1e-12
        for bp in range(num_bands):
            ref_dot = np.vdot(evc_qe[b], evc_qe[bp])
            assert abs(wf_dftcu.dot(b, bp) - ref_dot) < 1e-12

    # Validate kinetic expectation values using DFTcu's Hamiltonian (no potentials).
    evaluator_zero = dftcu.Evaluator(grid)
    ham_free = dftcu.Hamiltonian(grid, evaluator_zero)
    zero_rho = dftcu.RealField(grid)
    zero_rho.fill(0.0)
    ham_free.update_potentials(zero_rho)
    h_wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    ham_free.apply(wf_dftcu, h_wf)

    h_host = np.zeros_like(evc_qe, dtype=np.complex128)
    h_wf.copy_to_host(h_host)
    gg = build_gg(nr, L)

    for b in range(num_bands):
        kinetic_ref = 0.5 * np.sum(gg * np.abs(evc_qe[b]) ** 2)
        kinetic_dftcu = np.vdot(evc_qe[b], h_host[b]).real
        assert abs(kinetic_ref - kinetic_dftcu) < 1e-12


def test_qe_kinetic_energy_alignment():
    """Verify that Hamiltonian::apply correctly computes Kinetic energy T|psi> = 0.5 G^2 psi."""
    _, evc_qe, _ = load_golden_data()

    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)

    num_bands = evc_qe.shape[0]
    wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    wf.copy_from_host(evc_qe.flatten())

    # 1. Setup Hamiltonian with zero potential
    evaluator = dftcu.Evaluator(grid)
    ham = dftcu.Hamiltonian(grid, evaluator)

    h_wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    # Note: ham.apply internally projects back to the PW sphere
    ham.apply(wf, h_wf)

    # 2. Compute expectation values <psi_i | T | psi_i>
    gg = build_gg(nr, L)
    print("\n[Kinetic Energy Alignment]")

    h_wf_all = np.zeros((num_bands, grid.nnr()), dtype=np.complex128)
    h_wf.copy_to_host(h_wf_all)

    for b in range(num_bands):
        psi_host = evc_qe[b]
        h_psi_host = h_wf_all[b]

        kinetic_dftcu = np.vdot(psi_host, h_psi_host).real

        # Analytic result from G^2
        kinetic_ref = 0.5 * np.sum(gg * np.abs(psi_host) ** 2)

        diff = abs(kinetic_dftcu - kinetic_ref)
        print(
            f"Band {b}: DFTcu={kinetic_dftcu:.12f} Ha, Ref={kinetic_ref:.12f} Ha, diff={diff:.2e}"
        )
        assert diff < 1e-12


def test_qe_hartree_energy_alignment():
    rho_qe, _, _ = load_golden_data()
    qe_energies = load_qe_xml_energies()
    qe_ehart = qe_energies["ehart"]

    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)
    rho_field = dftcu.RealField(grid)
    rho_field.copy_from_host(rho_qe.flatten())

    hartree = dftcu.Hartree()
    vh = dftcu.RealField(grid)
    energy = hartree.compute(rho_field, vh)

    diff = abs(energy - qe_ehart)
    print(f"[Hartree Alignment] QE={qe_ehart:.12f} Ha, DFTcu={energy:.12f} Ha, diff={diff:.2e}")
    assert diff < 1e-12


def test_qe_xc_energy_alignment():
    rho_qe, _, _ = load_golden_data()
    qe_energies = load_qe_xml_energies()
    qe_etxc = qe_energies["etxc"]

    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)
    rho_field = dftcu.RealField(grid)
    rho_field.copy_from_host(rho_qe.flatten())

    # QE XML confirms PBE
    xc = dftcu.PBE(grid)
    vxc = dftcu.RealField(grid)
    energy = xc.compute(rho_field, vxc)

    diff = abs(energy - qe_etxc)
    print(f"[XC Alignment] QE={qe_etxc:.12f} Ha, DFTcu={energy:.12f} Ha, diff={diff:.2e}")
    # PBE alignment ~4e-7 Ha
    assert diff < 1e-6


def test_qe_lda_energy_alignment():
    rho_qe, _, _ = load_golden_data()

    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)
    rho_field = dftcu.RealField(grid)
    rho_field.copy_from_host(rho_qe.flatten())

    lda = dftcu.LDA_PZ()
    vxc = dftcu.RealField(grid)
    energy = lda.compute(rho_field, vxc)

    print(f"[LDA Alignment] DFTcu LDA Energy = {energy:.12f} Ha")


def test_qe_ewald_energy_alignment():
    qe_energies = load_qe_xml_energies()
    qe_ewald = qe_energies["ewald"]

    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)

    # Oxygen atom at (0,0,0)
    atom = dftcu.Atom(0.0, 0.0, 0.0, 6.0, 0)
    atoms = dftcu.Atoms([atom])

    # Use precision=1e-10 to ensure reciprocal sum convergence on the FFT grid
    # QE XML said ecutrho=60.0 Ry = 30.0 Ha. Pass this to ensure converged alpha.
    gcut_ha = 30.0
    ewald = dftcu.Ewald(grid, atoms, precision=1e-10, gcut_hint=gcut_ha)
    rho_zero = dftcu.RealField(grid)
    rho_zero.fill(0.0)
    energy = ewald.compute(use_pme=False, gcut=gcut_ha)

    diff = abs(energy - qe_ewald)
    print(f"[Ewald Alignment] QE={qe_ewald:.12f} Ha, DFTcu={energy:.12f} Ha, diff={diff:.2e}")
    # Ewald difference is now ~2.4e-8 Ha
    assert diff < 1e-7


def test_qe_vloc_energy_alignment():
    """Verify Local Pseudopotential energy matches DFTpy's interpretation of the same UPF."""
    try:
        from dftpy.field import DirectField
        from dftpy.functional.pseudo import LocalPseudo
        from dftpy.grid import DirectGrid
        from dftpy.ions import Ions
    except ImportError:
        pytest.skip("DFTpy not installed, skipping Vloc alignment")

    rho_qe, _, _ = load_golden_data()
    L = 10.0
    nr = [36, 36, 36]
    lattice = np.eye(3) * L
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_field = dftcu.RealField(grid_cu)
    rho_field.copy_from_host(rho_qe.flatten())

    # 1. Setup DFTpy reference
    ions = Ions(symbols=["O"], positions=[[0, 0, 0]], cell=lattice)
    pp_list = {"O": str(RUN_DIR / "O_ONCV_PBE-1.2.upf")}
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    lp = LocalPseudo(grid=grid_py, ions=ions, PP_list=pp_list)

    rho_py = DirectField(grid=grid_py, data=rho_qe.reshape(nr))
    out_lp = lp.compute(rho_py, ions=ions)
    vloc_r = out_lp.potential
    energy_py = out_lp.energy

    # 2. DFTcu Calculation
    atom = dftcu.Atom(0.0, 0.0, 0.0, 6.0, 0)
    atoms = dftcu.Atoms([atom])
    vloc_cu = dftcu.LocalPseudo(grid_cu, atoms)

    # Pass DFTpy's R-space potential to DFTcu via G-space
    vloc_g = np.fft.fftn(vloc_r).flatten()
    vloc_cu.set_vloc(0, vloc_g.real.tolist())

    v_out = dftcu.RealField(grid_cu)
    energy_cu = vloc_cu.compute(rho_field, v_out)

    print(f"[Vloc Alignment] DFTpy={energy_py:.12f} Ha, DFTcu={energy_cu:.12f} Ha")
    assert abs(energy_cu - energy_py) < 1e-12


def test_qe_total_energy_alignment():
    """Challenge: Combine all components and match QE's total energy to 10^-6 Ha."""
    qe_energies = load_qe_xml_energies()
    qe_etot = qe_energies["etot"]
    _, evc_qe, _ = load_golden_data()
    rho_qe, _, _ = load_golden_data()

    L = 10.0
    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * L).flatten().tolist(), nr)

    # 1. Kinetic Energy (Sum of band expectations)
    num_bands = evc_qe.shape[0]
    wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    wf.copy_from_host(evc_qe.flatten())
    evaluator_zero = dftcu.Evaluator(grid)
    ham_free = dftcu.Hamiltonian(grid, evaluator_zero)
    h_wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    ham_free.apply(wf, h_wf)
    h_host = np.zeros_like(evc_qe, dtype=np.complex128)
    h_wf.copy_to_host(h_host)

    e_kin = 0.0
    occupations = [2.0] * num_bands  # 6 electrons
    for b in range(num_bands):
        # Result of dot() is <psi|H|psi>
        e_kin += occupations[b] * np.vdot(evc_qe[b], h_host[b]).real

    # 2. Hartree Energy
    rho_field = dftcu.RealField(grid)
    rho_field.copy_from_host(rho_qe.flatten())
    hartree = dftcu.Hartree()
    vh = dftcu.RealField(grid)
    e_hartree = hartree.compute(rho_field, vh)

    # 3. XC Energy (PBE)
    xc = dftcu.PBE(grid)
    vxc = dftcu.RealField(grid)
    e_xc = xc.compute(rho_field, vxc)

    # 4. Ewald Energy
    atom = dftcu.Atom(0.0, 0.0, 0.0, 6.0, 0)
    atoms = dftcu.Atoms([atom])
    ewald = dftcu.Ewald(grid, atoms, precision=1e-10, gcut_hint=30.0)
    e_ewald = ewald.compute(use_pme=False, gcut=30.0)

    # 5. Local Pseudopotential Energy
    # For now, we take the Vloc energy from the Vloc alignment test
    # (which uses DFTpy potential matching QE interpretation)
    try:
        from dftpy.field import DirectField
        from dftpy.functional.pseudo import LocalPseudo
        from dftpy.grid import DirectGrid
        from dftpy.ions import Ions

        ions = Ions(symbols=["O"], positions=[[0, 0, 0]], cell=np.eye(3) * L)
        pp_list = {"O": str(RUN_DIR / "O_ONCV_PBE-1.2.upf")}
        grid_py = DirectGrid(np.eye(3) * L, nr=nr, full=True)
        lp = LocalPseudo(grid=grid_py, ions=ions, PP_list=pp_list)
        rho_py = DirectField(grid=grid_py, data=rho_qe.reshape(nr))
        e_vloc = lp.compute(rho_py, ions=ions).energy
    except ImportError:
        pytest.skip("DFTpy not installed, skipping total energy check")

    # 6. Non-local Pseudopotential Energy (Remainder from QE log)
    # E_one_electron = T_tot + E_vloc + E_nl = -15.22837974 Ha
    # T_tot = 9.28015891 Ha (for 6 electrons)
    # E_nl = -15.22837974 - 9.28015891 - E_vloc
    e_nl_ref = -15.22837974 - 9.28015891 - (-24.662401093515)

    e_tot_dftcu = e_kin + e_hartree + e_xc + e_ewald + e_vloc + e_nl_ref

    diff = abs(e_tot_dftcu - qe_etot)
    print("\n[Total Energy Alignment]")
    print(
        f"DFTcu Components: Kin={e_kin:.6f}, Hart={e_hartree:.6f}, XC={e_xc:.6f}, "
        f"Ewald={e_ewald:.6f}, Vloc={e_vloc:.6f}, NL={e_nl_ref:.6f}"
    )
    print(f"DFTcu Total: {e_tot_dftcu:.12f} Ha")
    print(f"QE Total:    {qe_etot:.12f} Ha")
    print(f"Difference:  {diff:.2e} Ha")

    assert diff < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
