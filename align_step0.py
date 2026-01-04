import numpy as np

import dftcu


def align_step0():
    # 1. Setup Grid and Atoms (Bohr -> Angstrom for Grid constructor)
    BOHR_TO_ANG = dftcu.constants.BOHR_TO_ANGSTROM
    a_bohr = 18.897261
    a_ang = a_bohr * BOHR_TO_ANG
    grid = dftcu.Grid([a_ang, 0, 0, 0, a_ang, 0, 0, 0, a_ang], [125, 125, 125])
    atoms = dftcu.Atoms([dftcu.Atom(a_ang / 2, a_ang / 2, a_ang / 2, 6.0, 0)])
    upf_file = "run_qe_v1.0/O_ONCV_PBE-1.0.UPF"

    # 2. Load QE Initial Density (rho_init)
    rho_init = dftcu.RealField(grid)
    with open("run_qe_v1.0/qe_rho_init.txt", "r") as f:
        f.readline()
        data = np.fromfile(f, sep=" ")
        nr = grid.nr()
        data = data.reshape((nr[2], nr[1], nr[0])).transpose(2, 1, 0).flatten()
        rho_init.copy_from_host(data)

    # 3. Setup Hamiltonian
    # Note: Use initialize_hamiltonian but ensure we use our loaded rho_init
    _, ham, __ = dftcu.initialize_hamiltonian(
        grid, atoms, [upf_file], ecutwfc=100.0, rho_init=rho_init
    )

    # 4. Load QE Initial Wavefunctions (psi_init)
    miller_file = "run_qe_v1.0/qe_wfc_g_atomic.dat"
    with open(miller_file, "r") as f:
        header = f.readline().split()
        npw_qe = int(header[0])
    miller = []
    with open(miller_file, "r") as f:
        f.readline()
        for _ in range(npw_qe):
            parts = f.readline().split()
            miller.append((int(parts[0]), int(parts[1]), int(parts[2])))
    miller = np.array(miller)
    with open("run_qe_v1.0/qe_psi_subspace.dat", "r") as f:
        header = f.readline().split()
        n_bands = int(header[0])
        data = np.fromfile(f, sep=" ").reshape(n_bands, npw_qe, 2)
        coeffs = data[:, :, 0] + 1j * data[:, :, 1]
    psi = dftcu.Wavefunction(grid, n_bands, 100.0)  # Match QE ecutwfc = 100.0 Ry
    psi.set_coefficients_miller(
        miller[:, 0].tolist(), miller[:, 1].tolist(), miller[:, 2].tolist(), coeffs.flatten()
    )

    # 5. Compute Components in DFTcu
    # eband
    h_psi = dftcu.Wavefunction(grid, n_bands, 100.0)  # Match ecutwfc
    ham.apply(psi, h_psi)

    # Calculate eigenvalues manually from <psi|H|psi>
    psi_np = np.zeros((n_bands, grid.nnr()), dtype=complex)
    h_psi_np = np.zeros((n_bands, grid.nnr()), dtype=complex)
    psi.copy_to_host(psi_np)
    h_psi.copy_to_host(h_psi_np)

    # Calculate eigenvalues using subspace diagonalization (like verify_native)
    # This achieves much higher precision than direct <psi|H|psi>
    h_m = np.zeros((n_bands, n_bands), dtype=complex)
    s_m = np.zeros((n_bands, n_bands), dtype=complex)
    for i in range(n_bands):
        for j in range(n_bands):
            h_m[i, j] = np.sum(np.conj(psi_np[i]) * h_psi_np[j])
            s_m[i, j] = np.sum(np.conj(psi_np[i]) * psi_np[j])

    from dftcu.utils import solve_generalized_eigenvalue_problem

    e_calc = solve_generalized_eigenvalue_problem(grid, h_m, s_m)

    # Debug: print eigenvalues for comparison
    if False:  # Set to True to debug
        print("\n--- DFTcu Eigenvalues ---")
        for i, e in enumerate(e_calc):
            print(f"Band {i}: ε = {e:.14f} Ha")

    occ = np.array([2.0, 1.3333333333, 1.3333333333, 1.3333333333])
    eband_cu = np.sum(e_calc * occ)

    # ehart, etxc
    # We need to compute these separately.
    # Use the functionals directly.
    hartree = dftcu.Hartree()
    hartree.set_gcut(-1.0)  # No spherical cutoff (QE doesn't use it for Hartree)
    lda = dftcu.LDA_PZ()
    lda.set_rho_threshold(1e-10)  # QE's hardcoded threshold
    vh = dftcu.RealField(grid)
    vxc = dftcu.RealField(grid)
    eh_cu = hartree.compute(rho_init, vh)
    exc_cu = lda.compute(rho_init, vxc)

    # deband = -∫ ρ * (V_H + V_XC) dr
    # NOTE: ham.v_loc() = V_H + V_XC + V_ps, so we need to compute V_H + V_XC separately
    vh_np = np.zeros(grid.nnr())
    vxc_np = np.zeros(grid.nnr())
    vh.copy_to_host(vh_np)
    vxc.copy_to_host(vxc_np)

    rho_np = np.zeros(grid.nnr())
    rho_init.copy_to_host(rho_np)

    deband_cu = -np.sum(rho_np * (vh_np + vxc_np)) * grid.dv_bohr()

    # ewald
    # QE uses ecutrho = 400.0 Ry = 200.0 Ha for Ewald
    ecutrho = 400.0  # Ry
    ecutrho_ha = ecutrho * 0.5  # Ha
    ewald_func = dftcu.Ewald(grid, atoms, precision=1e-10, gcut_hint=ecutrho_ha)
    ew_cu = ewald_func.compute(False, ecutrho_ha)

    print("\n--- Step 0 Alignment (DFTcu) ---")
    print(f"eband:    {eband_cu:20.12f}")
    print(f"deband:   {deband_cu:20.12f}")
    print(f"ehart:    {eh_cu:20.12f}")
    print(f"etxc:     {exc_cu:20.12f}")
    print(f"ewald:    {ew_cu:20.12f}")
    print(f"Total:    {eband_cu + deband_cu + eh_cu + exc_cu + ew_cu:20.12f}")

    # 6. QE Reference (Inferred from qe_energies_init.txt and qe_eig_step0.dat)
    eh_qe = 2.1805410709115197e01 * 0.5
    etxc_qe = -6.2985483971554892e00 * 0.5
    vtxc_qe = -8.2539204531667192e00 * 0.5
    deband_qe = -(2 * eh_qe + vtxc_qe)

    with open("run_qe_v1.0/qe_eig_step0.dat", "r") as f:
        f.readline()
        eig_ry = np.fromfile(f, sep="\n")
    eband_qe = np.sum(eig_ry * 0.5 * occ)

    # QE Ewald from XML output (direct, not inferred)
    # From run_qe_v1.0/pwscf.save/data-file-schema.xml: <ewald>-2.702579773768214e0</ewald>
    ew_qe = -2.702579773768214

    # QE Total Energy
    etot_qe = eband_qe + deband_qe + eh_qe + etxc_qe + ew_qe

    print("\n--- Step 0 Alignment (QE Reference) ---")
    print(f"eband:    {eband_qe:20.12f}")
    print(f"deband:   {deband_qe:20.12f}")
    print(f"ehart:    {eh_qe:20.12f}")
    print(f"etxc:     {etxc_qe:20.12f}")
    print(f"ewald:    {ew_qe:20.12f}")
    print(f"Total:    {etot_qe:20.12f}")

    print("\n--- Differences (DFTcu - QE) ---")
    print(f"eband:    {eband_cu - eband_qe:20.12e}")
    print(f"deband:   {deband_cu - deband_qe:20.12e}")
    print(f"ehart:    {eh_cu - eh_qe:20.12e}")
    print(f"etxc:     {exc_cu - etxc_qe:20.12e}")
    print(f"ewald:    {ew_cu - ew_qe:20.12e}")
    print(f"Total:    {(eband_cu + deband_cu + eh_cu + exc_cu + ew_cu) - etot_qe:20.12e}")


if __name__ == "__main__":
    align_step0()
