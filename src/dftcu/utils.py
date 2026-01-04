import os

import numpy as np

import dftcu
from dftcu._dftcu import constants


def build_initial_density(grid, atoms, upf_data_map, ecutrho):
    """
    Build initial charge density from atomic superposition (QE potinit style).
    """
    builder = dftcu.DensityBuilder(grid, atoms)
    builder.set_gcut(ecutrho)  # Rydberg

    for type_idx, data in upf_data_map.items():
        # Directly pass radial data to C++ for high-precision transform
        builder.set_atomic_rho_r(type_idx, data["r"], data["rho_at"], data["rab"])

    rho_val = dftcu.RealField(grid)
    builder.build_density(rho_val)
    return rho_val


def get_core_density(grid, atoms, upf_data_map, ecutrho):
    """
    Build core charge density for Non-Linear Core Correction (NLCC).
    """
    has_core = any("rho_core" in d and d["rho_core"] is not None for d in upf_data_map.values())
    if not has_core:
        return None

    builder = dftcu.DensityBuilder(grid, atoms)
    builder.set_gcut(ecutrho)

    for type_idx, data in upf_data_map.items():
        if "rho_core" not in data or data["rho_core"] is None:
            continue
        builder.set_atomic_rho_r(type_idx, data["r"], data["rho_core"], data["rab"])

    rho_core = dftcu.RealField(grid)
    builder.build_density(rho_core)
    return rho_core


def get_optimal_fft_grid(ecutrho, lattice, q_factor=1.0):
    """
    Compute optimal FFT grid dimensions matching QE logic.
    """
    BOHR_TO_ANG = constants.BOHR_TO_ANGSTROM
    lat = np.array(lattice).reshape(3, 3)
    g_max = np.sqrt(ecutrho)

    nr = []
    for i in range(3):
        a_len_bohr = np.linalg.norm(lat[i]) / BOHR_TO_ANG
        n_min = int(np.ceil(g_max * a_len_bohr / np.pi * q_factor))

        # Simple FFT friendly finder (powers of 2, 3, 5)
        n = n_min
        while True:
            m = n
            for p in [2, 3, 5]:
                while m > 0 and m % p == 0:
                    m //= p
            if m == 1:
                break
            n += 1
        nr.append(n)

    return nr


def initialize_wavefunctions(grid, atoms, upf_files, num_bands, ecutwfc=30.0):
    """
    Initialize wavefunctions using atomic superposition (starting_wfc='atomic').
    """
    upf_data_map = {}
    z_total = 0.0
    for i, path in enumerate(upf_files):
        data = dftcu.parse_upf(path)
        upf_data_map[i] = data

    for i in range(atoms.nat()):
        type_idx = atoms.h_type()[i]
        z_total += upf_data_map[type_idx]["zp"]

    # 1. Setup Wavefunction object
    psi = dftcu.Wavefunction(grid, num_bands, ecutwfc * 0.5)

    # 2. Setup WavefunctionBuilder
    builder = dftcu.WavefunctionBuilder(grid, atoms)
    has_chi = False
    for type_idx, data in upf_data_map.items():
        if "chi" in data and len(data["chi"]) > 0:
            has_chi = True
            for l_val, chi_r in zip(data["chi_l"], data["chi"]):
                builder.add_atomic_orbital(type_idx, l_val, data["r"], chi_r, data["rab"])

    # 3. Build
    if has_chi:
        builder.build_atomic_wavefunctions(psi)
        # Normalize wavefunctions to match QE convention: <psi|psi> = 1
        # (not <psi|psi> = nnr as in plane-wave codes)
        psi.orthonormalize()
    else:
        # Match QE: fallback to random if no atomic wavefunctions are provided
        psi.randomize()  # randomize() already calls orthonormalize()

    return psi


def initialize_hamiltonian(  # noqa: C901
    grid, atoms, upf_files, ecutwfc=30.0, ecutrho=None, rho_init=None, injection_path=None
):
    if ecutrho is None:
        ecutrho = 4.0 * ecutwfc

    upf_data_map = {}
    z_total = 0.0
    for i, path in enumerate(upf_files):
        data = dftcu.parse_upf(path)
        upf_data_map[i] = data

    for i in range(atoms.nat()):
        type_idx = atoms.h_type()[i]
        z_total += upf_data_map[type_idx]["zp"]

    # 1. Build Densities
    if rho_init is None:
        if injection_path and os.path.exists(os.path.join(injection_path, "qe_rho_init.txt")):
            target_file = os.path.join(injection_path, "qe_rho_init.txt")
            print(f"--> Injecting QE initial density from {target_file}")
            with open(target_file, "r") as f:
                f.readline()  # Skip header
                data = np.fromfile(f, sep=" ")
                # Handle layout if needed
                # For now assume matches DFTcu layout or handled by exporter
                rho_val = dftcu.RealField(grid)
                rho_val.copy_from_host(data)
        else:
            rho_val = build_initial_density(grid, atoms, upf_data_map, ecutrho)
            BOHR_TO_ANG = constants.BOHR_TO_ANGSTROM
            current_charge_electrons = rho_val.integral() / (BOHR_TO_ANG**3)

            if abs(current_charge_electrons - z_total) > 1e-7:
                scale = z_total / current_charge_electrons
                rho_np = np.zeros(grid.nnr())
                rho_val.copy_to_host(rho_np)
                rho_val.copy_from_host(rho_np * scale)
    else:
        if isinstance(rho_init, np.ndarray):
            rho_val = dftcu.RealField(grid)
            rho_val.copy_from_host(rho_init)
        else:
            rho_val = rho_init

    rho_core = get_core_density(grid, atoms, upf_data_map, ecutrho)
    evaluator = dftcu.Evaluator(grid)

    hartree = dftcu.Hartree()
    hartree.set_gcut(-1.0)
    lda = dftcu.LDA_PZ()
    lda.set_rho_threshold(1e-10)

    vloc = dftcu.LocalPseudo(grid, atoms)
    for type_idx, data in upf_data_map.items():
        vloc.init_tab_vloc(
            type_idx, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume()
        )
    vloc.set_gcut(ecutrho)

    evaluator.add_functional(hartree)
    evaluator.add_functional(lda)
    evaluator.add_functional(vloc)

    vh = dftcu.RealField(grid)
    hartree.compute(rho_val, vh)

    vxc = dftcu.RealField(grid)
    rho_val_np = np.zeros(grid.nnr())
    rho_val.copy_to_host(rho_val_np)
    if rho_core is not None:
        rho_core_np = np.zeros(grid.nnr())
        rho_core.copy_to_host(rho_core_np)
        rho_xc_np = np.abs(rho_val_np + rho_core_np)
    else:
        rho_xc_np = np.abs(rho_val_np)

    rho_xc = dftcu.RealField(grid)
    rho_xc.copy_from_host(rho_xc_np)
    lda.compute(rho_xc, vxc)

    vps = dftcu.RealField(grid)
    vloc.compute_potential(vps)

    nl_pseudo = None
    if any(len(d["betas"]) > 0 for d in upf_data_map.values()):
        nl_pseudo = dftcu.NonLocalPseudo(grid)
        for type_idx, data in upf_data_map.items():
            if len(data["betas"]) > 0:
                nl_pseudo.init_tab_beta(
                    type_idx,
                    data["r"],
                    data["betas"],
                    data["rab"],
                    data["l_list"],
                    data["kkbeta_list"],
                    grid.volume(),
                )

                # --- Phase 1 Comparison ---
                if injection_path:
                    target_file = os.path.join(injection_path, "qe_tab_beta.dat")
                    if os.path.exists(target_file):
                        all_tabs = []
                        curr = []
                        with open(target_file, "r") as f:
                            for line in f:
                                if line.startswith("#"):
                                    if curr:
                                        all_tabs.append(np.array(curr))
                                        curr = []
                                    continue
                                p = line.split()
                                if len(p) >= 3:
                                    curr.append(float(p[2]))
                            if curr:
                                all_tabs.append(np.array(curr))

                        qe_tab = all_tabs[type_idx * len(data["betas"])]  # simplify
                        cu_tab = nl_pseudo.get_tab_beta(type_idx, 0)
                        limit = min(len(qe_tab), len(cu_tab))
                        diff = np.max(np.abs(qe_tab[:limit] - cu_tab[:limit]))
                        print(f"--> Phase 1 (Radial) Diff: {diff:.6e}")

                        # Inject for Phase 2
                        for nb in range(len(data["betas"])):
                            idx = type_idx * len(data["betas"]) + nb
                            if idx < len(all_tabs):
                                nl_pseudo.set_tab_beta(type_idx, nb, all_tabs[idx].tolist())

                nl_pseudo.init_dij(type_idx, data["dij"])
        nl_pseudo.update_projectors(atoms)

    ham = dftcu.Hamiltonian(grid, evaluator)
    if nl_pseudo:
        ham.set_nonlocal(nl_pseudo)

    # Initialize potentials in Hamiltonian from rho_val
    ham.update_potentials(rho_val)

    # Return vloc so SCF can use it for potential updates
    return rho_val, ham, vloc


def solve_generalized_eigenvalue_problem(grid, h_matrix, s_matrix):
    solver = dftcu.SubspaceSolver(grid)
    return solver.solve_generalized(h_matrix, s_matrix)


def verify_native_subspace_alignment(qe_data_dir, grid, upf_files):
    import os

    BOHR_TO_ANG = constants.BOHR_TO_ANGSTROM
    a_ang = 18.897261 * BOHR_TO_ANG
    atoms = dftcu.Atoms([dftcu.Atom(a_ang / 2, a_ang / 2, a_ang / 2, 6.0, 0)])

    rho_init = dftcu.RealField(grid)
    rho_file = os.path.join(qe_data_dir, "qe_rho_init.txt")
    if os.path.exists(rho_file):
        with open(rho_file, "r") as f:
            f.readline()
            data = np.fromfile(f, sep=" ")
            rho_init.copy_from_host(data)

    _, ham, _ = initialize_hamiltonian(
        grid, atoms, upf_files, ecutwfc=100.0, rho_init=rho_init, injection_path=None
    )

    miller_file = os.path.join(qe_data_dir, "qe_wfc_g_atomic.dat")
    with open(miller_file, "r") as f:
        header = f.readline().split()
        npw_qe = int(header[0])

    def _load_miller(filename, npw):
        miller = []
        with open(filename, "r") as f:
            f.readline()
            for _ in range(npw):
                parts = f.readline().split()
                miller.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return np.array(miller)

    miller = _load_miller(miller_file, npw_qe)

    with open(os.path.join(qe_data_dir, "qe_psi_subspace.dat"), "r") as f:
        header = f.readline().split()
        n_bands = int(header[0])
        data = np.fromfile(f, sep=" ").reshape(n_bands, npw_qe, 2)
        coeffs = data[:, :, 0] + 1j * data[:, :, 1]

    psi = dftcu.Wavefunction(grid, n_bands, 50.0)
    psi.set_coefficients_miller(
        miller[:, 0].tolist(), miller[:, 1].tolist(), miller[:, 2].tolist(), coeffs.flatten()
    )

    h_psi = dftcu.Wavefunction(grid, n_bands, 50.0)
    ham.apply(psi, h_psi)

    # Use internalized direct solver (GPU-accelerated, no Python logic)
    solver = dftcu.SubspaceSolver(grid)
    eigenvalues = np.array(solver.solve_direct(ham, psi))

    # For debugging/breakdown printout ONLY (no part in eigenvalue computation)
    psi_np = np.zeros((n_bands, grid.nnr()), dtype=complex)
    psi.copy_to_host(psi_np)
    h_psi_np = np.zeros((n_bands, grid.nnr()), dtype=complex)
    h_psi.copy_to_host(h_psi_np)

    # Load QE reference H*psi
    h_psi_ref_file = os.path.join(qe_data_dir, "qe_hpsi_subspace.dat")
    with open(h_psi_ref_file, "r") as f:
        header = f.readline().split()
        nb_ref, npw_ref = int(header[0]), int(header[1])
        ref_data = np.fromfile(f, sep=" ").reshape(nb_ref, npw_ref, 2)
        ref_coeffs = ref_data[:, :, 0] + 1j * ref_data[:, :, 1]
        h_psi_ref_wf = dftcu.Wavefunction(grid, nb_ref, 50.0)
        h_psi_ref_wf.set_coefficients_miller(
            miller[:, 0].tolist(),
            miller[:, 1].tolist(),
            miller[:, 2].tolist(),
            ref_coeffs.flatten(),
        )
        h_psi_ref_np = np.zeros((nb_ref, grid.nnr()), dtype=complex)
        h_psi_ref_wf.copy_to_host(h_psi_ref_np)

    gg_bohr = np.array(grid.gg()) * (BOHR_TO_ANG**2)

    print("\n--- Detailed Operator Breakdown (Native vs QE Ha) ---")
    for b in range(n_bands):
        e_kin = np.sum(np.abs(psi_np[b]) ** 2 * 0.5 * gg_bohr).real
        nat_total = np.sum(np.conj(psi_np[b]) * h_psi_np[b]).real
        ref_total = np.sum(np.conj(psi_np[b]) * h_psi_ref_np[b]).real * 0.5
        print(
            f"Band {b}: Kinetic={e_kin:8.4f}, Pot_Total={nat_total - e_kin:8.4f}, "
            f"Total={nat_total:8.4f} | Ref={ref_total:8.4f} | Diff={nat_total - ref_total:8.4f}"
        )
    print("--------------------------------------------\n")

    with open(os.path.join(qe_data_dir, "qe_eig_step0.dat"), "r") as f:
        f.readline()
        eig_qe = np.fromfile(f, sep="\n") * 0.5
    print(
        "--> NATIVE Subspace Alignment Complete. Max Diff: "
        f"{np.max(np.abs(eigenvalues - eig_qe)):.6e} Ha"
    )
    return eigenvalues, eig_qe, np.max(np.abs(eigenvalues - eig_qe))


def verify_qe_subspace_alignment(qe_data_dir, grid):
    import os

    miller_file = os.path.join(qe_data_dir, "qe_wfc_g_atomic.dat")
    with open(miller_file, "r") as f:
        header = f.readline().split()
        npw_qe = int(header[0])

    def _load_miller(filename, npw):
        miller = []
        with open(filename, "r") as f:
            f.readline()
            for _ in range(npw):
                parts = f.readline().split()
                miller.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return np.array(miller)

    miller = _load_miller(miller_file, npw_qe)

    def _load_into_wf(filename):
        with open(os.path.join(qe_data_dir, filename), "r") as f:
            header = f.readline().split()
            n_bands, npw = int(header[0]), int(header[1])
            data = np.fromfile(f, sep=" ").reshape(n_bands, npw, 2)
            coeffs = data[:, :, 0] + 1j * data[:, :, 1]
            wf = dftcu.Wavefunction(grid, n_bands, 50.0)
            wf.set_coefficients_miller(
                miller[:, 0].tolist(),
                miller[:, 1].tolist(),
                miller[:, 2].tolist(),
                coeffs.flatten(),
            )
            return wf

    psi = _load_into_wf("qe_psi_subspace.dat")
    hpsi = _load_into_wf("qe_hpsi_subspace.dat")
    num_bands = psi.num_bands()
    psi_np = np.zeros((num_bands, grid.nnr()), dtype=complex)
    h_psi_np = np.zeros((num_bands, grid.nnr()), dtype=complex)
    psi.copy_to_host(psi_np)
    hpsi.copy_to_host(h_psi_np)
    omega = grid.volume_bohr()
    h_m = np.zeros((num_bands, num_bands), dtype=complex)
    s_m = np.zeros((num_bands, num_bands), dtype=complex)
    for i in range(num_bands):
        for j in range(num_bands):
            h_m[i, j] = np.sum(np.conj(psi_np[i]) * h_psi_np[j]) * 0.5 * omega
            s_m[i, j] = np.sum(np.conj(psi_np[i]) * psi_np[j]) * omega
    eig = solve_generalized_eigenvalue_problem(grid, h_m, s_m)
    with open(os.path.join(qe_data_dir, "qe_eig_step0.dat"), "r") as f:
        f.readline()
        eig_qe = np.fromfile(f, sep="\n") * 0.5
    print(f"--> Alignment Verification Complete. Max Diff: {np.max(np.abs(eig - eig_qe)):.6e} Ha")
    return eig, eig_qe, np.max(np.abs(eig - eig_qe))
