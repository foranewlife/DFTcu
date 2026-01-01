import numpy as np

import dftcu


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


def is_fft_friendly(n):
    """Check if n only has small prime factors (2, 3, 5, 7)."""
    if n <= 0:
        return False
    for p in [2, 3, 5, 7]:
        while n % p == 0:
            n //= p
    return n == 1


def next_fft_friendly(n):
    """Find the next integer >= n that is FFT friendly."""
    while not is_fft_friendly(n):
        n += 1
    return n


def get_optimal_fft_grid(ecutrho, lattice, q_factor=1.0):
    """
    Compute optimal FFT grid dimensions matching QE logic.
    Args:
        ecutrho (float): Charge density cutoff in Rydberg.
        lattice (list or np.array): 3x3 lattice matrix in Angstrom.
        q_factor (float): Additional scaling factor (default 1.0).
    Returns:
        list: [nr1, nr2, nr3]
    """
    BOHR_TO_ANG = 0.529177210903
    lat = np.array(lattice).reshape(3, 3)

    # g_max in Bohr^-1. (1 Ry = 1 Bohr^-2)
    g_max = np.sqrt(ecutrho)

    nr = []
    for i in range(3):
        # Length of lattice vector in Bohr
        a_len_bohr = np.linalg.norm(lat[i]) / BOHR_TO_ANG
        # Minimal nr to avoid aliasing: nr >= 2 * g_max * |a| / (2*pi) = g_max * |a| / pi
        n_min = int(np.ceil(g_max * a_len_bohr / np.pi * q_factor))
        nr.append(next_fft_friendly(n_min))

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
    else:
        # Match QE: fallback to random if no atomic wavefunctions are provided
        psi.randomize()

    # [NOTE] Orthonormalization is disabled here for direct alignment verification.
    # It should be called explicitly before solving if needed.
    # psi.orthonormalize()

    return psi


def initialize_hamiltonian(  # noqa: C901
    grid, atoms, upf_files, ecutwfc=30.0, ecutrho=None, rho_init=None, injection_path=None
):
    """
    Standard Hamiltonian initialization for dftcu, strictly aligned with QE units and flow.
    If rho_init is None, it generates density via atomic superposition and renormalizes.
    Args:
        injection_path (str): Optional path to load QE's high-precision radial tables
            (qe_tab_beta.dat).
    """
    if ecutrho is None:
        ecutrho = 4.0 * ecutwfc

    upf_data_map = {}
    z_total = 0.0
    for i, path in enumerate(upf_files):
        data = dftcu.parse_upf(path)
        upf_data_map[i] = data
        # Count total expected valence electrons
        # Note: In a real system, we need to count atoms of each type

    # Accurate z_total calculation
    for i in range(atoms.nat()):
        type_idx = atoms.h_type()[i]
        z_total += upf_data_map[type_idx]["zp"]

    # 1. Build Densities
    if rho_init is None:
        rho_val = build_initial_density(grid, atoms, upf_data_map, ecutrho)

        # --- QE RENORMALIZATION (Consistent Units) ---
        BOHR_TO_ANG = 0.529177210903
        # integral() is in Ang^3. Convert to Bohr^3 (physical electrons)
        current_charge_electrons = rho_val.integral() / (BOHR_TO_ANG**3)

        if abs(current_charge_electrons - z_total) > 1e-7:
            scale = z_total / current_charge_electrons
            rho_np = np.zeros(grid.nnr())
            rho_val.copy_to_host(rho_np)
            rho_val.copy_from_host(rho_np * scale)
        # ---------------------------
    else:
        if isinstance(rho_init, np.ndarray):
            rho_val = dftcu.RealField(grid)
            rho_val.copy_from_host(rho_init)
        else:
            rho_val = rho_init

    rho_core = get_core_density(grid, atoms, upf_data_map, ecutrho)

    evaluator = dftcu.Evaluator(grid)

    # 2. Component Setups
    hartree = dftcu.Hartree()
    hartree.set_gcut(-1.0)  # Full grid

    lda = dftcu.LDA_PZ()
    lda.set_rho_threshold(1e-10)

    vloc = dftcu.LocalPseudo(grid, atoms)
    for type_idx, data in upf_data_map.items():
        vloc.init_tab_vloc(
            type_idx, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume()
        )
    vloc.set_gcut(ecutrho)  # Rydberg

    # 3. Potential Update Logic
    # Calculate components independently then sum.
    v_eff = dftcu.RealField(grid)
    v_eff.fill(0.0)

    # Hartree
    vh = dftcu.RealField(grid)
    hartree.compute(rho_val, vh)

    # XC (Apply abs() only to XC input)
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

    # Local Pseudo
    vps = dftcu.RealField(grid)
    vloc.compute_potential(vps)

    # Non-local Pseudo
    nl_pseudo = None
    has_nl = any(len(d["betas"]) > 0 for d in upf_data_map.values())
    if has_nl:
        nl_pseudo = dftcu.NonLocalPseudo(grid)
        for type_idx, data in upf_data_map.items():
            if len(data["betas"]) > 0:
                nl_pseudo.init_tab_beta(
                    type_idx, data["r"], data["betas"], data["rab"], data["l_list"], grid.volume()
                )

                # --- QE TABLE INJECTION ---
                if injection_path:
                    import os

                    target_file = os.path.join(injection_path, "qe_tab_beta.dat")
                    if os.path.exists(target_file):
                        # Extract table for this type/projector
                        # Simplified loader: assuming 1 type for now, or match type in loop
                        all_tabs = []
                        with open(target_file, "r") as f:
                            curr = []
                            for line in f:
                                if line.startswith("#"):
                                    continue
                                p = line.split()
                                if len(p) == 3:
                                    curr.append(float(p[2]))
                                if len(curr) == 1004:  # Match QE nqx
                                    all_tabs.append(np.array(curr))
                                    curr = []
                        # Inject matching projectors
                        if type_idx < len(all_tabs):
                            # This is a bit simplified, but enough for current alignment
                            for nb_idx in range(len(data["betas"])):
                                global_nb_idx = type_idx * len(data["betas"]) + nb_idx
                                if global_nb_idx < len(all_tabs):
                                    nl_pseudo.set_tab_beta(
                                        type_idx, nb_idx, all_tabs[global_nb_idx].tolist()
                                    )
                # --------------------------

                nl_pseudo.init_dij(type_idx, data["dij"])
        nl_pseudo.update_projectors(atoms)

    # Sum potentials
    vh_np = np.zeros(grid.nnr())
    vxc_np = np.zeros(grid.nnr())
    vps_np = np.zeros(grid.nnr())
    vh.copy_to_host(vh_np)
    vxc.copy_to_host(vxc_np)
    vps.copy_to_host(vps_np)

    v_eff_np = vh_np + vxc_np + vps_np

    ham = dftcu.Hamiltonian(grid, evaluator)
    ham.v_loc().copy_from_host(v_eff_np)
    if nl_pseudo is not None:
        ham.set_nonlocal(nl_pseudo)

    return rho_val, ham


# === QE Alignment & Compatibility Utilities ===


def load_qe_miller_indices(filename, npw):
    """Load Miller indices (h, k, l) from QE debug export."""
    miller = []
    with open(filename, "r") as f:
        f.readline()  # header
        for _ in range(npw):
            parts = f.readline().split()
            miller.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return miller


def expand_qe_wfc_to_full_grid(raw_data, miller_indices, nr):
    """
    Expand QE half-sphere G-space coefficients to full FFT grid
    using Hermitian symmetry (C_-G = C_G*).
    """
    nnr = nr[0] * nr[1] * nr[2]
    coeffs = np.zeros(nnr, dtype=complex)
    for ig, (h, k, l) in enumerate(miller_indices):
        val = raw_data[ig]
        # Positive G placement
        n0, n1, n2 = h % nr[0], k % nr[1], l % nr[2]
        idx = n0 * nr[1] * nr[2] + n1 * nr[2] + n2
        coeffs[idx] = val
        # Hermitian Negative G placement
        if h != 0 or k != 0 or l != 0:
            n0_i, n1_i, n2_i = (-h) % nr[0], (-k) % nr[1], (-l) % nr[2]
            idx_i = n0_i * nr[1] * nr[2] + n1_i * nr[2] + n2_i
            coeffs[idx_i] = np.conj(val)
    return coeffs


def solve_generalized_eigenvalue_problem(h_matrix, s_matrix):
    """
    Solve Hc = epsilon Sc using Cholesky decomposition to match QE's diaghg.
    """
    L = np.linalg.cholesky(s_matrix)
    L_inv = np.linalg.inv(L)
    h_prime = L_inv @ h_matrix @ L_inv.T
    return np.linalg.eigvalsh(h_prime)


def verify_qe_subspace_alignment(qe_data_dir, grid):
    """
    Ultimate diagnostic tool to verify DFTcu's subspace math against QE.
    Expects qe_psi_subspace.dat, qe_hpsi_subspace.dat, qe_wfc_g_atomic.dat,
    and qe_eig_step0.dat in qe_data_dir.
    """
    import os

    # 1. Load Miller Mapping
    miller_file = os.path.join(qe_data_dir, "qe_wfc_g_atomic.dat")
    with open(miller_file, "r") as f:
        header = f.readline().split()
        npw_qe = int(header[0])
    miller = load_qe_miller_indices(miller_file, npw_qe)

    # 2. Helper to load and expand
    def _load_and_expand(filename):
        with open(os.path.join(qe_data_dir, filename), "r") as f:
            header = f.readline().split()
            n_bands = int(header[0])
            data = np.fromfile(f, sep=" ").reshape(n_bands, npw_qe, 2)
            expanded = []
            for b in range(n_bands):
                c_g = expand_qe_wfc_to_full_grid(
                    data[b, :, 0] + 1j * data[b, :, 1], miller, grid.nr()
                )
                expanded.append(c_g)
            return np.array(expanded)

    print(f"--> Loading subspace data from {qe_data_dir}...")
    psi_all = _load_and_expand("qe_psi_subspace.dat")
    hpsi_all = _load_and_expand("qe_hpsi_subspace.dat") * 0.5  # Ry -> Ha

    num_bands = psi_all.shape[0]
    h_matrix = np.zeros((num_bands, num_bands))
    s_matrix = np.zeros((num_bands, num_bands))

    for i in range(num_bands):
        for j in range(num_bands):
            h_matrix[i, j] = np.sum(np.conj(psi_all[i]) * hpsi_all[j]).real
            s_matrix[i, j] = np.sum(np.conj(psi_all[i]) * psi_all[j]).real

    eigenvalues = solve_generalized_eigenvalue_problem(h_matrix, s_matrix)

    # 3. Load Truth
    with open(os.path.join(qe_data_dir, "qe_eig_step0.dat"), "r") as f:
        f.readline()
        eig_qe = np.fromfile(f, sep="\n") * 0.5

    max_diff = np.max(np.abs(eigenvalues - eig_qe))
    print(f"--> Alignment Verification Complete. Max Diff: {max_diff:.6e} Ha")

    return eigenvalues, eig_qe, max_diff
