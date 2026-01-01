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


def initialize_hamiltonian(grid, atoms, upf_files, ecutwfc=30.0, ecutrho=None, rho_init=None):
    """
    Standard Hamiltonian initialization for dftcu, strictly aligned with QE units and flow.
    If rho_init is None, it generates density via atomic superposition and renormalizes.
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

    return rho_val, ham
