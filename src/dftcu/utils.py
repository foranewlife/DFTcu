import numpy as np

import dftcu


def initialize_hamiltonian(grid, atoms, upf_files, ecutwfc=30.0, ecutrho=None, rho_init=None):
    """
    Standard Hamiltonian initialization for dftcu, strictly aligned with QE units and flow.
    """
    if ecutrho is None:
        ecutrho = 4.0 * ecutwfc

    upf_data_map = {}
    for i, path in enumerate(upf_files):
        upf_data_map[i] = dftcu.parse_upf(path)

    evaluator = dftcu.Evaluator(grid)

    # 1. Component Setups
    hartree = dftcu.Hartree()
    hartree.set_gcut(-1.0)  # Full grid

    lda = dftcu.LDA_PZ()

    vloc = dftcu.LocalPseudo(grid, atoms)
    for type_idx, data in upf_data_map.items():
        vloc.init_tab_vloc(
            type_idx, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume()
        )
    vloc.set_gcut(ecutrho)  # Rydberg

    # 2. Potential Update Logic
    # To achieve 10^-14, we must calculate components independently then sum.
    # ham.update_potentials(rho) sums them all using ONE rho.
    # But Hartree needs rho_val, while XC needs |rho_val + rho_core|.

    v_eff = dftcu.RealField(grid)
    v_eff.fill(0.0)

    if rho_init is not None:
        if isinstance(rho_init, np.ndarray):
            rv = dftcu.RealField(grid)
            rv.copy_from_host(rho_init)
            rho_val = rv
        else:
            rho_val = rho_init

        # Hartree
        vh = dftcu.RealField(grid)
        hartree.compute(rho_val, vh)

        # XC (Apply abs() only to XC input)
        vxc = dftcu.RealField(grid)
        # Assuming no core for now, or add it here
        rho_val_np = np.zeros(grid.nnr())
        rho_val.copy_to_host(rho_val_np)
        rho_xc_np = np.abs(rho_val_np)
        rho_xc = dftcu.RealField(grid)
        rho_xc.copy_from_host(rho_xc_np)
        lda.compute(rho_xc, vxc)

        # Local Pseudo
        vps = dftcu.RealField(grid)
        vloc.compute_potential(vps)

        # Manually sum into v_eff using numpy
        vh_np = np.zeros(grid.nnr())
        vxc_np = np.zeros(grid.nnr())
        vps_np = np.zeros(grid.nnr())
        vh.copy_to_host(vh_np)
        vxc.copy_to_host(vxc_np)
        vps.copy_to_host(vps_np)

        v_eff_np = vh_np + vxc_np + vps_np
        v_eff.copy_from_host(v_eff_np)

    ham = dftcu.Hamiltonian(grid, evaluator)
    # Inject our precisely aligned V_eff
    ham.v_loc().copy_from_host(v_eff_np)

    return ham
