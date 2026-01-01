#!/usr/bin/env python3
import xml.etree.ElementTree as ET

import numpy as np

import dftcu

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_RY = 2.0


def parse_upf(filename):
    """Simple UPF v2 parser for ONCV pseudopotentials."""
    tree = ET.parse(filename)
    root = tree.getroot()

    grid_node = root.find(".//PP_MESH/PP_R")
    r_grid = np.fromstring(grid_node.text, sep=" ")
    rab_node = root.find(".//PP_MESH/PP_RAB")
    rab = np.fromstring(rab_node.text, sep=" ")
    vloc_node = root.find(".//PP_LOCAL")
    vloc_r = np.fromstring(vloc_node.text, sep=" ")
    betas = []
    l_list = []
    nl_node = root.find(".//PP_NONLOCAL")
    for child in nl_node:
        if child.tag.startswith("PP_BETA."):
            betas.append(np.fromstring(child.text, sep=" "))
            l_list.append(int(child.attrib["angular_momentum"]))
    dij_node = root.find(".//PP_NONLOCAL/PP_DIJ")
    dij_flat = np.fromstring(dij_node.text, sep=" ")
    header = root.find(".//PP_HEADER")
    z_valence = float(header.attrib["z_valence"])
    rho_at_node = root.find(".//PP_RHOATOM")
    rho_at_r = np.fromstring(rho_at_node.text, sep=" ")
    return {
        "r": r_grid,
        "rab": rab,
        "vloc": vloc_r,
        "betas": betas,
        "l_list": l_list,
        "dij": dij_flat,
        "zp": z_valence,
        "rho_at": rho_at_r,
    }


def get_rho_g_table(data, omega_bohr):
    """Perform radial Fourier transform of atomic density."""
    r = data["r"]
    rho_r = data["rho_at"]
    rab = data["rab"]
    qs = np.linspace(0, 15.0, 1000)
    rho_gs = []
    for q in qs:
        if q < 1e-12:
            rho_gs.append(np.sum(rho_r * rab))
        else:
            # Sinc function handle
            rho_gs.append(np.sum(rho_r * np.sinc(q * r / np.pi) * rab))
    return qs, np.array(rho_gs) / omega_bohr


def run_scf():
    a = 10.0
    grid = dftcu.Grid([a, 0, 0, 0, a, 0, 0, 0, a], [96, 96, 96])
    vol_bohr = grid.volume() / (BOHR_TO_ANGSTROM**3)
    data = parse_upf("run_qe_lda/O_ONCV_PBE-1.2.upf")
    print(f"Parsed Z_valence: {data['zp']}")
    print(f"Rho_at size:      {len(data['rho_at'])}")
    print(f"Rho_at integral:  {np.sum(data['rho_at'] * data['rab']):.5f}")
    pos_bohr = 9.4486
    pos_ang = pos_bohr * BOHR_TO_ANGSTROM
    atoms = dftcu.Atoms([dftcu.Atom(pos_ang, pos_ang, pos_ang, data["zp"], 0)])

    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.Hartree())
    evaluator.add_functional(dftcu.LDA_PZ())
    ewald = dftcu.Ewald(grid, atoms, 1e-10)
    ewald.set_eta(1.0)  # Match LocalPseudo's hardcoded eta
    evaluator.add_functional(ewald)
    vloc = dftcu.LocalPseudo(grid, atoms)
    vloc.init_tab_vloc(0, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume())
    evaluator.add_functional(vloc)

    nl_pseudo = dftcu.NonLocalPseudo(grid)
    nl_pseudo.init_tab_beta(0, data["r"], data["betas"], data["rab"], data["l_list"], grid.volume())
    nl_pseudo.init_dij(0, data["dij"])
    nl_pseudo.update_projectors(atoms)
    ham = dftcu.Hamiltonian(grid, evaluator)
    ham.set_nonlocal(nl_pseudo)

    options = dftcu.SCFOptions()
    options.max_iter = 100
    options.mixing_type = dftcu.MixingType.Broyden
    options.mixing_beta = 0.7  # Match QE
    options.mixing_history = 8
    options.davidson_max_iter = 50
    options.davidson_tol = 1e-8
    options.verbose = True
    solver = dftcu.SCFSolver(grid, options)

    builder = dftcu.DensityBuilder(grid, atoms)
    qs, rho_gs = get_rho_g_table(data, vol_bohr)
    builder.set_atomic_rho_g(0, list(qs), list(rho_gs))
    rho = dftcu.RealField(grid)
    builder.build_density(rho)

    # Corrected normalization check in Ha units
    print(f"Initial charge (electrons): {rho.integral() / (BOHR_TO_ANGSTROM**3):.5f}")

    nbnd = 4
    occupations = [2.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]
    psi = dftcu.Wavefunction(grid, nbnd, 15.0)  # Match QE ecutwfc=30Ry
    psi.randomize(1234)

    solver.solve(ham, psi, occupations, rho)

    # 6. Eigenvalue comparison
    final_davidson = dftcu.DavidsonSolver(grid, 50, 1e-9)
    ev_ha = final_davidson.solve(ham, psi)
    ev_ry = [e * HARTREE_TO_RY for e in ev_ha]

    # Apply Alpha shift for comparison with QE
    alpha_ha = vloc.get_alpha(0)
    ev_shifted_ry = [(e + alpha_ha) * HARTREE_TO_RY for e in ev_ha]

    print("\n" + "=" * 45)
    print("EIGENVALUE COMPARISON (Ry)")
    print("=" * 45)
    print(f"Alpha Shift (Ry):  {alpha_ha * HARTREE_TO_RY:15.8f}")
    for i, (ery, ery_s) in enumerate(zip(ev_ry, ev_shifted_ry)):
        print(f"Band {i}: Raw={ery:10.5f} Ry, Shifted={ery_s:10.5f} Ry")

    print("\nQE Reference (Ry):  -1.889, -0.612, -0.612, -0.612")

    # Final total charge check

    total_charge = rho.integral() / (BOHR_TO_ANGSTROM**3)
    print(f"\nFinal total charge:      {total_charge:.10f} e⁻")

    # Component Breakdown and Comparison
    # QE's one-electron contribution = E_kin + E_vloc + E_nl
    # DFTcu e_final_ha = E_kin + E_nl + E_vloc_sr + E_H + E_XC + E_ewald

    e_kin_ry = psi.compute_kinetic_energy(occupations) * HARTREE_TO_RY
    e_nl_ry = nl_pseudo.calculate_energy(psi, occupations) * HARTREE_TO_RY

    # Calculate components separately
    vh = dftcu.RealField(grid)
    vxc = dftcu.RealField(grid)
    e_h_ha = dftcu.Hartree().compute(rho, vh)
    e_xc_ha = dftcu.LDA_PZ().compute(rho, vxc)
    e_ewald_ha = ewald.compute(False)

    # Local potential energy
    vloc_field = dftcu.RealField(grid)
    vloc.compute_potential(vloc_field)  # Fix API name

    alpha_ha = vloc.get_alpha(0)
    e_vloc_sr_ry = (rho.dot(vloc_field) * grid.dv_bohr()) * HARTREE_TO_RY
    e_alpha_ry = (alpha_ha * 6.0) * HARTREE_TO_RY  # N_elec * alpha

    print("\n" + "=" * 45)
    print("DETAILED COMPONENT COMPARISON VS QE (Ry)")
    print("=" * 45)
    print(f"Kinetic Energy:     {e_kin_ry:15.8f}")
    print(f"Non-local Energy:   {e_nl_ry:15.8f}")
    print(f"V_loc SR Energy:    {e_vloc_sr_ry:15.8f}")
    print(f"Alpha Correction:   {e_alpha_ry:15.8f}")

    one_elec_ry = e_kin_ry + e_nl_ry + e_vloc_sr_ry + e_alpha_ry
    print(f"One-elec (Total):   {one_elec_ry:15.8f} (QE: -39.63496881)")
    print(f"Hartree Energy:     {e_h_ha * HARTREE_TO_RY:15.8f} (QE:  20.72013453)")
    print(f"XC Energy (PZ):     {e_xc_ha * HARTREE_TO_RY:15.8f} (QE:  -6.13101934)")
    print(f"Ewald Energy:       {e_ewald_ha * HARTREE_TO_RY:15.8f} (QE:  -5.40515945)")

    e_total_ry = one_elec_ry + (e_h_ha + e_xc_ha + e_ewald_ha) * HARTREE_TO_RY
    print("-" * 45)
    print(f"FINAL TOTAL ENERGY: {e_total_ry:15.10f} Ry")
    print(f"QE REFERENCE:       {-30.45101307:15.10f} Ry")
    print(f"ERROR:              {e_total_ry - (-30.45101307):15.10f} Ry")


if __name__ == "__main__":
    run_scf()
