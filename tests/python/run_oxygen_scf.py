#!/usr/bin/env python3
import xml.etree.ElementTree as ET

import dftcu
import numpy as np

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
    grid = dftcu.Grid([a, 0, 0, 0, a, 0, 0, 0, a], [72, 72, 72])
    vol_bohr = grid.volume() / (BOHR_TO_ANGSTROM**3)
    data = parse_upf("run_qe_lda/O_ONCV_PBE-1.2.upf")
    pos_bohr = 9.4486
    pos_ang = pos_bohr * BOHR_TO_ANGSTROM
    atoms = dftcu.Atoms([dftcu.Atom(pos_ang, pos_ang, pos_ang, data["zp"], 0)])

    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.Hartree())
    evaluator.add_functional(dftcu.LDA_PZ())
    ewald = dftcu.Ewald(grid, atoms, 1e-10)
    ewald.set_eta(1.6)
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
    options.mixing_beta = 0.1
    options.davidson_max_iter = 10
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
    occupations = [2.0, 2.0, 2.0, 0.0]
    psi = dftcu.Wavefunction(grid, nbnd, 60.0)
    psi.randomize(1234)

    e_final_ha = solver.solve(ham, psi, occupations, rho)

    # 6. Final Comparison
    alpha_ry = 0.0010267131 * 6.0  # N_elec * V_avg
    e_total_ry = e_final_ha * HARTREE_TO_RY + alpha_ry
    print(f"\nFINAL CORRECTED ENERGY: {e_total_ry:15.10f} Ry")
    print(f"QE Reference Value:     {-30.45101307:15.10f} Ry")


if __name__ == "__main__":
    run_scf()
