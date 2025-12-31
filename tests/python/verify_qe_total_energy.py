#!/usr/bin/env python3
import struct
import xml.etree.ElementTree as ET

import dftcu
import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_RY = 2.0


def read_fortran_record(f):
    header = f.read(4)
    if not header:
        return None
    size = struct.unpack("i", header)[0]
    data = f.read(size)
    f.read(4)  # footer
    return data


def parse_upf(filename):
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
    return {
        "r": r_grid,
        "rab": rab,
        "vloc": vloc_r,
        "betas": betas,
        "l_list": l_list,
        "dij": dij_flat,
        "zp": z_valence,
    }


def run_final_victory():
    # 1. Setup Grid
    nr = [72, 72, 72]
    a = 10.0
    lattice = [a, 0, 0, 0, a, 0, 0, 0, a]
    grid = dftcu.Grid(lattice, nr)
    nnr = grid.nnr()

    # 2. Load Data
    with open("run_qe_lda/qe_wfc.dat", "rb") as f:
        header = read_fortran_record(f)
        nbnd, npw = struct.unpack("ii", header)
        read_fortran_record(f)  # skip eigenvalues
        evc_qe = np.frombuffer(read_fortran_record(f), dtype=np.complex128).reshape((npw, nbnd))
    with open("run_qe_lda/qe_wfc_indices.dat", "rb") as f:
        read_fortran_record(f)  # skip npw
        indices_raw = np.frombuffer(read_fortran_record(f), dtype=np.int32).reshape((npw, 3))
    with open("run_qe_lda/qe_rho_r.dat", "rb") as f:
        read_fortran_record(f)  # skip nnr
        rho_r_data = np.frombuffer(read_fortran_record(f), dtype=np.float64)

    # 3. Inject Wavefunction (with 1/sqrt(Omega) normalization)
    def qe_to_linear(m1, m2, m3, nr):
        i1 = int(m1) % nr[0]
        i2 = int(m2) % nr[1]
        i3 = int(m3) % nr[2]
        return i1 * (nr[1] * nr[2]) + i2 * nr[2] + i3

    psi = dftcu.Wavefunction(grid, nbnd, 120.0)
    for ib in range(nbnd):
        full_coeffs = np.zeros(nnr, dtype=np.complex128)
        for ig in range(npw):
            m1, m2, m3 = indices_raw[ig]
            idx = qe_to_linear(m1, m2, m3, nr)
            full_coeffs[idx] = evc_qe[ig, ib]
            if m1 != 0 or m2 != 0 or m3 != 0:
                idx_mirror = qe_to_linear(-m1, -m2, -m3, nr)
                full_coeffs[idx_mirror] = np.conj(evc_qe[ig, ib])

        psi.set_coefficients(full_coeffs, ib)

    # 4. Calculate components in DFTcu (G=0 local potential is 0)
    occupations = [2.0] * 3 + [0.0] * (nbnd - 3)
    e_kin_ry = psi.compute_kinetic_energy(occupations) * HARTREE_TO_RY

    # Load UPF and Non-local
    data = parse_upf("run_qe_lda/O_ONCV_PBE-1.2.upf")
    atoms = dftcu.Atoms(
        [
            dftcu.Atom(
                9.4486 * BOHR_TO_ANGSTROM,
                9.4486 * BOHR_TO_ANGSTROM,
                9.4486 * BOHR_TO_ANGSTROM,
                data["zp"],
                0,
            )
        ]
    )
    nl_pseudo = dftcu.NonLocalPseudo(grid)
    nl_pseudo.init_tab_beta(0, data["r"], data["betas"], data["rab"], data["l_list"], grid.volume())
    nl_pseudo.init_dij(0, data["dij"])
    nl_pseudo.update_projectors(atoms)
    e_nl_ry = nl_pseudo.calculate_energy(psi, occupations) * HARTREE_TO_RY

    rho = dftcu.RealField(grid)
    rho.copy_from_host(rho_r_data)
    e_h_ry = dftcu.Hartree().compute(rho, dftcu.RealField(grid)) * 2.0
    e_xc_ry = dftcu.LDA_PZ().compute(rho, dftcu.RealField(grid)) * 2.0
    ewald_obj = dftcu.Ewald(grid, atoms, 1e-10)
    ewald_obj.set_eta(1.6)
    e_ew_ry = ewald_obj.compute(False) * 2.0

    vloc = dftcu.LocalPseudo(grid, atoms)
    vloc.init_tab_vloc(0, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume())
    e_vl_ry = vloc.compute(rho, dftcu.RealField(grid)) * 2.0

    # 5. Get Alpha term value from QE tab_vloc
    v_avg_ry = 0.0010267131
    alpha_energy_correction = 6.0 * v_avg_ry  # N_elec * V_avg

    print("=" * 70)
    print("FINAL VICTORY REPORT: OXYGEN ATOM TOTAL ENERGY")
    print("=" * 70)

    print(f"1. Kinetic Energy (T):      {e_kin_ry:18.10f} Ry")
    print(f"2. Non-local Energy (Vnl):  {e_nl_ry:18.10f} Ry")
    print(f"3. Local Pseudo (G>0 only): {e_vl_ry:18.10f} Ry")
    print(f"4. Hartree Energy (Eh):     {e_h_ry:18.10f} Ry")
    print(f"5. XC Energy (LDA):         {e_xc_ry:18.10f} Ry")
    print(f"6. Ewald Energy (Eew):      {e_ew_ry:18.10f} Ry")
    print("-" * 70)

    one_el_dftcu = e_kin_ry + e_nl_ry + e_vl_ry
    total_raw = one_el_dftcu + e_h_ry + e_xc_ry + e_ew_ry

    print(f"Alpha Correction Value:     {alpha_energy_correction:18.10f} Ry")
    print(f"Total Corrected (DFTcu):    {total_raw + alpha_energy_correction:18.10f} Ry")
    print(f"Total Reference (QE):       {-30.45101307:18.10f} Ry")
    print("-" * 70)

    final_err = (total_raw + alpha_energy_correction) - (-30.45101307)
    print(f"FINAL RESIDUAL ERROR:       {final_err:18.10e} Ry")

    if abs(final_err) < 1e-7:
        print("\n🏆 ALIGNMENT SUCCESSFUL! WE HAVE REPLICATED QE TOTAL ENERGY.")
    else:
        print("\n❌ STILL A TINY MISMATCH. CHECKING CONSTANTS...")


if __name__ == "__main__":
    run_final_victory()
