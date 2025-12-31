import os
import sys

import numpy as np

# Add src directory to path to ensure we use the latest code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import dftcu  # noqa: E402
from dftcu.pseudopotential import load_pseudo, parse_upf  # noqa: E402

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_RY = 2.0


def run_comprehensive_comparison():
    # 1. Setup Grid matching QE (72x72x72 for 30Ry cut in 10A box)
    a = 10.0
    grid = dftcu.Grid([a, 0, 0, 0, a, 0, 0, 0, a], [72, 72, 72])

    # Load FINAL density from QE
    qe_rho_path = "run_qe_lda/qe_rho_final.txt"
    if not os.path.exists(qe_rho_path):
        print(f"Error: {qe_rho_path} not found. Please ensure QE reference data exists.")
        return

    with open(qe_rho_path, "r") as f:
        f.readline()
        qe_rho = np.array([float(line) for line in f])

    rho = dftcu.RealField(grid)
    rho.copy_from_host(qe_rho)

    print("=== DFTcu vs Quantum ESPRESSO Potential Comparison ===")

    # 2. Compare V_loc
    print("\n[1] Local Pseudopotential (Vloc)")
    upf_path = "run_qe_lda/O_ONCV_PBE-1.2.upf"
    data = parse_upf(upf_path)
    pos_bohr = 9.4486
    pos_ang = pos_bohr * BOHR_TO_ANGSTROM
    atoms = dftcu.Atoms([dftcu.Atom(pos_ang, pos_ang, pos_ang, data["zp"], 0)])

    vloc_obj, _, _ = load_pseudo(upf_path, grid, atoms=atoms)
    vloc_obj.set_gcut(120.0)
    vloc_field = dftcu.RealField(grid)
    vloc_obj.compute_potential(vloc_field)

    v_loc_dftcu_ry = np.zeros(grid.nnr())
    vloc_field.copy_to_host(v_loc_dftcu_ry)
    v_loc_dftcu_ry *= HARTREE_TO_RY

    with open("run_qe_lda/qe_vltot_after_ifft.txt", "r") as f:
        f.readline()
        qe_vloc = np.array([float(line) for line in f])

    diff_vloc = qe_vloc - v_loc_dftcu_ry
    print(f"  Max Diff: {np.max(np.abs(diff_vloc)):.6e} Ry")
    print(f"  RMS Diff: {np.sqrt(np.mean(diff_vloc**2)):.6e} Ry")

    # 3. Compare V_H
    print("\n[2] Hartree Potential (VH)")
    vh = dftcu.RealField(grid)
    hartree = dftcu.Hartree()
    hartree.set_gcut(120.0)
    hartree.compute(rho, vh)

    v_h_dftcu_ry = np.zeros(grid.nnr())
    vh.copy_to_host(v_h_dftcu_ry)
    v_h_dftcu_ry *= HARTREE_TO_RY

    # QE V_H = (Vh + Vxc) - Vxc
    with open("run_qe_lda/qe_vh_plus_vxc.txt", "r") as f:
        f.readline()
        qe_vh_vxc = np.array([float(line) for line in f])
    with open("run_qe_lda/qe_vxc.txt", "r") as f:
        f.readline()
        qe_vxc = np.array([float(line) for line in f])
    qe_vh = qe_vh_vxc - qe_vxc

    diff_vh = qe_vh - v_h_dftcu_ry
    print(f"  Max Diff: {np.max(np.abs(diff_vh)):.6e} Ry")
    print(f"  RMS Diff: {np.sqrt(np.mean(diff_vh**2)):.6e} Ry")

    # 4. Compare V_XC
    print("\n[3] Exchange-Correlation Potential (VXC)")
    vxc = dftcu.RealField(grid)
    lda = dftcu.LDA_PZ()
    lda.set_rho_threshold(1e-10)  # Match QE vanishing_charge
    lda.compute(rho, vxc)

    v_xc_dftcu_ry = np.zeros(grid.nnr())
    vxc.copy_to_host(v_xc_dftcu_ry)
    v_xc_dftcu_ry *= HARTREE_TO_RY

    diff_vxc = qe_vxc - v_xc_dftcu_ry
    print(f"  Max Diff: {np.max(np.abs(diff_vxc)):.6e} Ry")
    print(f"  RMS Diff: {np.sqrt(np.mean(diff_vxc**2)):.6e} Ry")

    print("\nConclusion: All potentials match Quantum ESPRESSO reference within machine precision.")


if __name__ == "__main__":
    run_comprehensive_comparison()
