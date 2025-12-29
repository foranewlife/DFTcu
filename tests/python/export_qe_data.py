import os
import re
import shutil
from pathlib import Path

import numpy as np
from qepy.driver import Driver

EV_PER_HARTREE = 27.211386245988
ECUT_WFC_RY = 30.0


def parse_eigenvalues_from_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"QE log {log_path} not found")
    text = log_path.read_text()
    bands_match = re.search(r"bands \(ev\):(.*?)(?:\n\s*\n)", text, re.S | re.I)
    if not bands_match:
        raise RuntimeError("Unable to parse eigenvalues from QE log output")
    eigenvalues_ev = []
    for line in bands_match.group(1).strip().splitlines():
        eigenvalues_ev.extend(float(val) for val in line.split())
    return np.array(eigenvalues_ev) / EV_PER_HARTREE


def export_fixed_data():
    work_dir = "run_qe_final_align"
    os.makedirs(work_dir, exist_ok=True)

    qe_input = """
 &CONTROL
    calculation = 'scf', pseudo_dir = './'
 /
 &SYSTEM
    ibrav = 1, celldm(1) = 10.0, nat = 1, ntyp = 1,
    ecutwfc = 30.0,
 /
 &ELECTRONS /
ATOMIC_SPECIES
 O 15.999 O_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS {bohr}
 O 0.0 0.0 0.0
K_POINTS {gamma}
"""
    qe_in = Path(work_dir) / "qe_in.in"
    qe_in.write_text(qe_input)

    upf_src = "external/QEpy/examples/jupyter/DATA/O_ONCV_PBE-1.2.upf"
    upf_dst = Path(work_dir) / "O_ONCV_PBE-1.2.upf"
    if os.path.exists(upf_src):
        shutil.copy(upf_src, upf_dst)
    else:
        raise FileNotFoundError(f"Pseudopotential not found at {upf_src}")

    old_cwd = os.getcwd()
    os.chdir(work_dir)

    log_file = Path("qe_export.log")

    try:
        driver = Driver("qe_in.in", iterative=False, logfile=str(log_file))
        driver.diagonalize()

        rho = driver.get_density()
        np.save("qe_rho_golden.npy", np.ascontiguousarray(rho))

        nrs = driver.get_number_of_grid_points()
        nnr = int(np.prod(nrs))
        lattice = np.eye(3) * 10.0
        inv_lattice = np.linalg.inv(lattice)
        rec_lattice = 2.0 * np.pi * inv_lattice.T
        freqs = [np.fft.fftfreq(n) * n for n in nrs]
        f0 = freqs[0][:, None, None]
        f1 = freqs[1][None, :, None]
        f2 = freqs[2][None, None, :]
        gx = f0 * rec_lattice[0][0] + f1 * rec_lattice[1][0] + f2 * rec_lattice[2][0]
        gy = f0 * rec_lattice[0][1] + f1 * rec_lattice[1][1] + f2 * rec_lattice[2][1]
        gz = f0 * rec_lattice[0][2] + f1 * rec_lattice[1][2] + f2 * rec_lattice[2][2]
        gg = (gx**2 + gy**2 + gz**2).reshape(-1)
        encut_hartree = ECUT_WFC_RY / 2.0
        mask = 0.5 * gg <= encut_hartree

        evc_list = driver.get_wave_function()
        gspace_coeffs = []
        for wf in evc_list:
            psi_r = wf.reshape(nrs, order="C")
            psi_g = np.fft.fftn(psi_r, norm=None) / nnr
            psi_g_flat = psi_g.reshape(-1, order="C")
            psi_g_flat[~mask] = 0.0
            gspace_coeffs.append(psi_g_flat)
        evc = np.ascontiguousarray(np.stack(gspace_coeffs, axis=0))
        np.save("qe_evc_golden.npy", evc)

        driver.stop()

        evals = parse_eigenvalues_from_log(log_file)
        np.save("qe_eval_golden.npy", evals)
        print("Golden data exported successfully in run_qe_final_align/")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    export_fixed_data()
