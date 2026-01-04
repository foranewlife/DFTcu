import os

import numpy as np

import dftcu
from dftcu._dftcu import constants


def test_oxygen_scf_alignment():
    # 1. Setup Grid
    L = 15.0  # Bohr
    B = constants.BOHR_TO_ANGSTROM
    lattice_flat = [L * B, 0, 0, 0, L * B, 0, 0, 0, L * B]
    nr = [48, 48, 48]
    grid = dftcu.Grid(lattice_flat, nr)

    # 2. Setup Atom
    atoms = dftcu.Atoms([dftcu.Atom(L * B / 2, L * B / 2, L * B / 2, 6.0, 0)])
    upf_path = "run_qe_v1.0/O_ONCV_PBE-1.0.UPF"

    # 3. Run QE
    qe_in = f"""
 &CONTROL
    calculation = 'scf'
    prefix = 'o_scf'
    pseudo_dir = './run_qe_v1.0/'
    outdir = './tmp/'
 /
 &SYSTEM
    ibrav = 1, celldm(1) = {L}, nat = 1, ntyp = 1,
    ecutwfc = 25.0,
    occupations = 'from_input'
    nbnd = 4
    input_dft = 'PZ'
 /
 &ELECTRONS
    mixing_beta = 0.0
    startingpot = 'atomic'
 /
ATOMIC_SPECIES
 O  15.999  O_ONCV_PBE-1.0.UPF
ATOMIC_POSITIONS (crystal)
 O  0.5 0.5 0.5
K_POINTS (gamma)
OCCUPATIONS
2.0
1.333333333333
1.333333333333
1.333333333333
"""
    with open("run_qe_v1.0/qe_o_scf.in", "w") as f:
        f.write(qe_in)

    print("Running QE...")
    os.system("external/qe/build/bin/pw.x < run_qe_v1.0/qe_o_scf.in > run_qe_v1.0/qe_o_scf.out")

    # Load injected rho_init from QE
    with open("qe_rho_init.txt", "r") as f:
        f.readline()
        rho_raw = np.fromfile(f, sep=" ")
        rho_3d = rho_raw.reshape((nr[0], nr[1], nr[2]), order="F")
        rho_aligned = rho_3d.flatten(order="C")
        rho = dftcu.RealField(grid)
        rho.copy_from_host(rho_aligned)

    # 4. Initialize DFTcu Hamiltonian
    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.Hartree())
    evaluator.add_functional(dftcu.LDA_PZ())

    vloc = dftcu.LocalPseudo(grid, atoms)
    data = dftcu.parse_upf(upf_path)
    vloc.init_tab_vloc(0, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume())
    evaluator.add_functional(vloc)
    evaluator.add_functional(dftcu.Ewald(grid, atoms, 1e-12))

    ham = dftcu.Hamiltonian(grid, evaluator)
    if len(data["betas"]) > 0:
        nl = dftcu.NonLocalPseudo(grid)
        nl.init_tab_beta(
            0,
            data["r"],
            data["betas"],
            data["rab"],
            data["l_list"],
            data["kkbeta_list"],
            grid.volume(),
        )
        nl.init_dij(0, data["dij"].flatten().tolist())
        nl.update_projectors(atoms)
        ham.set_nonlocal(nl)

    # 5. Initialize SCF Solver
    opts = dftcu.SCFOptions()
    opts.max_iter = 0  # Match Step 0 precisely
    opts.verbose = True
    solver = dftcu.SCFSolver(grid, opts)

    # 6. Run Solve (Step 0)
    psi = dftcu.Wavefunction(grid, 4, 12.5)
    psi.randomize()
    occs = [2.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]
    # b_final = solver.solve(ham, psi, occs, rho)

    # Re-diagonalize for debug eigenvalues
    ham.update_potentials(rho)
    davidson = dftcu.DavidsonSolver(grid, 100, 1e-12)
    eigenvalues = davidson.solve(ham, psi)

    # Compute breakdown manually for comparison
    b = solver.compute_energy_breakdown(eigenvalues, occs, ham, psi, rho)

    # Parse QE
    qe = {}
    with open("run_qe_v1.0/qe_o_scf.out", "r") as f:
        for line in f:
            if "DEBUG Ry: hwf_energy =" in line:
                qe["hwf"] = float(line.split()[4]) * 0.5
            if "DEBUG INITIAL Ry: ehart =" in line:
                qe["ehart"] = float(line.split()[5]) * 0.5
            if "DEBUG INITIAL Ry: etxc  =" in line:
                qe["etxc"] = float(line.split()[5]) * 0.5
            if "DEBUG Ry: ewld       =" in line:
                qe["eewld"] = float(line.split()[4]) * 0.5
            if "DEBUG Ry: eband      =" in line:
                qe["eband"] = float(line.split()[4]) * 0.5
            if "DEBUG Ry: delta_e_raw =" in line:
                qe["deband"] = float(line.split()[4]) * 0.5
            if "DEBUG: Alpha-Z intensive" in line:
                qe["alpha"] = float(line.split()[8]) * 0.5
            if "DEBUG Ry: band  1" in line:
                p = line.split()
                for i_f, fld in enumerate(p):
                    if fld.startswith("eig="):
                        val = p[i_f + 1] if len(fld) == 4 else fld[4:]
                        qe["eig1"] = float(val) * 0.5

    print("\n--- QE vs DFTcu SCF Step 0 Alignment (Ha) ---")
    print(
        f"Hartree: QE={qe['ehart']:20.12f}, CU={b.ehart:20.12f}, Diff={b.ehart-qe['ehart']:12.4e}"
    )
    print(f"XC:      QE={qe['etxc']:20.12f}, CU={b.etxc:20.12f}, Diff={b.etxc-qe['etxc']:12.4e}")
    print(
        f"Ewald:   QE={qe['eewld']:20.12f}, CU={b.eewld:20.12f}, Diff={b.eewld-qe['eewld']:12.4e}"
    )

    shift = 6.0 * qe["alpha"]
    print(f"Alpha:   QE={shift:20.12f}, CU={b.alpha:20.12f}, Diff={b.alpha-shift:12.4e}")

    # Shifted comparisons
    eband_cu = b.eband
    eband_qe_shifted = qe["eband"] + shift
    print(
        f"Eband:   QE+Shift={eband_qe_shifted:20.12f}, \
            CU={eband_cu:20.12f}, Diff={eband_cu-eband_qe_shifted:12.4e}"
    )

    deband_cu = b.deband
    deband_qe_shifted = qe["deband"] - shift
    print(
        f"Deband:  QE-Shift={deband_qe_shifted:20.12f}, \
            CU={deband_cu:20.12f}, Diff={deband_cu-deband_qe_shifted:12.4e}"
    )

    print(f"Total:   QE={qe['hwf']:20.12f}, CU={b.etot:20.12f}, Diff={b.etot-qe['hwf']:12.4e}")

    print(
        f"\nBand 1:  QE+Alpha={qe['eig1']+qe['alpha']:20.12f}, CU={eigenvalues[0]:20.12f},\
              Diff={eigenvalues[0]-(qe['eig1']+qe['alpha']):12.4e}"
    )


if __name__ == "__main__":
    test_oxygen_scf_alignment()
