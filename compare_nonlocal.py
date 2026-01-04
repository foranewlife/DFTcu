import os

import numpy as np

import dftcu


def read_nl_verify_data(filename):  # noqa: C901
    data = {}
    if not os.path.exists(filename):
        return data

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if not parts:
            i += 1
            continue

        key = parts[0]

        if key.startswith("NPW"):
            data["npw"] = int(key[3:])
        elif key.startswith("NKB"):
            data["nkb"] = int(key[3:])
        elif key.startswith("NBND"):
            data["nbnd"] = int(key[4:])
        elif key.startswith("NAT"):
            data["nat"] = int(key[3:])
        elif key.startswith("NNR"):
            data["nnr"] = int(key[3:])
        elif key.startswith("GSTART"):
            data["gstart"] = int(key[6:])
        elif key == "TAU":
            i += 1
            tau = []
            for _ in range(data["nat"]):
                tau.append(list(map(float, lines[i].split())))
                i += 1
            data["tau"] = np.array(tau)
            i -= 1
        elif key == "MILLER_INDICES":
            i += 1
            miller = []
            for _ in range(data["npw"]):
                miller.append(list(map(int, lines[i].split())))
                i += 1
            data["miller"] = np.array(miller)
            i -= 1  # Adjust for outer loop increment

        elif key == "VKB":
            i += 1
            vkb = []
            for _ in range(data["nkb"]):
                proj = []
                for _ in range(data["npw"]):
                    p = list(map(float, lines[i].split()))
                    proj.append(complex(p[0], p[1]))
                    i += 1
                vkb.append(proj)
            data["vkb"] = np.array(vkb)
            i -= 1
        elif key == "EVC":
            i += 1
            evc = []
            for _ in range(data["nbnd"]):
                band = []
                for _ in range(data["npw"]):
                    p = list(map(float, lines[i].split()))
                    band.append(complex(p[0], p[1]))
                    i += 1
                evc.append(band)
            data["evc"] = np.array(evc)
            i -= 1
        elif key == "BECP":
            i += 1
            becp = []
            for _ in range(data["nbnd"] * data["nkb"]):
                becp.append(float(lines[i].strip()))
                i += 1
            data["becp"] = np.array(becp).reshape(data["nbnd"], data["nkb"])
            i -= 1
        elif key == "DEEQ":
            i += 1
            data["deeq"] = []
            for _ in range(data["nat"]):
                parts = lines[i].split()
                nh = int(parts[3])
                i += 1
                mat = []
                for _ in range(nh * nh):
                    mat.append(float(lines[i].strip()))
                    i += 1
                data["deeq"].append(np.array(mat).reshape(nh, nh))
            i -= 1
        i += 1

    return data


def compare_nonlocal_final():
    qe = read_nl_verify_data("si_test/nl_verify.dat")
    if not qe:
        print("Verification file not found!")
        return

    BOHR_TO_ANG = 0.529177210903
    a_bohr = 10.20
    a_ang = a_bohr * BOHR_TO_ANG

    nr = [16, 16, 16]
    v1 = [-a_ang / 2, 0, a_ang / 2]
    v2 = [0, a_ang / 2, a_ang / 2]
    v3 = [-a_ang / 2, a_ang / 2, 0]
    lattice = v1 + v2 + v3
    grid = dftcu.Grid(lattice, nr)
    grid.set_is_gamma(True)

    tau_ang = qe["tau"] * a_bohr * BOHR_TO_ANG
    atoms = dftcu.Atoms(
        [
            dftcu.Atom(tau_ang[0, 0], tau_ang[0, 1], tau_ang[0, 2], 4.0, 0),
            dftcu.Atom(tau_ang[1, 0], tau_ang[1, 1], tau_ang[1, 2], 4.0, 0),
        ]
    )

    nl = dftcu.NonLocalPseudo(grid)

    upf_file = "si_test/Si.pz-rrkj.UPF"
    upf_data = dftcu.parse_upf(upf_file)

    nl.init_tab_beta(
        0,
        upf_data["r"],
        upf_data["betas"],
        upf_data["rab"],
        upf_data["l_list"],
        upf_data["kkbeta_list"],
        grid.volume(),
    )
    nl.init_dij(0, upf_data["dij"])

    nl.update_projectors(atoms)

    # Inject wavefunctions from QE
    psi = dftcu.Wavefunction(grid, qe["nbnd"], 12.0)
    psi.set_coefficients_miller(
        qe["miller"][:, 0].tolist(),
        qe["miller"][:, 1].tolist(),
        qe["miller"][:, 2].tolist(),
        qe["evc"].flatten(),
    )

    # --- Run Calculation & Get Projections ---
    occ = [2.0] * qe["nbnd"]
    e_nl_cu_tot = nl.calculate_energy(psi, occ)
    cu_projections = np.array(nl.get_projections()).reshape(qe["nbnd"], qe["nkb"])

    # --- QE Reference Energy ---
    qe_enl_ry = 0.0
    for ibnd in range(qe["nbnd"]):
        band_enl = 0.0
        for na in range(qe["nat"]):
            deeq = qe["deeq"][na]
            offset = na * 4
            for ih in range(4):
                for jh in range(4):
                    band_enl += (
                        qe["becp"][ibnd, offset + ih] * deeq[ih, jh] * qe["becp"][ibnd, offset + jh]
                    )
        qe_enl_ry += band_enl * occ[ibnd]
    qe_enl_ha = qe_enl_ry * 0.5

    # --- Final Report ---
    print("\n--- Final Projection Comparison ---")
    max_proj_diff = np.max(np.abs(qe["becp"] - cu_projections.real))
    print(f"Max Projection Difference: {max_proj_diff:e}")
    for i in range(4):  # Print first 4 for brevity
        qe_val = qe["becp"][0, i]
        cu_val = cu_projections[0, i].real
        print(f"Proj {i:2d} (Band 0): QE={qe_val:16.12f}, CU={cu_val:16.12f}")

    print("\n--- Final Energy Comparison ---")
    print(f"DFTcu Total NL Energy: {e_nl_cu_tot:20.12f} Ha")
    print(f"QE Reference Energy:   {qe_enl_ha:20.12f} Ha")
    print(f"Difference:            {e_nl_cu_tot - qe_enl_ha:20.12f} Ha")

    if np.isclose(e_nl_cu_tot, qe_enl_ha, atol=1e-8):
        print("\nSUCCESS: Nonlocal energy is aligned with QE.")
    else:
        print("\nFAILURE: Nonlocal energy is NOT aligned.")


if __name__ == "__main__":
    compare_nonlocal_final()
