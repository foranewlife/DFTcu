import subprocess

import dftcu


def test_bit_perfect_initial_step():
    L = 15.0
    N = 80
    ecut = 30.0

    # 2. QE Run Step 0
    qe_input = f"""&CONTROL
    calculation = 'scf', pseudo_dir = './', outdir = './', prefix = 'pwscf',
    verbosity = 'high'
 /
 &SYSTEM
    ibrav = 1, celldm(1) = {L}, nat = 1, ntyp = 1, ecutwfc = {ecut},
    nr1 = {N}, nr2 = {N}, nr3 = {N}, nbnd = 4, input_dft = 'sla+pz',
    occupations = 'from_input'
 /
 &ELECTRONS
    startingpot = 'atomic', startingwfc = 'atomic', electron_maxstep = 1,
 /
ATOMIC_SPECIES
 O 15.999 O_ONCV_PBE-1.0.UPF
ATOMIC_POSITIONS {{alat}}
 O 0.5 0.5 0.5
OCCUPATIONS
 2.0 1.3333333333333333 1.3333333333333333 1.3333333333333333
K_POINTS {{gamma}}
"""
    with open("run_qe_v1.0/qe_bit.in", "w") as f:
        f.write(qe_input)
    subprocess.run(
        "cd run_qe_v1.0 && ../external/qe/build/bin/pw.x -in qe_bit.in > qe_bit.out", shell=True
    )

    qe = {}
    with open("run_qe_v1.0/qe_bit.out", "r") as f:
        for line in f:
            if "DEBUG:" in line and "=" in line:
                name = line.split("DEBUG:")[1].split("=")[0].strip()
                val_str = line.split("=")[1].split("Ry")[0].strip()
                if val_str:
                    val = float(val_str)
                    if "Ry" in line:
                        val /= 2.0
                    qe[name] = val

    # 3. DFTcu Run
    B = dftcu.constants.BOHR_TO_ANGSTROM
    a = L * B
    grid = dftcu.Grid([a, 0, 0, 0, a, 0, 0, 0, a], [N, N, N])
    atoms = dftcu.Atoms([dftcu.Atom(a / 2, a / 2, a / 2, 6.0, 0)])
    upf = ["run_qe_v1.0/O_ONCV_PBE-1.0.UPF"]
    rho_in, ham = dftcu.initialize_hamiltonian(grid, atoms, upf, injection_path="./run_qe_v1.0")

    ecutrho = 4 * ecut
    hartree = dftcu.Hartree()
    hartree.set_gcut(ecutrho)
    vh = dftcu.RealField(grid)
    eh_val = hartree.compute(rho_in, vh)

    vxc = dftcu.RealField(grid)
    lda = dftcu.LDA_PZ()
    lda.set_rho_threshold(1e-10)
    exc_val = lda.compute(rho_in, vxc)

    # Recalculate eband
    davidson = dftcu.DavidsonSolver(grid, 100, 1e-12)
    # Re-setup evaluator to use gcut
    eval_new = dftcu.Evaluator(grid)
    eval_new.add_functional(hartree)
    eval_new.add_functional(lda)
    eval_new.add_functional(dftcu.Ewald(grid, atoms, 1e-12))
    lp = dftcu.LocalPseudo(grid, atoms)
    upf_data = dftcu.parse_upf(upf[0])
    lp.init_tab_vloc(
        0, upf_data["r"], upf_data["vloc"], upf_data["rab"], upf_data["zp"], grid.volume()
    )
    lp.set_gcut(ecutrho)
    eval_new.add_functional(lp)

    ham_bit = dftcu.Hamiltonian(grid, eval_new)
    ham_bit.update_potentials(rho_in)
    psi = dftcu.initialize_wavefunctions(grid, atoms, upf, num_bands=4, ecutwfc=ecut)
    eigenvalues = davidson.solve(ham_bit, psi)

    occ = [2.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]
    eband = sum(o * e for o, e in zip(occ, eigenvalues))

    v_h_int = rho_in.dot(vh) * grid.dv_bohr()
    v_xc_int = rho_in.dot(vxc) * grid.dv_bohr()

    # Since tab_vloc[0] is now 0, eband matches QE (periodic part)
    cu_eband = eband
    cu_deband = -(v_h_int + v_xc_int)

    print("\n--- Components Alignment (Ha) ---")
    print("{ 'Component':<12} | { 'QE':<24} | { 'DFTcu':<24} | { 'Diff':<12}")
    print("-" * 80)
    print(
        f"{ 'eband':<12} | {qe['eband']:24.14f} | \
            {cu_eband:24.14f} | {cu_eband - qe['eband']:12.4e}"
    )
    print(
        f"{ 'deband':<12} | {qe['deband']:24.14f} | \
            {cu_deband:24.14f} | {cu_deband - qe['deband']:12.4e}"
    )
    print(
        f"{ 'ehart':<12} | {qe['ehart']:24.14f} | \
          {eh_val:24.14f} | {eh_val - qe['ehart']:12.4e}"
    )
    print(
        f"{ 'etxc':<12} | {qe['etxc']:24.14f} | \
          {exc_val:24.14f} | {exc_val - qe['etxc']:12.4e}"
    )


if __name__ == "__main__":
    test_bit_perfect_initial_step()
