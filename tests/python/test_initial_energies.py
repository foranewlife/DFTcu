#!/usr/bin/env python3
import os

import dftcu
import numpy as np
from dftpy.density import DensityGenerator
from dftpy.functional import Functional
from dftpy.functional import Hartree as DFTpy_Hartree
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_initial_energy_components():
    """High precision comparison of energy components at Step 0"""
    # 1. Setup System (Al bulk)
    lattice = np.eye(3) * 7.65
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    generator = DensityGenerator()
    rho_init = generator.guess_rho(ions, grid=dftpy_grid)

    # 2. DFTpy Components
    tf_dftpy = Functional(type="KEDF", name="TF")
    lda_dftpy = Functional(type="XC", name="LDA")
    hartree_dftpy = DFTpy_Hartree()

    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.recpot")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_dftpy = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})

    e_tf_py = tf_dftpy(rho_init).energy
    e_xc_py = lda_dftpy(rho_init).energy
    e_h_py = hartree_dftpy(rho_init).energy
    e_ps_py = pseudo_dftpy.compute(rho_init, calcType={"E"}).energy

    # 3. DFTcu Components
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    v_tmp = dftcu.RealField(grid_cu, 1)

    # TF
    tf_cu = dftcu.ThomasFermi(coeff=1.0)
    e_tf_cu = tf_cu.compute(rho_cu, v_tmp)

    # XC
    lda_cu = dftcu.LDA_PZ()
    e_xc_cu = lda_cu.compute(rho_cu, v_tmp)

    # Hartree
    hartree_cu = dftcu.Hartree(grid_cu)
    # Hartree energy = 0.5 * integral(rho * V_h)
    e_h_cu = hartree_cu.compute(rho_cu, v_tmp)
    # The compute method returns potential in v_tmp and energy in e_h_cu
    # Let's double check Hartree energy calculation in our API

    # Pseudo
    v_ps = dftcu.RealField(grid_cu, 1)
    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc(0, pseudo_dftpy.vlines["Al"].flatten(order="C").tolist())
    pseudo_cu.compute(v_ps)
    e_ps_cu = rho_cu.dot(v_ps) * grid_cu.dv()

    print("\n" + "=" * 80)
    print(f"{ 'Component':<15} | { 'DFTpy (Ha)':<25} | { 'DFTcu (Ha)':<25} | { 'Diff':<12}")
    print("-" * 80)
    components = [
        ("Thomas-Fermi", e_tf_py, e_tf_cu),
        ("LDA XC", e_xc_py, e_xc_cu),
        ("Hartree", e_h_py, e_h_cu),
        ("Local Pseudo", e_ps_py, e_ps_cu),
    ]

    for name, py, cu in components:
        diff = abs(py - cu)
        print(f"{name:<15} | {py:<25.15f} | {cu:<25.15f} | {diff:<12.2e}")

    total_py = sum(c[1] for c in components)
    total_cu = sum(c[2] for c in components)
    diff_total = abs(total_py - total_cu)
    print("-" * 80)
    print(f"{ 'TOTAL':<15} | {total_py:<25.15f} | {total_cu:<25.15f} | {diff_total:<12.2e}")
    print("=" * 80)


if __name__ == "__main__":
    test_initial_energy_components()
