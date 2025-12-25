#!/usr/bin/env python3
import os
import time

import dftcu
import numpy as np
from dftpy.density import DensityGenerator
from dftpy.ewald import ewald as DFTpy_Ewald
from dftpy.functional import Functional
from dftpy.functional import Hartree as DFTpy_Hartree
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_initial_energy_components():
    """High precision comparison of all energy components using overlapping atomic densities"""
    # 1. Setup System (FCC Al from fcc.vasp)
    # 4.05 Angstrom converted to Bohr
    a_bohr = 4.05 / 0.529177249
    lattice = np.eye(3) * a_bohr
    nr = [32, 32, 32]

    # 4 atoms in cubic cell
    pos_dir = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    all_pos = pos_dir * a_bohr

    ions = Ions(symbols=["Al"] * 4, positions=all_pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Setup DFTpy Components and Overlapping Atomic Density
    print("\n1. Calculating DFTpy Components...")
    # Use UPF for atomic density data
    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})

    # Use reciprocal-space generator (direct=False) for smooth non-uniform density
    generator = DensityGenerator(pseudo=pseudo_py, direct=False)
    rho_init = generator.guess_rho(ions, grid=dftpy_grid)

    print(
        f"Density Stats: Min={rho_init.min():.2e}, Max={rho_init.max():.2e}, "
        f"Mean={rho_init.mean():.2e}"
    )
    print(f"Total Electrons (DFTpy): {rho_init.integral():.4f}")

    tf_py = Functional(type="KEDF", name="TF")
    vw_py = Functional(type="KEDF", name="vW")
    wtnl_py = Functional(type="KEDF", name="WT-NL")
    lda_py = Functional(type="XC", name="LDA")
    hartree_py = DFTpy_Hartree()
    ewald_py = DFTpy_Ewald(grid=dftpy_grid, ions=ions)

    t0 = time.time()
    e_tf_py = tf_py(rho_init).energy
    e_vw_py = vw_py(rho_init).energy
    e_wtnl_py = wtnl_py(rho_init).energy
    e_lda_py = lda_py(rho_init).energy
    e_h_py = hartree_py(rho_init).energy
    e_ps_py = pseudo_py.compute(rho_init, calcType={"E": ""}).energy
    e_ii_py = ewald_py.energy
    t_py = time.time() - t0

    # 3. DFTcu Components
    print("2. Calculating DFTcu Components (CUDA)...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_list = [dftcu.Atom(p[0], p[1], p[2], 3.0, 0) for p in all_pos]
    atoms_cu = dftcu.Atoms(atoms_list)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    tf_cu = dftcu.ThomasFermi(coeff=1.0)
    vw_cu = dftcu.vonWeizsacker(coeff=1.0)
    wt_cu = dftcu.WangTeter(coeff=1.0)
    lda_cu = dftcu.LDA_PZ()
    hartree_cu = dftcu.Hartree(grid_cu)
    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc(0, pseudo_py.vlines["Al"].flatten(order="C").tolist())
    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)

    v_tmp = dftcu.RealField(grid_cu, 1)
    v_ps = dftcu.RealField(grid_cu, 1)

    # Warmup
    tf_cu.compute(rho_cu, v_tmp)

    t0 = time.time()
    e_tf_cu = tf_cu.compute(rho_cu, v_tmp)
    e_vw_cu = vw_cu.compute(rho_cu, v_tmp)
    e_wt_cu = wt_cu.compute(rho_cu, v_tmp)
    e_lda_cu = lda_cu.compute(rho_cu, v_tmp)
    e_h_cu = hartree_cu.compute(rho_cu, v_tmp)
    pseudo_cu.compute(v_ps)
    e_ps_cu = rho_cu.dot(v_ps) * grid_cu.dv()
    e_ii_cu = ewald_cu.compute()
    t_cu = time.time() - t0

    print("\n" + "=" * 90)
    print(f"{ 'Component':<20} | { 'DFTpy (Ha)':<20} | { 'DFTcu (Ha)':<20} | { 'Abs Diff':<12}")
    print("-" * 90)
    components = [
        ("Thomas-Fermi", e_tf_py, e_tf_cu),
        ("von Weizsacker", e_vw_py, e_vw_cu),
        ("Wang-Teter (NL)", e_wtnl_py, e_wt_cu),
        ("LDA XC", e_lda_py, e_lda_cu),
        ("Hartree", e_h_py, e_h_cu),
        ("Local Pseudo", e_ps_py, e_ps_cu),
        ("Ewald (Exact)", e_ii_py, e_ii_cu),
    ]

    for name, py, cu in components:
        diff = abs(py - cu)
        print(f"{name:<20} | {py:<20.10f} | {cu:<20.10f} | {diff:<12.2e}")

    total_py = sum(c[1] for c in components)
    total_cu = sum(c[2] for c in components)
    print("-" * 90)
    diff_total = abs(total_py - total_cu)
    print(f"{ 'TOTAL':<20} | {total_py:<20.10f} | {total_cu:<20.10f} | {diff_total:<12.2e}")
    print("=" * 90)
    print(f"Time DFTpy (All): {t_py:.4f} s")
    print(f"Time DFTcu (All): {t_cu:.4f} s")
    print(f"Speedup:          {t_py/t_cu:.2f}x")
    print("=" * 90)

    assert abs(total_py - total_cu) < 1e-4


if __name__ == "__main__":
    test_initial_energy_components()
