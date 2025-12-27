import time

import dftcu
from dftpy.density import DensityGenerator
from dftpy.ewald import ewald as DFTpy_Ewald
from dftpy.functional import Functional
from dftpy.functional import Hartree as DFTpy_Hartree
from dftpy.grid import DirectGrid
from test_utils import get_pp_path, get_system, setup_pseudo, to_dftcu_atoms


def test_initial_energy_components():
    """High precision comparison of all energy components using overlapping atomic densities"""
    # 1. Setup System
    nr = [32, 32, 32]
    ions = get_system("Al_fcc", a=4.05, cubic=True)
    lattice = ions.cell.array

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Setup DFTpy Components and Overlapping Atomic Density
    print("\n1. Calculating DFTpy Components...")
    pp_file = get_pp_path("al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})

    generator = DensityGenerator(pseudo=pseudo_py, direct=False)
    rho_init = generator.get_3d_value_recipe(ions, grid=dftpy_grid)

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
    e_ps_py = pseudo_py.compute(rho_init, calcType={"E"}).energy
    e_ii_py = ewald_py.energy
    t_py = time.time() - t0

    # 3. DFTcu Components
    print("2. Calculating DFTcu Components (CUDA)...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = to_dftcu_atoms(ions)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    tf_cu = dftcu.ThomasFermi(coeff=1.0)
    vw_cu = dftcu.vonWeizsacker(coeff=1.0)
    wt_cu = dftcu.WangTeter(coeff=1.0)
    lda_cu = dftcu.LDA_PZ()
    hartree_cu = dftcu.Hartree(grid_cu)
    pseudo_cu, _ = setup_pseudo(grid_cu, atoms_cu, "al.lda.upf", ions)
    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)
    ewald_cu.set_eta(ewald_py.eta)

    v_tmp = dftcu.RealField(grid_cu, 1)

    # Warmup
    tf_cu.compute(rho_cu, v_tmp)

    t0 = time.time()
    # Test individual components directly
    e_tf_cu = tf_cu.compute(rho_cu, v_tmp)
    e_vw_cu = vw_cu.compute(rho_cu, v_tmp)
    e_wt_cu = wt_cu.compute(rho_cu, v_tmp)
    e_lda_cu = lda_cu.compute(rho_cu, v_tmp)
    e_h_cu = hartree_cu.compute(rho_cu, v_tmp)
    e_ps_cu = pseudo_cu.compute(rho_cu, v_tmp)
    e_ii_cu = ewald_cu.compute(False)  # Exact
    t_cu = time.time() - t0

    # Also verify the Evaluator (Composition)
    evaluator_cu = dftcu.Evaluator(grid_cu)
    evaluator_cu.add_functional(tf_cu)
    evaluator_cu.add_functional(vw_cu)
    evaluator_cu.add_functional(wt_cu)
    evaluator_cu.add_functional(lda_cu)
    evaluator_cu.add_functional(hartree_cu)
    evaluator_cu.add_functional(pseudo_cu)
    evaluator_cu.add_functional(ewald_cu)
    v_tot = dftcu.RealField(grid_cu, 1)
    e_total_evaluator = evaluator_cu.compute(rho_cu, v_tot)

    print("\n" + "=" * 98)
    print(
        f"{'Component':<22} | {'DFTpy (Ha)':>18} | {'DFTcu (Ha)':>18} | "
        f"{'Abs Diff':>12} | {'Status':>8}"
    )
    print("-" * 98)
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
        status = "✓" if diff < 1e-6 else "⚠"
        print(f"{name:<22} | {py:>18.10f} | {cu:>18.10f} | {diff:>12.2e} | {status:>8}")

    total_py = sum(c[1] for c in components)
    total_cu = sum(c[2] for c in components)
    print("-" * 98)
    diff_total = abs(total_py - total_cu)
    status_total = "✓" if diff_total < 1e-4 else "⚠"
    print(
        f"{'TOTAL (Sum)':<22} | {total_py:>18.10f} | {total_cu:>18.10f} | "
        f"{diff_total:>12.2e} | {status_total:>8}"
    )

    diff_eval = abs(e_total_evaluator - total_cu)
    status_eval = "✓" if diff_eval < 1e-10 else "⚠"
    print(
        f"{'TOTAL (Evaluator)':<22} | {'':<18} | {e_total_evaluator:>18.10f} | "
        f"{diff_eval:>12.2e} | {status_eval:>8}"
    )
    print("=" * 98)
    print(f"Time DFTpy (All): {t_py:.4f} s")
    print(f"Time DFTcu (All): {t_cu:.4f} s")
    print(f"Speedup:          {t_py/t_cu:.2f}x")
    print("=" * 98)

    # Validate energy consistency
    assert abs(total_py - total_cu) < 1e-10, "DFTpy vs DFTcu total energy mismatch"
    assert abs(e_total_evaluator - total_cu) < 1e-10, "Evaluator total energy mismatch"


if __name__ == "__main__":
    test_initial_energy_components()
