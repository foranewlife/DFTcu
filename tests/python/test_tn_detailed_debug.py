"""
TN Optimizer Detailed Debug Script
详细对比 DFTpy 和 DFTcu 每一步的中间变量
"""

import dftcu
import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.optimization import Optimization
from test_utils import get_pp_path, get_system, setup_pseudo, to_dftcu_atoms


def analyze_density(rho, grid, label=""):
    """分析密度场的统计信息"""
    rho_np = rho if isinstance(rho, np.ndarray) else rho.copy()

    integral = np.sum(rho_np) * grid.dV if hasattr(grid, "dV") else np.sum(rho_np) * grid.dv()

    print(f"\n  [{label}] Density Statistics:")
    print(f"    Min:      {np.min(rho_np):.10e}")
    print(f"    Max:      {np.max(rho_np):.10e}")
    print(f"    Mean:     {np.mean(rho_np):.10e}")
    print(f"    Integral: {integral:.10f} (should be N_electrons)")
    print(f"    Std:      {np.std(rho_np):.10e}")


def compare_fields(field1, field2, name, tolerance=1e-10):
    """对比两个场的差异"""
    f1 = field1 if isinstance(field1, np.ndarray) else field1.copy()
    f2 = field2 if isinstance(field2, np.ndarray) else field2.copy()

    diff = np.abs(f1 - f2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / (np.max(np.abs(f1)) + 1e-20)

    status = "✅" if max_diff < tolerance else "❌"
    print(f"\n  {name} Comparison {status}:")
    print(f"    Max Diff:     {max_diff:.2e}")
    print(f"    Mean Diff:    {mean_diff:.2e}")
    print(f"    Relative:     {rel_diff:.2e}")
    print(f"    Tolerance:    {tolerance:.2e}")

    if max_diff > tolerance:
        # 找出差异最大的位置
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Max diff at:  {idx}")
        print(f"    DFTpy value:  {f1[idx]:.10e}")
        print(f"    DFTcu value:  {f2[idx]:.10e}")

    return max_diff < tolerance


def test_tn_step_by_step():
    """逐步对比 DFTpy 和 DFTcu 的 TN 优化过程"""

    # ==================== 系统设置 ====================
    print("=" * 80)
    print(" " * 20 + "TN Optimizer Detailed Debug")
    print("=" * 80)

    ions = get_system("Al_fcc", a=4.05, cubic=True)
    lattice = ions.cell.array

    # Use nr=[32, 32, 32]
    nr = [32, 32, 32]

    print("\n[System Setup]")
    print(f"  Lattice constant: {4.05:.6f} Angstrom")
    print(f"  Grid size:        {nr}")
    print(f"  N_atoms:          {ions.nat}")
    print(f"  N_electrons:      {ions.nat * 3.0}")
    print(f"  Volume:           {ions.cell.volume:.6f} Bohr³")

    # ==================== DFTpy Setup ====================
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    pp_file = get_pp_path("al.lda.recpot")

    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=grid_py, ions=ions, PP_list={"Al": pp_file})

    ke_py = Functional(type="KEDF", name="TFvW", grid=grid_py)
    hartree_py = Functional(type="HARTREE", grid=grid_py)
    xc_py = Functional(type="XC", name="LDA", grid=grid_py)

    evaluator_py = TotalFunctional(
        KineticEnergyFunctional=ke_py, XCFunctional=xc_py, HARTREE=hartree_py, PSEUDO=pseudo_py
    )

    # 初始密度
    rho_py = DirectField(grid=grid_py, data=np.full(nr, ions.nat * 3.0 / grid_py.volume))

    # ==================== DFTcu Setup ====================
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = to_dftcu_atoms(ions)

    pseudo_cu, _ = setup_pseudo(grid_cu, atoms_cu, "al.lda.recpot", ions)

    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)

    evaluator_cu = dftcu.Evaluator(grid_cu)
    # TFvW is TF + vW
    evaluator_cu.add_functional(dftcu.ThomasFermi(1.0))
    evaluator_cu.add_functional(dftcu.vonWeizsacker(1.0))
    evaluator_cu.add_functional(pseudo_cu)
    evaluator_cu.add_functional(dftcu.Hartree(grid_cu))
    evaluator_cu.add_functional(dftcu.LDA_PZ())
    evaluator_cu.add_functional(ewald_cu)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.fill(ions.nat * 3.0 / grid_cu.volume())

    # ==================== 初始状态对比 ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "Initial State Comparison")
    print("=" * 80)

    # 初始密度对比
    rho_cu_host = np.zeros(nr, dtype=np.float64, order="C")
    rho_cu.copy_to_host(rho_cu_host)

    compare_fields(rho_py, rho_cu_host, "Initial Density")
    analyze_density(rho_py, grid_py, "DFTpy")
    analyze_density(rho_cu_host, grid_cu, "DFTcu")

    # 初始能量对比
    print("\n[Initial Energy Components]")
    components_py_0 = {}
    for name, func in [
        ("Kinetic", ke_py),
        ("Hartree", hartree_py),
        ("XC", xc_py),
        ("Pseudo", pseudo_py),
    ]:
        result = func.compute(rho_py, calcType=["E"])
        components_py_0[name] = result.energy
        print(f"  DFTpy {name:10s}: {result.energy:15.10f} Ha")

    ewald_py = pseudo_py.get_ewald()
    e_corr_py = ewald_py.Energy_corr()
    e_real_py = ewald_py.Energy_real()
    e_rec_py = ewald_py.Energy_rec()
    ewald_py_0 = e_corr_py + e_real_py + e_rec_py

    print(f"  DFTpy Ewald Real: {e_real_py:15.10f} Ha")
    print(f"  DFTpy Ewald Rec : {e_rec_py:15.10f} Ha")
    print(f"  DFTpy Ewald Corr: {e_corr_py:15.10f} Ha")
    print(f"  DFTpy Ewald     : {ewald_py_0:15.10f} Ha")

    total_py_0 = sum(components_py_0.values()) + ewald_py_0
    print(f"  DFTpy Total     : {total_py_0:15.10f} Ha")

    print("\n[DFTcu Energy Components]")
    v_tmp = dftcu.RealField(grid_cu, 1)
    e_ke_cu = dftcu.ThomasFermi(1.0).compute(rho_cu, v_tmp)
    print(f"  DFTcu Kinetic   : {e_ke_cu:15.10f} Ha")

    e_vh_cu = dftcu.Hartree(grid_cu).compute(rho_cu, v_tmp)
    print(f"  DFTcu Hartree   : {e_vh_cu:15.10f} Ha")

    e_xc_cu = dftcu.LDA_PZ().compute(rho_cu, v_tmp)
    print(f"  DFTcu XC        : {e_xc_cu:15.10f} Ha")

    e_ps_cu = pseudo_cu.compute(rho_cu, v_tmp)
    print(f"  DFTcu Pseudo    : {e_ps_cu:15.10f} Ha")

    e_ii_cu = ewald_cu.compute(False)  # use_pme=False
    print(f"  DFTcu Ewald     : {e_ii_cu:15.10f} Ha")

    total_cu_0 = e_ke_cu + e_vh_cu + e_xc_cu + e_ps_cu + e_ii_cu
    print(f"  DFTcu Total     : {total_cu_0:15.10f} Ha")
    print(f"  Diff Total      : {abs(total_py_0 - total_cu_0):.2e} Ha")

    # 获取 DFTpy 的总势能用于对比
    result_py = evaluator_py.compute(rho_py, calcType=["E", "V"])
    v_tot_cu = dftcu.RealField(grid_cu, 1)
    evaluator_cu.compute(rho_cu, v_tot_cu)

    print(f"  DFTpy Eta       : {pseudo_py.ewald.eta:.6f}")

    # 分项对比势能
    print("\n[Potential Components Comparison]")
    v_tmp_cu = dftcu.RealField(grid_cu, 1)

    # 1. Hartree
    dftcu.Hartree(grid_cu).compute(rho_cu, v_tmp_cu)
    v_tmp_host = np.zeros(nr, dtype=np.float64, order="C")
    v_tmp_cu.copy_to_host(v_tmp_host)
    compare_fields(
        hartree_py.compute(rho_py, calcType=["V"]).potential, v_tmp_host, "Hartree Potential"
    )

    # 2. XC
    dftcu.LDA_PZ().compute(rho_cu, v_tmp_cu)
    v_tmp_cu.copy_to_host(v_tmp_host)
    compare_fields(xc_py.compute(rho_py, calcType=["V"]).potential, v_tmp_host, "XC Potential")

    # 3. Pseudo
    pseudo_cu.compute(v_tmp_cu)  # This calculates V_ext(r)
    v_tmp_cu.copy_to_host(v_tmp_host)
    compare_fields(
        pseudo_py.compute(rho_py, calcType=["V"]).potential, v_tmp_host, "Pseudo Potential"
    )

    # 4. KEDF (TF)
    dftcu.ThomasFermi(1.0).compute(rho_cu, v_tmp_cu)
    v_tmp_cu.copy_to_host(v_tmp_host)
    compare_fields(
        ke_py.compute(rho_py, calcType=["V"], split=True)["TF"].potential,
        v_tmp_host,
        "TF Potential",
    )

    # 5. KEDF (vW)
    dftcu.vonWeizsacker(1.0).compute(rho_cu, v_tmp_cu)
    v_tmp_cu.copy_to_host(v_tmp_host)
    compare_fields(
        ke_py.compute(rho_py, calcType=["V"], split=True)["VW"].potential,
        v_tmp_host,
        "vW Potential",
    )

    # 初始势能对比
    v_tot_cu_host = np.zeros(nr, dtype=np.float64, order="C")
    v_tot_cu.copy_to_host(v_tot_cu_host)
    compare_fields(result_py.potential, v_tot_cu_host, "Initial Potential")

    # ==================== 开始优化 ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "Optimization Process (Step 1)")
    print("=" * 80)

    # DFTpy 优化 (1 step)
    print("\n[DFTpy TN Optimization - Step 1]")
    print("-" * 80)
    opt_py = Optimization(
        optimization_method="TN",
        EnergyEvaluator=evaluator_py,
        optimization_options={"maxiter": 1, "econv": 1e-12, "verbose": True},
    )
    rho_opt_py = opt_py.optimize_rho(guess_rho=rho_py)
    e_step1_py = evaluator_py.compute(rho_opt_py).energy

    # DFTcu 优化 (1 step)
    print("\n[DFTcu TN Optimization - Step 1]")
    print("-" * 80)

    options = dftcu.OptimizationOptions()
    options.max_iter = 1
    options.econv = 1e-12
    optimizer_cu = dftcu.TNOptimizer(grid_cu, options)

    rho_cu_opt = dftcu.RealField(grid_cu, 1)
    rho_cu_opt.fill(ions.get_ncharges() / ions.cell.volume)

    optimizer_cu.solve(rho_cu_opt, evaluator_cu)
    e_step1_cu = evaluator_cu.compute(rho_cu_opt, dftcu.RealField(grid_cu, 1))

    print(f"\nDFTpy Step 1 Energy: {e_step1_py:.12f} Ha")
    print(f"DFTcu Step 1 Energy: {e_step1_cu:.12f} Ha")
    print(f"Diff Step 1:        {abs(e_step1_py - e_step1_cu):.2e} Ha")

    # ==================== Line Search Comparison ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "Line Search Comparison (Step 1)")
    print("=" * 80)

    # Deriving the search direction and functions from the initial state
    rho_0_py = DirectField(grid=grid_py, data=np.full(nr, ions.get_ncharges() / grid_py.volume))
    phi_0_py = np.sqrt(rho_0_py)
    res_py = evaluator_py.compute(rho_0_py, calcType=["E", "V"])
    v_0_py = res_py.potential
    mu_0 = np.sum(v_0_py * rho_0_py) * grid_py.dV / ions.get_ncharges()

    # Initial residual and direction (Steepest Descent for step 1)
    residual_0 = (v_0_py - mu_0) * phi_0_py
    p_0 = -residual_0.copy()

    # Orthogonalize and normalize p_0
    p_dot_phi = np.sum(p_0 * phi_0_py) * grid_py.dV
    p_0 -= (p_dot_phi / ions.get_ncharges()) * phi_0_py
    p_norm = np.sqrt(np.sum(p_0 * p_0) * grid_py.dV)
    p_0 *= np.sqrt(ions.get_ncharges()) / p_norm

    e0 = res_py.energy
    derphi0 = 2.0 * np.sum(v_0_py * phi_0_py * p_0) * grid_py.dV

    def phi_func(theta):
        phi_t = np.cos(theta) * phi_0_py + np.sin(theta) * p_0
        rho_t = DirectField(grid=grid_py, data=phi_t * phi_t)
        return evaluator_py.compute(rho_t, calcType=["E"]).energy

    def derphi_func(theta):
        phi_t = np.cos(theta) * phi_0_py + np.sin(theta) * p_0
        rho_t = DirectField(grid=grid_py, data=phi_t * phi_t)
        res_t = evaluator_py.compute(rho_t, calcType=["V"])
        p_rot = np.cos(theta) * p_0 - np.sin(theta) * phi_0_py
        return 2.0 * np.sum(res_t.potential * phi_t * p_rot) * grid_py.dV

    print("Line Search Inputs:")
    print(f"  phi0:    {e0:.12f}")
    print(f"  derphi0: {derphi0:.12f}")

    from scipy.optimize._linesearch import scalar_search_wolfe1 as scipy_ls

    # SciPy (Reference) Results
    # scalar_search_wolfe1(phi, derphi, phi0, old_phi0, derphi0, c1=1e-4, c2=0.1, amax=50, ...)
    alpha_py, phi_py, phi0_py = scipy_ls(
        phi_func,
        derphi_func,
        phi0=e0,
        derphi0=derphi0,
        c1=1e-4,
        c2=0.2,
        amax=np.pi,
        amin=0.0,
        xtol=1e-14,
    )
    print("\nSciPy (Reference) Results:")
    print(f"  Alpha:  {alpha_py:.12f}")
    print(f"  Energy: {phi_py:.12f}")

    # DFTcu (C++) Line Search
    conv_cu, alpha_cu, phi_cu = dftcu.scalar_search_wolfe1(
        phi_func,
        derphi_func,
        phi0=e0,
        derphi0=derphi0,
        c1=1e-4,
        c2=0.2,
        amax=np.pi,
        amin=0.0,
        xtol=1e-14,
    )
    print("\nDFTcu (C++) Results:")
    print(f"  Alpha:  {alpha_cu:.12f}")
    print(f"  Energy: {phi_cu:.12f}")
    print(f"  Conv:   {conv_cu}")
    print(f"  Diff Alpha: {abs(alpha_py - alpha_cu):.2e}")

    # ==================== 继续优化到收敛 ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "Full Optimization")
    print("=" * 80)

    # Reset and run to convergence
    opt_py = Optimization(
        optimization_method="TN",
        EnergyEvaluator=evaluator_py,
        optimization_options={"maxiter": 50, "econv": 1e-10, "verbose": False},
    )
    rho_opt_py = opt_py.optimize_rho(guess_rho=rho_py)
    e_final_py = evaluator_py.compute(rho_opt_py).energy

    options.max_iter = 50
    options.econv = 1e-10
    optimizer_cu = dftcu.TNOptimizer(grid_cu, options)
    rho_cu_opt.fill(ions.get_ncharges() / ions.cell.volume)
    optimizer_cu.solve(rho_cu_opt, evaluator_cu)
    e_final_cu = evaluator_cu.compute(rho_cu_opt, dftcu.RealField(grid_cu, 1))

    print(f"\nDFTcu Final Energy: {e_final_cu:.12f} Ha")

    # ==================== 最终对比 ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "Final Comparison")
    print("=" * 80)

    # 最终密度对比
    rho_final_cu_host = np.zeros(nr, dtype=np.float64, order="C")
    rho_cu_opt.copy_to_host(rho_final_cu_host)

    compare_fields(rho_opt_py, rho_final_cu_host, "Final Density", tolerance=1e-6)
    analyze_density(rho_opt_py, grid_py, "DFTpy Final")
    analyze_density(rho_final_cu_host, grid_cu, "DFTcu Final")

    # 最终能量对比
    print("\n[Final Energy Comparison]")
    diff_energy = abs(e_final_py - e_final_cu)
    status = "✅" if diff_energy < 1e-6 else "❌"

    print(f"  DFTpy:  {e_final_py:.12f} Ha")
    print(f"  DFTcu:  {e_final_cu:.12f} Ha")
    print(f"  Diff:   {diff_energy:.2e} Ha {status}")
    print("  Target: < 1.0e-06 Ha")

    # 最终势能对比
    v_final_cu = dftcu.RealField(grid_cu, 1)
    evaluator_cu.compute(rho_cu_opt, v_final_cu)
    v_final_cu_host = np.zeros(nr, dtype=np.float64, order="C")
    v_final_cu.copy_to_host(v_final_cu_host)

    result_final_py = evaluator_py.compute(rho_opt_py)
    compare_fields(result_final_py.potential, v_final_cu_host, "Final Potential", tolerance=1e-5)

    # ==================== 能量组分分析 ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "Energy Components Analysis")
    print("=" * 80)

    # DFTpy 组分
    print("\n[DFTpy Energy Components]")
    components_py = {}
    for name, func in [
        ("Kinetic", ke_py),
        ("Hartree", hartree_py),
        ("XC", xc_py),
        ("Pseudo", pseudo_py),
    ]:
        try:
            if hasattr(func, "compute"):
                result = func.compute(rho_opt_py, calcType=["E"])
                if hasattr(result, "energy"):
                    components_py[name] = result.energy
                    print(f"  {name:10s}: {result.energy:15.10f} Ha")
        except Exception as e:
            print(f"  {name:10s}: Failed ({e})")

    # TODO: DFTcu 的组分输出需要在 C++ 端实现

    print("\n" + "=" * 80)
    print(" " * 25 + "Analysis Complete")
    print("=" * 80)

    return diff_energy


if __name__ == "__main__":
    diff = test_tn_step_by_step()
    exit(0 if diff < 1e-6 else 1)
