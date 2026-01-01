import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from test_utils import get_pp_path, get_system, setup_pseudo, to_dftcu_atoms

import dftcu


def test_tn_alignment_multi_steps():
    nr = [32, 32, 32]
    ions = get_system("Al_single", a=7.6)
    lattice = ions.cell.array

    grid_py = DirectGrid(lattice, nr=nr, full=True)
    pp_file = get_pp_path("al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=grid_py, ions=ions, PP_list={"Al": pp_file})
    evaluator_py = TotalFunctional(
        KineticEnergyFunctional=Functional(type="KEDF", name="TF", grid=grid_py),
        XCFunctional=Functional(type="XC", name="LDA", grid=grid_py),
        HARTREE=Functional(type="HARTREE", grid=grid_py),
        PSEUDO=pseudo_py,
    )
    rho_py = DirectField(grid=grid_py, data=np.full(nr, ions.nat * 3.0 / grid_py.volume))

    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = to_dftcu_atoms(ions)
    pseudo_cu, _ = setup_pseudo(grid_cu, atoms_cu, "al.lda.upf", ions)
    from dftpy.ewald import ewald as DFTpy_Ewald

    ewald_py = DFTpy_Ewald(grid=grid_py, ions=ions)
    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)
    ewald_cu.set_eta(ewald_py.eta)
    evaluator_cu = dftcu.Evaluator(grid_cu)
    evaluator_cu.add_functional(dftcu.ThomasFermi(1.0))
    evaluator_cu.add_functional(pseudo_cu)
    evaluator_cu.add_functional(dftcu.Hartree(grid_cu))
    evaluator_cu.add_functional(dftcu.LDA_PZ())
    evaluator_cu.add_functional(ewald_cu)
    rho_cu = dftcu.RealField(grid_cu)
    rho_cu.fill(ions.nat * 3.0 / grid_cu.volume())

    from dftpy.optimization import Optimization as DFTpy_Opt

    opt_py = DFTpy_Opt(
        optimization_method="TN",
        EnergyEvaluator=evaluator_py,
        optimization_options={"maxiter": 10, "econv": 1e-15},
    )
    opt_py.nspin = 1
    opt_py.mp = grid_py.mp
    opt_py.lphi = False

    options = dftcu.OptimizationOptions()
    options.max_iter = 1
    options.econv = 1e-15
    optimizer_cu = dftcu.TNOptimizer(grid_cu, options)

    print("\n--- TN Alignment ---")
    theta_last = 0.1
    for step in range(5):
        # 1. Energy
        e_py_obj = evaluator_py.compute(rho_py, calcType=["E", "V"])
        e_cu = evaluator_cu.compute(rho_cu, dftcu.RealField(grid_cu))

        # 2. Chemical Potential
        mu_py = (e_py_obj.potential * rho_py).integral() / (ions.nat * 3.0)
        v_tot_cu = dftcu.RealField(grid_cu)
        evaluator_cu.compute(rho_cu, v_tot_cu)
        mu_cu = rho_cu.dot(v_tot_cu) * grid_cu.dv() / (ions.nat * 3.0)

        # 3. Residual Norm
        phi_py_curr = np.sqrt(rho_py)
        res_py = (e_py_obj.potential - mu_py) * phi_py_curr
        res_norm_py = np.sqrt((res_py * res_py).integral())

        phi_cu_data = np.zeros(nr)
        rho_cu.copy_to_host(phi_cu_data)
        phi_cu_data = np.sqrt(phi_cu_data)
        v_tot_cu_data = np.zeros(nr)
        v_tot_cu.copy_to_host(v_tot_cu_data)
        res_cu_data = (v_tot_cu_data - mu_cu) * phi_cu_data
        res_norm_cu = np.sqrt(np.sum(res_cu_data * res_cu_data) * grid_cu.dv())

        print(f"Step {step}:")
        print(
            f"  E: py={e_py_obj.energy:.14f}, cu={e_cu:.14f}, diff={abs(e_py_obj.energy-e_cu):.1e}"
        )
        print(f"  mu:     py={mu_py:.14f}, cu={mu_cu:.14f}, diff={abs(mu_py-mu_cu):.2e}")
        print(f"  |r|: py={res_norm_py:.4f}, cu={res_norm_cu:.4f}")

        # 4. Get direction
        direction_py, n_inner_py = opt_py.get_direction(
            [res_py], phi=phi_py_curr, mu=mu_py, method="TN"
        )
        p_py, theta0_py = opt_py.OrthogonalNormalization(direction_py, phi_py_curr)

        if step == 1:
            print(f"Step 1 Direction p_py (sample): {np.array(p_py.data).flatten()[:5]}")

        # 5. DFTpy step
        from functools import partial

        from dftpy.math_utils import line_search as dftpy_ls

        fun_ls = partial(opt_py.ValueAndDerivative, phi_py_curr, p_py)
        func0 = fun_ls(0.0, func=e_py_obj)

        theta_py_search = min(theta0_py, theta_last)
        theta_taken, task, n_ls, vd = dftpy_ls(fun_ls, alpha0=theta_py_search, func0=func0)
        theta_last = theta_taken

        phi_py_next = phi_py_curr.data * np.cos(theta_taken) + p_py.data * np.sin(theta_taken)
        rho_py = DirectField(grid=grid_py, data=phi_py_next * phi_py_next)

        # 6. DFTcu step
        optimizer_cu.solve(rho_cu, evaluator_cu)

        print(f"  theta:  py={theta_taken:.10f}")


if __name__ == "__main__":
    test_tn_alignment_multi_steps()
