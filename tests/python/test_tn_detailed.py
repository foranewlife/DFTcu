import os

import dftcu
import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_step_by_step_alignment():
    # 1. Setup
    a0 = 7.6
    lattice = np.eye(3) * a0
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)
    grid_py = DirectGrid(lattice, nr=nr, full=True)
    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=grid_py, ions=ions, PP_list={"Al": pp_file})
    evaluator_py = TotalFunctional(
        KineticEnergyFunctional=Functional(type="KEDF", name="TF", grid=grid_py),
        XCFunctional=Functional(type="XC", name="LDA", grid=grid_py),
        HARTREE=Functional(type="HARTREE", grid=grid_py),
        PSEUDO=pseudo_py,
    )
    rho_py = DirectField(grid=grid_py, data=np.full(nr, ions.nat * 3.0 / grid_py.volume))
    phi_py = np.sqrt(rho_py)

    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0.0, 0.0, 0.0, 3.0, 0)])
    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc_radial(
        0, pseudo_py.readpp._gp["Al"].tolist(), pseudo_py.readpp._vp["Al"].tolist()
    )
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

    # 2. Get DFTpy Step 1
    from dftpy.optimization import Optimization as DFTpy_Opt

    opt_py = DFTpy_Opt(optimization_method="TN", EnergyEvaluator=evaluator_py)
    opt_py.optimization_options["maxiter"] = 2
    rho_opt_py = opt_py.optimize_rho(guess_rho=rho_py)
    e_py_step1 = evaluator_py.compute(rho_opt_py).energy

    # 3. Get DFTcu Step 1
    options = dftcu.OptimizationOptions()
    options.max_iter = 1
    optimizer_cu = dftcu.TNOptimizer(grid_cu, options)
    rho_cu_step1 = dftcu.RealField(grid_cu)
    rho_cu_step1.copy_from_host(rho_py.data)
    optimizer_cu.solve(rho_cu_step1, evaluator_cu)
    e_cu_step1 = evaluator_cu.compute(rho_cu_step1, dftcu.RealField(grid_cu))

    print(f"E1: py={e_py_step1:.14f}, cu={e_cu_step1:.14f}, diff={abs(e_py_step1-e_cu_step1):.1e}")

    # 4. Investigate divergence
    e_py_obj = evaluator_py.compute(rho_py, calcType=["E", "V"])
    mu_py = (e_py_obj.potential * rho_py).integral() / (ions.nat * 3.0)
    res_py = (e_py_obj.potential - mu_py) * phi_py
    direction_py, num_inner_py = opt_py.get_direction([res_py], phi=phi_py, mu=mu_py, method="TN")
    p_py, theta0_py = opt_py.OrthogonalNormalization(direction_py, phi_py)

    # ValueAndDerivative at 0 and alpha0
    vd0 = opt_py.ValueAndDerivative(phi_py, p_py, 0.0)
    vd_alpha = opt_py.ValueAndDerivative(phi_py, p_py, theta0_py)

    print(f"DFTpy at 0: f={vd0[0]:.14f}, df={vd0[1]:.14f}")
    print(f"DFTpy at alpha0: f={vd_alpha[0]:.14f}, df={vd_alpha[1]:.14f}")

    f0, df0 = vd0[0], vd0[1]
    f1, df1 = vd_alpha[0], vd_alpha[1]
    alpha = theta0_py

    d1 = df0 + df1 - 3.0 * (f1 - f0) / alpha
    d2 = np.sqrt(d1**2 - df0 * df1)
    theta_new = alpha * (1.0 - (df1 + d2 - d1) / (df1 - df0 + 2.0 * d2))
    print(f"Manual Cubic min: {theta_new:.14f}")

    # Scaling check
    thetas = np.linspace(0, 1.5, 1000)
    diffs = []
    for t in thetas:
        phit = phi_py.data + p_py.data * t
        norm = np.sum(phit * phit) * grid_py.dV
        phit *= np.sqrt((ions.nat * 3.0) / norm)
        rhot = phit * phit
        diffs.append(np.max(np.abs(rho_opt_py.data - rhot)))
    idx = np.argmin(diffs)
    print(f"Scaling Model: min_diff={diffs[idx]:.2e} at theta={thetas[idx]:.4f}")


if __name__ == "__main__":
    test_step_by_step_alignment()
