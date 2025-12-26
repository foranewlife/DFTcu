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
from dftpy.optimization.optimization import Optimization


def test_scf_comparison_dftpy():
    """Compare a full SCF cycle convergence between DFTpy and DFTcu including Local Pseudo"""
    # 1. Setup System
    a0 = 7.65
    lattice = np.eye(3) * a0
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # 2. Setup DFTpy Components
    print("\n1. Running DFTpy SCF...")
    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})

    # Initial density guess
    generator = DensityGenerator(pseudo=pseudo_py, direct=False)
    rho_init = generator.get_3d_value_recipe(ions, grid=dftpy_grid)

    tf_py = Functional(type="KEDF", name="TF")
    lda_py = Functional(type="XC", name="LDA")
    hartree_py = DFTpy_Hartree()
    ewald_py = DFTpy_Ewald(grid=dftpy_grid, ions=ions)

    t0 = time.time()

    class MultiFunctional:
        def __init__(self, funcs):
            self.funcs = funcs

        def __call__(self, rho, **kwargs):
            total = self.funcs[0](rho, **kwargs)
            for f in self.funcs[1:]:
                res = f(rho, **kwargs)
                total.energy += res.energy
                total.potential += res.potential
            return total

    evaluator_py = MultiFunctional([tf_py, lda_py, hartree_py, pseudo_py])
    opt_py = Optimization(
        EnergyEvaluator=evaluator_py,
        optimization_options={"max_iter": 200, "econv": 1e-7, "ncheck": 2},
    )

    # Run DFTpy Optimization
    res_py = opt_py.optimize_rho(guess_rho=rho_init)

    # Robustly get energy from MultiFunctional output
    final_res = evaluator_py(res_py)
    e_py_val = float(final_res.energy)
    e_total_py = e_py_val + ewald_py.energy

    t_py = time.time() - t0
    print(f"DFTpy Energy: {e_total_py:.10f} Ha, Time: {t_py:.4f} s")

    # 3. DFTcu Implementation
    print("\n2. Running DFTcu (CUDA) SCF...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])

    tf_cu = dftcu.ThomasFermi()
    lda_cu = dftcu.LDA_PZ()
    hartree_cu = dftcu.Hartree(grid_cu)
    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc(0, pseudo_py.vlines["Al"].flatten(order="C").tolist())
    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)
    ewald_cu.set_eta(ewald_py.eta)

    evaluator_cu = dftcu.Evaluator(grid_cu)
    evaluator_cu.add_functional(tf_cu)
    evaluator_cu.add_functional(lda_cu)
    evaluator_cu.add_functional(hartree_cu)
    evaluator_cu.add_functional(pseudo_cu)
    evaluator_cu.add_functional(ewald_cu)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    options = dftcu.OptimizationOptions()
    options.max_iter = 200
    options.econv = 1e-7
    options.step_size = 0.1
    optimizer = dftcu.CGOptimizer(grid_cu, options)

    t0 = time.time()
    optimizer.solve(rho_cu, evaluator_cu)
    v_tot = dftcu.RealField(grid_cu, 1)
    e_total_cu = evaluator_cu.compute(rho_cu, v_tot)
    t_cu = time.time() - t0
    print(f"DFTcu Energy: {e_total_cu:.10f} Ha, Time: {t_cu:.4f} s")

    # 4. Final Comparison
    diff = abs(e_total_py - e_total_cu)
    print(f"\nFinal Difference: {diff:.2e} Ha")
    assert diff < 1e-6


if __name__ == "__main__":
    test_scf_comparison_dftpy()
