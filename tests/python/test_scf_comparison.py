#!/usr/bin/env python3
import os

import dftcu
import numpy as np
import pytest
from dftpy.density import DensityGenerator
from dftpy.functional import Functional
from dftpy.functional import Hartree as DFTpy_Hartree
from dftpy.functional import TotalFunctional
from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.optimization import Optimization


def test_scf_vs_dftpy():
    """Compare converged SCF energy between DFTcu and DFTpy using WT, LDA, and Ewald"""
    print("\n" + "=" * 70)
    print("SCF Verification: DFTcu vs DFTpy (WT + LDA + Ewald)")
    print("=" * 70)

    # 1. Setup System
    lattice = np.eye(3) * 7.65
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.recpot")
    if not os.path.exists(pp_file):
        pytest.skip(f"Pseudopotential file {pp_file} not found")

    # Common convergence options (Units: Hartree)
    econv = 1.0e-7
    maxiter = 100
    ncheck = 2

    # 2. Run DFTpy SCF
    print("\n1. Running DFTpy Density Optimization...")
    pseudo_dftpy = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})
    # WT in DFTpy is (TF, vW, WT-NL)
    wt_dftpy = Functional(type="KEDF", name="WT")
    lda_dftpy = Functional(type="XC", name="LDA")
    hartree_dftpy = DFTpy_Hartree()

    evaluator_dftpy = TotalFunctional(
        KineticEnergyFunctional=wt_dftpy,
        XCFunctional=lda_dftpy,
        HARTREE=hartree_dftpy,
        PSEUDO=pseudo_dftpy,
    )

    generator = DensityGenerator()
    rho_init = generator.guess_rho(ions, grid=dftpy_grid)

    opt_dftpy = Optimization(
        EnergyEvaluator=evaluator_dftpy,
        optimization_method="CG-HS",
        optimization_options={"econv": econv, "maxiter": maxiter, "ncheck": ncheck},
    )
    rho_conv_dftpy = opt_dftpy.optimize_rho(guess_rho=rho_init.copy())
    energy_dftpy = opt_dftpy.functional.energy
    print(f"   DFTpy Total Energy: {energy_dftpy:.10f} Ha")

    # 3. Run DFTcu SCF
    print("\n2. Running DFTcu CGOptimizer (Conjugate Gradient HS)...")
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])

    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc(0, pseudo_dftpy.vlines["Al"].flatten(order="C").tolist())

    tf_cu = dftcu.ThomasFermi(coeff=1.0)
    vw_cu = dftcu.vonWeizsacker(coeff=1.0)
    wt_cu = dftcu.WangTeter(coeff=1.0)
    lda_cu = dftcu.LDA_PZ()
    hartree_cu = dftcu.Hartree(grid_cu)
    ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)

    evaluator_cu = dftcu.Evaluator(grid_cu)
    evaluator_cu.set_tf(tf_cu)
    evaluator_cu.set_vw(vw_cu)
    evaluator_cu.set_wt(wt_cu)
    evaluator_cu.set_xc(lda_cu)
    evaluator_cu.set_hartree(hartree_cu)
    evaluator_cu.set_pseudo(pseudo_cu)
    evaluator_cu.set_ewald(ewald_cu)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    options = dftcu.OptimizationOptions()
    options.max_iter = maxiter
    options.econv = econv
    options.ncheck = ncheck
    options.step_size = 0.1

    optimizer = dftcu.CGOptimizer(grid_cu, options)
    optimizer.solve(rho_cu, evaluator_cu)

    v_tot = dftcu.RealField(grid_cu, 1)
    energy_cu = evaluator_cu.compute(rho_cu, v_tot)
    print(f"   DFTcu Total Energy: {energy_cu:.10f} Ha")

    # 4. Final Comparison
    energy_abs_error = abs(energy_cu - energy_dftpy)

    rho_host_cu = np.zeros(grid_cu.nnr())
    rho_cu.copy_to_host(rho_host_cu)
    rho_host_cu = rho_host_cu.reshape(nr, order="C")
    density_max_abs_diff = np.abs(rho_host_cu - rho_conv_dftpy).max()

    print("\n3. Final Comparison (Absolute):")
    print(f"   Energy Abs Error: {energy_abs_error:.2e} Ha")
    print(f"   Density Max Abs Diff: {density_max_abs_diff:.2e}")

    # Thresholds
    assert energy_abs_error < 5e-4
    assert density_max_abs_diff < 0.05
    print("\n✓ SCF Verification (WT + Ewald) Passed")


if __name__ == "__main__":
    test_scf_vs_dftpy()
