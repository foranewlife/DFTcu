import os

import dftcu
import numpy as np
import pytest
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_tn_alignment_step10():
    """Verify TN alignment for 10 steps to 10+ digits."""
    # Setup system
    a0 = 4
    lattice = np.eye(3) * a0
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    # DFTpy Setup
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

    from dftpy.optimization import Optimization as DFTpy_Opt

    opt_py = DFTpy_Opt(
        optimization_method="TN",
        EnergyEvaluator=evaluator_py,
        optimization_options={"maxiter": 100, "econv": 1e-12},
    )
    rho_opt_py = opt_py.optimize_rho(guess_rho=rho_py)
    e_py_step10 = evaluator_py.compute(rho_opt_py).energy

    # DFTcu Setup
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

    options = dftcu.OptimizationOptions()
    options.max_iter = 100
    options.econv = 1e-12
    optimizer_cu = dftcu.TNOptimizer(grid_cu, options)

    optimizer_cu.solve(rho_cu, evaluator_cu)
    e_cu_step10 = evaluator_cu.compute(rho_cu, dftcu.RealField(grid_cu))

    print(f"Energy Step 10: py={e_py_step10:.14f}, cu={e_cu_step10:.14f}")
    diff = abs(e_py_step10 - e_cu_step10)
    print(f"Difference: {diff:.2e}")

    # Note: Cumulative differences in line search might cause drift over many steps
    assert diff < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
