#!/usr/bin/env python3
import os

import dftcu
import numpy as np
import pytest
from dftpy.density import DensityGenerator
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


@pytest.mark.comparison
def test_simple_scf():
    """Test a simple SCF loop for Aluminum bulk"""
    # 1. Setup Grid and Ions
    lattice = np.eye(3) * 7.65  # Bohr, Al lattice constant ~4.05 Angstrom
    nr = [32, 32, 32]

    pos = np.array([[0.0, 0.0, 0.0]])  # FCC Al
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_list = [dftcu.Atom(0, 0, 0, 3.0, 0)]
    atoms_cu = dftcu.Atoms(atoms_list)

    # 2. Setup Functionals
    # KEDF: Thomas-Fermi
    tf = dftcu.ThomasFermi(coeff=1.0)

    # XC: LDA
    lda = dftcu.LDA_PZ()

    # Hartree
    hartree = dftcu.Hartree(grid_cu)

    # Pseudo
    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.recpot")
    if not os.path.exists(pp_file):
        pytest.skip(f"Pseudopotential file {pp_file} not found")

    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    pseudo_dftpy = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})
    vloc_Al_g = pseudo_dftpy.vlines["Al"]

    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    pseudo_cu.set_vloc(0, vloc_Al_g.flatten(order="C").tolist())

    # 3. Setup Evaluator
    evaluator = dftcu.Evaluator(grid_cu)
    evaluator.set_kedf(tf)
    evaluator.set_xc(lda)
    evaluator.set_hartree(hartree)
    evaluator.set_pseudo(pseudo_cu)

    # 4. Initial Density
    generator = DensityGenerator()
    rho_init = generator.guess_rho(ions, grid=dftpy_grid)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    # 5. Optimize
    options = dftcu.OptimizationOptions()
    options.max_iter = 50
    options.econv = 1e-7
    options.step_size = 0.05

    optimizer = dftcu.SimpleOptimizer(grid_cu, options)
    optimizer.solve(rho_cu, evaluator)

    # 6. Final Energy Evaluation
    v_tot = dftcu.RealField(grid_cu, 1)
    final_energy = evaluator.compute(rho_cu, v_tot)
    print(f"Final SCF Energy: {final_energy} Ha")

    # Simple check for stability
    assert final_energy < 0.0  # Should be negative for Al system


if __name__ == "__main__":
    test_simple_scf()
