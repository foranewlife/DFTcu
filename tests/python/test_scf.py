#!/usr/bin/env python3
import dftcu
import numpy as np
from dftpy.density import DensityGenerator
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_scf_cycle():
    """Verify a complete SCF cycle on Aluminum bulk supercell"""
    # 1. Setup System
    a0 = 7.65
    lattice = np.eye(3) * a0
    nr = [32, 32, 32]
    pos = np.array([[0.0, 0.0, 0.0]])
    ions = Ions(symbols=["Al"], positions=pos, cell=lattice)
    ions.set_charges(3.0)

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    generator = DensityGenerator()
    rho_init = generator.guess_rho(ions, grid=dftpy_grid)

    # 2. DFTcu Implementation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    atoms_cu = dftcu.Atoms([dftcu.Atom(0, 0, 0, 3.0, 0)])

    tf = dftcu.ThomasFermi()
    lda = dftcu.LDA_PZ()
    hartree = dftcu.Hartree(grid_cu)
    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)

    # Note: We need a UPF to get Vloc, or manually set it.
    # Here we skip actual pseudo computation or use uniform potential for test logic.
    pseudo_cu.set_vloc(0, [0.0] * grid_cu.nnr())

    # 3. Setup Evaluator
    evaluator = dftcu.Evaluator(grid_cu)
    evaluator.add_functional(tf)
    evaluator.add_functional(lda)
    evaluator.add_functional(hartree)
    evaluator.add_functional(pseudo_cu)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    # 4. Run Optimizer
    options = dftcu.OptimizationOptions()
    options.max_iter = 10
    options.econv = 1e-5
    options.step_size = 0.1

    optimizer = dftcu.CGOptimizer(grid_cu, options)

    print("\nStarting SCF cycle test...")
    optimizer.solve(rho_cu, evaluator)
    print("SCF cycle test completed.")

    # 5. Final validation
    total_electrons = rho_cu.integral()
    print(f"Total electrons: {total_electrons:.4f}")
    assert abs(total_electrons - 3.0) < 1e-4


if __name__ == "__main__":
    test_scf_cycle()
