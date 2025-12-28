import os

import dftcu
import numpy as np
from dftpy.density import DensityGenerator
from dftpy.functional.kedf.hc import revHC as DFTpy_revHC
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def test_revhc_scaling():
    """Verify revHC across different lattice scales (densities)"""
    scaling_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    a_bohr_base = 10.0
    nr = [32, 32, 32]

    pp_file = os.path.join("external", "DFTpy", "examples", "DATA", "al.lda.upf")

    print("\nrevHC Lattice Scaling Test:")
    print(f"{'Scale':<8} | {'DFTpy Energy':<15} | {'DFTcu Energy':<15} | {'Diff (Ha)':<12}")
    print("-" * 60)

    for scale in scaling_factors:
        a_bohr = a_bohr_base * scale
        lattice = np.eye(3) * a_bohr
        dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
        grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)

        pos = np.array([[a_bohr / 2.0, a_bohr / 2.0, a_bohr / 2.0]])
        ions = Ions(symbols=["Al"], positions=pos, cell=lattice)

        from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

        pseudo_py = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})
        generator = DensityGenerator(pseudo=pseudo_py, direct=False)
        rho_init = generator.get_3d_value_recipe(ions, grid=dftpy_grid)

        rho_cu = dftcu.RealField(grid_cu, 1)
        rho_cu.copy_from_host(rho_init.flatten(order="C"))

        # 1. DFTpy
        ke_kernel_saved = {"shape": None, "rho0": 0.0, "Kernel": None, "kfmin": 1e8, "kfmax": -1e8}
        res_py = DFTpy_revHC(rho_init, calcType={"E"}, ke_kernel_saved=ke_kernel_saved)

        # 2. DFTcu
        revhc_cu = dftcu.revHC()
        v_cu = dftcu.RealField(grid_cu, 1)
        energy_cu = revhc_cu.compute(rho_cu, v_cu)

        diff = abs(res_py.energy - energy_cu)
        print(f"{scale:<8.1f} | {res_py.energy:<15.8f} | {energy_cu:<15.8f} | {diff:<12.2e}")

        # Assert high precision holds across scales after optimization
        assert diff < 1e-12


if __name__ == "__main__":
    test_revhc_scaling()
