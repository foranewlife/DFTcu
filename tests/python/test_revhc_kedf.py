import dftcu
from dftpy.density import DensityGenerator
from dftpy.functional.kedf.hc import revHC as DFTpy_revHC
from dftpy.grid import DirectGrid
from test_utils import get_pp_path, get_system


def test_revhc_functional():
    """
    Verify revHC non-local KEDF implementation against DFTpy.
    Matches energy within 10^-12 Ha after formula optimization.
    """
    nr = [32, 32, 32]
    ions = get_system("Al_single", a=10.0)
    lattice = ions.cell.array

    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)

    pp_file = get_pp_path("al.lda.upf")
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo

    pseudo_py = DFTpy_LocalPseudo(grid=dftpy_grid, ions=ions, PP_list={"Al": pp_file})
    generator = DensityGenerator(pseudo=pseudo_py, direct=False)
    rho_init = generator.get_3d_value_recipe(ions, grid=dftpy_grid)

    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_init.flatten(order="C"))

    # 1. DFTpy revHC (alpha=2, beta=2/3)
    ke_kernel_saved = {"shape": None, "rho0": 0.0, "Kernel": None, "kfmin": 1e8, "kfmax": -1e8}
    res_py = DFTpy_revHC(rho_init, calcType={"E", "V"}, ke_kernel_saved=ke_kernel_saved)

    # 2. DFTcu revHC
    revhc_cu = dftcu.revHC()
    v_cu = dftcu.RealField(grid_cu, 1)
    energy_cu = revhc_cu.compute(rho_cu, v_cu)

    print("\nrevHC Energy Comparison:")
    print(f"DFTpy Energy: {res_py.energy:.12f} Ha")
    print(f"DFTcu Energy: {energy_cu:.12f} Ha")
    print(f"Diff:         {abs(res_py.energy - energy_cu):.2e} Ha")

    # After performance and formula optimization, precision reaches 10^-12 Ha
    assert abs(res_py.energy - energy_cu) < 1e-12


if __name__ == "__main__":
    test_revhc_functional()
