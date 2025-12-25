#!/usr/bin/env python3
"""
Test Wang-Teter KEDF implementation against DFTpy
"""
import numpy as np
import pytest
import dftcu
from dftpy.field import DirectField
from dftpy.functional.kedf.wt import WT as DFTpy_WT
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.density import DensityGenerator

@pytest.mark.comparison
def test_wt_kedf():
    """Compare DFTcu Wang-Teter implementation with DFTpy"""
    # Setup
    lattice = np.eye(3) * 10.0
    nr = [32, 32, 32]
    dftpy_grid = DirectGrid(lattice, nr=nr, full=True)

    # Create test density from atomic charges
    pos = np.array([[5.0, 5.0, 5.0]])
    ions = Ions(symbols=['Al'], positions=pos, cell=lattice)
    ions.set_charges(3.0)
    generator = DensityGenerator()
    rho_dftpy = generator.guess_rho(ions, grid=dftpy_grid)

    # DFTpy calculation
    result_dftpy = DFTpy_WT(rho_dftpy, calcType={"E", "V"})
    energy_dftpy = result_dftpy.energy
    potential_dftpy = np.array(result_dftpy.potential)

    # DFTcu calculation
    grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
    rho_cu = dftcu.RealField(grid_cu, 1)
    rho_cu.copy_from_host(rho_dftpy.flatten(order="C"))
    v_kedf_cu = dftcu.RealField(grid_cu, 1)

    wt_cu = dftcu.WangTeter(coeff=1.0)
    energy_cu = wt_cu.compute(rho_cu, v_kedf_cu)

    potential_cu = np.zeros(grid_cu.nnr())
    v_kedf_cu.copy_to_host(potential_cu)
    potential_cu = potential_cu.reshape(nr, order="C")

    # Comparison
    energy_rel_error = abs(energy_cu - energy_dftpy) / max(abs(energy_dftpy), 1e-10)
    potential_max_diff = np.abs(potential_cu - potential_dftpy).max()

    # Thresholds
    assert energy_rel_error < 1e-5
    assert potential_max_diff < 1e-4

if __name__ == "__main__":
    test_wt_kedf()
