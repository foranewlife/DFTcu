import numpy as np
import pytest

import dftcu


def test_grid_gvectors():
    lattice = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    nr = [8, 8, 8]
    grid = dftcu.Grid(lattice, nr)

    # We can't directly check gx/gy/gz from Python yet as they are not bound.
    # But we can check nnr and volume.
    assert grid.nnr() == 512
    assert abs(grid.volume() - 1000.0) < 1e-12


def test_pseudo_interpolation():
    lattice = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice, nr)

    # Define a smooth radial function v(q) = exp(-q^2/2)
    q_radial = np.linspace(0, 20, 5000)
    v_radial = np.exp(-(q_radial**2) / 2.0)

    atoms = dftcu.Atoms([dftcu.Atom(0.0, 0.0, 0.0, 3.0, 0)])
    pseudo = dftcu.LocalPseudo(grid, atoms)
    pseudo.set_vloc_radial(0, q_radial.tolist(), v_radial.tolist())

    v_r = dftcu.RealField(grid)
    pseudo.compute(v_r)

    v_r_data = np.zeros(grid.nnr())
    v_r.copy_to_host(v_r_data)

    # V(0) = 1/V_cell * sum_G V(G)
    # We need G vectors to verify. Since they are not bound,
    # we can't easily verify the exact sum here.
    # But we can check if it's reasonable.
    assert np.abs(v_r_data[0]) > 0

    # Check that it's consistent with a manual sum if we knew G
    # For now, this test just ensures it runs and produces non-zero results.
    # The detailed alignment in test_tn_detailed.py already verified pseudo components.


if __name__ == "__main__":
    pytest.main([__file__])
