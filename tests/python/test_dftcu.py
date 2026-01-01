import numpy as np

import dftcu


def test_grid_initialization():
    """Test basic Grid creation"""
    lattice = np.eye(3) * 10.0
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)
    assert grid.nnr() == 16**3
    assert np.allclose(grid.volume(), 1000.0)


def test_real_field_operations():
    """Test basic RealField operations"""
    lattice = np.eye(3) * 10.0
    nr = [16, 16, 16]
    grid = dftcu.Grid(lattice.flatten().tolist(), nr)

    field = dftcu.RealField(grid, 1)
    field.fill(1.0)

    # Test integral (1.0 * volume)
    assert np.allclose(field.integral(), 1000.0)

    # Test copy from/to host
    data = np.random.rand(grid.nnr())
    field.copy_from_host(data)
    out = np.zeros(grid.nnr())
    field.copy_to_host(out)
    assert np.allclose(data, out)


def test_atoms_creation():
    """Test Atoms collection creation"""
    atoms_list = [dftcu.Atom(0, 0, 0, 1.0, 0), dftcu.Atom(5, 5, 5, 2.0, 1)]
    atoms = dftcu.Atoms(atoms_list)
    assert atoms.nat() == 2
