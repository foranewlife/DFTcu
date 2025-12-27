import os

import dftcu
import numpy as np
from ase.build import bulk
from dftpy.ions import Ions

# Project root is expected to be 2 levels up from this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_pp_path(name="al.lda.upf"):
    """Get absolute path to a pseudopotential file."""
    # Common PPs used in tests
    paths = {
        "al.lda.upf": os.path.join(
            PROJECT_ROOT, "external", "DFTpy", "examples", "DATA", "al.lda.upf"
        ),
        "al.lda.recpot": os.path.join(
            PROJECT_ROOT, "external", "DFTpy", "examples", "DATA", "al.lda.recpot"
        ),
    }
    path = paths.get(name)
    if path and not os.path.exists(path):
        # Try a relative path if absolute fails (e.g. running from different dir)
        rel_path = os.path.join("external", "DFTpy", "examples", "DATA", name)
        if os.path.exists(rel_path):
            return rel_path
    return path


def get_system(name="Al_fcc", **kwargs):
    """Get standard test systems."""
    if name == "Al_fcc":
        # Bulk Al cubic fcc with 4 atoms
        a = kwargs.get("a", 4.05)
        cubic = kwargs.get("cubic", True)
        atoms = bulk("Al", "fcc", a=a, cubic=cubic)
        return Ions.from_ase(atoms)
    elif name == "Al_single":
        # Single atom in a cubic box
        a = kwargs.get("a", 7.6)
        lattice = np.eye(3) * a
        ions = Ions(symbols=["Al"], positions=[[0, 0, 0]], cell=lattice)
        ions.set_charges(3.0)
        return ions
    elif name == "Al_primitive":
        # Primitive FCC cell (1 atom)
        a = kwargs.get("a", 7.6)
        lattice = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
        ions = Ions(symbols=["Al"], positions=[[0, 0, 0]], cell=lattice)
        ions.set_charges(3.0)
        return ions
    else:
        raise ValueError(f"Unknown system: {name}")


def to_dftcu_atoms(ions):
    """Convert dftpy Ions to dftcu Atoms."""
    cu_atoms_list = []
    for i in range(ions.nat):
        # type is 0 for now as we mostly deal with single species in tests
        cu_atoms_list.append(
            dftcu.Atom(
                ions.positions[i, 0],
                ions.positions[i, 1],
                ions.positions[i, 2],
                ions.charges[i],
                0,
            )
        )
    return dftcu.Atoms(cu_atoms_list)


def setup_pseudo(grid_cu, atoms_cu, pp_name="al.lda.upf", ions_py=None):
    """Convenience helper to setup LocalPseudo for DFTcu."""
    from dftpy.functional.pseudo import LocalPseudo as DFTpy_LocalPseudo
    from dftpy.grid import DirectGrid

    pp_file = get_pp_path(pp_name)
    if ions_py is None:
        # Create a dummy ions object if not provided (rarely needed if we just need PP data)
        ions_py = Ions(symbols=["Al"], positions=[[0, 0, 0]], cell=np.eye(3) * 10)
        ions_py.set_charges(3.0)

    # Use a dummy grid for dftpy to read PP
    dummy_grid = DirectGrid(lattice=np.eye(3) * 10, nr=[10, 10, 10])
    pseudo_py = DFTpy_LocalPseudo(grid=dummy_grid, ions=ions_py, PP_list={"Al": pp_file})

    pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
    g_radial = pseudo_py.readpp._gp["Al"]
    v_radial = pseudo_py.readpp._vp["Al"]
    pseudo_cu.set_vloc_radial(0, g_radial.tolist(), v_radial.tolist())

    return pseudo_cu, pseudo_py
