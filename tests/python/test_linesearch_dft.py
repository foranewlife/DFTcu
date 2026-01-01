#!/usr/bin/env python3
"""
Test C++ line search implementation on DFT energy landscape
"""
import numpy as np
from dftpy.ewald import ewald as DFTpy_Ewald
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional.pseudo import LocalPseudo as LP
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from scipy.optimize._linesearch import scalar_search_wolfe1

import dftcu

# Setup
a0, lattice, nr = 7.6, np.eye(3) * 7.6, [32, 32, 32]
ions = Ions(symbols=["Al"], positions=np.array([[0.0, 0.0, 0.0]]), cell=lattice)
ions.set_charges(3.0)

# DFTpy setup
grid_py = DirectGrid(lattice, nr=nr, full=True)
pseudo_py = LP(grid=grid_py, ions=ions, PP_list={"Al": "external/DFTpy/examples/DATA/al.lda.upf"})
evaluator_py = TotalFunctional(
    KineticEnergyFunctional=Functional(type="KEDF", name="TF", grid=grid_py),
    XCFunctional=Functional(type="XC", name="LDA", grid=grid_py),
    HARTREE=Functional(type="HARTREE", grid=grid_py),
    PSEUDO=pseudo_py,
)

# DFTcu setup
grid_cu = dftcu.Grid(lattice.flatten().tolist(), nr)
atoms_cu = dftcu.Atoms([dftcu.Atom(0.0, 0.0, 0.0, 3.0, 0)])
pseudo_cu = dftcu.LocalPseudo(grid_cu, atoms_cu)
pseudo_cu.set_vloc_radial(
    0, pseudo_py.readpp._gp["Al"].tolist(), pseudo_py.readpp._vp["Al"].tolist()
)


ewald_py = DFTpy_Ewald(grid=grid_py, ions=ions)
ewald_cu = dftcu.Ewald(grid_cu, atoms_cu)
ewald_cu.set_eta(ewald_py.eta)

evaluator_cu = dftcu.Evaluator(grid_cu)
evaluator_cu.add_functional(dftcu.ThomasFermi(1.0))
evaluator_cu.add_functional(pseudo_cu)
evaluator_cu.add_functional(dftcu.Hartree(grid_cu))
evaluator_cu.add_functional(dftcu.LDA_PZ())
evaluator_cu.add_functional(ewald_cu)

# Initial state
rho_py = DirectField(grid=grid_py, data=np.full(nr, ions.nat * 3.0 / grid_py.volume))
phi_py = np.sqrt(rho_py)
f_py = evaluator_py.compute(rho_py)
v_tot_py = f_py.potential
ne = np.sum(rho_py) * grid_py.dV
mu = np.sum(rho_py * v_tot_py) * grid_py.dV / ne

# Gradient and search direction
g_py = (v_tot_py - mu) * phi_py
p_py = -g_py.copy()

# Orthogonalize and normalize
p_dot_phi = np.sum(p_py * phi_py) * grid_py.dV
p_py -= (p_dot_phi / ne) * phi_py
p_norm = np.sqrt(np.sum(p_py * p_py) * grid_py.dV)
p_py = np.sqrt(ne) / p_norm * p_py

e0 = f_py.energy
g_dot_p = np.sum(g_py * p_py) * grid_py.dV
derphi0 = 2.0 * g_dot_p

print("=" * 70)
print("DFT Line Search Test: DFTcu vs SciPy")
print("=" * 70)
print(f"Initial energy: {e0:.12f} Ha")
print(f"g_dot_p: {g_dot_p:.12e}")
print(f"derphi(0): {derphi0:.12e}")


# Scipy reference
def phi_scipy(theta):
    phi_trial = np.cos(theta) * phi_py + np.sin(theta) * p_py
    rho_trial = phi_trial * phi_trial
    f_trial = evaluator_py.compute(DirectField(grid=grid_py, data=rho_trial))
    return f_trial.energy


def derphi_scipy(theta):
    phi_trial = np.cos(theta) * phi_py + np.sin(theta) * p_py
    rho_trial = phi_trial * phi_trial
    f_trial = evaluator_py.compute(DirectField(grid=grid_py, data=rho_trial))
    v_trial = f_trial.potential
    p_rot = np.cos(theta) * p_py - np.sin(theta) * phi_py
    return 2.0 * np.sum(v_trial * phi_trial * p_rot) * grid_py.dV


print("\n" + "-" * 70)
print("SciPy scalar_search_wolfe1")
print("-" * 70)

alpha_scipy, phi_scipy, phi0_scipy = scalar_search_wolfe1(
    phi_scipy,
    derphi_scipy,
    phi0=e0,
    derphi0=derphi0,
    c1=1e-4,
    c2=0.2,
    amax=np.pi,
    amin=0.0,
    xtol=1e-14,
)

print(f"  alpha = {alpha_scipy:.12e}")
print(f"  energy = {phi_scipy:.12f} Ha")
print(f"  dE = {phi_scipy - e0:.12e} Ha")

# TODO: Test DFTcu line search with same phi/derphi functions
# This would require exposing the line search function to Python

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Target: alpha={alpha_scipy:.6f}, E={phi_scipy:.12f} Ha")
