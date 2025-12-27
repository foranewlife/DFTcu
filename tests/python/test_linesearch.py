#!/usr/bin/env python3
"""
Isolated test for line search: compare DFTcu vs scipy DCSRCH
"""
import numpy as np
from scipy.optimize._linesearch import scalar_search_wolfe1


def test_simple_quadratic():
    """Test line search on a simple quadratic function: f(x) = (x-2)^2"""

    def phi(alpha):
        return (alpha - 2.0) ** 2

    def derphi(alpha):
        return 2.0 * (alpha - 2.0)

    phi0 = phi(0.0)
    derphi0 = derphi(0.0)

    print("=" * 70)
    print("Test 1: Simple Quadratic f(x) = (x-2)^2")
    print("=" * 70)
    print(f"phi(0) = {phi0}")
    print(f"derphi(0) = {derphi0}")
    print("Expected minimum at x=2.0")

    # scipy result
    alpha_scipy, phi_scipy, phi0_scipy = scalar_search_wolfe1(
        phi, derphi, phi0=phi0, derphi0=derphi0, c1=1e-4, c2=0.2, amax=10.0, amin=0.0, xtol=1e-14
    )

    print("\nSciPy result:")
    print(f"  alpha = {alpha_scipy:.12e}")
    print(f"  phi   = {phi_scipy:.12e}")
    print(f"  derphi = {derphi(alpha_scipy):.12e}")

    # TODO: DFTcu line search test will go here


def test_dft_energy():
    """Test line search on actual DFT energy landscape"""
    from dftpy.field import DirectField
    from dftpy.functional import Functional, TotalFunctional
    from dftpy.functional.pseudo import LocalPseudo as LP
    from dftpy.grid import DirectGrid
    from dftpy.ions import Ions

    # Setup
    lattice, nr = 7.6, np.eye(3) * 7.6, [32, 32, 32]
    ions = Ions(symbols=["Al"], positions=np.array([[0.0, 0.0, 0.0]]), cell=lattice)
    ions.set_charges(3.0)

    grid_py = DirectGrid(lattice, nr=nr, full=True)
    pseudo_py = LP(
        grid=grid_py, ions=ions, PP_list={"Al": "external/DFTpy/examples/DATA/al.lda.upf"}
    )
    evaluator_py = TotalFunctional(
        KineticEnergyFunctional=Functional(type="KEDF", name="TF", grid=grid_py),
        XCFunctional=Functional(type="XC", name="LDA", grid=grid_py),
        HARTREE=Functional(type="HARTREE", grid=grid_py),
        PSEUDO=pseudo_py,
    )

    rho_py = DirectField(grid=grid_py, data=np.full(nr, ions.nat * 3.0 / grid_py.volume))
    phi_py = np.sqrt(rho_py)
    f_py = evaluator_py.compute(rho_py)
    v_tot_py = f_py.potential
    ne = np.sum(rho_py) * grid_py.dV
    mu = np.sum(rho_py * v_tot_py) * grid_py.dV / ne

    # Gradient
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

    def phi(theta):
        phi_trial = np.cos(theta) * phi_py + np.sin(theta) * p_py
        rho_trial = phi_trial * phi_trial
        f_trial = evaluator_py.compute(DirectField(grid=grid_py, data=rho_trial))
        return f_trial.energy

    def derphi(theta):
        phi_trial = np.cos(theta) * phi_py + np.sin(theta) * p_py
        rho_trial = phi_trial * phi_trial
        f_trial = evaluator_py.compute(DirectField(grid=grid_py, data=rho_trial))
        v_trial = f_trial.potential
        p_rot = np.cos(theta) * p_py - np.sin(theta) * phi_py
        return 2.0 * np.sum(v_trial * phi_trial * p_rot) * grid_py.dV

    print("\n" + "=" * 70)
    print("Test 2: DFT Energy Landscape (Al FCC)")
    print("=" * 70)
    print(f"phi(0) = {e0:.12f} Ha")
    print(f"derphi(0) = {derphi0:.12e}")

    # scipy result
    alpha_scipy, phi_scipy, phi0_scipy = scalar_search_wolfe1(
        phi, derphi, phi0=e0, derphi0=derphi0, c1=1e-4, c2=0.2, amax=np.pi, amin=0.0, xtol=1e-14
    )

    print("\nSciPy result:")
    print(f"  alpha = {alpha_scipy:.12e}")
    print(f"  phi   = {phi_scipy:.12f} Ha")
    print(f"  derphi = {derphi(alpha_scipy):.12e}")
    print(f"  Energy reduction = {phi_scipy - e0:.12e} Ha")

    # Verify Wolfe conditions
    armijo = phi_scipy <= e0 + 1e-4 * alpha_scipy * derphi0
    curvature = abs(derphi(alpha_scipy)) <= 0.2 * abs(derphi0)
    print(f"\n  Armijo condition satisfied: {armijo}")
    print(f"  Curvature condition satisfied: {curvature}")

    return alpha_scipy, phi_scipy


if __name__ == "__main__":
    test_simple_quadratic()
    alpha_target, phi_target = test_dft_energy()

    print("\n" + "=" * 70)
    print("Target for DFTcu implementation:")
    print("=" * 70)
    print(f"  alpha = {alpha_target:.12e}")
    print(f"  energy = {phi_target:.12f} Ha")
