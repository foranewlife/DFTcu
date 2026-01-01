#!/usr/bin/env python3
"""
Test script to verify DFTcu Hamiltonian using QE initial data
"""
import numpy as np

import dftcu

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_RY = 2.0


def load_qe_rho(filename):
    """Load QE density from text file"""
    with open(filename, "r") as f:
        line = f.readline()
        nnr, nspin = map(int, line.split())
        print(f"Loading rho: nnr={nnr}, nspin={nspin}")

        rho_data = []
        for line in f:
            rho_data.append(float(line.strip()))

        rho = np.array(rho_data)
        print(f"Loaded {len(rho)} density values")
        return rho, nnr, nspin


def load_qe_wfc(wfc_file, indices_file):
    """Load QE wavefunctions and G-vector indices"""
    with open(wfc_file, "r") as f:
        line = f.readline()
        npw, nbnd, nks = map(int, line.split())
        print(f"Loading wfc: npw={npw}, nbnd={nbnd}, nks={nks}")

        wfc = np.zeros((npw, nbnd), dtype=complex)
        for ibnd in range(nbnd):
            # Read real part
            real_part = []
            for _ in range(npw):
                real_part.append(float(f.readline().strip()))
            # Read imaginary part
            imag_part = []
            for _ in range(npw):
                imag_part.append(float(f.readline().strip()))

            wfc[:, ibnd] = np.array(real_part) + 1j * np.array(imag_part)

        print(f"Loaded wfc with shape {wfc.shape}")

    # Load indices
    with open(indices_file, "r") as f:
        line = f.readline()
        npw_check = int(line.strip())
        assert npw_check == npw, f"NPW mismatch: {npw_check} vs {npw}"

        indices = []
        for _ in range(npw):
            line = f.readline()
            i, j, k = map(int, line.split())
            indices.append((i, j, k))

    return wfc, np.array(indices), npw, nbnd


def main():
    print("=" * 60)
    print("DFTcu QE Alignment Test")
    print("=" * 60)

    # System parameters
    a = 10.0  # Angstrom
    ecutwfc = 30.0  # Ry

    # Setup grid
    grid = dftcu.Grid([a, 0, 0, 0, a, 0, 0, 0, a], [96, 96, 96])
    vol_bohr = grid.volume() / (BOHR_TO_ANGSTROM**3)
    print(f"Grid volume (Bohr^3): {vol_bohr}")

    # Load QE initial data
    print("\nLoading QE initial data...")
    rho_qe, nnr, nspin = load_qe_rho("run_qe_lda/qe_rho_init.txt")
    wfc_qe, indices_qe, npw, nbnd = load_qe_wfc(
        "run_qe_lda/qe_wfc_init.txt", "run_qe_lda/qe_indices_init.txt"
    )

    print(f"\nQE initial rho integral: {np.sum(rho_qe) * grid.dv_bohr():.6f} electrons")

    # Setup DFTcu components
    print("\nSetting up DFTcu components...")

    # Atom
    pos_bohr = 9.4486
    pos_ang = pos_bohr * BOHR_TO_ANGSTROM
    z_valence = 6.0
    atoms = dftcu.Atoms([dftcu.Atom(pos_ang, pos_ang, pos_ang, z_valence, 0)])

    # Parse pseudopotential
    import xml.etree.ElementTree as ET

    tree = ET.parse("run_qe_lda/O_ONCV_PBE-1.2.upf")
    root = tree.getroot()

    grid_node = root.find(".//PP_MESH/PP_R")
    r_grid = np.fromstring(grid_node.text, sep=" ")
    rab_node = root.find(".//PP_MESH/PP_RAB")
    rab = np.fromstring(rab_node.text, sep=" ")
    vloc_node = root.find(".//PP_LOCAL")
    vloc_r = np.fromstring(vloc_node.text, sep=" ")

    betas = []
    l_list = []
    nl_node = root.find(".//PP_NONLOCAL")
    for child in nl_node:
        if child.tag.startswith("PP_BETA."):
            betas.append(np.fromstring(child.text, sep=" "))
            l_list.append(int(child.attrib["angular_momentum"]))

    dij_node = root.find(".//PP_NONLOCAL/PP_DIJ")
    dij_flat = np.fromstring(dij_node.text, sep=" ")

    # Setup functionals
    evaluator = dftcu.Evaluator(grid)
    evaluator.add_functional(dftcu.Hartree())
    evaluator.add_functional(dftcu.LDA_PZ())

    ewald = dftcu.Ewald(grid, atoms, 1e-10)
    ewald.set_eta(1.0)
    evaluator.add_functional(ewald)

    vloc = dftcu.LocalPseudo(grid, atoms)
    vloc.init_tab_vloc(0, r_grid, vloc_r, rab, z_valence, grid.volume())
    evaluator.add_functional(vloc)

    nl_pseudo = dftcu.NonLocalPseudo(grid)
    nl_pseudo.init_tab_beta(0, r_grid, betas, rab, l_list, grid.volume())
    nl_pseudo.init_dij(0, dij_flat)
    nl_pseudo.update_projectors(atoms)

    ham = dftcu.Hamiltonian(grid, evaluator)
    ham.set_nonlocal(nl_pseudo)

    # Load QE density into DFTcu
    print("\nLoading QE density into DFTcu...")
    # TODO: Need to implement a way to load data from numpy array into RealField
    # For now, let's just compute DFTcu's potential with its own density

    # Create DFTcu wavefunction with QE data
    psi_dftcu = dftcu.Wavefunction(grid, nbnd, ecutwfc / HARTREE_TO_RY)  # Convert to Ha

    # TODO: Need to implement a way to load wfc data from numpy array
    # For now, let's just randomize and see if the framework works
    psi_dftcu.randomize(1234)

    print("\nComputing Hamiltonian action...")
    # Apply Hamiltonian
    # ham.apply(psi_dftcu, h_psi)  # TODO: Need this API

    print("\n" + "=" * 60)
    print("Test framework ready!")
    print("Next steps:")
    print("1. Implement RealField.set_data() to load QE density")
    print("2. Implement Wavefunction.set_data() to load QE wavefunctions")
    print("3. Implement Hamiltonian.apply() to compute H|psi>")
    print("4. Compare eigenvalues and matrix elements with QE")
    print("=" * 60)


if __name__ == "__main__":
    main()
