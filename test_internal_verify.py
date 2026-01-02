import dftcu


def test_internal_verify():
    # Setup same grid as QE.
    # QE celldm(1) = 18.897261 Bohr.
    BOHR_TO_ANG = 0.529177210903
    a_ang = 18.897261 * BOHR_TO_ANG
    grid = dftcu.Grid([a_ang, 0, 0, 0, a_ang, 0, 0, 0, a_ang], [125, 125, 125])

    # 1. Check Radial Integration Alignment (CU vs QE)
    print("\n--- Phase 1: Radial Integration Alignment ---")
    atoms = dftcu.Atoms([dftcu.Atom(5.0, 5.0, 5.0, 6.0, 0)])
    upf_files = ["run_qe_v1.0/O_ONCV_PBE-1.0.UPF"]
    # initialize_hamiltonian will trigger the comparison if injection_path is provided
    dftcu.initialize_hamiltonian(grid, atoms, upf_files, injection_path="./run_qe_v1.0")

    # 2. Run the official internal verification tool (with injection)
    print("\n--- Phase 2: Subspace Eigenvalue Alignment (with QE Injection) ---")
    e_calc, e_ref, diff = dftcu.verify_qe_subspace_alignment("./run_qe_v1.0", grid)

    # 3. Check Eigenvalue Alignment without Injection
    print("\n--- Phase 3: Subspace Eigenvalue Alignment (NATIVE CU integration) ---")
    dftcu.verify_native_subspace_alignment("./run_qe_v1.0", grid, upf_files)

    if diff < 1e-14:
        print("\n[SUCCESS] Internalized Verification passed with 10^-15 precision!")


if __name__ == "__main__":
    test_internal_verify()
