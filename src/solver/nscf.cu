#include <algorithm>
#include <iostream>

#include "solver/nscf.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"

namespace dftcu {

NonSCFSolver::NonSCFSolver(Grid& grid) : grid_(grid), subspace_solver_(grid) {}

EnergyBreakdown NonSCFSolver::solve(Hamiltonian& ham, Wavefunction& psi, double nelec,
                                    std::shared_ptr<Atoms> atoms, double ecutrho,
                                    const RealField* rho_core, double alpha_energy) {
    std::cout << "     TRACE: entering NonSCFSolver::solve" << std::endl;
    std::cout << "     Band Structure Calculation" << std::endl;

    // Phase 1: Diagonalization (equivalent to c_bands_nscf)
    // For now, we assume a single k-point as in the Silicon test case.
    std::cout << "     TRACE: calling subspace_solver_.solve_direct" << std::endl;
    std::vector<double> eigenvalues = subspace_solver_.solve_direct(ham, psi);

    // Phase 2: Weight Calculation (equivalent to weights_only / weights)
    // Silicon at Gamma is an insulator.
    int nbands = psi.num_bands();
    std::vector<double> occupations(nbands, 0.0);
    double ef = 0.0;
    std::cout << "     TRACE: calling compute_weights_insulator" << std::endl;
    compute_weights_insulator(nbands, nelec, eigenvalues, occupations, ef);

    // Phase 3: Density Construction (equivalent to sum_band)
    RealField rho(grid_, 1);
    std::cout << "     TRACE: calling psi.compute_density" << std::endl;
    psi.compute_density(occupations, rho);
    grid_.synchronize();

    // Phase 4: Energy Breakdown and Reporting
    SCFSolver::Options options;
    options.verbose = false;
    SCFSolver scf_helper(grid_, options);

    // Total energy correction term (QE style): Nelec * <V_eff>
    // Note: eigenvalues from solve_direct already include ham.get_v_of_0()
    double v0 = ham.get_v_of_0();
    double total_alpha_energy = nelec * v0;

    scf_helper.set_alpha_energy(total_alpha_energy);
    scf_helper.set_atoms(atoms);
    scf_helper.set_ecutrho(ecutrho);

    std::cout << "     End of band structure calculation" << std::endl;

    return scf_helper.compute_energy_breakdown(eigenvalues, occupations, ham, psi, rho, rho_core);
}

void NonSCFSolver::compute_weights_insulator(int nbands, double nelec,
                                             const std::vector<double>& eigenvalues,
                                             std::vector<double>& occupations, double& ef) {
    double degspin = 2.0;  // Assume nspin=1 and not noncollinear
    int n_filled = static_cast<int>(std::round(nelec / degspin));

    for (int i = 0; i < nbands; ++i) {
        if (i < n_filled) {
            occupations[i] = degspin;
        } else {
            occupations[i] = 0.0;
        }
    }

    if (n_filled > 0) {
        ef = eigenvalues[n_filled - 1];
    } else {
        ef = -1e10;
    }
}

}  // namespace dftcu
