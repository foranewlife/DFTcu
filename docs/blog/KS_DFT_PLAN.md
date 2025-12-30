# Kohn-Sham DFT Implementation Roadmap: Moving to SCF

Following the successful high-precision alignment of all energy components (Kinetic, Hartree, XC, Local Pseudo, and Non-local Pseudo) against Quantum ESPRESSO, the project is now entering the phase of implementing a fully self-consistent field (SCF) loop.

## 1. Phase 1: SCF Infrastructure & Loop Control
The immediate priority is to wrap the existing Hamiltonian and Solver components into a coherent SCF cycle.

*   **Initial Density Generation**: Implement a routine to generate the starting charge density $\rho(r)$ using the superposition of atomic densities (atomic starting guess) instead of random wavefunctions.
*   **Occupation Logic**: Implement Fermi-Dirac distribution and Methfessel-Paxton smearing for calculating band occupations $f_n$, supporting both insulating and metallic systems.
*   **Convergence Criteria**: Introduce robust stopping criteria based on:
    *   Total energy difference: $|E_{i} - E_{i-1}| < \epsilon_E$
    *   Density residual (DRHO): $\int |\rho_{i} - \rho_{i-1}| dr < \epsilon_\rho$
*   **Total Energy Reconstruction**: Ensure the double-counting terms in the KS energy functional are correctly handled during the SCF.

## 2. Phase 2: Density Mixing & Stability
To ensure convergence for complex systems, advanced mixing schemes are required.

*   **Linear Mixing**: Implement as a baseline (Python/C++).
*   **Pulay Mixing (DIIS)**: Implement the Direct Inversion in the Iterative Subspace (DIIS) algorithm to accelerate convergence by using information from previous iterations.
*   **Broyden Mixing**: Explore quasi-Newton methods for density acceleration.

## 3. Phase 3: Performance & Solver Optimization
*   **Davidson Preconditioning**: Optimize the kinetic energy-based preconditioner to speed up eigenvalue convergence in large basis sets.
*   **GPU Memory Management**: Refactor temporary buffer allocations within the SCF loop to minimize fragmentation and overhead.
*   **Orthogonalization**: Verify and optimize the stability of Modified Gram-Schmidt or Cholesky-based orthogonalization on GPU.

## 4. Phase 4: End-to-End Validation
Benchmark the full SCF process against Quantum ESPRESSO for standard systems:
*   **Oxygen Atom (O)**: Single point SCF convergence.
*   **Silicon Bulk (Si)**: 8-atom cell, verifying periodic boundary conditions and total energy per atom.
*   **Oxygen Molecule (O₂)**: Verifying energy as a function of bond length.

## 5. Future Outlook: Forces and Stress
Once the SCF cycle is stable and validated:
*   **Hellmann-Feynman Forces**: Calculate $\frac{dE}{dR}$ for geometry optimization.
*   **Stress Tensor**: Implement the calculation of the stress tensor for cell relaxation.

---
**Current Status**: Energy component alignment 100% completed.
**Next Action**: Implement `scf_loop.py` and density mixing logic.
