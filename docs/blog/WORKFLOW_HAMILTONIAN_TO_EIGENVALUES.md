# Technical Workflow: From Hamiltonian Construction to Eigenvalue Solution

**Document ID**: WF-H2E
**Date**: 2026-01-02
**Author**: Gemini Agent

## 1. Introduction
This document provides a comprehensive technical overview of the computational workflow within `DFTcu` to determine the electronic energy levels (eigenvalues) of a system. The process begins with the construction of the Kohn-Sham (KS) Hamiltonian operator and culminates in the solution of the generalized eigenvalue problem within a given subspace.

## 2. Step 1: Construction of the Kohn-Sham Hamiltonian
The KS Hamiltonian ($H$) is the central operator in DFT calculations. It is composed of several distinct components, each representing a physical interaction.

$$
H = T + V_{eff} = T + V_{loc} + V_H + V_{XC} + V_{NL}
$$

### 2.1 Kinetic Energy Operator ($T$)
The kinetic energy is most simply represented in reciprocal (G-space) as a diagonal operator. For a given plane-wave basis function $|G\rangle$, the kinetic energy is:

$$
T(G) = \frac{1}{2} |G|^2
$$

- **Implementation**: This is applied directly to the wavefunction coefficients in G-space.
- **File**: `src/solver/hamiltonian.cu` (in `apply_kinetic_kernel`)

### 2.2 Local Potential ($V_{loc}$)
This term describes the local interaction between the ionic cores and the valence electrons. It is read from the pseudopotential file (UPF format) and processed in several steps.

1.  **1D Radial Table**: A radial table, `tab_vloc`, is generated from the UPF data. This table stores the Fourier transform of the *short-range* part of the potential.
2.  **Alpha Term (G=0 Limit)**: A special value, the "Alpha term", is computed for the G=0 component and stored at `tab_vloc[0]`. This term is crucial for correct absolute energy levels. It is calculated from the integral $\int (rV_{loc}(r) + 2Ze^2) dr$.
3.  **G-space Construction**: For any G-vector, the potential $V_{loc}(G)$ is constructed by:
    a. Interpolating the value from `tab_vloc` (for $q > 0$, starting from index 1).
    b. Subtracting the analytical Fourier transform of the long-range part (an `erf` function).
    c. Multiplying by the structure factor $S(G) = \sum_i e^{-iG \cdot R_i}$.
4.  **Real Space**: The final potential is obtained by an inverse FFT of $V_{loc}(G)$.

- **Implementation**: `src/functional/pseudo.cu`
- **Key Functions**: `init_tab_vloc`, `vloc_gspace_kernel`

### 2.3 Hartree Potential ($V_H$)
This term accounts for the classical electrostatic repulsion between electrons. It is derived from the electronic charge density $\rho(r)$.

1.  **Poisson's Equation**: The potential is calculated by solving the Poisson equation in G-space:
    $$
    V_H(G) = \frac{4\pi \rho(G)}{|G|^2}
    $$
2.  **Real Space**: An inverse FFT is performed on $V_H(G)$ to get the real-space potential $V_H(r)$. The G=0 component is ignored as it corresponds to an arbitrary constant potential.

- **Implementation**: `src/functional/hartree.cu`
- **Key Function**: `hartree_kernel`

### 2.4 Exchange-Correlation Potential ($V_{XC}$)
This term captures all the many-body quantum mechanical effects. In `DFTcu`, we use the Local Density Approximation (LDA) with the Perdew-Zunger (PZ) parameterization.

- **Calculation**: $V_{XC}(r)$ is a non-linear function of the local charge density, $V_{XC}(r) = f(\rho(r))$.
- **Implementation**: `src/functional/xc/lda_pz.cu`
- **Key Function**: `lda_pz_kernel`

### 2.5 Non-Local Pseudopotential ($V_{NL}$)
This is the most complex part, accounting for the orthogonality of valence wavefunctions to the core states. It is implemented using Kleinman-Bylander projectors.

$$
V_{NL} = \sum_{i,j} D_{ij} |\beta_i\rangle \langle\beta_j|
$$

1.  **Projector Generation**: For each atom, the radial projector functions $\beta_l(r)$ from the UPF file are transformed to G-space, combined with spherical harmonics $Y_{lm}(\hat{G})$, and multiplied by the structure factor to form the full projector $|eta_{ilm}\rangle$.
2.  **Operator Application**: The action on a wavefunction $|\psi\rangle$ is a two-step process:
    a. **Projection**: Calculate the overlaps $\langle\beta_j | \psi \rangle$.
    b. **Accumulation**: Sum the contributions: $\sum_{i,j} D_{ij} |\beta_i\rangle \langle\beta_j | \psi \rangle$.
    This is implemented efficiently using `cublasZgemm`.

- **Implementation**: `src/functional/nonlocal_pseudo.cu`
- **Key Functions**: `update_projectors`, `apply`

## 3. Step 2: Operator Application ($H|\psi\rangle$)
Once the Hamiltonian components are assembled (with $V_{eff} = V_{loc} + V_H + V_{XC}$ being the total local potential in real space), it is applied to a set of wavefunctions $|\psi_n\rangle$.

The process for each band `n` is:
1.  Transform $|\psi_n\rangle$ from G-space to R-space.
2.  Multiply the real-space wavefunction by $V_{eff}(r)$.
3.  Transform the result back to G-space.
4.  Add the kinetic energy term $\frac{1}{2}G^2 |\psi_n(G)\rangle$.
5.  Add the non-local contribution $V_{NL}|\psi_n\rangle$.

- **Implementation**: `src/solver/hamiltonian.cu`
- **Key Function**: `apply`

## 4. Step 3: Subspace Projection and Diagonalization
To find the energy levels, the Hamiltonian is projected onto the subspace spanned by the current wavefunctions.

1.  **Matrix Construction**: Two matrices are constructed:
    - **Hamiltonian Matrix**: $H_{ij} = \langle \psi_i | H | \psi_j \rangle$
    - **Overlap Matrix**: $S_{ij} = \langle \psi_i | \psi_j \rangle$
    These inner products are computed as sums over G-space coefficients or integrals over the real-space grid.

2.  **Generalized Eigenvalue Problem**: The energy levels are the eigenvalues of the equation:
    $$
    H C = E S C
    $$
    where $E$ is a diagonal matrix of eigenvalues and $C$ is the matrix of eigenvectors.

This workflow, from constructing individual potential terms to solving the final eigensystem, forms the core of a single "diagonalization" step within a larger self-consistent field (SCF) cycle. The bit-perfect alignment achieved ensures that each of these steps is numerically identical to reference codes like Quantum ESPRESSO.
