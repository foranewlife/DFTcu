# Non-local Pseudopotential Alignment Plan (DFTcu vs. Quantum ESPRESSO)

To align the non-local pseudopotential ($V_{NL}$), we follow the Kleinman-Bylander (KB) form. This document outlines the technical investigation of QE's implementation and the systematic steps for matching it in `dftcu`.

## 1. QE Implementation Investigation

### 1.1 Mathematical Formulation
QE implements the non-local operator as:
$$\hat{V}_{NL} = \sum_{I, \ell, m} D_{I, \ell} |\beta_{I, \ell, m}\rangle \langle \beta_{I, \ell, m}|$$
The G-space projector $\beta_{\ell, m}(\mathbf{G})$ is computed as:
$$\beta_{\ell, m}(\mathbf{G}) = \frac{4\pi}{\Omega} i^{\ell} Y_{\ell, m}(\hat{G}) S(\mathbf{G}) \int_0^\infty r \beta_{\ell}(r) j_{\ell}(Gr) dr$$
where:
- $j_{\ell}(Gr)$ is the spherical Bessel function of order $\ell$.
- $Y_{\ell, m}(\hat{G})$ are **real spherical harmonics**.
- $S(\mathbf{G}) = e^{-i\mathbf{G} \cdot \mathbf{R}_I}$ is the structure factor.
- $D_{I, \ell}$ is the coupling constant (derived from `PP_DIJ` in UPF).

### 1.2 Key Source Files in QE
- `external/qe/upflib/pseudo_types.f90`: Definition of the `upf` structure and $V_{NL}$ parameters.
- `external/qe/PW/src/init_us_2.f90`: Main routine generating the radial interpolation table for projectors.
- `external/qe/UtilXlib/ylmr2.f90`: Real spherical harmonics $Y_{\ell m}$ implementation.
- `external/qe/PW/src/vloc_psi.f90`: Application of $V_{NL}$ to a wavefunction.

## 2. Alignment Strategy

### Phase 1: 1D Radial Projectors (`tab_nl`)
- **Objective**: Match the radial integration $\int r \beta_{\ell}(r) j_{\ell}(Gr) dr$.
- **Validation**: Compare `dftcu` interpolation table with QE's internal table (exposed via debug prints in `init_us_2.f90`).
- **Precision target**: $10^{-10}$ Relative error.
- **Key Task**: Implement high-precision spherical Bessel functions $j_0, j_1, j_2, j_3$ on GPU.

### Phase 2: 3D Projector Field $\beta(\mathbf{G})$
- **Objective**: Match the full 3D projector values on the FFT grid.
- **Validation**: Export $\beta(\mathbf{G})$ for a single atom at a non-zero position.
- **Verification**: Check the angular part ($Y_{\ell m}$) and the phase from the structure factor.
- **Note**: QE's real spherical harmonics convention (order of $m$) must be strictly followed.

### Phase 3: Projector Overlaps (Projected Coefficients)
- **Objective**: Match $c_{lm} = \sum_{\mathbf{G}} \beta_{lm}^*(\mathbf{G}) \psi(\mathbf{G})$.
- **Validation**: Using a reference plane-wave $\psi$, compare the resulting scalars $c_{lm}$.
- **Verification**: Ensure the G-space summation normalization (factors of $\Omega$ or $N_{nr}$) is consistent.

### Phase 4: Non-local Energy ($E_{NL}$)
- **Objective**: Match the final energy contribution.
- **Equation**: $E_{NL} = \sum_{I, \ell, m} D_{\ell} |c_{lm}|^2$.
- **Precision target**: Machine precision (similar to $V_{\text{loc}}$).

## 3. Implementation Checklist
1. [ ] Parse `PP_BETA` and `PP_DIJ` from UPF (using `upf_to_json` or manual parser).
2. [ ] Implement CUDA kernel for radial integration (Simpson rule + Bessel).
3. [ ] Implement Real Spherical Harmonics $Y_{lm}$ in CUDA.
4. [ ] Implement G-space overlap reduction kernel.
5. [ ] Integrate into `Evaluator` and verify against QE total energy.

## 4. Risks & Mitigations
- **Bessel Instability**: At $G \to 0$, $j_\ell(x)/x$ needs careful handling (Taylor expansion).
- **$Y_{\ell m}$ Mapping**: QE's internal order of $m$ (e.g., $p_y, p_z, p_x$ vs $p_x, p_y, p_z$) varies by version; must verify against `UtilXlib/ylmr2.f90`.
- **Memory Bandwidth**: $V_{NL}$ is more expensive than $V_{\text{loc}}$. Optimization via constant memory or shared memory for projectors might be needed.
