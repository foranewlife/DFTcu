# Local Pseudopotential Alignment Report (DFTcu vs. Quantum ESPRESSO)

This report documents the alignment of the local pseudopotential ($V_{\text{loc}}$) calculation between `dftcu` and Quantum ESPRESSO (QE). As of December 31, 2025, the implementation has achieved **machine precision alignment** (error $\approx 10^{-12}$ Ry).

## 1. Core Issues and Solutions

### 1.1 Indexing Offset in `tab_vloc`
**Problem**: QE's interpolation table `tab_vloc` has a non-standard indexing convention. `tab_vloc(iq)` corresponds to $q = (iq-1) \times dq$ for $iq \ge 1$, while `tab_vloc(0)` is reserved for the Alpha correction term ($\int V_{\text{loc}}(r) + Ze^2/r dr$). DFTcu initially used a 0-based index for $q=0$, leading to a systematic shift and loss of the Alpha term.
**Solution**: Re-aligned DFTcu's internal storage to match QE:
- `tab_vloc[0]`: Alpha term.
- `tab_vloc[1]`: $q=0$ limit.
- `tab_vloc[iq]`: $q = (iq-1) \times dq$ for $iq > 1$.

### 1.2 Coordinate Precision and Sub-grid Displacement
**Problem**: In the test system, the QE input `9.4486 Bohr` was slightly different from the simplified `10.0 Angstrom` ($9.4486 \times 0.529177 \approx 4.99998...$). This tiny displacement caused the potential peak to shift between grid points, resulting in a $\approx 0.2\%$ discrepancy in the real-space peak value.
**Solution**: Used the exact QE-exported coordinates and physical constants ($BOHR\_TO\_ANGSTROM = 0.529177210903$) in the validation scripts.

### 1.3 Cutoff Energy Logic ($G^2$ Units)
**Problem**: In electronic structure codes using Bohr⁻¹ for $G$-vectors, the numerical value of $|G|^2$ is equivalent to energy in **Rydberg**, not Hartree.
Initially, a truncation of `if (g2 > 60.0)` was used (thinking Hartree), which actually cut the grid at `60 Ry`, whereas QE's `ecutrho` was `120 Ry`. This removed half of the high-frequency components.
**Solution**: Corrected the truncation logic to `if (g2 > 120.0)` to match QE's `ecutrho`.

### 1.4 FFT Normalization
**Problem**: Uncertainty whether IFFT requires a $1/N_{nr}$ scaling.
**Solution**: Verified that since the G-space potential $V(G)$ is already normalized by the cell volume $\Omega$ during the 1D Fourier transform, the standard cuFFT backward transform (which is unscaled) provides the physical real-space potential directly, without requiring an additional $1/N$ factor (provided the forward/backward logic is consistent).

## 2. Final Results

The following values were compared using an Oxygen ONCV pseudopotential in a 10 Å cubic cell with a $72 \times 72 \times 72$ grid.

| Quantity | QE Value (Ry) | DFTcu Value (Ry) | Absolute Diff |
| :--- | :--- | :--- | :--- |
| **$V_{\text{loc}}$ Average** | $1.026713 \times 10^{-3}$ | $1.026713 \times 10^{-3}$ | $< 10^{-15}$ |
| **G-space Max Diff** | - | - | $1.1 \times 10^{-14}$ |
| **$V_{\text{loc}}$ Peak (at atom)** | $-28.059955$ | $-28.059955$ | $1.5 \times 10^{-12}$ |
| **Max Abs Difference (Real)** | - | - | $1.5 \times 10^{-12}$ |

## 3. Implementation Checklist for Future Pseudo Alignment
1. [x] Match Simpson integration rule (handling of even/odd mesh).
2. [x] Align 1D interpolation grid indices (Alpha term at index 0).
3. [x] Use consistent physical constants (Bohr to Angstrom).
4. [x] Ensure $G$-vector truncation matches `ecutrho` in Rydberg.
5. [x] Verify IFFT scaling factors.
