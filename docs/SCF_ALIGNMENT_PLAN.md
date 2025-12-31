# SCF Alignment Plan

To align the Total Energy and SCF convergence with Quantum ESPRESSO (QE) to machine precision.

## Step 1: Static Energy Terms (Fixed Rho)
- [x] **Hartree Energy**: Align $E_H$ using fixed density.
- [x] **Ewald Energy**: Align ion-ion interaction (including background correction).
- [ ] **XC Energy**: Align $E_{XC}$ for LDA/PBE.

## Step 2: Hamiltonian Action (Fixed Psi)
- [ ] **Kinetic Energy**: Align $\langle \psi | -\frac{1}{2}\nabla^2 | \psi \rangle$.
- [ ] **Local Potential Action**: $V_{loc} \psi$.
- [ ] **Non-local Potential Action**: $V_{NL} \psi$.

## Step 3: Total Energy Formula
- [ ] Align the double-counting correction formula: $E_{tot} = E_{band} - E_{H} + E_{XC} - \int V_{XC}\rho + E_{ewald}$.

## Step 4: SCF Convergence
- [ ] Align density mixing (Broyden/Pulay).
- [ ] Align convergence thresholds and error estimates.
