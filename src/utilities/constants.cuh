#pragma once

namespace dftcu {

namespace constants {

// Physical constants
const double D_PI = 3.14159265358979323846;
const double D_PI2 = D_PI * D_PI;

// Unit conversion constants (Hartree Atomic Units)
const double BOHR_TO_ANGSTROM = 0.529177210903;          // 1 Bohr = 0.5292 Angstrom
const double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;  // 1 Angstrom = 1.8897 Bohr
const double HA_TO_EV = 27.211386245988;                 // 1 Ha = 27.2114 eV
const double HA_TO_RY = 2.0;                             // 1 Ha = 2 Ry (exact)
const double RY_TO_HA = 0.5;                             // 1 Ry = 0.5 Ha (exact)

// Common density thresholds for numerical stability
const double RHO_THRESHOLD = 1e-16;
const double SIGMA_THRESHOLD = 1e-20;
const double PHI_THRESHOLD = 1e-15;

// Functional specific constants
// Thomas-Fermi constant: C_TF = (3/10) * (3π²)^(2/3)
const double C_TF_BASE = 2.8712340001881918;  // (3.0/10.0) * pow(3.0*D_PI*D_PI, 2.0/3.0)

// Exchange-correlation constants
const double EX_LDA_COEFF = -0.7385587663820224;  // -3/4 * (3/pi)^(1/3)
// Value computed using double-precision arithmetic for consistency with other constants.
// Note: The last digit may differ due to floating-point calculation differences across platforms.

// Numerical limits
const int MAX_ATOMS_PSEUDO = 512;

}  // namespace constants

}  // namespace dftcu
