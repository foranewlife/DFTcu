#include <cmath>

#include "fft/fft_solver.cuh"
#include "pseudo.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

// Simpson integration (QE style)
// QE's simpson routine from upflib/simpsn.f90
// CRITICAL: For even mesh, last point is NOT included (weight=0)
double simpson_integrate(const std::vector<double>& f, const std::vector<double>& rab) {
    int mesh = f.size();
    if (mesh < 2)
        return 0.0;
    if (mesh == 2)
        return 0.5 * (f[0] * rab[0] + f[1] * rab[1]);

    double asum = 0.0;
    const double r12 = 1.0 / 3.0;

    // QE's loop: DO i = 2, mesh-1 (Fortran 1-based)
    // In C++ 0-based: i = 1 to mesh-2
    for (int i = 1; i < mesh - 1; ++i) {
        // fct = DBLE(ABS(MOD(i,2)-2)*2)
        // For 0-based: i=1 (odd) -> fct=4, i=2 (even) -> fct=2
        // For 1-based: i=2 (even) -> fct=4, i=3 (odd) -> fct=2
        // We need to match Fortran's 1-based behavior
        int i_fortran = i + 1;  // Convert to 1-based
        double fct = static_cast<double>(std::abs((i_fortran % 2) - 2) * 2);
        asum += fct * f[i] * rab[i];
    }

    // IF (MOD(mesh,2)==1) THEN
    //   asum = (asum + func(1)*rab(1) + func(mesh)*rab(mesh)) * r12
    // ELSE
    //   asum = (asum + func(1)*rab(1) - func(mesh-1)*rab(mesh-1)) * r12
    // ENDIF
    if (mesh % 2 == 1) {
        // Odd mesh
        asum = (asum + f[0] * rab[0] + f[mesh - 1] * rab[mesh - 1]) * r12;
    } else {
        // Even mesh: last point (mesh-1 in Fortran, mesh-2 in C++) is subtracted
        asum = (asum + f[0] * rab[0] - f[mesh - 2] * rab[mesh - 2]) * r12;
    }

    return asum;
}

// Error function (erf)
inline double erf_approx(double x) {
    return std::erf(x);
}

// Constant memory for atom positions and types
__constant__ double c_pseudo_atom_x[constants::MAX_ATOMS_PSEUDO];
__constant__ double c_pseudo_atom_y[constants::MAX_ATOMS_PSEUDO];
__constant__ double c_pseudo_atom_z[constants::MAX_ATOMS_PSEUDO];
__constant__ int c_pseudo_atom_type[constants::MAX_ATOMS_PSEUDO];

// Kernel for V_loc in G-space with QE-style interpolation and erf correction
__global__ void vloc_gspace_kernel(int nnr, const double* gx, const double* gy, const double* gz,
                                   const double* gg,  // G² in Å^-2 (Grid uses Angstrom)
                                   int nat,
                                   const double* tab_vloc,  // [nqx * num_types]
                                   const double* zp,        // [num_types]
                                   int nqx,
                                   double dq,     // Bohr^-1
                                   double omega,  // Bohr³
                                   gpufftComplex* v_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nnr)
        return;

    // Convert G² from Å^-2 to Bohr^-2.
    // Note: G^2 in Bohr^-2 is numerically equal to energy in Rydberg.
    const double BOHR_TO_ANGSTROM = 0.529177210903;
    double g2_angstrom = gg[i];                                       // Å^-2
    double g2 = g2_angstrom * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);  // Bohr^-2 (Rydberg)
    double gmod = sqrt(g2);                                           // Bohr^-1

    // Strictly match QE's ecutrho = 120.0 Ry
    if (g2 > 120.0000000001) {
        v_g[i].x = 0.0;
        v_g[i].y = 0.0;
        return;
    }

    double sum_re = 0.0;
    double sum_im = 0.0;

    for (int iat = 0; iat < nat; ++iat) {
        int type = c_pseudo_atom_type[iat];

        // Step 1: Interpolate V_short(g) using 4-point cubic Lagrange
        double vlocg = 0.0;

        if (gmod < 1e-8) {
            // G=0 case: use tab_vloc[0]
            vlocg = tab_vloc[type * nqx + 0];
        } else {
            // 4-point cubic Lagrange interpolation
            double gx_val = gmod;
            double px = gx_val / dq - floor(gx_val / dq);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;

            int i0 = (int)(gx_val / dq) + 1;
            i0 = min(max(i0, 1), nqx - 4);  // nqx is stride (size of table)
            int i1 = i0 + 1;
            int i2 = i0 + 2;
            int i3 = i0 + 3;

            double w0 = ux * vx * wx / 6.0;
            double w1 = px * vx * wx / 2.0;
            double w2 = -px * ux * wx / 2.0;
            double w3 = px * ux * vx / 6.0;

            vlocg = tab_vloc[type * nqx + i0] * w0 + tab_vloc[type * nqx + i1] * w1 +
                    tab_vloc[type * nqx + i2] * w2 + tab_vloc[type * nqx + i3] * w3;
        }

        // Step 2: Subtract analytical erf Fourier transform
        // vloc(G) = vloc_short(G) - (4π * Z * e² / (Ω * G²)) * exp(-G²/4)
        // In Hartree atomic units: e² = 1
        if (g2 > 1e-8) {
            const double fpi = 4.0 * constants::D_PI;
            double fac = fpi * zp[type] / omega;  // Hartree / Bohr²
            vlocg -= fac * exp(-0.25 * g2) / g2;
        }

        // Step 3: Multiply by structure factor exp(-i G·R)
        // Note: gx, gy, gz are in Å^-1, atom positions need to be in Å
        double phase = -(gx[i] * c_pseudo_atom_x[iat] + gy[i] * c_pseudo_atom_y[iat] +
                         gz[i] * c_pseudo_atom_z[iat]);
        double s, c;
        sincos(phase, &s, &c);

        sum_re += vlocg * c;
        sum_im += vlocg * s;
    }

    v_g[i].x = sum_re;
    v_g[i].y = sum_im;
}

}  // anonymous namespace

LocalPseudo::LocalPseudo(Grid& grid, std::shared_ptr<Atoms> atoms) : grid_(grid), atoms_(atoms) {}

void LocalPseudo::initialize_buffers(Grid& grid) {
    if (grid_ptr_ == &grid)
        return;

    grid_ptr_ = &grid;
    if (!fft_solver_) {
        fft_solver_ = std::make_unique<FFTSolver>(grid);
    }
    if (!v_g_) {
        v_g_ = std::make_unique<ComplexField>(grid);
    }
}

void LocalPseudo::init_tab_vloc(int type, const std::vector<double>& r_grid,
                                const std::vector<double>& vloc_r, const std::vector<double>& rab,
                                double zp, double omega_angstrom) {
    // Convert omega from Angstrom³ to Bohr³ for QE formulas
    const double BOHR_TO_ANGSTROM = 0.529177210903;
    omega_ = omega_angstrom / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

    // Resize storage if needed
    if (type >= static_cast<int>(zp_.size())) {
        zp_.resize(type + 1, 0.0);
    }
    zp_[type] = zp;

    // Determine qmax from grid (matching QE's logic)
    // Convert Grid's g2max from Å^-2 to Bohr^-2
    double g2max_angstrom = grid_.g2max();                                       // Å^-2
    double g2max_bohr = g2max_angstrom * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);  // Bohr^-2

    // QE logic: qmax = sqrt(ecutrho) * cell_factor
    // Here we use sqrt(g2max) * 1.2 as a reasonable equivalent
    double qmax = sqrt(g2max_bohr) * 1.2;

    // QE defines nqx such that the table goes from 0 to nqx
    // tab_vloc(0:nqx) -> size nqx + 1
    int required_nqx = static_cast<int>(qmax / dq_) + 4;
    if (required_nqx > nqx_) {
        nqx_ = required_nqx;
        // Resize all existing tables. Each table has nqx_ + 1 elements.
        for (auto& table : tab_vloc_) {
            table.resize(nqx_ + 1, 0.0);
        }
    }

    if (type >= static_cast<int>(tab_vloc_.size())) {
        tab_vloc_.resize(type + 1);
    }
    tab_vloc_[type].resize(nqx_ + 1, 0.0);

    const double fpi = 4.0 * constants::D_PI;
    const double e2 = 1.0;  // Hartree atomic units

    int msh = r_grid.size();
    if (msh != static_cast<int>(vloc_r.size()) || msh != static_cast<int>(rab.size())) {
        throw std::runtime_error("init_tab_vloc: r_grid, vloc_r, and rab size mismatch");
    }

    std::vector<double> aux(msh);

    // Generate interpolation table for q-grid (indices 1 to nqx_)
    // QE convention: tab_vloc(iq) corresponds to q = (iq-1) * dq
    for (int iq = 1; iq <= nqx_; ++iq) {
        double q = (iq - 1) * dq_;

        if (iq == 1) {  // q = 0 case
            // q=0 case: continuous limit
            for (int ir = 0; ir < msh; ++ir) {
                double r = r_grid[ir];
                aux[ir] = r * (r * vloc_r[ir] + zp * e2 * erf_approx(r));
            }
        } else {
            // q>0 case: remove erf(r)/r term
            for (int ir = 0; ir < msh; ++ir) {
                double r = r_grid[ir];
                double qr = q * r;
                aux[ir] = (r * vloc_r[ir] + zp * e2 * erf_approx(r)) * sin(qr) / q;
            }
        }

        // Simpson integration
        tab_vloc_[type][iq] = simpson_integrate(aux, rab) * fpi / omega_;
    }

    // Compute G=0 term (alpha term) - stored at index 0
    // This is ∫ r² (V_loc(r) + Z*e²/r) dr
    for (int ir = 0; ir < msh; ++ir) {
        double r = r_grid[ir];
        aux[ir] = r * (r * vloc_r[ir] + zp * e2);
    }
    tab_vloc_[type][0] = simpson_integrate(aux, rab) * fpi / omega_;
}

std::vector<double> LocalPseudo::get_vloc_g_shells(int type,
                                                   const std::vector<double>& g_shells) const {
    if (type >= static_cast<int>(tab_vloc_.size())) {
        return {};
    }

    const auto& table = tab_vloc_[type];
    std::vector<double> results(g_shells.size());

    const double fpi = 4.0 * constants::D_PI;

    for (size_t i = 0; i < g_shells.size(); ++i) {
        double gmod = g_shells[i];
        double g2 = gmod * gmod;
        double vlocg = 0.0;

        if (gmod < 1e-8) {
            vlocg = table[0];  // G=0 term (Alpha term) is at index 0
        } else {
            // 4-point cubic Lagrange interpolation
            double gx_val = gmod;
            double px = gx_val / dq_ - floor(gx_val / dq_);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;

            int i0 = (int)(gx_val / dq_) + 1;
            // i0, i1, i2, i3 will use table[i0...i3]
            // We must ensure i3 <= nqx_
            i0 = std::max(1, std::min(i0, nqx_ - 3));
            int i1 = i0 + 1;
            int i2 = i0 + 2;
            int i3 = i0 + 3;

            double w0 = ux * vx * wx / 6.0;
            double w1 = px * vx * wx / 2.0;
            double w2 = -px * ux * wx / 2.0;
            double w3 = px * ux * vx / 6.0;

            vlocg = table[i0] * w0 + table[i1] * w1 + table[i2] * w2 + table[i3] * w3;
        }

        // Subtract analytical erf Fourier transform
        if (g2 > 1e-8) {
            double fac = fpi * zp_[type] / omega_;
            vlocg -= fac * exp(-0.25 * g2) / g2;
        }

        results[i] = vlocg;
    }

    return results;
}

void LocalPseudo::set_valence_charge(int type, double zp) {
    if (type >= static_cast<int>(zp_.size())) {
        zp_.resize(type + 1, 0.0);
    }
    zp_[type] = zp;
}

void LocalPseudo::compute(RealField& v) {
    // Initialize buffers (lazy initialization like Hartree)
    initialize_buffers(v.grid());

    if (atoms_->nat() > constants::MAX_ATOMS_PSEUDO) {
        throw std::runtime_error("Number of atoms exceeds MAX_ATOMS_PSEUDO");
    }

    // Copy atom positions and types to constant memory
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_x, atoms_->h_pos_x().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_y, atoms_->h_pos_y().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_z, atoms_->h_pos_z().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_type, atoms_->h_type().data(),
                                  atoms_->nat() * sizeof(int), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));

    // Flatten tab_vloc for GPU transfer
    int num_types = tab_vloc_.size();
    int stride = nqx_ + 1;
    std::vector<double> tab_vloc_flat(stride * num_types, 0.0);
    for (int t = 0; t < num_types; ++t) {
        if (!tab_vloc_[t].empty()) {
            std::copy(tab_vloc_[t].begin(), tab_vloc_[t].end(), tab_vloc_flat.begin() + t * stride);
        }
    }

    GPU_Vector<double> d_tab_vloc(tab_vloc_flat.size());
    GPU_Vector<double> d_zp(zp_.size());
    d_tab_vloc.copy_from_host(tab_vloc_flat.data(), grid_.stream());
    d_zp.copy_from_host(zp_.data(), grid_.stream());

    // Debug: Check host values
    /*
    std::cout << "DEBUG compute: num_types=" << num_types << " stride=" << stride << std::endl;
    std::cout << "DEBUG compute: zp[0]=" << zp_[0] << " omega=" << omega_ << std::endl;
    std::cout << "DEBUG compute: tab_vloc[0][0]=" << tab_vloc_[0][0] << std::endl;
    */

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (grid_.nnr() + block_size - 1) / block_size;

    vloc_gspace_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        grid_.nnr(), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), atoms_->nat(),
        d_tab_vloc.data(), d_zp.data(), stride, dq_, omega_, v_g_->data());
    GPU_CHECK_KERNEL;

    // Inverse FFT
    fft_solver_->backward(*v_g_);

    // Convert complex to real
    size_t nnr = grid_.nnr();
    complex_to_real(nnr, v_g_->data(), v.data(), grid_.stream());

    // Synchronize to ensure all GPU operations complete before local vectors are destroyed
    grid_.synchronize();
}

double LocalPseudo::compute(const RealField& rho, RealField& v_out) {
    RealField v_ps(grid_);
    compute(v_ps);

    grid_.synchronize();
    double energy = rho.dot(v_ps) * grid_.dv();

    size_t n = grid_.nnr();
    v_add(n, v_out.data(), v_ps.data(), v_out.data());

    return energy;
}

}  // namespace dftcu
