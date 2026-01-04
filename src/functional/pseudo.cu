#include <algorithm>
#include <cmath>

#include "fft/fft_solver.cuh"
#include "math/bessel.cuh"
#include "pseudo.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "utilities/math_utils.cuh"

namespace dftcu {

namespace {

__device__ __constant__ double c_pseudo_atom_x[256];
__device__ __constant__ double c_pseudo_atom_y[256];
__device__ __constant__ double c_pseudo_atom_z[256];
__device__ __constant__ int c_pseudo_atom_type[256];

__global__ void vloc_gspace_kernel(int n, const double* gx, const double* gy, const double* gz,
                                   const double* gg, int nat, const double* flat_tab,
                                   const double* zp, int stride, double dq, double omega,
                                   double gcut, gpufftComplex* v_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;

    const double B = constants::BOHR_TO_ANGSTROM;
    double g2_ang = gg[i];
    double g2 = g2_ang * (B * B);  // Bohr^-2

    // Apply G-vector cutoff (ecutrho) to prevent Gibbs oscillations
    // gcut is ecutrho in Rydberg, g2 is in Bohr^-2
    // In atomic units: |G|^2 (Bohr^-2) = energy (Ry) numerically
    if (gcut > 0 && g2 > gcut) {
        v_g[i].x = 0.0;
        v_g[i].y = 0.0;
        return;
    }

    double gmod = sqrt(g2);
    const double fpi = 4.0 * constants::D_PI;

    double sum_re = 0;
    double sum_im = 0;

    for (int iat = 0; iat < nat; ++iat) {
        int type = c_pseudo_atom_type[iat];
        const double* table_short = flat_tab + type * stride;

        double vlocg = 0;
        if (g2 < 1e-12) {
            vlocg = table_short[0];
        } else {
            int i0 = (int)(gmod / dq) + 1;
            i0 = min(max(i0, 1), stride - 4);
            double px = gmod / dq - (double)(i0 - 1);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;

            vlocg =
                table_short[i0] * ux * vx * wx / 6.0 + table_short[i0 + 1] * px * vx * wx / 2.0 -
                table_short[i0 + 2] * px * ux * wx / 2.0 + table_short[i0 + 3] * px * ux * vx / 6.0;

            // vlocg intensive in Hartree
            vlocg -= (fpi * zp[type] / (omega * g2)) * exp(-0.25 * g2);
        }

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
}  // namespace

LocalPseudo::LocalPseudo(Grid& grid, std::shared_ptr<Atoms> atoms) : grid_(grid), atoms_(atoms) {}

void LocalPseudo::initialize_buffers(Grid& grid) {
    if (grid_ptr_ == &grid)
        return;
    grid_ptr_ = &grid;
    if (!fft_solver_)
        fft_solver_ = std::make_unique<FFTSolver>(grid);
    if (!v_g_)
        v_g_ = std::make_unique<ComplexField>(grid);
}

void LocalPseudo::init_tab_vloc(int type, const std::vector<double>& r_grid,
                                const std::vector<double>& vloc_r, const std::vector<double>& rab,
                                double zp, double omega_ang) {
    const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
    omega_ = omega_ang / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

    if (type >= static_cast<int>(zp_.size()))
        zp_.resize(type + 1, 0.0);
    zp_[type] = zp;

    if (type >= static_cast<int>(tab_vloc_.size()))
        tab_vloc_.resize(type + 1);

    dq_ = 0.0001;
    double g2max_bohr = grid_.g2max() * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
    nqx_ = (int)(sqrt(g2max_bohr) / dq_) + 10;
    tab_vloc_[type].resize(nqx_ + 1);

    int msh = r_grid.size();
    std::vector<double> aux(msh);
    const double fpi = 4.0 * constants::D_PI;

    for (int iq = 1; iq <= nqx_; ++iq) {
        double q = (iq - 1) * dq_;
        if (iq == 1) {
            for (int ir = 0; ir < msh; ++ir)
                aux[ir] = r_grid[ir] * (r_grid[ir] * vloc_r[ir] + 2.0 * zp * erf(r_grid[ir]));
        } else {
            for (int ir = 0; ir < msh; ++ir)
                aux[ir] = (r_grid[ir] * vloc_r[ir] + 2.0 * zp * erf(r_grid[ir])) *
                          sin(q * r_grid[ir]) / q;
        }
        // Multiply by 0.5 to convert Rydberg -> Hartree. Restore 1/Omega for intensive V(G).
        tab_vloc_[type][iq] = (simpson_integrate(aux, rab) * fpi / omega_) * 0.5;
    }

    for (int ir = 0; ir < msh; ++ir)
        aux[ir] = r_grid[ir] * (r_grid[ir] * vloc_r[ir] + 2.0 * zp);
    tab_vloc_[type][0] = (simpson_integrate(aux, rab) * fpi / omega_) * 0.5;
}

void LocalPseudo::compute(RealField& v) {
    Grid& grid_ = v.grid();
    initialize_buffers(grid_);
    size_t nnr = grid_.nnr();

    v_g_->fill({0, 0});
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

    int stride = nqx_ + 1;
    if (d_tab_.size() != stride * tab_vloc_.size()) {
        std::vector<double> flat(stride * tab_vloc_.size(), 0.0);
        for (size_t t = 0; t < tab_vloc_.size(); ++t)
            std::copy(tab_vloc_[t].begin(), tab_vloc_[t].end(), flat.begin() + t * stride);
        d_tab_.resize(flat.size());
        d_tab_.copy_from_host(flat.data(), grid_.stream());
    }

    if (d_zp_.size() != zp_.size()) {
        d_zp_.resize(zp_.size());
        d_zp_.copy_from_host(zp_.data(), grid_.stream());
    }

    vloc_gspace_kernel<<<(grid_.nnr() + 255) / 256, 256, 0, grid_.stream()>>>(
        (int)grid_.nnr(), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), (int)atoms_->nat(),
        d_tab_.data(), d_zp_.data(), stride, dq_, omega_, gcut_, v_g_->data());

    fft_solver_->backward(*v_g_);
    complex_to_real(grid_.nnr(), v_g_->data(), v.data(), grid_.stream());
    grid_.synchronize();
}

double LocalPseudo::compute(const RealField& rho, RealField& v_out) {
    RealField v_ps(grid_);
    compute(v_ps);
    double energy = rho.dot(v_ps) * grid_.dv_bohr();
    v_add(grid_.nnr(), v_out.data(), v_ps.data(), v_out.data(), grid_.stream());
    return energy;
}

std::vector<double> LocalPseudo::get_vloc_g_shells(int type,
                                                   const std::vector<double>& g_shells) const {
    if (type >= static_cast<int>(tab_vloc_.size()) || tab_vloc_[type].empty())
        return {};
    const double fpi = 4.0 * constants::D_PI;
    std::vector<double> results(g_shells.size());
    for (size_t i = 0; i < g_shells.size(); ++i) {
        double gmod = g_shells[i];
        double g2 = gmod * gmod;
        if (g2 < 1e-12) {
            results[i] = tab_vloc_[type][0];
        } else {
            int i0 = (int)(gmod / dq_) + 1;
            i0 = std::min(std::max(i0, 1), nqx_ - 4);
            double px = gmod / dq_ - (double)(i0 - 1);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;
            double vlocg = tab_vloc_[type][i0] * ux * vx * wx / 6.0 +
                           tab_vloc_[type][i0 + 1] * px * vx * wx / 2.0 -
                           tab_vloc_[type][i0 + 2] * px * ux * wx / 2.0 +
                           tab_vloc_[type][i0 + 3] * px * ux * vx / 6.0;
            results[i] = vlocg - (fpi * zp_[type] / (omega_ * g2)) * std::exp(-0.25 * g2);
        }
    }
    return results;
}

void LocalPseudo::set_valence_charge(int t, double z) {
    if (t >= static_cast<int>(zp_.size()))
        zp_.resize(t + 1);
    zp_[t] = z;
}
}  // namespace dftcu
