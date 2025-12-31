#include <cmath>

#include "fft/fft_solver.cuh"
#include "pseudo.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "utilities/math_utils.cuh"

namespace dftcu {
namespace {
__constant__ double c_pseudo_atom_x[constants::MAX_ATOMS_PSEUDO];
__constant__ double c_pseudo_atom_y[constants::MAX_ATOMS_PSEUDO];
__constant__ double c_pseudo_atom_z[constants::MAX_ATOMS_PSEUDO];
__constant__ int c_pseudo_atom_type[constants::MAX_ATOMS_PSEUDO];

__global__ void vloc_gspace_kernel(int nnr, const double* gx, const double* gy, const double* gz,
                                   const double* gg, int nat, const double* tab_vloc,
                                   const double* zp, int nqx, double dq, double omega,
                                   gpufftComplex* v_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nnr)
        return;
    const double BOHR_TO_ANGSTROM = 0.529177210903;
    double g2 = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
    double gmod = sqrt(g2);

    if (g2 > 120.0000000001) {
        v_g[i].x = 0.0;
        v_g[i].y = 0.0;
        return;
    }

    const double fpi = 4.0 * constants::D_PI;
    double sum_re = 0.0;
    double sum_im = 0.0;

    for (int iat = 0; iat < nat; ++iat) {
        int type = c_pseudo_atom_type[iat];
        double vlocg = 0.0;
        if (gmod < 1e-8) {
            vlocg = 0.0;  // QE convention: G=0 is zero in the potential field
        } else {
            double gx_val = gmod;
            int i0 = (int)(gx_val / dq) + 1;
            i0 = min(max(i0, 1), nqx - 4);
            double px = gx_val / dq - floor(gx_val / dq);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;
            double w0 = ux * vx * wx / 6.0;
            double w1 = px * vx * wx / 2.0;
            double w2 = -px * ux * wx / 2.0;
            double w3 = px * ux * vx / 6.0;
            vlocg = tab_vloc[type * nqx + i0] * w0 + tab_vloc[type * nqx + i0 + 1] * w1 +
                    tab_vloc[type * nqx + i0 + 2] * w2 + tab_vloc[type * nqx + i0 + 3] * w3;
            vlocg -= (fpi * zp[type] / omega) * exp(-0.25 * g2) / g2;
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
    const double BOHR_TO_ANGSTROM = 0.529177210903;
    omega_ = omega_ang / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
    if (type >= static_cast<int>(zp_.size()))
        zp_.resize(type + 1, 0.0);
    zp_[type] = zp;
    double qmax = sqrt(grid_.g2max() * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM) * 1.2;
    int required_nqx = static_cast<int>(qmax / dq_) + 4;
    if (required_nqx > nqx_) {
        nqx_ = required_nqx;
        for (auto& t : tab_vloc_)
            t.resize(nqx_ + 1, 0.0);
    }
    if (type >= static_cast<int>(tab_vloc_.size()))
        tab_vloc_.resize(type + 1);
    tab_vloc_[type].resize(nqx_ + 1, 0.0);
    const double fpi = 4.0 * constants::D_PI;
    int msh = r_grid.size();
    std::vector<double> aux(msh);
    for (int iq = 1; iq <= nqx_; ++iq) {
        double q = (iq - 1) * dq_;
        if (iq == 1)
            for (int ir = 0; ir < msh; ++ir)
                aux[ir] = r_grid[ir] * (r_grid[ir] * vloc_r[ir] * 0.5 + zp * erf(r_grid[ir]));
        else
            for (int ir = 0; ir < msh; ++ir)
                aux[ir] = (r_grid[ir] * vloc_r[ir] * 0.5 + zp * erf(r_grid[ir])) *
                          sin(q * r_grid[ir]) / q;
        tab_vloc_[type][iq] = simpson_integrate(aux, rab) * fpi / omega_;
    }
    for (int ir = 0; ir < msh; ++ir)
        aux[ir] = r_grid[ir] * (r_grid[ir] * vloc_r[ir] * 0.5 + zp);
    tab_vloc_[type][0] = simpson_integrate(aux, rab) * fpi / omega_;
}

void LocalPseudo::compute(RealField& v) {
    initialize_buffers(v.grid());
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
    std::vector<double> flat(stride * tab_vloc_.size(), 0.0);
    for (size_t t = 0; t < tab_vloc_.size(); ++t)
        std::copy(tab_vloc_[t].begin(), tab_vloc_[t].end(), flat.begin() + t * stride);
    GPU_Vector<double> d_tab(flat.size());
    d_tab.copy_from_host(flat.data(), grid_.stream());
    GPU_Vector<double> d_zp(zp_.size());
    d_zp.copy_from_host(zp_.data(), grid_.stream());
    vloc_gspace_kernel<<<(grid_.nnr() + 255) / 256, 256, 0, grid_.stream()>>>(
        grid_.nnr(), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), atoms_->nat(), d_tab.data(),
        d_zp.data(), stride, dq_, omega_, v_g_->data());
    fft_solver_->backward(*v_g_);
    complex_to_real(grid_.nnr(), v_g_->data(), v.data(), grid_.stream());
}

double LocalPseudo::compute(const RealField& rho, RealField& v_out) {
    RealField v_ps(grid_);
    compute(v_ps);
    double energy = rho.dot(v_ps) * grid_.dv_bohr();
    v_add(grid_.nnr(), v_out.data(), v_ps.data(), v_out.data(), grid_.stream());
    return energy;
}
std::vector<double> LocalPseudo::get_vloc_g_shells(int type, const std::vector<double>& g) const {
    return {};
}
void LocalPseudo::set_valence_charge(int t, double z) {
    if (t >= static_cast<int>(zp_.size()))
        zp_.resize(t + 1);
    zp_[t] = z;
}
}  // namespace dftcu
