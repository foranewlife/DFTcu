#include "math/bessel.cuh"
#include "math/ylm.cuh"
#include "nonlocal_pseudo.cuh"
#include "utilities/constants.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "utilities/math_utils.cuh"

#include <thrust/complex.h>

namespace dftcu {

namespace {

__device__ __constant__ double c_nl_atom_x[256];
__device__ __constant__ double c_nl_atom_y[256];
__device__ __constant__ double c_nl_atom_z[256];
__device__ __constant__ int c_nl_atom_type[256];

__global__ void interpolate_beta_kernel(int n, const double* gx, const double* gy, const double* gz,
                                        const double* gg, int l, int m, int iat, const double* tab,
                                        int stride, double dq, double omega,
                                        gpufftComplex* beta_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        const double B = constants::BOHR_TO_ANGSTROM;
        double gmod = sqrt(gg[i]) * B;  // Bohr^-1
        double betag = 0.0;

        if (gmod < 1e-12) {
            if (l == 0)
                betag = tab[0];
            else
                betag = 0.0;
        } else {
            int i0 = (int)(gmod / dq) + 1;
            if (i0 < stride - 4) {
                double px = gmod / dq - (double)(i0 - 1);
                double ux = 1.0 - px;
                double vx = 2.0 - px;
                double wx = 3.0 - px;

                betag = tab[i0 - 1] * ux * vx * wx / 6.0 + tab[i0] * px * vx * wx / 2.0 -
                        tab[i0 + 1] * px * ux * wx / 2.0 + tab[i0 + 2] * px * ux * vx / 6.0;
            }
        }

        double ylm_val = get_ylm(l, m, gx[i], gy[i], gz[i], gmod / B);
        double val = betag * ylm_val;

        // Apply phase factor exp(-iG.R)
        double phase =
            -(gx[i] * c_nl_atom_x[iat] + gy[i] * c_nl_atom_y[iat] + gz[i] * c_nl_atom_z[iat]);
        double s, c;
        sincos(phase, &s, &c);

        double re_part = val * c;
        double im_part = val * s;

        // Apply (-i)^l factor (QE convention)
        if (l == 0) {
            beta_g[i].x = re_part;
            beta_g[i].y = im_part;
        } else if (l == 1) {
            beta_g[i].x = im_part;
            beta_g[i].y = -re_part;
        } else if (l == 2) {
            beta_g[i].x = -re_part;
            beta_g[i].y = -im_part;
        } else if (l == 3) {
            beta_g[i].x = -im_part;
            beta_g[i].y = re_part;
        }
    }
}
}  // namespace

NonLocalPseudo::NonLocalPseudo(Grid& grid) : grid_(grid), omega_(grid.volume_bohr()) {}

void NonLocalPseudo::init_tab_beta(int type, const std::vector<double>& r,
                                   const std::vector<std::vector<double>>& betas,
                                   const std::vector<double>& rab, const std::vector<int>& l_list,
                                   const std::vector<int>& kkbeta_list, double omega_ang) {
    if (type >= static_cast<int>(tab_beta_.size())) {
        tab_beta_.resize(type + 1);
        l_list_.resize(type + 1);
    }
    int n_betas = betas.size();
    tab_beta_[type].resize(n_betas);
    l_list_[type] = l_list;

    const double fpi = 4.0 * constants::D_PI;
    const double B = constants::BOHR_TO_ANGSTROM;
    double omega_b = omega_ang / (B * B * B);

    double g2max_bohr = grid_.g2max() * (B * B);
    nqx_ = (int)(sqrt(g2max_bohr) / dq_) + 10;

    for (int nb = 0; nb < n_betas; ++nb) {
        tab_beta_[type][nb].resize(nqx_ + 1);
        int l = l_list[nb];
        int kkbeta = kkbeta_list[nb];
        std::vector<double> aux(kkbeta);
        for (int iq = 0; iq < nqx_; ++iq) {
            double q = iq * dq_;
            for (int ir = 0; ir < kkbeta; ++ir) {
                double x = q * r[ir];
                double jl = spherical_bessel_jl(l, x);
                aux[ir] = betas[nb][ir] * r[ir] * jl;
            }
            // QE table in file has 4pi/sqrt(Omega) scaling
            tab_beta_[type][nb][iq] = simpson_integrate(aux, rab) * fpi / sqrt(omega_b);
        }
    }
}

void NonLocalPseudo::init_dij(int type, const std::vector<double>& dij) {
    if (type >= static_cast<int>(d_ij_.size()))
        d_ij_.resize(type + 1);
    int n_proj = static_cast<int>(sqrt(dij.size()));
    d_ij_[type].resize(n_proj, std::vector<double>(n_proj));
    for (int i = 0; i < n_proj; ++i)
        for (int j = 0; j < n_proj; ++j)
            d_ij_[type][i][j] = dij[i * n_proj + j];
}

void NonLocalPseudo::update_projectors(const Atoms& atoms) {
    int total_projectors = 0;
    for (size_t i = 0; i < atoms.nat(); ++i) {
        int type = atoms.h_type()[i];
        for (int l : l_list_[type])
            total_projectors += (2 * l + 1);
    }
    num_projectors_ = total_projectors;
    if (num_projectors_ == 0)
        return;

    size_t n = grid_.nnr();
    d_projectors_.resize(n * num_projectors_);
    d_coupling_.resize(num_projectors_ * num_projectors_);
    d_coupling_.fill(0.0);

    CHECK(cudaMemcpyToSymbolAsync(c_nl_atom_x, atoms.h_pos_x().data(), atoms.nat() * sizeof(double),
                                  0, cudaMemcpyHostToDevice, grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_nl_atom_y, atoms.h_pos_y().data(), atoms.nat() * sizeof(double),
                                  0, cudaMemcpyHostToDevice, grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_nl_atom_z, atoms.h_pos_z().data(), atoms.nat() * sizeof(double),
                                  0, cudaMemcpyHostToDevice, grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_nl_atom_type, atoms.h_type().data(), atoms.nat() * sizeof(int),
                                  0, cudaMemcpyHostToDevice, grid_.stream()));

    std::vector<double> h_coupling(num_projectors_ * num_projectors_, 0.0);
    int proj_idx = 0;
    for (size_t iat = 0; iat < atoms.nat(); ++iat) {
        int type = atoms.h_type()[iat];
        int n_radial = tab_beta_[type].size();
        std::vector<std::vector<int>> radial_to_proj(n_radial);
        int current_p = proj_idx;
        for (int nb = 0; nb < n_radial; ++nb) {
            int l = l_list_[type][nb];
            for (int m = 0; m < (2 * l + 1); ++m)
                radial_to_proj[nb].push_back(current_p++);
        }

        for (int nb = 0; nb < n_radial; ++nb) {
            int l = l_list_[type][nb];
            GPU_Vector<double> d_tab(tab_beta_[type][nb].size());
            d_tab.copy_from_host(tab_beta_[type][nb].data(), grid_.stream());

            for (int m_idx = 0; m_idx < (2 * l + 1); ++m_idx) {
                int p_idx = radial_to_proj[nb][m_idx];
                interpolate_beta_kernel<<<(n + 255) / 256, 256, 0, grid_.stream()>>>(
                    (int)n, grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), l, m_idx, (int)iat,
                    d_tab.data(), nqx_ + 1, dq_, omega_, d_projectors_.data() + p_idx * n);
            }
        }

        for (int nb = 0; nb < n_radial; ++nb) {
            int l1 = l_list_[type][nb];
            for (int mb = 0; mb < n_radial; ++mb) {
                int l2 = l_list_[type][mb];
                if (l1 == l2) {
                    for (int m = 0; m < (2 * l1 + 1); ++m) {
                        int p1 = radial_to_proj[nb][m];
                        int p2 = radial_to_proj[mb][m];
                        h_coupling[p1 * num_projectors_ + p2] =
                            d_ij_[type][nb][mb] * 0.5;  // Ry -> Ha
                    }
                }
            }
        }
        proj_idx = current_p;
    }
    d_coupling_.copy_from_host(h_coupling.data(), grid_.stream());
    grid_.synchronize();
}

void NonLocalPseudo::apply(Wavefunction& psi, Wavefunction& h_psi) {
    if (num_projectors_ == 0)
        return;
    size_t n = grid_.nnr();
    int nb = psi.num_bands();
    if (d_projections_.size() < (size_t)num_projectors_ * nb)
        d_projections_.resize(num_projectors_ * nb);

    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, grid_.stream()));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);

    // Physical scaling: <beta|psi> = sum(beta* psi)
    cuDoubleComplex alpha_nl = {1.0, 0.0}, beta_zero = {0.0, 0.0}, alpha_one = {1.0, 0.0};
    CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_C, CUBLAS_OP_N, num_projectors_, nb, (int)n,
                                 &alpha_nl, (const cuDoubleComplex*)d_projectors_.data(), (int)n,
                                 (const cuDoubleComplex*)psi.data(), (int)n, &beta_zero,
                                 (cuDoubleComplex*)d_projections_.data(), num_projectors_));

    GPU_Vector<gpufftComplex> dps(num_projectors_ * nb);
    {
        GPU_Vector<cuDoubleComplex> d_coup_c(num_projectors_ * num_projectors_);
        std::vector<double> h_coup(num_projectors_ * num_projectors_);
        d_coupling_.copy_to_host(h_coup.data(), grid_.stream());
        std::vector<cuDoubleComplex> h_coup_complex(h_coup.size());
        for (size_t i = 0; i < h_coup.size(); ++i)
            h_coup_complex[i] = {h_coup[i], 0.0};
        d_coup_c.copy_from_host(h_coup_complex.data(), grid_.stream());

        CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, num_projectors_, nb,
                                     num_projectors_, &alpha_one,
                                     (const cuDoubleComplex*)d_coup_c.data(), num_projectors_,
                                     (const cuDoubleComplex*)d_projections_.data(), num_projectors_,
                                     &beta_zero, (cuDoubleComplex*)dps.data(), num_projectors_));
    }
    // Accumulate: h_psi += Beta * dps
    {
        CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, nb, num_projectors_,
                                     &alpha_one, (const cuDoubleComplex*)d_projectors_.data(),
                                     (int)n, (const cuDoubleComplex*)dps.data(), num_projectors_,
                                     &alpha_one, (cuDoubleComplex*)h_psi.data(), (int)n));
    }
}

double NonLocalPseudo::calculate_energy(const Wavefunction& psi,
                                        const std::vector<double>& occupations) {
    if (num_projectors_ == 0)
        return 0.0;
    size_t n = grid_.nnr();
    int nb = psi.num_bands();
    if (d_projections_.size() < (size_t)num_projectors_ * nb)
        d_projections_.resize(num_projectors_ * nb);
    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, grid_.stream()));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);
    cuDoubleComplex alpha_nl = {omega_, 0.0}, beta_zero = {0.0, 0.0};
    CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_C, CUBLAS_OP_N, num_projectors_, nb, (int)n,
                                 &alpha_nl, (const cuDoubleComplex*)d_projectors_.data(), (int)n,
                                 (const cuDoubleComplex*)psi.data(), (int)n, &beta_zero,
                                 (cuDoubleComplex*)d_projections_.data(), num_projectors_));
    std::vector<gpufftComplex> hp(num_projectors_ * nb);
    d_projections_.copy_to_host(hp.data(), grid_.stream());
    std::vector<double> h_coup(num_projectors_ * num_projectors_);
    d_coupling_.copy_to_host(h_coup.data(), grid_.stream());
    double energy = 0.0;
    for (int ib = 0; ib < nb; ++ib) {
        double band_energy = 0.0;
        for (int i = 0; i < num_projectors_; ++i) {
            thrust::complex<double> pi(hp[ib * num_projectors_ + i].x,
                                       hp[ib * num_projectors_ + i].y);
            for (int j = 0; j < num_projectors_; ++j) {
                thrust::complex<double> pj(hp[ib * num_projectors_ + j].x,
                                           hp[ib * num_projectors_ + j].y);
                band_energy += (thrust::conj(pi) * h_coup[i * num_projectors_ + j] * pj).real();
            }
        }
        energy += band_energy * occupations[ib];
    }
    return energy;
}

void NonLocalPseudo::set_tab_beta(int type, int nb, const std::vector<double>& tab) {
    if (type < (int)tab_beta_.size() && nb < (int)tab_beta_[type].size())
        tab_beta_[type][nb] = tab;
}

std::vector<std::complex<double>> NonLocalPseudo::get_projector(int idx) const {
    size_t nnr = grid_.nnr();
    std::vector<std::complex<double>> host(nnr);
    CHECK(cudaMemcpy(host.data(), d_projectors_.data() + idx * nnr, nnr * sizeof(gpufftComplex),
                     cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::complex<double>> NonLocalPseudo::get_projections() const {
    std::vector<std::complex<double>> host(d_projections_.size());
    CHECK(cudaMemcpy(host.data(), d_projections_.data(),
                     d_projections_.size() * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));
    return host;
}

void NonLocalPseudo::add_projector(const std::vector<std::complex<double>>& beta_g,
                                   double coupling_constant) {}
void NonLocalPseudo::clear() {
    tab_beta_.clear();
    l_list_.clear();
    d_ij_.clear();
    num_projectors_ = 0;
}
}  // namespace dftcu
