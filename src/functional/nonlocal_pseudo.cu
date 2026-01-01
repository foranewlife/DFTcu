#include <fstream>
#include <iostream>

#include "functional/nonlocal_pseudo.cuh"
#include "math/bessel.cuh"
#include "math/ylm.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "utilities/math_utils.cuh"

#include <cublas_v2.h>

namespace dftcu {

namespace {

__global__ void build_projectors_kernel(int nnr, const double* gx, const double* gy,
                                        const double* gz, const double* gg, int nat,
                                        const double* c_atom_x, const double* c_atom_y,
                                        const double* c_atom_z, const int* c_atom_type,
                                        const double* tab_beta, const int* l_list_flat,
                                        const int* rad_offsets, const int* num_rad_list, int stride,
                                        double dq, gpufftComplex* d_projectors) {
    int ig = blockDim.x * blockIdx.x + threadIdx.x;
    if (ig >= nnr)
        return;

    const double BOHR_TO_ANGSTROM = 0.529177210903;
    double g2_ang = gg[ig];
    double gmod_ang = sqrt(g2_ang);
    double gmod = gmod_ang * BOHR_TO_ANGSTROM;
    double g2 = gmod * gmod;
    bool truncated = (g2 > 120.0000000001);

    int iproj_global = 0;
    for (int iat = 0; iat < nat; ++iat) {
        int type = c_atom_type[iat];
        int r_off = rad_offsets[type];
        int n_rad = num_rad_list[type];
        double phase = -(gx[ig] * c_atom_x[iat] + gy[ig] * c_atom_y[iat] + gz[ig] * c_atom_z[iat]);
        double s_feat, c_feat;
        sincos(phase, &s_feat, &c_feat);

        for (int irad = 0; irad < n_rad; ++irad) {
            int rad_idx = r_off + irad;
            int l = l_list_flat[rad_idx];
            double vq = 0.0;
            if (!truncated) {
                int i0 = (int)(gmod / dq) + 1;
                i0 = min(max(i0, 1), stride - 4);
                double px = gmod / dq - floor(gmod / dq);
                double ux = 1.0 - px;
                double vx = 2.0 - px;
                double wx = 3.0 - px;
                double w0 = ux * vx * wx / 6.0;
                double w1 = px * vx * wx / 2.0;
                double w2 = -px * ux * wx / 2.0;
                double w3 = px * ux * vx / 6.0;
                const double* table = &tab_beta[rad_idx * stride];
                vq = table[i0] * w0 + table[i0 + 1] * w1 + table[i0 + 2] * w2 + table[i0 + 3] * w3;
            }
            for (int m = 0; m < 2 * l + 1; ++m) {
                double ylm = get_ylm(l, m, gx[ig], gy[ig], gz[ig], gmod_ang);
                double val = vq * ylm;

                double re = 0, im = 0;
                // Use (-i)^l convention matching QE forward FT
                if (l % 4 == 0) {
                    re = val * c_feat;
                    im = val * s_feat;
                } else if (l % 4 == 1) {
                    re = val * s_feat;
                    im = -val * c_feat;
                } else if (l % 4 == 2) {
                    re = -val * c_feat;
                    im = -val * s_feat;
                } else if (l % 4 == 3) {
                    re = -val * s_feat;
                    im = val * c_feat;
                }
                d_projectors[iproj_global * nnr + ig].x = re;
                d_projectors[iproj_global * nnr + ig].y = im;
                iproj_global++;
            }
        }
    }
}
}  // namespace

NonLocalPseudo::NonLocalPseudo(Grid& grid) : grid_(grid) {}

void NonLocalPseudo::init_tab_beta(int type, const std::vector<double>& r,
                                   const std::vector<std::vector<double>>& b,
                                   const std::vector<double>& rb, const std::vector<int>& l,
                                   double o) {
    const double B = 0.529177210903;
    omega_ = o / (B * B * B);
    if (type >= (int)tab_beta_.size()) {
        tab_beta_.resize(type + 1);
        l_list_.resize(type + 1);
    }
    tab_beta_[type].resize(b.size());
    l_list_[type] = l;
    double qmax = sqrt(grid_.g2max() * B * B) * 1.2;
    if ((int)(qmax / dq_) + 4 > nqx_)
        nqx_ = (int)(qmax / dq_) + 4;

    // Normalization matching QE: beta(q) = (4pi/sqrt(Omega)) * \int r j_l(qr) (r*beta) dr
    double pref = 4.0 * constants::D_PI / sqrt(omega_);
    for (size_t nb = 0; nb < b.size(); ++nb) {
        tab_beta_[type][nb].resize(nqx_ + 1);
        for (int iq = 1; iq <= nqx_; ++iq) {
            double q = (iq - 1) * dq_;
            std::vector<double> aux(r.size());
            for (size_t ir = 0; ir < r.size(); ++ir)
                aux[ir] = b[nb][ir] * spherical_bessel_jl(l[nb], q * r[ir]) * r[ir];
            tab_beta_[type][nb][iq] = simpson_integrate(aux, rb) * pref;
        }
    }
}

void NonLocalPseudo::init_dij(int type, const std::vector<double>& dij) {
    if (type >= (int)d_ij_.size())
        d_ij_.resize(type + 1);
    int nb = l_list_[type].size();
    d_ij_[type].assign(nb, std::vector<double>(nb));
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j)
            d_ij_[type][i][j] = dij[i * nb + j] * 0.5;
}

void NonLocalPseudo::update_projectors(const Atoms& atoms) {
    num_projectors_ = 0;
    for (int i = 0; i < atoms.nat(); ++i) {
        int t = atoms.h_type()[i];
        for (int l : l_list_[t])
            num_projectors_ += (2 * l + 1);
    }
    d_projectors_.resize((size_t)num_projectors_ * grid_.nnr());

    int stride = nqx_ + 1;
    int total_rad = 0;
    for (const auto& t : tab_beta_)
        total_rad += t.size();
    std::vector<double> flat_tab(total_rad * stride);
    std::vector<int> l_flat(total_rad), r_off(tab_beta_.size()), n_rad(tab_beta_.size());
    int curr = 0;
    for (size_t t = 0; t < tab_beta_.size(); ++t) {
        r_off[t] = curr;
        n_rad[t] = tab_beta_[t].size();
        for (size_t nb = 0; nb < tab_beta_[t].size(); ++nb) {
            std::copy(tab_beta_[t][nb].begin(), tab_beta_[t][nb].end(),
                      flat_tab.begin() + curr * stride);
            l_flat[curr++] = l_list_[t][nb];
        }
    }
    GPU_Vector<double> dt(flat_tab.size());
    dt.copy_from_host(flat_tab.data(), grid_.stream());
    GPU_Vector<int> dl(l_flat.size());
    dl.copy_from_host(l_flat.data(), grid_.stream());
    GPU_Vector<int> doff(r_off.size());
    doff.copy_from_host(r_off.data(), grid_.stream());
    GPU_Vector<int> dnr(n_rad.size());
    dnr.copy_from_host(n_rad.data(), grid_.stream());

    build_projectors_kernel<<<(grid_.nnr() + 255) / 256, 256, 0, grid_.stream()>>>(
        (int)grid_.nnr(), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), atoms.nat(),
        atoms.pos_x(), atoms.pos_y(), atoms.pos_z(), atoms.type(), dt.data(), dl.data(),
        doff.data(), dnr.data(), stride, dq_, d_projectors_.data());

    CHECK(cudaStreamSynchronize(grid_.stream()));

    std::vector<double> coup(num_projectors_ * num_projectors_, 0.0);
    int idx_i = 0;
    for (int iat = 0; iat < atoms.nat(); ++iat) {
        int t = atoms.h_type()[iat];
        int n_rad = (int)l_list_[t].size();
        int atom_proj_offset = idx_i;
        int curr_atom_total_proj = 0;
        for (int ih = 0; ih < n_rad; ++ih)
            curr_atom_total_proj += (2 * l_list_[t][ih] + 1);

        int r_off_ih = 0;
        for (int ih = 0; ih < n_rad; ++ih) {
            int l_h = l_list_[t][ih];
            for (int m = 0; m < 2 * l_h + 1; ++m) {
                int r_off_jh = 0;
                for (int jh = 0; jh < n_rad; ++jh) {
                    int l_j = l_list_[t][jh];
                    if (l_h == l_j) {
                        for (int mj = 0; mj < 2 * l_j + 1; ++mj) {
                            if (m == mj) {
                                int global_i = atom_proj_offset + r_off_ih + m;
                                int global_j = atom_proj_offset + r_off_jh + mj;
                                coup[global_i * num_projectors_ + global_j] = d_ij_[t][ih][jh];
                            }
                        }
                    }
                    r_off_jh += (2 * l_j + 1);
                }
            }
            r_off_ih += (2 * l_h + 1);
        }
        idx_i += curr_atom_total_proj;
    }
    d_coupling_.resize(num_projectors_ * num_projectors_);
    d_coupling_.copy_from_host(coup.data(), grid_.stream());
    grid_.synchronize();
}

void NonLocalPseudo::apply(Wavefunction& psi, Wavefunction& h_psi) {
    size_t n = grid_.nnr();
    int nb = psi.num_bands();
    if (num_projectors_ == 0)
        return;

    if (d_projections_.size() < (size_t)num_projectors_ * nb)
        d_projections_.resize(num_projectors_ * nb);

    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, grid_.stream()));
    cuDoubleComplex alpha_complex = {1.0, 0.0}, beta_complex = {0.0, 0.0};

    // 1. Projections: P = Beta^H * Psi (num_projectors x nb)
    {
        CUBLAS_SAFE_CALL(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_C, CUBLAS_OP_N, num_projectors_, nb, (int)n,
                                     &alpha_complex, (const cuDoubleComplex*)d_projectors_.data(),
                                     (int)n, (const cuDoubleComplex*)psi.data(), (int)n,
                                     &beta_complex, (cuDoubleComplex*)d_projections_.data(),
                                     num_projectors_));
    }

    // 2. Apply D_ij: P' = D * P
    GPU_Vector<gpufftComplex> dps(num_projectors_ * nb);
    {
        std::vector<double> h_coup(num_projectors_ * num_projectors_);
        d_coupling_.copy_to_host(h_coup.data(), grid_.stream());
        grid_.synchronize();
        std::vector<gpufftComplex> h_coup_c(h_coup.size());
        for (size_t i = 0; i < h_coup.size(); ++i)
            h_coup_c[i] = {h_coup[i], 0.0};
        GPU_Vector<gpufftComplex> d_coupling_complex(h_coup_c.size());
        d_coupling_complex.copy_from_host(h_coup_c.data(), grid_.stream());

        CUBLAS_SAFE_CALL(cublasZgemm(
            h, CUBLAS_OP_N, CUBLAS_OP_N, num_projectors_, nb, num_projectors_, &alpha_complex,
            (const cuDoubleComplex*)d_coupling_complex.data(), num_projectors_,
            (const cuDoubleComplex*)d_projections_.data(), num_projectors_, &beta_complex,
            (cuDoubleComplex*)dps.data(), num_projectors_));
        CHECK(cudaStreamSynchronize(grid_.stream()));
    }

    // 3. Accumulate back to H_psi: H_psi += Beta * P'
    {
        CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, nb, num_projectors_,
                                     &alpha_complex, (const cuDoubleComplex*)d_projectors_.data(),
                                     (int)n, (const cuDoubleComplex*)dps.data(), num_projectors_,
                                     &alpha_complex, (cuDoubleComplex*)h_psi.data(), (int)n));
        CUBLAS_SAFE_CALL(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
    }
}

double NonLocalPseudo::calculate_energy(const Wavefunction& psi, const std::vector<double>& occ) {
    size_t n = grid_.nnr();
    int nb = psi.num_bands();
    if (num_projectors_ == 0)
        return 0.0;
    if (d_projections_.size() < (size_t)num_projectors_ * nb)
        d_projections_.resize(num_projectors_ * nb);

    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, grid_.stream()));
    cuDoubleComplex a = {1.0, 0.0}, b = {0.0, 0.0};
    {
        CUBLAS_SAFE_CALL(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_SAFE_CALL(cublasZgemm(h, CUBLAS_OP_C, CUBLAS_OP_N, num_projectors_, nb, (int)n, &a,
                                     (const cuDoubleComplex*)d_projectors_.data(), (int)n,
                                     (const cuDoubleComplex*)psi.data(), (int)n, &b,
                                     (cuDoubleComplex*)d_projections_.data(), num_projectors_));
        CUBLAS_SAFE_CALL(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
    }
    std::vector<gpufftComplex> hp(num_projectors_ * nb);
    d_projections_.copy_to_host(hp.data(), grid_.stream());
    std::vector<double> hc(num_projectors_ * num_projectors_);
    d_coupling_.copy_to_host(hc.data(), grid_.stream());
    grid_.synchronize();

    double energy = 0.0;
    for (int i_b = 0; i_b < nb; ++i_b) {
        double be = 0.0;
        const gpufftComplex* p = &hp[i_b * num_projectors_];
        for (int i = 0; i < num_projectors_; ++i) {
            for (int j = 0; j < num_projectors_; ++j) {
                double dij = hc[i * num_projectors_ + j];
                be += dij * (p[i].x * p[j].x + p[i].y * p[j].y);
            }
        }
        energy += occ[i_b] * be;
    }
    return energy;
}

void NonLocalPseudo::add_projector(const std::vector<std::complex<double>>& b, double c) {}
void NonLocalPseudo::clear() {}

}  // namespace dftcu
