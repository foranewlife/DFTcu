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

// Kernel for building 3D projectors Beta_lm(G)
__global__ void build_projectors_kernel(
    int nnr, const double* gx, const double* gy, const double* gz, const double* gg, int nat,
    const double* c_atom_x, const double* c_atom_y, const double* c_atom_z, const int* c_atom_type,
    const double* tab_beta,       // [total_radial_betas][stride]
    const int* l_list,            // [total_radial_betas]
    const int* radial_offsets,    // Offset to first radial beta for each type
    const int* num_radial_betas,  // Number of radial betas for each type
    int stride,                   // nqx + 1
    double dq,
    gpufftComplex* d_projectors,  // [total_nh_all_atoms][nnr]
    int nh_total                  // Total projectors across all atoms
) {
    int ig = blockDim.x * blockIdx.x + threadIdx.x;
    if (ig >= nnr)
        return;

    const double BOHR_TO_ANGSTROM = 0.529177210903;
    double g2_ang = gg[ig];
    double gmod_ang = sqrt(g2_ang);
    double gmod = gmod_ang * BOHR_TO_ANGSTROM;  // Bohr^-1
    double g2 = gmod * gmod;

    // Truncate at ecutrho (120 Ry)
    bool truncated = (g2 > 120.0000000001);

    int iproj_global = 0;
    for (int iat = 0; iat < nat; ++iat) {
        int type = c_atom_type[iat];
        int r_off = radial_offsets[type];
        int n_rad = num_radial_betas[type];

        // Structure factor: exp(-i G.R)
        // Positions in Angstrom, G in Angstrom^-1
        double phase = -(gx[ig] * c_atom_x[iat] + gy[ig] * c_atom_y[iat] + gz[ig] * c_atom_z[iat]);
        double s_feat, c_feat;
        sincos(phase, &s_feat, &c_feat);

        for (int irad = 0; irad < n_rad; ++irad) {
            int radial_idx = r_off + irad;
            int l = l_list[radial_idx];

            // Interpolate radial part
            double vq = 0.0;
            if (!truncated) {
                double px = gmod / dq - floor(gmod / dq);
                double ux = 1.0 - px;
                double vx = 2.0 - px;
                double wx = 3.0 - px;

                int i0 = (int)(gmod / dq) + 1;
                i0 = min(max(i0, 1), stride - 4);
                int i1 = i0 + 1;
                int i2 = i0 + 2;
                int i3 = i0 + 3;

                double w0 = ux * vx * wx / 6.0;
                double w1 = px * vx * wx / 2.0;
                double w2 = -px * ux * wx / 2.0;
                double w3 = px * ux * vx / 6.0;

                const double* table = &tab_beta[radial_idx * stride];
                vq = table[i0] * w0 + table[i1] * w1 + table[i2] * w2 + table[i3] * w3;
            }

            // Loop over m
            for (int m = 0; m < 2 * l + 1; ++m) {
                double ylm = get_ylm(l, m, gx[ig], gy[ig], gz[ig], gmod_ang);
                double val = vq * ylm;

                // Apply (-i)^l * exp(-i G.R)
                // phase_feat = exp(-i G.R) = (c_feat, s_feat) since phase = -G.R
                double res_re = 0.0, res_im = 0.0;
                if (l % 4 == 0) {
                    res_re = val * c_feat;
                    res_im = val * s_feat;
                } else if (l % 4 == 1) {
                    // (-i)*(c + is) = s - ic
                    res_re = val * s_feat;
                    res_im = -val * c_feat;
                } else if (l % 4 == 2) {
                    // (-1)*(c + is) = -c - is
                    res_re = -val * c_feat;
                    res_im = -val * s_feat;
                } else if (l % 4 == 3) {
                    // (i)*(c + is) = -s + ic
                    res_re = -val * s_feat;
                    res_im = val * c_feat;
                }

                d_projectors[iproj_global * nnr + ig].x = res_re;
                d_projectors[iproj_global * nnr + ig].y = res_im;
                iproj_global++;
            }
        }
    }
}

// Kernel to apply D_i coupling constants to projections
__global__ void scale_projections_kernel(int num_proj, int num_bands, const double* d_coupling,
                                         gpufftComplex* projections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_proj * num_bands) {
        int proj_idx = i % num_proj;
        projections[i].x *= d_coupling[proj_idx];
        projections[i].y *= d_coupling[proj_idx];
    }
}

__global__ void manual_projection_kernel(int num_proj, int nbands, int n,
                                         const gpufftComplex* projectors, const gpufftComplex* psi,
                                         gpufftComplex* projections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_proj * nbands) {
        int iproj = idx % num_proj;
        int iband = idx / num_proj;
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int i = 0; i < n; ++i) {
            gpufftComplex p = projectors[iproj * n + i];
            gpufftComplex w = psi[iband * n + i];
            sum_x += p.x * w.x + p.y * w.y;
            sum_y += p.x * w.y - p.y * w.x;
        }
        projections[idx].x = sum_x;
        projections[idx].y = sum_y;
    }
}

__global__ void manual_apply_nl_kernel(int num_proj, int nbands, int n, double omega,
                                       const gpufftComplex* projectors,
                                       const gpufftComplex* projections, gpufftComplex* hpsi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * nbands) {
        int ig = idx % n;
        int iband = idx / n;
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int iproj = 0; iproj < num_proj; ++iproj) {
            gpufftComplex p = projectors[iproj * n + ig];
            gpufftComplex proj_val = projections[iband * num_proj + iproj];
            sum_x += p.x * proj_val.x - p.y * proj_val.y;
            sum_y += p.x * proj_val.y + p.y * proj_val.x;
        }
        // Physical projectors Beta_lm(G) already include 1/sqrt(Omega)
        // Overlaps P = sum beta* psi. Energy E = D |P|^2.
        // The action H|psi> = V_nl|psi> = Sum D |beta><beta|psi>
        // No extra Omega factor is needed for discrete G-sum logic.
        hpsi[idx].x += sum_x;
        hpsi[idx].y += sum_y;
    }
}

}  // namespace

NonLocalPseudo::NonLocalPseudo(Grid& grid) : grid_(grid) {}

void NonLocalPseudo::init_tab_beta(int type, const std::vector<double>& r_grid,
                                   const std::vector<std::vector<double>>& beta_r,
                                   const std::vector<double>& rab, const std::vector<int>& l_list,
                                   double omega_angstrom) {
    const double BOHR_TO_ANGSTROM = 0.529177210903;
    omega_ = omega_angstrom / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

    int nbeta = beta_r.size();
    if (type >= static_cast<int>(tab_beta_.size())) {
        tab_beta_.resize(type + 1);
        l_list_.resize(type + 1);
    }
    tab_beta_[type].resize(nbeta);
    l_list_[type] = l_list;

    // Determine qmax from grid (matching QE's logic)
    double g2max_angstrom = grid_.g2max();
    double g2max_bohr = g2max_angstrom * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
    double qmax = sqrt(g2max_bohr) * 1.2;

    int required_nqx = static_cast<int>(qmax / dq_) + 4;
    if (required_nqx > nqx_) {
        nqx_ = required_nqx;
    }

    const double fpi = 4.0 * constants::D_PI;
    const double pref = fpi / sqrt(omega_);

    int msh = r_grid.size();

    // Determine kkbeta: the maximum index with non-zero beta for this atom type
    int kkbeta = 0;
    for (int nb = 0; nb < nbeta; ++nb) {
        int last_non_zero = msh;
        while (last_non_zero > 0 && std::abs(beta_r[nb][last_non_zero - 1]) < 1e-15) {
            last_non_zero--;
        }
        if (last_non_zero > kkbeta)
            kkbeta = last_non_zero;
    }
    if (kkbeta < 2)
        kkbeta = 2;

    for (int nb = 0; nb < nbeta; ++nb) {
        tab_beta_[type][nb].resize(nqx_ + 1, 0.0);
        int l = l_list[nb];

        for (int iq = 1; iq <= nqx_; ++iq) {
            double q = (iq - 1) * dq_;
            std::vector<double> aux_nb(kkbeta);
            std::vector<double> rab_nb(kkbeta);

            for (int ir = 0; ir < kkbeta; ++ir) {
                double r = r_grid[ir];
                // beta_r[nb][ir] is expected to be r * beta_l(r)
                aux_nb[ir] = beta_r[nb][ir] * spherical_bessel_jl(l, q * r) * r;
                rab_nb[ir] = rab[ir];
            }

            tab_beta_[type][nb][iq] = simpson_integrate(aux_nb, rab_nb) * pref;
        }
    }
}

void NonLocalPseudo::init_dij(int type, const std::vector<double>& dij) {
    if (type >= static_cast<int>(d_ij_.size())) {
        d_ij_.resize(type + 1);
    }

    int nbeta = l_list_[type].size();
    if (dij.size() != static_cast<size_t>(nbeta * nbeta)) {
        throw std::runtime_error("init_dij: Matrix size mismatch with nbeta");
    }

    d_ij_[type].resize(nbeta, std::vector<double>(nbeta));
    for (int i = 0; i < nbeta; ++i) {
        for (int j = 0; j < nbeta; ++j) {
            // Convert Ry (QE) to Ha (DFTcu)
            d_ij_[type][i][j] = dij[i * nbeta + j] * 0.5;
        }
    }
}

void NonLocalPseudo::update_projectors(const Atoms& atoms) {
    int nat = atoms.nat();
    if (nat == 0)
        return;

    // 1. Calculate total number of projectors nh across all atoms
    // And prepare the expanded coupling vector (assuming diagonal for now)
    int nh_total = 0;
    std::vector<double> h_coupling_expanded;

    for (int iat = 0; iat < nat; ++iat) {
        int type = atoms.h_type()[iat];
        for (size_t nb = 0; nb < l_list_[type].size(); ++nb) {
            int l = l_list_[type][nb];
            double d_val = d_ij_[type][nb][nb];  // Use diagonal term
            for (int m = 0; m < 2 * l + 1; ++m) {
                h_coupling_expanded.push_back(d_val);
                nh_total++;
            }
        }
    }
    num_projectors_ = nh_total;

    // 2. Prepare radial data for GPU
    int total_rad_betas = 0;
    std::vector<int> radial_offsets(tab_beta_.size(), 0);
    std::vector<int> num_rad_betas(tab_beta_.size(), 0);
    for (size_t t = 0; t < tab_beta_.size(); ++t) {
        radial_offsets[t] = total_rad_betas;
        num_rad_betas[t] = tab_beta_[t].size();
        total_rad_betas += tab_beta_[t].size();
    }

    int stride = nqx_ + 1;
    std::vector<double> tab_beta_flat(total_rad_betas * stride, 0.0);
    std::vector<int> l_list_flat(total_rad_betas, 0);
    for (size_t t = 0; t < tab_beta_.size(); ++t) {
        for (size_t nb = 0; nb < tab_beta_[t].size(); ++nb) {
            int idx = radial_offsets[t] + nb;
            std::copy(tab_beta_[t][nb].begin(), tab_beta_[t][nb].end(),
                      tab_beta_flat.begin() + idx * stride);
            l_list_flat[idx] = l_list_[t][nb];
        }
    }

    GPU_Vector<double> d_tab_beta_vec(tab_beta_flat.size());
    GPU_Vector<int> d_l_list(l_list_flat.size());
    GPU_Vector<int> d_radial_offsets(radial_offsets.size());
    GPU_Vector<int> d_num_rad_betas(num_rad_betas.size());

    d_tab_beta_vec.copy_from_host(tab_beta_flat.data(), grid_.stream());
    d_l_list.copy_from_host(l_list_flat.data(), grid_.stream());
    d_radial_offsets.copy_from_host(radial_offsets.data(), grid_.stream());
    d_num_rad_betas.copy_from_host(num_rad_betas.data(), grid_.stream());

    GPU_Vector<double> d_pos_x(nat), d_pos_y(nat), d_pos_z(nat);
    GPU_Vector<int> d_type(nat);
    d_pos_x.copy_from_host(atoms.h_pos_x().data(), grid_.stream());
    d_pos_y.copy_from_host(atoms.h_pos_y().data(), grid_.stream());
    d_pos_z.copy_from_host(atoms.h_pos_z().data(), grid_.stream());
    d_type.copy_from_host(atoms.h_type().data(), grid_.stream());

    d_projectors_.resize(nh_total * grid_.nnr());
    d_coupling_.resize(num_projectors_);
    d_coupling_.copy_from_host(h_coupling_expanded.data(), grid_.stream());

    const int block_size = 256;
    const int grid_size = (grid_.nnr() + block_size - 1) / block_size;

    build_projectors_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        grid_.nnr(), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), nat, d_pos_x.data(),
        d_pos_y.data(), d_pos_z.data(), d_type.data(), d_tab_beta_vec.data(), d_l_list.data(),
        d_radial_offsets.data(), d_num_rad_betas.data(), stride, dq_, d_projectors_.data(),
        nh_total);
    GPU_CHECK_KERNEL;
    grid_.synchronize();
}

void NonLocalPseudo::clear() {
    num_projectors_ = 0;
    d_projectors_.resize(0);
    d_coupling_.resize(0);
}

void NonLocalPseudo::add_projector(const std::vector<std::complex<double>>& beta_g,
                                   double coupling_constant) {
    size_t n = grid_.nnr();
    if (beta_g.size() != n) {
        throw std::runtime_error("Projector size mismatch with grid");
    }

    int old_num = num_projectors_;
    num_projectors_++;

    GPU_Vector<gpufftComplex> next_projectors(num_projectors_ * n);
    if (old_num > 0) {
        CHECK(cudaMemcpy(next_projectors.data(), d_projectors_.data(),
                         old_num * n * sizeof(gpufftComplex), cudaMemcpyDeviceToDevice));
    }
    CHECK(cudaMemcpy(next_projectors.data() + old_num * n, beta_g.data(), n * sizeof(gpufftComplex),
                     cudaMemcpyHostToDevice));
    d_projectors_ = std::move(next_projectors);

    std::vector<double> h_coupling(num_projectors_);
    if (old_num > 0) {
        d_coupling_.copy_to_host(h_coupling.data());
    }
    h_coupling[old_num] = coupling_constant;
    d_coupling_.resize(num_projectors_);
    d_coupling_.copy_from_host(h_coupling.data());
}

void NonLocalPseudo::apply(Wavefunction& psi_in, Wavefunction& h_psi_out) {
    if (num_projectors_ == 0)
        return;

    size_t n = grid_.nnr();
    int nbands = psi_in.num_bands();

    d_projections_.resize(num_projectors_ * nbands);

    const int block_size_p = 64;
    const int grid_size_p = (num_projectors_ * nbands + block_size_p - 1) / block_size_p;

    manual_projection_kernel<<<grid_size_p, block_size_p, 0, grid_.stream()>>>(
        num_projectors_, nbands, (int)n, d_projectors_.data(), psi_in.data(),
        d_projections_.data());

    const int block_size_s = 256;
    const int grid_size_s = (num_projectors_ * nbands + block_size_s - 1) / block_size_s;
    scale_projections_kernel<<<grid_size_s, block_size_s, 0, grid_.stream()>>>(
        num_projectors_, nbands, d_coupling_.data(), d_projections_.data());

    const int block_size_a = 256;
    const int grid_size_a = (int)(n * nbands + block_size_a - 1) / block_size_a;

    manual_apply_nl_kernel<<<grid_size_a, block_size_a, 0, grid_.stream()>>>(
        num_projectors_, nbands, (int)n, 1.0, d_projectors_.data(), d_projections_.data(),
        h_psi_out.data());

    grid_.synchronize();
}

double NonLocalPseudo::calculate_energy(const Wavefunction& psi,
                                        const std::vector<double>& occupations) {
    if (num_projectors_ == 0)
        return 0.0;

    size_t n = grid_.nnr();
    int nbands = psi.num_bands();

    if (d_projections_.size() < (size_t)num_projectors_ * nbands) {
        d_projections_.resize(num_projectors_ * nbands);
    }

    const int block_size = 64;
    const int grid_size = (num_projectors_ * nbands + block_size - 1) / block_size;

    manual_projection_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        num_projectors_, nbands, (int)n, d_projectors_.data(), psi.data(), d_projections_.data());

    std::vector<gpufftComplex> h_projections(num_projectors_ * nbands);
    d_projections_.copy_to_host(h_projections.data(), grid_.stream());

    std::vector<double> h_coupling(num_projectors_);
    d_coupling_.copy_to_host(h_coupling.data(), grid_.stream());

    grid_.synchronize();

    double energy = 0.0;

    for (int n_idx = 0; n_idx < nbands; ++n_idx) {
        double band_energy = 0.0;
        for (int i = 0; i < num_projectors_; ++i) {
            gpufftComplex p = h_projections[n_idx * num_projectors_ + i];
            double p2 = p.x * p.x + p.y * p.y;
            band_energy += h_coupling[i] * p2;
        }
        energy += occupations[n_idx] * band_energy;
    }

    return energy;
}

}  // namespace dftcu
