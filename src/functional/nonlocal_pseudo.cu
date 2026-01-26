#include "math/bessel.cuh"
#include "math/ylm.cuh"
#include "nonlocal_pseudo.cuh"
#include "pseudopotential_data.cuh"
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

// Scatter packed array (npw) to FFT grid (nnr) using nl_d mapping and accumulate
// Input: data_packed[npw] - compact data on Smooth grid
// Output: data_fft[nnr] - FFT grid (accumulate: += )
__global__ void scatter_and_accumulate_kernel(int npw, const int* nl_d,
                                              const gpufftComplex* data_packed,
                                              gpufftComplex* data_fft) {
    int ig = blockDim.x * blockIdx.x + threadIdx.x;
    if (ig < npw) {
        int ifft = nl_d[ig];  // Map Smooth grid index to FFT grid position
        // Accumulate (not replace!)
        data_fft[ifft].x += data_packed[ig].x;
        data_fft[ifft].y += data_packed[ig].y;
    }
}

// Extract packed array from FFT grid using nl_d mapping
// Input: data_fft[nnr] - sparse data on FFT grid (only nl_d[0:npw-1] are non-zero)
// Output: data_packed[npw] - packed contiguous array
__global__ void extract_smooth_to_packed_kernel(int npw, const int* nl_d,
                                                const gpufftComplex* data_fft,
                                                gpufftComplex* data_packed) {
    int ig = blockDim.x * blockIdx.x + threadIdx.x;
    if (ig < npw) {
        int ifft = nl_d[ig];  // Map Smooth grid index to FFT grid position
        data_packed[ig] = data_fft[ifft];
    }
}

// NEW: Smooth grid version - only compute on npw G-vectors, use nl_d mapping
__global__ void interpolate_beta_smooth_kernel(int npw, const int* nl_d, const double* gx,
                                               const double* gy, const double* gz, const double* gg,
                                               int l, int m, int iat, const double* tab, int stride,
                                               double dq, double omega_bohr,
                                               gpufftComplex* beta_g) {
    int ig = blockDim.x * blockIdx.x + threadIdx.x;
    if (ig < npw) {
        const double TWO_PI = constants::D_PI * 2.0;
        int ifft = nl_d[ig];  // Map Smooth grid index to FFT grid position

        double gmod_phys = sqrt(gg[ifft]) * TWO_PI;  // Physical G in Bohr^-1
        double betag = 0.0;

        if (gmod_phys < 1e-12) {
            if (l == 0)
                betag = tab[0];
            else
                betag = 0.0;
        } else {
            int i0 = (int)(gmod_phys / dq) + 1;
            if (i0 < stride - 4) {
                double px = gmod_phys / dq - (double)(i0 - 1);
                double ux = 1.0 - px;
                double vx = 2.0 - px;
                double wx = 3.0 - px;

                betag = tab[i0 - 1] * ux * vx * wx / 6.0 + tab[i0] * px * vx * wx / 2.0 -
                        tab[i0 + 1] * px * ux * wx / 2.0 + tab[i0 + 2] * px * ux * vx / 6.0;
            }
        }

        double ylm_val = get_ylm(l, m, gx[ifft], gy[ifft], gz[ifft], sqrt(gg[ifft]));
        double val = betag * ylm_val;

        // Apply phase factor exp(-iG_phys.R)
        double phase = -TWO_PI * (gx[ifft] * c_nl_atom_x[iat] + gy[ifft] * c_nl_atom_y[iat] +
                                  gz[ifft] * c_nl_atom_z[iat]);
        double s, c;
        sincos(phase, &s, &c);

        double re_part = val * c;
        double im_part = val * s;

        // Apply (-i)^l factor (QE convention)
        if (l == 0) {
            beta_g[ifft].x = re_part;
            beta_g[ifft].y = im_part;
        } else if (l == 1) {
            beta_g[ifft].x = im_part;
            beta_g[ifft].y = -re_part;
        } else if (l == 2) {
            beta_g[ifft].x = -re_part;
            beta_g[ifft].y = -im_part;
        } else if (l == 3) {
            beta_g[ifft].x = -im_part;
            beta_g[ifft].y = re_part;
        }
    }
}

__global__ void interpolate_beta_kernel(int n, const double* gx, const double* gy, const double* gz,
                                        const double* gg, int l, int m, int iat, const double* tab,
                                        int stride, double dq, double omega_bohr,
                                        gpufftComplex* beta_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        const double TWO_PI = constants::D_PI * 2.0;
        double gmod_phys = sqrt(gg[i]) * TWO_PI;  // Physical G in Bohr^-1
        double betag = 0.0;

        if (gmod_phys < 1e-12) {
            if (l == 0)
                betag = tab[0];
            else
                betag = 0.0;
        } else {
            int i0 = (int)(gmod_phys / dq) + 1;
            if (i0 < stride - 4) {
                double px = gmod_phys / dq - (double)(i0 - 1);
                double ux = 1.0 - px;
                double vx = 2.0 - px;
                double wx = 3.0 - px;

                betag = tab[i0 - 1] * ux * vx * wx / 6.0 + tab[i0] * px * vx * wx / 2.0 -
                        tab[i0 + 1] * px * ux * wx / 2.0 + tab[i0 + 2] * px * ux * vx / 6.0;
            }
        }

        double ylm_val = get_ylm(l, m, gx[i], gy[i], gz[i], sqrt(gg[i]));
        double val = betag * ylm_val;

        // Apply phase factor exp(-iG_phys.R)
        double phase = -TWO_PI * (gx[i] * c_nl_atom_x[iat] + gy[i] * c_nl_atom_y[iat] +
                                  gz[i] * c_nl_atom_z[iat]);
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
                                   const std::vector<int>& kkbeta_list, double omega_bohr) {
    if (type >= static_cast<int>(tab_beta_.size())) {
        tab_beta_.resize(type + 1);
        l_list_.resize(type + 1);
    }
    int n_betas = betas.size();
    tab_beta_[type].resize(n_betas);
    l_list_[type] = l_list;

    const double fpi = 4.0 * constants::D_PI;

    dq_ = 0.01;  // Match QE dq
    const double TWO_PI = constants::D_PI * 2.0;
    double g2max_phys = grid_.g2max() * (TWO_PI * TWO_PI);
    nqx_ = (int)(sqrt(g2max_phys) / dq_) + 10;

    for (int nb = 0; nb < n_betas; ++nb) {
        tab_beta_[type][nb].resize(nqx_ + 1);
        int l = l_list[nb];
        int kkbeta = kkbeta_list[nb];
        std::vector<double> aux(kkbeta);
        std::vector<double> rab_sub(kkbeta);
        for (int ir = 0; ir < kkbeta; ++ir)
            rab_sub[ir] = rab[ir];

        for (int iq = 0; iq < nqx_ + 1; ++iq) {
            double q = iq * dq_;
            for (int ir = 0; ir < kkbeta; ++ir) {
                double x = q * r[ir];
                double jl = spherical_bessel_jl(l, x);
                // UPF beta is already r*beta or beta?
                // QE logic: aux = beta(r) * jl(qr) * r
                aux[ir] = betas[nb][ir] * r[ir] * jl;
            }
            tab_beta_[type][nb][iq] = simpson_integrate(aux, rab_sub) * fpi / sqrt(omega_bohr);
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

    // CRITICAL FIX: Only compute projectors on Smooth grid (npw), not full FFT grid (nnr)
    // This matches QE's init_us_2, which only computes vkb(npw, nkb)
    size_t npw = grid_.ngw();  // Smooth grid size
    size_t nnr = grid_.nnr();  // FFT grid size (for storage stride)

    // Allocate storage with FFT grid stride (for nl_d mapping)
    d_projectors_.resize(nnr * num_projectors_);
    d_coupling_.resize(num_projectors_ * num_projectors_);

    // Initialize all projectors to zero (important for FFT grid points outside Smooth grid)
    d_projectors_.fill({0.0, 0.0});
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
                // FIX: Use new Smooth grid kernel with nl_d mapping
                interpolate_beta_smooth_kernel<<<(npw + 255) / 256, 256, 0, grid_.stream()>>>(
                    (int)npw, grid_.nl_d(), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), l,
                    m_idx, (int)iat, d_tab.data(), nqx_ + 1, dq_, omega_,
                    d_projectors_.data() + p_idx * nnr);
            }
            grid_.synchronize();
        }

        // DEBUG: Print D_ij matrix for first atom type
        if (type == 0) {
            printf("[DEBUG D_ij] D_ij matrix (type=%d, size=%dx%d, in Hartree after ×0.5):\n", type,
                   n_radial, n_radial);
            for (int i = 0; i < std::min(4, n_radial); ++i) {
                for (int j = 0; j < std::min(4, n_radial); ++j) {
                    printf("  D[%d,%d] = %.10e\n", i, j, d_ij_[type][i][j] * 0.5);
                }
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
                            d_ij_[type][nb][mb] * 0.5;  // RESTORE: Ry→Ha conversion
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

    int npw = grid_.ngw();     // Smooth grid size (85 for Si Gamma)
    size_t nnr = grid_.nnr();  // FFT grid size (3375 for Si Gamma)
    int nb = psi.num_bands();

    if (d_projections_.size() < (size_t)num_projectors_ * nb)
        d_projections_.resize(num_projectors_ * nb);

    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, grid_.stream()));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);

    // ========================================================================
    // NEW IMPLEMENTATION: Use DGEMM on packed arrays (matching QE calbec_gamma)
    // ========================================================================

    // Step 1: Pack beta projectors from FFT grid (nnr) to compact array (npw)
    // Beta projectors are stored on FFT grid, but only nl_d[0:npw-1] are non-zero
    GPU_Vector<gpufftComplex> beta_packed(npw * num_projectors_);

    for (int iproj = 0; iproj < num_projectors_; ++iproj) {
        extract_smooth_to_packed_kernel<<<(npw + 255) / 256, 256, 0, grid_.stream()>>>(
            npw, grid_.nl_d(),
            d_projectors_.data() + iproj * nnr,  // Input: beta on FFT grid
            beta_packed.data() + iproj * npw     // Output: beta packed
        );
    }
    grid_.synchronize();

    // DEBUG: Export beta_packed for comparison with QE vkb
    static bool beta_exported = false;
    if (!beta_exported) {
        std::vector<gpufftComplex> h_beta(npw * num_projectors_);
        beta_packed.copy_to_host(h_beta.data());

        FILE* fp = fopen("dftcu_beta_packed.txt", "w");
        if (fp) {
            fprintf(fp, "# DFTcu: beta_packed(npw, num_projectors)\n");
            fprintf(fp, "# Format: ig iproj Re Im\n");
            for (int ip = 0; ip < num_projectors_; ++ip) {
                for (int ig = 0; ig < npw; ++ig) {
                    fprintf(fp, "%5d %5d %25.16e %25.16e\n", ig + 1, ip + 1,
                            h_beta[ip * npw + ig].x, h_beta[ip * npw + ig].y);
                }
            }
            fclose(fp);
            printf("[DEBUG] Exported beta_packed to dftcu_beta_packed.txt\n");
        }
        beta_exported = true;
    }

    // Step 2: Pack wavefunctions from FFT grid (nnr) to compact array (npw)
    GPU_Vector<gpufftComplex> psi_packed(npw * nb);

    for (int ib = 0; ib < nb; ++ib) {
        extract_smooth_to_packed_kernel<<<(npw + 255) / 256, 256, 0, grid_.stream()>>>(
            npw, grid_.nl_d(),
            psi.data() + ib * nnr,        // Input: psi on FFT grid
            psi_packed.data() + ib * npw  // Output: psi packed
        );
    }
    grid_.synchronize();

    // Step 3: Compute becp = <beta|psi> using DGEMM (Gamma-only formula)
    // QE formula: becp(i,j) = 2*Re(Σ_k beta^*(k,i) psi(k,j)) - beta^*(0,i)psi(0,j)
    // Implementation: treat complex arrays as real with size 2*npw

    double* beta_real = reinterpret_cast<double*>(beta_packed.data());
    double* psi_real = reinterpret_cast<double*>(psi_packed.data());

    // Allocate real becp array (num_projectors × nb)
    GPU_Vector<double> becp_real(num_projectors_ * nb);

    // DGEMM: becp = 2.0 * beta^T * psi
    // beta_real: (2*npw, num_projectors) in column-major
    // psi_real:  (2*npw, nb) in column-major
    // becp_real: (num_projectors, nb) in column-major
    double alpha_gamma = 2.0;  // Gamma-only factor
    double beta_zero_d = 0.0;

    CUBLAS_SAFE_CALL(
        cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, num_projectors_, nb, 2 * npw,  // m, n, k
                    &alpha_gamma, beta_real, 2 * npw,  // A: (2*npw, num_projectors), lda=2*npw
                    psi_real, 2 * npw,                 // B: (2*npw, nb), ldb=2*npw
                    &beta_zero_d, becp_real.data(),
                    num_projectors_));  // C: (num_projectors, nb), ldc=num_projectors

    // DEBUG: Print becp after DGEMM, before G=0 correction (first band only)
    int debug_count = std::min(8, num_projectors_);
    if (debug_count > 0) {
        std::vector<double> h_becp_all(num_projectors_ * nb);
        becp_real.copy_to_host(h_becp_all.data(), grid_.stream());
        grid_.synchronize();
        printf("[DEBUG becp] After DGEMM (before G=0 correction, first %d projectors, band 0):\n",
               debug_count);
        for (int i = 0; i < debug_count; ++i) {
            printf("  becp[%d,0] = %.10e\n", i, h_becp_all[i]);
        }
    }

    // Step 4: Subtract G=0 overcounting (gstart=2 means G=0 exists)
    // becp -= beta(G=0) * psi(G=0)
    int gstart = grid_.gstart();  // 2 if G=0 exists (Gamma-only), 1 otherwise

    if (gstart == 2) {
        // DGER: becp -= beta_real[0,:] * psi_real[0,:]^T
        // beta_real[0,:] is the first row (G=0, real part) of beta_packed
        // psi_real[0,:] is the first row (G=0, real part) of psi_packed
        double alpha_correction = -1.0;

        CUBLAS_SAFE_CALL(
            cublasDger(h, num_projectors_, nb, &alpha_correction, beta_real,
                       2 * npw,            // x: beta_real[0::2*npw] (stride 2*npw, starting at 0)
                       psi_real, 2 * npw,  // y: psi_real[0::2*npw] (stride 2*npw, starting at 0)
                       becp_real.data(), num_projectors_));

        // DEBUG: Print becp after G=0 correction
        if (debug_count > 0) {
            std::vector<double> h_becp_all_after(num_projectors_ * nb);
            becp_real.copy_to_host(h_becp_all_after.data(), grid_.stream());
            grid_.synchronize();
            printf("[DEBUG becp] After G=0 correction (first %d projectors, band 0):\n",
                   debug_count);
            for (int i = 0; i < debug_count; ++i) {
                printf("  becp[%d,0] = %.10e\n", i, h_becp_all_after[i]);
            }
        }
    }

    // Step 5: Convert becp from real to complex (imaginary part = 0 for Gamma-only)
    // Allocate separate GPU storage for becp (don't overwrite d_projectors_ which holds beta!)
    GPU_Vector<cuDoubleComplex> becp_complex(num_projectors_ * nb);

    // Convert real becp to complex on GPU (more efficient than host conversion)
    std::vector<double> h_becp_real(num_projectors_ * nb);
    becp_real.copy_to_host(h_becp_real.data(), grid_.stream());
    grid_.synchronize();

    std::vector<cuDoubleComplex> h_becp_complex(num_projectors_ * nb);
    for (size_t i = 0; i < h_becp_real.size(); ++i) {
        h_becp_complex[i] = {h_becp_real[i], 0.0};
    }
    becp_complex.copy_from_host(h_becp_complex.data(), grid_.stream());

    // ========================================================================
    // D-matrix coupling and final application
    // ========================================================================

    if (d_dps_.size() < (size_t)num_projectors_ * nb)
        d_dps_.resize(num_projectors_ * nb);

    // Step 6: D-matrix coupling: dps = D * becp
    // CRITICAL: QE uses DGEMM for Gamma-only! Not ZGEMM!
    // deeq (real) × becp%r (real) → ps (real)

    GPU_Vector<double> dps_real(num_projectors_ * nb);

    // D_coupling is already real (no imaginary part)
    std::vector<double> h_coupling_real(num_projectors_ * num_projectors_);
    d_coupling_.copy_to_host(h_coupling_real.data(), grid_.stream());
    GPU_Vector<double> d_coupling_real(num_projectors_ * num_projectors_);
    d_coupling_real.copy_from_host(h_coupling_real.data(), grid_.stream());

    // Use DGEMM: dps_real = D_real × becp_real
    double alpha_dmat = 1.0;
    double beta_dmat = 0.0;

    CUBLAS_SAFE_CALL(
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, num_projectors_, nb, num_projectors_, &alpha_dmat,
                    d_coupling_real.data(), num_projectors_,  // D: (num_projectors, num_projectors)
                    becp_real.data(), num_projectors_,        // becp: (num_projectors, nb)
                    &beta_dmat, dps_real.data(), num_projectors_));  // dps: (num_projectors, nb)

    // Step 7: Final contribution to H*psi: hpsi += beta * dps
    // CRITICAL: QE uses DGEMM treating complex as real (2*npw)!
    // QE's add_vuspsi_gamma formula: hpsi_real += vkb_real * ps_real

    // Allocate compact V_NL|ψ> result (npw × nb)
    GPU_Vector<gpufftComplex> vnl_packed(npw * nb);

    // dps_real is already computed in Step 6 via DGEMM (no conversion needed)

    // Use DGEMM: vnl_real = beta_real * dps_real (treating complex as 2*npw real)
    double* beta_real_final = reinterpret_cast<double*>(beta_packed.data());
    double* vnl_real = reinterpret_cast<double*>(vnl_packed.data());

    double alpha_vnl = 1.0;
    double beta_vnl = 0.0;

    // DEBUG: Check beta_packed values (all 8 projectors, G=0 only)
    std::vector<cuDoubleComplex> h_beta_g0(num_projectors_);
    for (int iproj = 0; iproj < num_projectors_; ++iproj) {
        CHECK(cudaMemcpy(&h_beta_g0[iproj], beta_packed.data() + iproj * npw,
                         sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }
    printf("[DEBUG V_NL] beta_packed at G=0 (all 8 projectors):\n");
    for (int i = 0; i < num_projectors_; ++i) {
        printf("  beta[0,proj=%d] = (%.10e, %.10e)\n", i, h_beta_g0[i].x, h_beta_g0[i].y);
    }

    // DEBUG: Check dps_real values (all 8 projectors, band 0)
    std::vector<double> h_dps_all(num_projectors_ * nb);
    dps_real.copy_to_host(h_dps_all.data(), grid_.stream());
    grid_.synchronize();
    printf("[DEBUG V_NL] dps_real (all 8 projectors, band 0):\n");
    for (int i = 0; i < num_projectors_; ++i) {
        // dps_real is (num_projectors, nb) in column-major
        printf("  dps[proj=%d,band=0] = %.10e\n", i, h_dps_all[i]);
    }

    // Manual check: compute vnl[0,0] = Σ_i beta[0,i] * dps[i,0]
    double vnl_g0_manual = 0.0;
    for (int i = 0; i < num_projectors_; ++i) {
        vnl_g0_manual += h_beta_g0[i].x * h_dps_all[i];  // Real part only for G=0
    }
    printf("[DEBUG V_NL] Manual vnl[G=0,band=0] = %.10e\n", vnl_g0_manual);

    printf("[DEBUG V_NL] Final DGEMM:\n");
    printf("  beta_real: (%d, %d), lda=%d\n", 2 * npw, num_projectors_, 2 * npw);
    printf("  dps_real: (%d, %d), ldb=%d\n", num_projectors_, nb, num_projectors_);
    printf("  vnl_real: (%d, %d), ldc=%d\n", 2 * npw, nb, 2 * npw);

    CUBLAS_SAFE_CALL(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2 * npw, nb,
                                 num_projectors_,  // m, n, k (treating complex as real)
                                 &alpha_vnl, beta_real_final,
                                 2 * npw,                           // A: (2*npw, num_projectors)
                                 dps_real.data(), num_projectors_,  // B: (num_projectors, nb)
                                 &beta_vnl, vnl_real, 2 * npw));    // C: (2*npw, nb)

    // DEBUG: Check vnl_packed values
    std::vector<cuDoubleComplex> h_vnl_check(std::min(10, npw));
    CHECK(cudaMemcpy(h_vnl_check.data(), vnl_packed.data(),
                     h_vnl_check.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    printf("[DEBUG V_NL] vnl_packed after DGEMM (band 0, first 10):\n");
    for (size_t i = 0; i < h_vnl_check.size(); ++i) {
        printf("  [%zu] = (%.6e, %.6e)\n", i, h_vnl_check[i].x, h_vnl_check[i].y);
    }

    // Scatter V_NL|ψ> from compact (npw) to FFT grid (nnr) and accumulate to h_psi
    for (int ib = 0; ib < nb; ++ib) {
        scatter_and_accumulate_kernel<<<(npw + 255) / 256, 256, 0, grid_.stream()>>>(
            npw, grid_.nl_d(),
            vnl_packed.data() + ib * npw,  // Input: compact V_NL
            h_psi.data() + ib * nnr        // Output: FFT grid h_psi (accumulate)
        );
    }
    grid_.synchronize();
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
    cuDoubleComplex alpha_nl = {1.0, 0.0}, beta_zero = {0.0, 0.0};
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
    if (type < (int)tab_beta_.size() && nb < (int)tab_beta_[type].size()) {
        tab_beta_[type][nb] = tab;
        nqx_ = (int)tab.size() - 1;
    }
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

std::vector<double> NonLocalPseudo::get_coupling() const {
    std::vector<double> host(d_coupling_.size());
    d_coupling_.copy_to_host(host.data());
    return host;
}

std::vector<std::complex<double>> NonLocalPseudo::get_d_projections() const {
    std::vector<std::complex<double>> host(d_dps_.size());
    d_dps_.copy_to_host(reinterpret_cast<gpufftComplex*>(host.data()));
    return host;
}

void NonLocalPseudo::debug_projections(const Wavefunction& psi, const std::vector<double>& qe_becp,
                                       const std::vector<std::complex<double>>& qe_vkb,
                                       const std::vector<std::complex<double>>& qe_evc,
                                       const std::vector<std::vector<int>>& miller) {}

void NonLocalPseudo::add_projector(const std::vector<std::complex<double>>& beta_g,
                                   double coupling_constant) {}

void NonLocalPseudo::set_projectors(const std::vector<std::complex<double>>& projectors) {
    if (projectors.size() != d_projectors_.size()) {
        throw std::runtime_error("set_projectors size mismatch");
    }
    d_projectors_.copy_from_host(reinterpret_cast<const gpufftComplex*>(projectors.data()),
                                 grid_.stream());
    grid_.synchronize();
}

void NonLocalPseudo::clear() {
    tab_beta_.clear();
    l_list_.clear();
    d_ij_.clear();
    num_projectors_ = 0;
}
}  // namespace dftcu
