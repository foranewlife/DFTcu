#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "fft/fft_solver.cuh"
#include "fft/gamma_fft_solver.cuh"
#include "solver/hamiltonian.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

// ============================================================================
// Debug Export Functions
// ============================================================================

/**
 * @brief Export complex array to text file (for debugging)
 *
 * @param filename Output file name
 * @param data Device pointer to complex array
 * @param n Array size
 * @param description Description of the data
 */
void export_debug_complex(const char* filename, const gpufftComplex* data, size_t n,
                          const char* description) {
    std::vector<gpufftComplex> host_data(n);
    CHECK(cudaMemcpy(host_data.data(), data, n * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));

    std::ofstream f(filename);
    f << std::scientific << std::setprecision(16);
    f << "# DFTcu Debug: " << description << "\n";
    f << "# Format: index, Re, Im\n";
    f << "# n = " << n << "\n";

    for (size_t i = 0; i < n; ++i) {
        f << std::setw(8) << i << " " << std::setw(24) << host_data[i].x << " " << std::setw(24)
          << host_data[i].y << "\n";
    }
    f.close();
    std::cout << "  [DFTcu Debug] Exported " << description << " to " << filename << std::endl;
}

// Scale kernel for FFT normalization
__global__ void scale_complex_kernel(size_t n, gpufftComplex* data, double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

/**
 * @brief Apply kinetic energy operator on Smooth grid.
 *
 * T|ψ> = ½|G|² |ψ(G)>
 *
 * Note: Wavefunction stores data with stride=nnr (FFT grid), not npw!
 * Each band occupies nnr elements, only first npw are valid plane waves.
 *
 * @param npw Number of plane waves (Smooth grid G-vectors)
 * @param num_bands Number of bands
 * @param lda Leading dimension (stride) = nnr
 * @param g2kin Kinetic energy coefficients (npw, Hartree)
 * @param psi Input wavefunction (nbands × lda, stride=lda)
 * @param h_psi Output H|ψ> (nbands × lda, stride=lda)
 */
__global__ void apply_kinetic_kernel(int npw, int num_bands, int lda, const int* nl_d,
                                     const double* g2kin, const gpufftComplex* psi,
                                     gpufftComplex* h_psi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = npw * num_bands;
    if (i < total_size) {
        int band = i / npw;           // Which band
        int ig = i % npw;             // Which G-vector (Smooth grid index)
        int ifft = nl_d[ig];          // Map to FFT grid index
        int idx = band * lda + ifft;  // Actual index with stride

        // g2kin is already in Hartree (½|G|²_physical)
        double t = g2kin[ig];
        h_psi[idx].x = t * psi[idx].x;
        h_psi[idx].y = t * psi[idx].y;
    }
}

__global__ void apply_vloc_kernel(size_t n, const double* v_loc, gpufftComplex* psi_r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        psi_r[i].x *= v_loc[i];
        psi_r[i].y *= v_loc[i];
    }
}

/**
 * @brief Scatter Smooth grid to FFT grid (zero-padding).
 *
 * @param npw Number of Smooth grid G-vectors
 * @param nl_d Smooth G → FFT grid mapping
 * @param psi_smooth Input on Smooth grid (npw)
 * @param psi_fft Output on FFT grid (nnr, zero-initialized)
 */
__global__ void scatter_smooth_to_fft_kernel(int npw, const int* nl_d,
                                             const gpufftComplex* psi_smooth,
                                             gpufftComplex* psi_fft) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        int ifft = nl_d[ig];
        psi_fft[ifft] = psi_smooth[ig];
    }
}

/**
 * @brief Scatter Smooth grid to FFT grid with Hermitian symmetry (Gamma-only).
 *
 * For Gamma-only calculations, wavefunctions satisfy ψ(-G) = ψ*(G).
 * This kernel fills both +G and -G (conjugate) to ensure correct IFFT results.
 *
 * @param npw Number of Smooth grid G-vectors
 * @param nl_d Smooth G → FFT grid mapping (+G)
 * @param nlm_d Smooth -G → FFT grid mapping (-G conjugate)
 * @param psi_smooth Input on Smooth grid (npw)
 * @param psi_fft Output on FFT grid (nnr, zero-initialized)
 */
__global__ void scatter_smooth_to_fft_gamma_kernel(int npw, const int* nl_d, const int* nlm_d,
                                                   const gpufftComplex* psi_smooth,
                                                   gpufftComplex* psi_fft) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        int ifft_pos = nl_d[ig];   // +G index in FFT grid
        int ifft_neg = nlm_d[ig];  // -G index in FFT grid

        // ✅ CRITICAL FIX: psi_smooth 是顺序存储的 Smooth grid，索引应该用 ig
        // psi_smooth[npw] - Smooth grid，顺序存储 (ig=0,1,2,...,npw-1)
        // nl_d[ig] 给出 FFT grid 的位置，不是 psi_smooth 的索引
        gpufftComplex val = psi_smooth[ig];

        // Fill +G in output grid
        psi_fft[ifft_pos] = val;

        // Fill -G (conjugate): ψ(-G) = ψ*(G)
        // Special case: G=0 should not be duplicated (ifft_pos == ifft_neg)
        if (ifft_pos != ifft_neg) {
            psi_fft[ifft_neg].x = val.x;   // Real part same
            psi_fft[ifft_neg].y = -val.y;  // Imaginary part negated
        }
    }
}

/**
 * @brief Remove √2 normalization from Gamma-only wavefunction (Smooth grid).
 *
 * QE stores G≠0 coefficients with √2 factor for Gamma-only calculations.
 * After gathering from FFT grid, we need to divide by √2 to get physical coefficients.
 *
 * @param npw Number of Smooth grid G-vectors
 * @param psi_smooth Smooth grid wavefunction (npw, in-place modification)
 */
__global__ void remove_sqrt2_normalization_kernel(int npw, gpufftComplex* psi_smooth) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        if (ig == 0) {
            // G=0: no √2 factor, keep as-is
            return;
        }
        // G≠0: divide by √2
        const double INV_SQRT2 = 0.7071067811865475;  // 1/√2
        psi_smooth[ig].x *= INV_SQRT2;
        psi_smooth[ig].y *= INV_SQRT2;
    }
}

/**
 * @brief Add √2 normalization back to Gamma-only wavefunction (Smooth grid).
 *
 * QE expects G≠0 coefficients to have √2 factor for Gamma-only storage.
 * Before accumulating back to h_psi, we need to multiply by √2.
 */
__global__ void add_sqrt2_normalization_kernel(int npw, gpufftComplex* psi_smooth) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        if (ig == 0) {
            return;
        }
        const double SQRT2 = 1.4142135623730951;
        psi_smooth[ig].x *= SQRT2;
        psi_smooth[ig].y *= SQRT2;
    }
}

/**
 * @brief Add back √2 normalization AND factor 2.0 for Gamma-only coefficients.
 */
__global__ void add_sqrt2_and_factor2_kernel(int npw, gpufftComplex* psi_smooth) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        if (ig == 0)
            return;
        const double SQRT2 = 1.4142135623730951;
        psi_smooth[ig].x *= 2.0 * SQRT2;
        psi_smooth[ig].y *= 2.0 * SQRT2;
    }
}

/**
 * @brief Gather FFT grid to Smooth grid (truncation).
 * @param psi_smooth Output on Smooth grid (npw)
 */
__global__ void gather_fft_to_smooth_kernel(int npw, const int* nl_d, const gpufftComplex* psi_fft,
                                            gpufftComplex* psi_smooth) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        int ifft = nl_d[ig];
        psi_smooth[ig] = psi_fft[ifft];
    }
}

__global__ void accumulate_hpsi_kernel(int npw, const int* nl_d, const gpufftComplex* tmp,
                                       gpufftComplex* h_psi) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        int ifft = nl_d[ig];  // Map Smooth grid index to FFT grid index
        h_psi[ifft].x += tmp[ig].x;
        h_psi[ifft].y += tmp[ig].y;
    }
}

__global__ void scale_vloc_kernel(size_t n, gpufftComplex* data, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

}  // namespace

// Base constructor - dfp is optional
Hamiltonian::Hamiltonian(Grid& grid)
    : grid_(grid),
      dfp_(nullptr),
      nonlocal_(nullptr),
      v_loc_tot_(grid),
      v_ps_(grid),
      v_h_(grid),
      v_xc_(grid) {
    v_loc_tot_.fill(0.0);
    v_ps_.fill(0.0);
    v_h_.fill(0.0);
    v_xc_.fill(0.0);
}

// Backward compatibility constructor
Hamiltonian::Hamiltonian(Grid& grid, std::shared_ptr<DensityFunctionalPotential> dfp,
                         std::shared_ptr<NonLocalPseudoOperator> nl_pseudo)
    : grid_(grid),
      dfp_(dfp),
      nonlocal_(nl_pseudo),
      v_loc_tot_(grid),
      v_ps_(grid),
      v_h_(grid),
      v_xc_(grid) {
    v_loc_tot_.fill(0.0);
    v_ps_.fill(0.0);
    v_h_.fill(0.0);
    v_xc_.fill(0.0);
}

void Hamiltonian::copy_from(const Hamiltonian& other) {
    if (this != &other) {
        if (&grid_ != &other.grid_) {
            throw std::invalid_argument("Hamiltonian must share the same grid for copy_from");
        }
        dfp_ = other.dfp_;
        nonlocal_ = other.nonlocal_;
        v_loc_tot_.copy_from(other.v_loc_tot_);
        v_ps_.copy_from(other.v_ps_);
        v_h_.copy_from(other.v_h_);
        v_xc_.copy_from(other.v_xc_);
        v_of_0_ = other.v_of_0_;
    }
}

void Hamiltonian::update_potentials_inplace(const RealField& rho) {
    if (!dfp_) {
        throw std::runtime_error(
            "Hamiltonian::update_potentials_inplace: DensityFunctionalPotential not set");
    }

    // Clear all potential components
    v_loc_tot_.fill(0.0);
    v_ps_.fill(0.0);
    v_h_.fill(0.0);
    v_xc_.fill(0.0);

    // Get functional components from DFP
    const auto& components = dfp_->get_components();

    // Compute each component separately
    // Assumption based on run_nscf.py order:
    //   components[0] = Hartree
    //   components[1] = LDA_PZ (XC)
    //   components[2] = LocalPseudoOperator (V_ps)
    for (size_t i = 0; i < components.size(); ++i) {
        if (i == 0) {
            // Hartree potential
            components[i].compute(rho, v_h_);
        } else if (i == 1) {
            // XC potential
            components[i].compute(rho, v_xc_);
        } else if (i == 2) {
            // Pseudopotential local component
            components[i].compute(rho, v_ps_);
        } else {
            // Fallback: add to v_loc_tot_ directly for unknown components
            RealField v_tmp(grid_);
            components[i].compute(rho, v_tmp);
            v_loc_tot_ = v_loc_tot_ + v_tmp;
        }
    }

    // Sum all components to get total local potential
    v_loc_tot_ = v_ps_ + v_h_ + v_xc_;

    double vol_bohr = grid_.volume_bohr();
    double v_ps_mean = v_ps_.integral() / vol_bohr;
    double v_h_mean = v_h_.integral() / vol_bohr;
    double v_xc_mean = v_xc_.integral() / vol_bohr;

    // Calculate v_of_0 (QE's vltot(G=0) equivalent in Hartree)
    double total_v_ang = v_loc_tot_.integral();
    v_of_0_ = total_v_ang / grid_.volume();

    grid_.synchronize();
}

void Hamiltonian::set_ecutrho(double ecutrho) {}

void Hamiltonian::apply_kinetic(Wavefunction& psi, Wavefunction& h_psi) {
    int npw = psi.num_pw();
    int nbands = psi.num_bands();
    size_t nnr = grid_.nnr();
    int lda = nnr;

    // Zero output
    cudaMemsetAsync(h_psi.data(), 0, nnr * nbands * sizeof(gpufftComplex), grid_.stream());

    // Apply kinetic energy
    const int block_size = 256;
    const int grid_size_k = (npw * nbands + block_size - 1) / block_size;

    apply_kinetic_kernel<<<grid_size_k, block_size, 0, grid_.stream()>>>(
        npw, nbands, lda, grid_.nl_d(), grid_.g2kin(), psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;

    h_psi.enforce_gamma_constraint_inplace();
    grid_.synchronize();
}

void Hamiltonian::apply_local(Wavefunction& psi, Wavefunction& h_psi) {
    int npw = psi.num_pw();
    int nbands = psi.num_bands();
    size_t nnr = grid_.nnr();

    const int block_size = 256;
    const int grid_size_smooth = (npw + block_size - 1) / block_size;
    const int grid_size_fft = (nnr + block_size - 1) / block_size;

    GammaFFTSolver fft(grid_);

    std::vector<int> h = grid_.get_miller_h();
    std::vector<int> k = grid_.get_miller_k();
    std::vector<int> l = grid_.get_miller_l();

    // ✅ 使用 QE 的 Gamma-only 打包策略：一次处理两个 band
    // 匹配 QE vloc_psi_gamma.f90:112-200
    for (int ibnd = 0; ibnd < nbands; ibnd += 2) {
        int ebnd = ibnd;
        int brange = 1;
        double fac = 1.0;

        if (ibnd < nbands - 1) {
            ebnd = ibnd + 1;
            brange = 2;
            fac = 0.5;  // ⚠️ QE 的关键因子：两个 band 打包时需要 0.5
        }

        // 准备两个 band 的 G 空间数据
        // ✅ 关键修复：Wavefunction 使用 FFT grid 布局 (nnr=3375)
        // 但 pack_two_kernel 期望紧凑数组 (npw=85)
        // 需要先使用 nl_d 映射提取 Smooth grid 数据

        GPU_Vector<gpufftComplex> psi1_smooth(npw);
        GPU_Vector<gpufftComplex> psi2_smooth(npw);
        ComplexField psi_r_packed(grid_);

        // 从 FFT grid 提取 Smooth grid 数据（使用 nl_d 映射）
        gather_fft_to_smooth_kernel<<<grid_size_smooth, block_size, 0, grid_.stream()>>>(
            npw, grid_.nl_d(), psi.band_data(ibnd), psi1_smooth.data());
        GPU_CHECK_KERNEL;

        if (brange == 2) {
            gather_fft_to_smooth_kernel<<<grid_size_smooth, block_size, 0, grid_.stream()>>>(
                npw, grid_.nl_d(), psi.band_data(ebnd), psi2_smooth.data());
            GPU_CHECK_KERNEL;
        } else {
            cudaMemsetAsync(psi2_smooth.data(), 0, npw * sizeof(gpufftComplex), grid_.stream());
        }

        // G -> R (Gamma-only 打包: psi_packed = psi1 + i*psi2)
        // 现在 psi1_smooth 和 psi2_smooth 是正确的紧凑数组
        fft.wave_g2r_pair_compact(psi1_smooth.data(), psi2_smooth.data(), h, k, l, psi_r_packed);

        // V(r) * ψ(r) - 在打包的实空间数据上应用局域势
        apply_vloc_kernel<<<grid_size_fft, block_size, 0, grid_.stream()>>>(nnr, v_loc_tot_.data(),
                                                                            psi_r_packed.data());
        GPU_CHECK_KERNEL;

        // R -> G (解包: 从 psi_packed 恢复 psi1 和 psi2)
        ComplexField vpsi1_g(grid_);
        ComplexField vpsi2_g(grid_);

        fft.wave_r2g_pair(psi_r_packed, h, k, l, vpsi1_g, vpsi2_g);

        // ✅ 关键修复：wave_r2g_pair 的 unpack_two_kernel 输出紧凑数组！
        // vpsi1_g 和 vpsi2_g 的前 npw 个元素就是结果（不是 FFT grid 布局）
        // 直接应用 fac 缩放，然后散射到 h_psi

        // 应用 fac 缩放（直接在紧凑数组的前 npw 个元素上）
        scale_vloc_kernel<<<grid_size_smooth, block_size, 0, grid_.stream()>>>(npw, vpsi1_g.data(),
                                                                               fac);
        GPU_CHECK_KERNEL;

        // 散射到 h_psi（从紧凑数组到 FFT grid）
        accumulate_hpsi_kernel<<<grid_size_smooth, block_size, 0, grid_.stream()>>>(
            npw, grid_.nl_d(), vpsi1_g.data(), h_psi.band_data(ibnd));
        GPU_CHECK_KERNEL;

        if (brange == 2) {
            scale_vloc_kernel<<<grid_size_smooth, block_size, 0, grid_.stream()>>>(
                npw, vpsi2_g.data(), fac);
            GPU_CHECK_KERNEL;

            accumulate_hpsi_kernel<<<grid_size_smooth, block_size, 0, grid_.stream()>>>(
                npw, grid_.nl_d(), vpsi2_g.data(), h_psi.band_data(ebnd));
            GPU_CHECK_KERNEL;
        }

        // 🔍 Diagnostic: final scaled result (simplified)
        // Removed to avoid potential segfault
    }

    h_psi.enforce_gamma_constraint_inplace();
    grid_.synchronize();
}

void Hamiltonian::apply_nonlocal(Wavefunction& psi, Wavefunction& h_psi) {
    if (nonlocal_) {
        nonlocal_->apply(psi, h_psi);
        h_psi.enforce_gamma_constraint_inplace();
        grid_.synchronize();
    }
}

void Hamiltonian::apply(Wavefunction& psi, Wavefunction& h_psi) {
    size_t nnr = grid_.nnr();
    int nbands = psi.num_bands();
    int npw = psi.num_pw();

    // Zero output
    cudaMemsetAsync(h_psi.data(), 0, nnr * nbands * sizeof(gpufftComplex), grid_.stream());

    // Apply T|ψ>
    apply_kinetic(psi, h_psi);

    // 保存 T|ψ> 用于后续计算 V_loc|ψ>
    Wavefunction tpsi_copy(grid_, nbands, psi.encut());
    CHECK(cudaMemcpy(tpsi_copy.data(), h_psi.data(), nnr * nbands * sizeof(gpufftComplex),
                     cudaMemcpyDeviceToDevice));

    // Apply V_loc|ψ> (accumulates)
    apply_local(psi, h_psi);

    // Apply V_NL|ψ> (accumulates)
    apply_nonlocal(psi, h_psi);

    h_psi.enforce_gamma_constraint_inplace();
    grid_.synchronize();
}

}  // namespace dftcu
