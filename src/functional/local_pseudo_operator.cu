#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "fft/fft_solver.cuh"
#include "local_pseudo_kernels.cuh"
#include "local_pseudo_operator.cuh"
#include "math/bessel.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "utilities/math_utils.cuh"

namespace dftcu {

// ============================================================================
// Internal Kernels
// ============================================================================

namespace {

__device__ __constant__ double c_pseudo_atom_x[256];
__device__ __constant__ double c_pseudo_atom_y[256];
__device__ __constant__ double c_pseudo_atom_z[256];
__device__ __constant__ int c_pseudo_atom_type[256];

// Helper kernel to add scalar to all elements of array
__global__ void add_scalar_kernel(size_t n, double* data, double scalar) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i] += scalar;
    }
}

// Helper kernel to scale real array
__global__ void scale_vloc_kernel(size_t n, double* data, double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i] *= scale;
    }
}

// Helper kernel to scale complex array
__global__ void scale_complex_kernel(size_t n, gpufftComplex* data, double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

__global__ void vloc_gspace_kernel(int n, const double* gx, const double* gy, const double* gz,
                                   const double* gg, int nat, const double* flat_tab,
                                   const double* zp, int stride, double dq, double omega,
                                   double gcut, gpufftComplex* v_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;

    double g2_cryst = gg[i];
    if (gcut > 0 && g2_cryst > gcut) {
        v_g[i].x = 0.0;
        v_g[i].y = 0.0;
        return;
    }

    const double tpiba_factor = 4.0 * constants::D_PI * constants::D_PI;
    double gmod = sqrt(g2_cryst * tpiba_factor);

    double sum_re = 0;
    double sum_im = 0;

    for (int iat = 0; iat < nat; ++iat) {
        int type = c_pseudo_atom_type[iat];
        const double* table_short = flat_tab + type * stride;

        // 1. 调用纯插值函数
        double vlocg = interpolate_vloc_phys(gmod, table_short, stride, zp[type], omega);

        // 2. 调用纯相位函数
        gpufftComplex phase = compute_structure_factor_phase(
            gx[i], gy[i], gz[i], c_pseudo_atom_x[iat], c_pseudo_atom_y[iat], c_pseudo_atom_z[iat]);

        sum_re += vlocg * phase.x;
        sum_im += vlocg * phase.y;
    }
    v_g[i].x = sum_re;
    v_g[i].y = sum_im;
}

/**
 * @brief Scatter Dense grid V_loc(G) to FFT grid (Gamma-only with Hermitian symmetry).
 *
 * Matches QE's fftx_oned2threed for gamma-only:
 *   psi(nl_d(ig)) = c(ig)              # positive-G hemisphere
 *   psi(nlm_d(ig)) = CONJG(c(ig))      # negative-G hemisphere (conjugate)
 *
 * @param ngm_dense Number of Dense grid G-vectors (730 for Si Gamma)
 * @param nl_dense Dense +G → FFT grid index mapping
 * @param nlm_dense Dense -G → FFT grid index mapping
 * @param v_dense V_loc(G) on Dense grid (input)
 * @param v_fft V_loc(G) on FFT grid (output, zero-initialized)
 */
__global__ void scatter_dense_to_fft_kernel(int ngm_dense, const int* nl_dense,
                                            const int* nlm_dense, const gpufftComplex* v_dense,
                                            gpufftComplex* v_fft) {
    int ig = blockDim.x * blockIdx.x + threadIdx.x;
    if (ig >= ngm_dense)
        return;

    // Positive-G hemisphere
    int ifft_pos = nl_dense[ig];
    v_fft[ifft_pos] = v_dense[ig];

    // Negative-G hemisphere (conjugate for Hermitian symmetry)
    // SKIP G=0 to avoid duplication (nl_dense[0] == nlm_dense[0])
    if (ig > 0) {
        int ifft_neg = nlm_dense[ig];
        v_fft[ifft_neg].x = v_dense[ig].x;   // Real part: same
        v_fft[ifft_neg].y = -v_dense[ig].y;  // Imaginary part: negated (conjugate)
    }
}

/**
 * @brief Apply Hermitian symmetry to ensure V_loc(r) is real.
 *
 * For a real field in R-space, G-space must satisfy: V(-G) = conj(V(G))
 *
 * @param nx, ny, nz FFT grid dimensions
 * @param v_fft V_loc(G) on FFT grid (modified in-place)
 */
__global__ void apply_hermitian_symmetry_kernel(int nx, int ny, int nz, gpufftComplex* v_fft) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    // Skip G=0
    if (ix == 0 && iy == 0 && iz == 0)
        return;

    // Compute -G indices with periodic boundary conditions
    int mx = (ix == 0) ? 0 : (nx - ix);
    int my = (iy == 0) ? 0 : (ny - iy);
    int mz = (iz == 0) ? 0 : (nz - iz);

    // Compute 1D indices
    int idx_G = ix * ny * nz + iy * nz + iz;
    int idx_mG = mx * ny * nz + my * nz + mz;

    // V(-G) = conj(V(G))
    // Only set if -G position is currently zero (to avoid overwriting already set values)
    if (v_fft[idx_mG].x == 0.0 && v_fft[idx_mG].y == 0.0 && v_fft[idx_G].x != 0.0) {
        v_fft[idx_mG].x = v_fft[idx_G].x;
        v_fft[idx_mG].y = -v_fft[idx_G].y;  // Conjugate: negate imaginary part
    }
}
}  // namespace

LocalPseudoOperator::LocalPseudoOperator(Grid& grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms) {}

void LocalPseudoOperator::initialize_buffers(Grid& grid) {
    if (grid_ptr_ == &grid)
        return;
    grid_ptr_ = &grid;
    if (!fft_solver_)
        fft_solver_ = std::make_unique<FFTSolver>(grid);
    if (!v_g_)
        v_g_ = std::make_unique<ComplexField>(grid);
}

void LocalPseudoOperator::init_tab_vloc(int type, const std::vector<double>& r_grid,
                                        const std::vector<double>& vloc_r,
                                        const std::vector<double>& rab, double zp,
                                        double omega_bohr, int mesh_cutoff) {
    // 记录此算符对应的原子类型
    atom_type_ = type;

    // FIXME: This function still uses mixed units for historical reasons
    // omega_ is stored in Bohr³ internally, but gg[] from grid is in Angstrom⁻²
    // This will be fixed when Grid units are unified to pure atomic units
    omega_ = omega_bohr;

    if (type >= static_cast<int>(zp_.size()))
        zp_.resize(type + 1, 0.0);
    zp_[type] = zp;

    if (type >= static_cast<int>(tab_vloc_.size()))
        tab_vloc_.resize(type + 1);

    dq_ = 0.01;  // Match QE dq = 0.01

    // CRITICAL: Use Dense grid's max G² for interpolation range
    // grid_.g2max() returns FFT grid's max (too large), use Dense grid max instead
    auto gl_shells = grid_.get_gl_shells();
    double g2max_dense_cryst =
        gl_shells.empty() ? 0.0 : *std::max_element(gl_shells.begin(), gl_shells.end());
    double g2max_dense_phys =
        g2max_dense_cryst * (2.0 * constants::D_PI) * (2.0 * constants::D_PI);  // Cryst → Phys

    nqx_ = (int)(std::sqrt(g2max_dense_phys) / dq_) + 10;
    tab_vloc_[type].resize(nqx_ + 1);

    int msh = mesh_cutoff > 0 ? mesh_cutoff : (int)r_grid.size();
    std::vector<double> aux(msh);
    std::vector<double> rab_sub(msh);
    for (int i = 0; i < msh; ++i)
        rab_sub[i] = rab[i];

    const double fpi = 4.0 * constants::D_PI;
    const double e2 = 2.0;  // Ry units

    // Calculate G=0 term (alpha) separately - matches QE vloc_mod.f90:159-163
    // alpha = ∫ r*(r*vloc(r) + Z*e2) * 4π/Ω dr
    for (int ir = 0; ir < msh; ++ir)
        aux[ir] = r_grid[ir] * (r_grid[ir] * vloc_r[ir] + e2 * zp);
    tab_vloc_[type][0] = (simpson_integrate(aux, rab_sub) * fpi / omega_) * 0.5;

    // Calculate G≠0 terms with erf(r) correction
    for (int iq = 1; iq <= nqx_; ++iq) {
        double q = (iq - 1) * dq_;
        for (int ir = 0; ir < msh; ++ir)
            aux[ir] =
                (r_grid[ir] * vloc_r[ir] + e2 * zp * erf(r_grid[ir])) * sin(q * r_grid[ir]) / q;
        tab_vloc_[type][iq] = (simpson_integrate(aux, rab_sub) * fpi / omega_) * 0.5;
    }
}

void LocalPseudoOperator::compute(RealField& v) {
    Grid& grid_ = v.grid();
    initialize_buffers(grid_);
    size_t nnr = grid_.nnr();

    v_g_->fill({0, 0});

    // 检查此类型是否已初始化
    if (atom_type_ < 0) {
        throw std::runtime_error(
            "LocalPseudoOperator::compute: atom_type_ not set. Call init_tab_vloc first.");
    }

    // 只复制属于此类型的原子坐标到常量内存
    std::vector<double> atom_x, atom_y, atom_z;
    std::vector<int> atom_type_list;
    for (size_t i = 0; i < atoms_->nat(); ++i) {
        int type = atoms_->h_type()[i];
        if (type == atom_type_) {
            atom_x.push_back(atoms_->h_pos_x()[i]);
            atom_y.push_back(atoms_->h_pos_y()[i]);
            atom_z.push_back(atoms_->h_pos_z()[i]);
            atom_type_list.push_back(type);
        }
    }

    int num_atoms_this_type = atom_x.size();
    if (num_atoms_this_type == 0) {
        // 没有此类型的原子，直接返回零势
        v.fill(0.0);
        return;
    }

    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_x, atom_x.data(),
                                  num_atoms_this_type * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_y, atom_y.data(),
                                  num_atoms_this_type * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_z, atom_z.data(),
                                  num_atoms_this_type * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_type, atom_type_list.data(),
                                  num_atoms_this_type * sizeof(int), 0, cudaMemcpyHostToDevice,
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

    // ============================================================================
    // CRITICAL FIX: Use Dense grid (ecutrho) instead of FFT grid
    // ============================================================================
    // Create temporary buffer for Dense grid V_loc(G)
    int ngm_dense = grid_.ngm_dense();
    GPU_Vector<gpufftComplex> v_g_dense(ngm_dense);
    v_g_dense.fill({0, 0});

    // Step 1: Compute V_loc(G) on Dense grid (730 G-vectors for Si Gamma)
    vloc_gspace_kernel<<<(ngm_dense + 255) / 256, 256, 0, grid_.stream()>>>(
        ngm_dense, grid_.gx_dense(), grid_.gy_dense(), grid_.gz_dense(), grid_.gg_dense(),
        num_atoms_this_type, d_tab_.data(), d_zp_.data(), stride, dq_, omega_, gcut_,
        v_g_dense.data());

    // Step 2: Scatter Dense grid to FFT grid (730 → 4096) with Hermitian symmetry
    v_g_->fill({0, 0});

    // DEBUG: Check nl_dense and nlm_dense mapping
    std::vector<int> nl_test(std::min(10, ngm_dense));
    std::vector<int> nlm_test(std::min(10, ngm_dense));
    CHECK(cudaMemcpy(nl_test.data(), grid_.nl_dense(), nl_test.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(nlm_test.data(), grid_.nlm_dense(), nlm_test.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    // std::cout << "[DEBUG LocalPseudoOperator] nl_dense and nlm_dense mapping (first 10):"
    //           << std::endl;
    // for (size_t ig = 0; ig < nl_test.size(); ++ig) {
    //     std::cout << "  ig=" << ig << ": nl=" << nl_test[ig] << ", nlm=" << nlm_test[ig]
    //               << std::endl;
    // }

    scatter_dense_to_fft_kernel<<<(ngm_dense + 255) / 256, 256, 0, grid_.stream()>>>(
        ngm_dense, grid_.nl_dense(), grid_.nlm_dense(), v_g_dense.data(), v_g_->data());
    GPU_CHECK_KERNEL;

    // DEBUG: Export v_g_dense (Dense grid V_loc(G)) for comparison with QE
    // grid_.synchronize();
    // std::vector<gpufftComplex> v_dense_host(ngm_dense);
    // CHECK(cudaMemcpy(v_dense_host.data(), v_g_dense.data(), ngm_dense * sizeof(gpufftComplex),
    //                  cudaMemcpyDeviceToHost));
    // std::ofstream v_dense_file("dftcu_vloc_dense_debug.txt");
    // v_dense_file << "# DFTcu V_loc(G) on Dense grid (Hartree units)\n";
    // v_dense_file << "# ngm_dense = " << ngm_dense << "\n";
    // v_dense_file << "# Format: ig (0-based), Re(V) (Ha), Im(V) (Ha), |V| (Ha)\n";
    // for (int ig = 0; ig < ngm_dense; ++ig) {
    //     double re = v_dense_host[ig].x;
    //     double im = v_dense_host[ig].y;
    //     double mag = std::sqrt(re * re + im * im);
    //     v_dense_file << ig << " " << std::scientific << std::setprecision(16) << re << " " << im
    //                  << " " << mag << "\n";
    // }
    // v_dense_file.close();
    // std::cout << "[DEBUG LocalPseudoOperator] Exported v_dense to dftcu_vloc_dense_debug.txt"
    //           << std::endl;

    // DEBUG: Export v_g_ (FFT grid after scatter) for comparison with QE
    // std::vector<gpufftComplex> v_fft_host(grid_.nnr());
    // CHECK(cudaMemcpy(v_fft_host.data(), v_g_->data(), grid_.nnr() * sizeof(gpufftComplex),
    //                  cudaMemcpyDeviceToHost));
    // std::ofstream v_fft_file("dftcu_vloc_fft_grid_debug.txt");
    // v_fft_file << "# DFTcu V_loc FFT grid (after scatter, before IFFT, Hartree units)\n";
    // v_fft_file << "# nnr = " << grid_.nnr() << "\n";
    // v_fft_file << "# Format: ifft (0-based), Re (Ha), Im (Ha)\n";
    // for (size_t ifft = 0; ifft < grid_.nnr(); ++ifft) {
    //     v_fft_file << ifft << " " << std::scientific << std::setprecision(16) <<
    //     v_fft_host[ifft].x
    //                << " " << v_fft_host[ifft].y << "\n";
    // }
    // v_fft_file.close();
    // std::cout << "[DEBUG LocalPseudoOperator] Exported v_fft_grid to
    // dftcu_vloc_fft_grid_debug.txt"
    //           << std::endl;

    // 1. Extract Alpha term (G=0 contribution) as a scalar in Hartree.
    // tab_vloc_[type][0] matches QE v_of_0 * 0.5 per atom.
    // 只累加属于此类型的原子
    v_of_0_ = 0.0;
    for (size_t iat = 0; iat < atoms_->nat(); ++iat) {
        int type = atoms_->h_type()[iat];
        if (type == atom_type_) {
            v_of_0_ += tab_vloc_[type][0];
        }
    }
    // Note: tab_vloc[0] is already integrated and scaled by 4pi/Omega in init_tab_vloc.
    // So sum(tab_vloc[0]) is the total local potential shift.

    // 2. Zero out G=0 component in reciprocal space to make real-space field zero-mean.
    gpufftComplex zero_val = {0.0, 0.0};
    CHECK(cudaMemcpyAsync(v_g_->data(), &zero_val, sizeof(gpufftComplex), cudaMemcpyHostToDevice,
                          grid_.stream()));

    // DEBUG: Check values before FFT
    grid_.synchronize();
    // std::vector<gpufftComplex> v_g_before(std::min(10, (int)grid_.nnr()));
    // CHECK(cudaMemcpy(v_g_before.data(), v_g_->data(), v_g_before.size() * sizeof(gpufftComplex),
    //                  cudaMemcpyDeviceToHost));
    // std::cout << "[DEBUG LocalPseudoOperator] V_loc(G) before IFFT (first 10):" << std::endl;
    // for (size_t i = 0; i < v_g_before.size(); ++i) {
    //     std::cout << "  [" << i << "] = (" << v_g_before[i].x << ", " << v_g_before[i].y << ")"
    //               << std::endl;
    // }

    fft_solver_->backward(*v_g_);

    complex_to_real(grid_.nnr(), v_g_->data(), v.data(), grid_.stream());

    // NOTE: The 0.5 factor was previously applied here due to a misunderstanding
    // of Hermitian symmetry. The correct behavior is:
    // - We fill both +G and -G with conjugate values (Hermitian symmetry)
    // - The IFFT naturally produces the correct real-space values
    // - No additional scaling is needed
    //
    // The 0.5 factor has been REMOVED to match QE's output.
    // If you see a 2x discrepancy with QE, check:
    // 1. init_tab_vloc() Ry→Ha conversion (should be ×0.5)
    // 2. FFT normalization convention

    // 3. Add back the alpha shift (v_of_0_) to all R-space points
    const int bs_vloc = 256;
    const int gs_vloc = (grid_.nnr() + bs_vloc - 1) / bs_vloc;
    add_scalar_kernel<<<gs_vloc, bs_vloc, 0, grid_.stream()>>>(grid_.nnr(), v.data(), v_of_0_);
    GPU_CHECK_KERNEL;

    grid_.synchronize();
}

double LocalPseudoOperator::compute(const RealField& rho, RealField& v_out) {
    RealField v_ps(grid_);
    compute(v_ps);
    double energy = rho.dot(v_ps) * grid_.dv_bohr();
    v_add(grid_.nnr(), v_out.data(), v_ps.data(), v_out.data(), grid_.stream());
    return energy;
}

std::vector<double> LocalPseudoOperator::get_vloc_g_shells(
    int type, const std::vector<double>& g_shells) const {
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

void LocalPseudoOperator::set_valence_charge(int t, double z) {
    if (t >= static_cast<int>(zp_.size()))
        zp_.resize(t + 1);
    zp_[t] = z;
}
}  // namespace dftcu
