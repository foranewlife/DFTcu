#include <algorithm>
#include <cmath>

#include "functional/wavefunction_builder.cuh"
#include "math/bessel.cuh"
#include "math/interpolation.cuh"
#include "math/ylm.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/math_utils.cuh"

#include <curand_kernel.h>

namespace dftcu {

namespace {

__global__ void build_atomic_band_kernel(int nnr, const double* atom_x, const double* atom_y,
                                         const double* atom_z, int iat, int l, int m_idx,
                                         const double* gx, const double* gy, const double* gz,
                                         const double* gg, const double* tab_chi,
                                         const double* tab_q, const double* tab_M,
                                         const double* tab_h, int nqx, double dq, double omega_bohr,
                                         double encut_hartree, gpufftComplex* band_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        // gg is already in Bohr^-2, convert to Rydberg: multiply by 2
        double g2_ry = gg[i] * 2.0;

        if (g2_ry > encut_hartree * 2.0 + 1e-5)
            return;

        double gmod_phys = sqrt(gg[i]) * 2.0 * constants::D_PI;  // |G| in Bohr^-1
        double chi_g = 0.0;

        // 使用三次样条插值（预计算系数）
        chi_g = math::cubic_spline_interpolate_device(gmod_phys, tab_q, tab_chi, tab_M, tab_h, nqx);

        double ylm = get_ylm(l, m_idx, gx[i], gy[i], gz[i], sqrt(gg[i]));
        double phase = -2.0 * constants::D_PI *
                       (gx[i] * atom_x[iat] + gy[i] * atom_y[iat] + gz[i] * atom_z[iat]);
        double s, c;
        sincos(phase, &s, &c);

        // ✅ FIX (2026-01-26): 使用 i^l 而不是 (-i)^l
        // QE 约定: lphase = i^l，确保 Gamma-only 波函数在实空间是实数
        // 参考: QE Modules/wavefunctions.f90:100-102
        double re_pre = 0, im_pre = 0;
        if (l == 0) {
            re_pre = 1.0;  // i^0 = 1
        } else if (l == 1) {
            im_pre = 1.0;  // i^1 = i
        } else if (l == 2) {
            re_pre = -1.0;  // i^2 = -1
        } else if (l == 3) {
            im_pre = -1.0;  // i^3 = -i
        }

        // Normalization: chi_q 已经包含了 1/sqrt(Ω) 归一化
        // 这里只需要乘以球谐函数和相位因子
        double norm = chi_g * ylm;
        // ✅ FIX (2026-01-26): 直接写入，不累加（每个 band 对应一个原子）
        // QE 方式：为每个原子的每个轨道创建独立的 band
        band_out[i].x = norm * (re_pre * c - im_pre * s);
        band_out[i].y = norm * (re_pre * s + im_pre * c);
    }
}

__global__ void apply_random_phase_kernel(int nnr, gpufftComplex* band, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        curandState state;
        curand_init(seed, i, 0, &state);
        double phase = curand_uniform(&state) * 2.0 * constants::D_PI;
        double s, c;
        sincos(phase, &s, &c);
        double re = band[i].x;
        double im = band[i].y;
        band[i].x = re * c - im * s;
        band[i].y = re * s + im * c;
    }
}

}  // namespace

WavefunctionBuilder::WavefunctionBuilder(Grid& grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms) {}

void WavefunctionBuilder::add_atomic_orbital(int type, int l, const std::vector<double>& r,
                                             const std::vector<double>& chi,
                                             const std::vector<double>& rab) {
    if (type >= (int)orbital_tables_.size())
        orbital_tables_.resize(type + 1);

    // [QE 对齐] 自动计算 msh: 找到第一个 r > 10 Bohr 的点
    // QE 使用 rcut=10 Bohr 截断积分网格，避免大 r 处的数值噪声
    // 参考：QE upflib/init_tab_atwfc.f90
    const double rcut = 10.0;  // Bohr
    int mesh_size = r.size();  // 默认：使用全部点
    for (size_t ir = 0; ir < r.size(); ++ir) {
        if (r[ir] > rcut) {
            mesh_size = ir;
            break;
        }
    }

    // [QE 对齐] 强制使用奇数网格点（Simpson 积分要求）
    // QE 的 Simpson 积分在奇数网格上精度最高
    // 参考：QE upflib/simpsn.f90 的实现
    if (mesh_size % 2 == 0) {
        mesh_size += 1;
        if (mesh_size > (int)r.size()) {
            mesh_size = (int)r.size();
        }
    }

    // QE 的 qmax 计算方式（参考 QE upflib/qrad_mod.f90）
    // qmax 不是基于 ecutwfc，而是基于原子轨道的实际截断半径
    // 对于大多数赝势，qmax = 4.5 Bohr^-1 就足够了
    // 这里使用更保守的估计：基于径向网格的最大半径
    double r_max = r[mesh_size - 1];  // 最大径向距离 (Bohr)

    // qmax 应该足够大以覆盖原子轨道的傅里叶变换
    // 经验公式：qmax ≈ 2π / r_min，但不超过 sqrt(2*ecutwfc)
    // 对于典型的赝势，qmax ≈ 4-5 Bohr^-1 就足够了
    double qmax_from_rmax = 2.0 * constants::D_PI / (r[0] * 10.0);  // 基于最小半径的估计
    double qmax_from_ecutwfc = sqrt(2.0 * grid_.ecutwfc()) * 2.0 * constants::D_PI;  // 基于截断能
    double qmax = std::min(qmax_from_rmax, qmax_from_ecutwfc);

    // 为了与 QE 对齐，使用固定的 qmax（QE 的经验值）
    // 这个值对于大多数赝势都足够了
    qmax = 4.5;  // Bohr^-1 (与 QE 数据一致)

    // QE 的 nqx 计算公式：nqx = INT(qmax / dq + 4)
    // 但 QE 的数组索引是 0 到 nqx-1，所以实际数据点数是 nqx
    // 从 QE 数据反推：nqx = 451, qmax = 4.5
    // 451 = INT(4.5 / 0.01 + 4) = INT(450 + 4) = 454 (不对！)
    // 实际上 QE 的公式应该是：nqx = INT(qmax / dq) + 1
    int nqx = static_cast<int>(qmax / dq_) + 1;  // 451 = 450 + 1

    std::vector<double> chi_q(nqx, 0.0);  // 注意：不是 nqx + 1
    std::vector<double> aux(mesh_size);   // 使用截断后的大小
    const double fpi = 4.0 * constants::D_PI;

    // QE 的归一化因子：pref = 4π / sqrt(Ω)
    double omega_bohr = grid_.volume_bohr();
    double pref = fpi / sqrt(omega_bohr);

    // QE 的循环：iq = 0 到 nqx-1
    for (int iq = 0; iq < nqx; ++iq) {
        double q = iq * dq_;  // q = 0, 0.01, 0.02, ..., 4.50
        if (q < 1e-12) {
            if (l == 0) {
                // q=0: j_0(0) = 1, 积分 ∫ r * chi(r) dr
                for (int ir = 0; ir < mesh_size; ++ir)
                    aux[ir] = r[ir] * chi[ir];
            } else {
                // q=0: j_l(0) = 0 for l > 0
                for (int ir = 0; ir < mesh_size; ++ir)
                    aux[ir] = 0.0;
            }
        } else {
            // q>0: 积分 ∫ r * chi(r) j_l(qr) dr
            for (int ir = 0; ir < mesh_size; ++ir) {
                aux[ir] = r[ir] * chi[ir] * spherical_bessel_jl(l, q * r[ir]);
            }
        }
        // 使用截断后的 rab 进行积分
        std::vector<double> rab_truncated(rab.begin(), rab.begin() + mesh_size);
        chi_q[iq] = simpson_integrate(aux, rab_truncated) * pref;  // 使用 pref 而不是 fpi
    }

    // 预计算三次样条插值系数
    std::vector<double> q_data(nqx);
    for (int iq = 0; iq < nqx; ++iq) {
        q_data[iq] = iq * dq_;
    }

    auto spline_coeff = math::precompute_cubic_spline_coefficients(q_data, chi_q);

    orbital_tables_[type].push_back({l, chi_q, spline_coeff.M, spline_coeff.h});
}

void WavefunctionBuilder::build_atomic_wavefunctions(Wavefunction& psi, bool randomize_phase) {
    // ✅ FIX: Delegate to internal implementation (DRY principle)
    build_atomic_wavefunctions_internal(psi, randomize_phase);
}

int WavefunctionBuilder::calculate_num_bands() const {
    int total = 0;
    for (int iat = 0; iat < (int)atoms_->nat(); ++iat) {
        int type = atoms_->h_type()[iat];
        if (type >= (int)orbital_tables_.size())
            continue;

        for (const auto& orb : orbital_tables_[type]) {
            total += (2 * orb.l + 1);  // Each orbital has (2l+1) magnetic quantum numbers
        }
    }
    return total;
}

int WavefunctionBuilder::num_bands() const {
    return calculate_num_bands();
}

const std::vector<double>& WavefunctionBuilder::get_chi_q(int type, int orbital_idx) const {
    if (type < 0 || type >= static_cast<int>(orbital_tables_.size())) {
        throw std::runtime_error("WavefunctionBuilder::get_chi_q: Invalid type index");
    }
    if (orbital_idx < 0 || orbital_idx >= static_cast<int>(orbital_tables_[type].size())) {
        throw std::runtime_error("WavefunctionBuilder::get_chi_q: Invalid orbital index");
    }
    return orbital_tables_[type][orbital_idx].chi_q;
}

std::unique_ptr<Wavefunction> WavefunctionBuilder::build(bool randomize_phase) {
    int n_bands = calculate_num_bands();

    if (n_bands == 0) {
        throw std::runtime_error("WavefunctionBuilder::build: No atomic orbitals added. "
                                 "Call add_atomic_orbital() before build().");
    }

    // Create Wavefunction object
    auto psi = std::make_unique<Wavefunction>(grid_, n_bands, grid_.ecutwfc());

    // Fill with atomic orbitals (reuse existing implementation)
    build_atomic_wavefunctions_internal(*psi, randomize_phase);

    return psi;
}

void WavefunctionBuilder::build_atomic_wavefunctions_internal(Wavefunction& psi,
                                                              bool randomize_phase) {
    // This is the same as the old build_atomic_wavefunctions, just renamed
    int n_bands = psi.num_bands();
    int nnr = grid_.nnr();
    double omega_bohr = grid_.volume_bohr();

    grid_.synchronize();
    cudaMemset(psi.data(), 0, n_bands * nnr * sizeof(gpufftComplex));

    // Combine all tables into flat contiguous buffers
    std::vector<double> h_total_chi_q;
    std::vector<double> h_total_q;
    std::vector<double> h_total_M;
    std::vector<double> h_total_h;
    std::vector<std::vector<size_t>> table_offsets(orbital_tables_.size());

    for (int t = 0; t < (int)orbital_tables_.size(); ++t) {
        for (const auto& orb : orbital_tables_[t]) {
            table_offsets[t].push_back(h_total_chi_q.size());

            // chi_q 数据
            h_total_chi_q.insert(h_total_chi_q.end(), orb.chi_q.begin(), orb.chi_q.end());

            // q 坐标（均匀网格）
            int nqx = orb.chi_q.size();
            for (int iq = 0; iq < nqx; ++iq) {
                h_total_q.push_back(iq * dq_);
            }

            // 三次样条系数
            h_total_M.insert(h_total_M.end(), orb.spline_M.begin(), orb.spline_M.end());
            h_total_h.insert(h_total_h.end(), orb.spline_h.begin(), orb.spline_h.end());
        }
    }

    // 传递数据到 GPU
    d_tab_.resize(h_total_chi_q.size());
    d_tab_.copy_from_host(h_total_chi_q.data());

    GPU_Vector<double> d_q(h_total_q.size());
    d_q.copy_from_host(h_total_q.data());

    d_spline_M_.resize(h_total_M.size());
    d_spline_M_.copy_from_host(h_total_M.data());

    d_spline_h_.resize(h_total_h.size());
    d_spline_h_.copy_from_host(h_total_h.data());

    grid_.synchronize();

    int current_band = 0;
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    // 遍历所有原子
    for (int iat = 0; iat < (int)atoms_->nat(); ++iat) {
        int type = atoms_->h_type()[iat];

        // 遍历该原子的所有轨道
        for (int i_orb = 0; i_orb < (int)orbital_tables_[type].size(); ++i_orb) {
            const auto& orb = orbital_tables_[type][i_orb];
            size_t offset = table_offsets[type][i_orb];

            // 遍历所有磁量子数
            for (int m = 0; m < 2 * orb.l + 1; ++m) {
                if (current_band >= n_bands) {
                    throw std::runtime_error(
                        "WavefunctionBuilder: band count exceeds limit (current_band=" +
                        std::to_string(current_band) + ", n_bands=" + std::to_string(n_bands) +
                        ")");
                }

                build_atomic_band_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
                    nnr, atoms_->pos_x(), atoms_->pos_y(), atoms_->pos_z(), iat, orb.l, m,
                    grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), d_tab_.data() + offset,
                    d_q.data() + offset, d_spline_M_.data() + offset, d_spline_h_.data() + offset,
                    (int)orb.chi_q.size(), dq_, omega_bohr, psi.encut(),
                    psi.band_data(current_band));

                if (randomize_phase) {
                    unsigned int seed = 42 + current_band;
                    apply_random_phase_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
                        nnr, psi.band_data(current_band), seed);
                }

                current_band++;
            }
        }
    }

    grid_.synchronize();
}

}  // namespace dftcu
