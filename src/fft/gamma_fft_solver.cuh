#pragma once

#include <cufft.h>

#include <vector>

#include "model/field.cuh"
#include "model/grid.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

// CUFFT error checking macro
#define CUFFT_CHECK(condition)                                                              \
    do {                                                                                    \
        cufftResult status = condition;                                                     \
        if (status != CUFFT_SUCCESS) {                                                      \
            std::cerr << "cuFFT error: " << status << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                         \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

namespace dftcu {

/**
 * @brief Gamma-only FFT Solver using QE's packing strategy
 *
 * QE Gamma-only 优化：
 * - 利用实波函数的 Hermitian 对称性 ψ(-G) = ψ*(G)
 * - 将两个波函数打包成一个复数：ψ_packed = ψ₁ + i·ψ₂
 * - 一次 FFT 同时处理两个波函数，减少计算量 50%
 *
 * 参考：
 * - QE: FFTXlib/src/fft_helper_subroutines.f90 (fftx_c2psi_gamma, fftx_psi2c_gamma)
 * - QE: Modules/fft_wave.f90 (wave_g2r, wave_r2g)
 *
 * 验证状态：Phase 0b.1 ✅ (机器精度 ~1e-16)
 */
class GammaFFTSolver {
  public:
    GammaFFTSolver(Grid& grid);
    ~GammaFFTSolver();

    /**
     * @brief 单个波函数 G -> R (利用 Hermitian 对称性)
     *
     * 输入: psi_g_half 只包含 G >= 0 的系数（通过 Miller 指数设置）
     * 输出: psi_r 全空间实波函数（虚部应该 ~0）
     *
     * QE 对应: wave_g2r(...) 单波函数路径
     */
    void wave_g2r_single(const ComplexField& psi_g_half,  // 输入：半球系数
                         const std::vector<int>& miller_h, const std::vector<int>& miller_k,
                         const std::vector<int>& miller_l,
                         ComplexField& psi_r);  // 输出：全空间

    /**
     * @brief 单个波函数 R -> G (强制 Hermitian 对称性)
     *
     * 输入: psi_r 全空间波函数
     * 输出: psi_g_half 波函数（已应用 Gamma 约束）
     *
     * QE 对应: wave_r2g(...) 单波函数路径
     */
    void wave_r2g_single(const ComplexField& psi_r,  // 输入：全空间
                         const std::vector<int>& miller_h, const std::vector<int>& miller_k,
                         const std::vector<int>& miller_l,
                         ComplexField& psi_g_half);  // 输出：半球系数

    /**
     * @brief 两个波函数打包 G -> R (QE 优化方法)
     *
     * 将两个波函数打包成一个复数，一次 FFT 同时处理：
     * ψ_packed(G) = ψ₁(G) + i·ψ₂(G)
     *
     * 输入: psi1_g, psi2_g 两个独立波函数（半球）
     * 输出: psi_r_packed 打包的实空间波函数
     *
     * QE 对应: wave_g2r(psi(:,ibnd:ebnd), psic, ...) 双波函数路径
     */
    void wave_g2r_pair(const ComplexField& psi1_g,  // 波函数 1（半球）
                       const ComplexField& psi2_g,  // 波函数 2（半球）
                       const std::vector<int>& miller_h, const std::vector<int>& miller_k,
                       const std::vector<int>& miller_l,
                       ComplexField& psi_r_packed);  // 打包的全空间

    /**
     * @brief 两个波函数打包 G -> R (紧凑数组版本)
     *
     * ✅ 修复版本：接受紧凑数组 (npw) 而不是 FFT grid (nnr)
     *
     * 输入: psi1_smooth, psi2_smooth 紧凑数组指针 (npw 个元素)
     * 输出: psi_r_packed 打包的实空间波函数 (FFT grid)
     */
    void wave_g2r_pair_compact(const gpufftComplex* psi1_smooth,  // 紧凑数组 (npw)
                               const gpufftComplex* psi2_smooth,  // 紧凑数组 (npw)
                               const std::vector<int>& miller_h, const std::vector<int>& miller_k,
                               const std::vector<int>& miller_l,
                               ComplexField& psi_r_packed);  // FFT grid (nnr)

    /**
     * @brief 从打包的 FFT 结果解包两个波函数 R -> G
     *
     * 解包公式（QE fftx_psi2c_gamma:39-42）：
     * fp = [ψ_packed(G) + ψ_packed(-G)] / 2
     * fm = [ψ_packed(G) - ψ_packed(-G)] / 2
     * ψ₁(G) = Re(fp) + i·Im(fm)
     * ψ₂(G) = Im(fp) - i·Re(fm)
     *
     * 输入: psi_r_packed 打包的实空间波函数
     * 输出: psi1_g, psi2_g 解包后的两个波函数（半球）
     *
     * QE 对应: wave_r2g(psic, vpsi(:,1:2), ...) 双波函数路径
     */
    void wave_r2g_pair(const ComplexField& psi_r_packed,  // 打包的全空间
                       const std::vector<int>& miller_h, const std::vector<int>& miller_k,
                       const std::vector<int>& miller_l,
                       ComplexField& psi1_g,   // 波函数 1（半球）
                       ComplexField& psi2_g);  // 波函数 2（半球）

  private:
    Grid& grid_;
    cufftHandle plan_z2z_;

    // 设备端 Miller 指数缓存（避免每次传输）
    int* d_miller_h_;
    int* d_miller_k_;
    int* d_miller_l_;
    int cached_npw_;

    void cache_miller_indices(const std::vector<int>& h, const std::vector<int>& k,
                              const std::vector<int>& l);
};

}  // namespace dftcu
