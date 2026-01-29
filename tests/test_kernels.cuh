/**
 * @file test_kernels.cuh
 * @brief 测试辅助：暴露内部 kernel 供单元测试使用
 *
 * 注意：这些 kernel 仅用于测试，不应在生产代码中直接调用
 */

#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace dftcu {
namespace test {

// 类型别名：与 DFTcu 内部一致
using gpufftComplex = cufftDoubleComplex;

/**
 * @brief 实空间乘法 kernel：V(r) * ψ(r)
 *
 * 公式：(V·ψ)(r) = V(r) · ψ(r)
 *
 * @param n 数组大小
 * @param v_loc 局域势 V(r) [n]
 * @param psi_r 波函数 ψ(r) [n]，原地修改
 */
__global__ void apply_vloc_kernel_test(size_t n, const double* v_loc, gpufftComplex* psi_r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        psi_r[i].x *= v_loc[i];
        psi_r[i].y *= v_loc[i];
    }
}

/**
 * @brief 缩放 kernel：data *= scale
 *
 * @param n 数组大小
 * @param data 复数数组 [n]，原地修改
 * @param scale 缩放因子
 */
__global__ void scale_complex_kernel_test(size_t n, gpufftComplex* data, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

/**
 * @brief 动能算符 kernel：T|ψ⟩ = ½|G|² |ψ(G)⟩
 *
 * 公式：h_psi[idx] = g2kin[ig] * psi[idx]
 * 其中 g2kin[ig] = ½|G|² (Hartree)
 *
 * @param npw 平面波数量
 * @param num_bands band 数量
 * @param lda Leading dimension (stride) = nnr
 * @param nl_d G-vector → FFT grid 映射 [npw]
 * @param g2kin 动能系数 ½|G|² [npw] (Hartree)
 * @param psi 输入波函数 [num_bands × lda]
 * @param h_psi 输出 T|ψ⟩ [num_bands × lda]
 */
__global__ void apply_kinetic_kernel_test(int npw, int num_bands, int lda, const int* nl_d,
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

}  // namespace test
}  // namespace dftcu
