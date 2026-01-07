#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace dftcu {

// 使用 cufft 的复数类型别名
using gpufftComplex = cufftDoubleComplex;

/*
 * Gamma-only 优化工具函数
 *
 * 实现 QE 的 Gamma-only 优化逻辑：
 * 1. 使用实数 BLAS (cublasDgemm) 替代复数 BLAS
 * 2. 利用 Hermitian 对称性，因子 2.0
 * 3. 修正 G=0 项（避免重复计数）
 *
 * 目标精度：1e-15 (与 QE 的 regterg.f90 对齐)
 */

// ============================================================================
// 核心函数：Gamma-only 子空间矩阵计算
// ============================================================================

/**
 * 计算子空间矩阵: M_ij = <psi_a_i | psi_b_j>
 *
 * @param npw        平面波数量
 * @param nbands     能带数量
 * @param gstart     G=0 起始索引 (1=不含G=0, 2=含G=0)
 * @param psi_a      第一组波函数 (npw x nbands)
 * @param lda_a      psi_a 的 leading dimension
 * @param psi_b      第二组波函数 (npw x nbands)
 * @param lda_b      psi_b 的 leading dimension
 * @param matrix_out 输出矩阵 (nbands x nbands, 实对称矩阵)
 * @param ldr        matrix_out 的 leading dimension
 * @param stream     CUDA stream
 */
void compute_subspace_matrix_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi_a,
                                   int lda_a, const gpufftComplex* psi_b, int lda_b,
                                   double* matrix_out, int ldr, cudaStream_t stream);

/**
 * 计算哈密顿子空间矩阵: H_sub[i,j] = <psi_i | H | psi_j>
 */
void compute_h_subspace_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi,
                              int lda_psi, const gpufftComplex* hpsi, int lda_hpsi, double* h_sub,
                              int ldh, cudaStream_t stream);

/**
 * 计算重叠子空间矩阵: S_sub[i,j] = <psi_i | psi_j>
 */
void compute_s_subspace_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi,
                              int lda_psi, double* s_sub, int lds, cudaStream_t stream);

// ============================================================================
// CUDA Kernel
// ============================================================================

/**
 * G=0 修正 kernel
 *
 * 修正 DGEMM 中对 G=0 项的重复计数
 * matrix_out[i,j] -= Re[psi_a(G=0,i)] * Re[psi_b(G=0,j)]
 */
__global__ void correct_g0_term_kernel(int npw, int nbands, int gstart, const gpufftComplex* psi_a,
                                       int lda_a, const gpufftComplex* psi_b, int lda_b,
                                       double* matrix_out, int ldr);

}  // namespace dftcu
