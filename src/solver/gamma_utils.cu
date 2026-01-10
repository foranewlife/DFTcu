#include "solver/gamma_utils.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"

namespace dftcu {

// ============================================================================
// Gamma-only 内积计算：使用实数 BLAS + G=0 修正
// ============================================================================

void compute_subspace_matrix_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi_a,
                                   int lda_a, const gpufftComplex* psi_b, int lda_b,
                                   double* matrix_out, int ldr, cudaStream_t stream) {
    /*
     * 计算子空间矩阵: M_ij = <psi_a_i | psi_b_j>
     *
     * QE Gamma-only 优化逻辑:
     * 1. 将复数数组视为实数 (长度翻倍): npw2 = 2*npw
     * 2. 利用 Hermitian 对称性: <a|b> = 2*Re[Σ a*(G)b(G)] - a*(0)b(0)
     * 3. 使用实数 DGEMM 计算，因子 2.0
     * 4. 修正 G=0 项（避免重复乘以2）
     *
     * 参考: QE regterg.f90:204
     */

    cublasHandle_t cb_handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(cb_handle, stream));
    CublasPointerModeGuard guard(cb_handle, CUBLAS_POINTER_MODE_HOST);

    int npw2 = 2 * npw;
    int lda_a2 = 2 * lda_a;
    int lda_b2 = 2 * lda_b;

    // 将复数数组视为实数数组
    const double* psi_a_real = reinterpret_cast<const double*>(psi_a);
    const double* psi_b_real = reinterpret_cast<const double*>(psi_b);

    double alpha = 2.0;  // 利用 Hermitian 对称性，乘以 2
    double beta = 0.0;

    // 调用实数 DGEMM: matrix_out = alpha * psi_a^T * psi_b + beta * matrix_out
    CUBLAS_SAFE_CALL(cublasDgemm(cb_handle,
                                 CUBLAS_OP_T,  // psi_a 转置
                                 CUBLAS_OP_N,  // psi_b 不转置
                                 nbands,       // M: matrix_out 的行数
                                 nbands,       // N: matrix_out 的列数
                                 npw2,         // K: 求和维度 (2*npw)
                                 &alpha, psi_a_real, lda_a2, psi_b_real, lda_b2, &beta, matrix_out,
                                 ldr));

    // G=0 修正：如果包含 G=0，需要减去虚部的重复计数
    if (gstart == 2) {
        // 在 GPU 上执行修正
        // 使用 nbands 个 block，每个 block nbands 个线程，正好处理 nbands x nbands 矩阵
        correct_g0_term_kernel<<<nbands, nbands, 0, stream>>>(npw, nbands, gstart, psi_a, lda_a,
                                                              psi_b, lda_b, matrix_out, ldr);
        CHECK(cudaGetLastError());
    }
}

// ============================================================================
// G=0 修正 kernel
// ============================================================================

__global__ void correct_g0_term_kernel(int npw, int nbands, int gstart, const gpufftComplex* psi_a,
                                       int lda_a, const gpufftComplex* psi_b, int lda_b,
                                       double* matrix_out, int ldr) {
    /*
     * 修正 G=0 项：matrix_out[i,j] -= Re[psi_a[0,i]] * Re[psi_b[0,j]]
     *
     * 因为在 DGEMM 中我们对所有项乘以了 2，但 G=0 项只应计数一次
     */
    int i = blockIdx.x;   // 行索引
    int j = threadIdx.x;  // 列索引

    if (i < nbands && j < nbands) {
        // G=0 对应索引 0，只取实部
        // lda_a 是波函数的 leading dimension
        double psi_a_g0 = psi_a[i * lda_a].x;  // Re[psi_a(G=0, band_i)]
        double psi_b_g0 = psi_b[j * lda_b].x;  // Re[psi_b(G=0, band_j)]

        // 减去重复计数的 G=0 项
        // matrix_out 是 column-major (由 cublasDgemm 输出)
        // 索引 [i, j] 对应 i + j * ldr
        matrix_out[i + j * ldr] -= psi_a_g0 * psi_b_g0;
    }
}

// ============================================================================
// 便利接口：直接计算 H_sub 或 S_sub
// ============================================================================

void compute_h_subspace_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi,
                              int lda_psi, const gpufftComplex* hpsi, int lda_hpsi, double* h_sub,
                              int ldh, cudaStream_t stream) {
    /*
     * H_sub[i,j] = <psi_i | H | psi_j> = <psi_i | hpsi_j>
     */
    compute_subspace_matrix_gamma(npw, nbands, gstart, psi, lda_psi, hpsi, lda_hpsi, h_sub, ldh,
                                  stream);
}

void compute_s_subspace_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi,
                              int lda_psi, double* s_sub, int lds, cudaStream_t stream) {
    /*
     * S_sub[i,j] = <psi_i | psi_j>
     */
    compute_subspace_matrix_gamma(npw, nbands, gstart, psi, lda_psi, psi, lda_psi, s_sub, lds,
                                  stream);
}

}  // namespace dftcu
