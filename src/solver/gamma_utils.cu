#include "solver/gamma_utils.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

// ============================================================================
// 打包 kernel: 从 FFT grid 提取 nl_d 映射的数据到紧凑数组
// ============================================================================

__global__ void pack_wavefunction_kernel(int npw, int nbands, const int* nl_d,
                                         const gpufftComplex* psi_fft, int lda_fft,
                                         gpufftComplex* psi_packed, int lda_packed) {
    /*
     * 从 FFT grid 稀疏数组打包到紧凑数组
     *
     * psi_fft: (lda_fft, nbands) - FFT grid 上的波函数，lda_fft = nnr
     * psi_packed: (lda_packed, nbands) - 紧凑数组，lda_packed = npw
     * nl_d: (npw) - G-vector 到 FFT grid 的映射
     *
     * 对于每个 G-vector ig (0 <= ig < npw)：
     *   ifft = nl_d[ig]  // FFT grid 索引
     *   psi_packed[ig + ib*npw] = psi_fft[ifft + ib*lda_fft]
     */
    int ig = blockIdx.x * blockDim.x + threadIdx.x;  // G-vector 索引
    int ib = blockIdx.y;                             // band 索引

    if (ig < npw && ib < nbands) {
        int ifft = nl_d[ig];                    // FFT grid 索引
        int idx_fft = ifft + ib * lda_fft;      // FFT grid 中的位置
        int idx_packed = ig + ib * lda_packed;  // 紧凑数组中的位置
        psi_packed[idx_packed] = psi_fft[idx_fft];
    }
}

// ============================================================================
// Gamma-only 内积计算：使用实数 BLAS + G=0 修正（修复版）
// ============================================================================

void compute_subspace_matrix_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi_a,
                                   int lda_a, const gpufftComplex* psi_b, int lda_b,
                                   double* matrix_out, int ldr, const int* nl_d,
                                   cudaStream_t stream) {
    /*
     * 计算子空间矩阵: M_ij = <psi_a_i | psi_b_j>
     *
     * ⚠️ 修复说明：
     * - 波函数数据在 FFT grid (lda=nnr) 上是稀疏存储的
     * - 只有 nl_d 映射的 npw 个点包含有效数据
     * - 需要先打包成紧凑数组 (lda=npw)，然后执行 DGEMM
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

    // [1] 分配紧凑数组（临时 GPU 内存）
    GPU_Vector<gpufftComplex> psi_a_packed(npw * nbands);
    GPU_Vector<gpufftComplex> psi_b_packed(npw * nbands);

    // [2] 打包波函数数据：FFT grid → 紧凑数组
    dim3 threads(256);
    dim3 blocks_a((npw + threads.x - 1) / threads.x, nbands);
    pack_wavefunction_kernel<<<blocks_a, threads, 0, stream>>>(npw, nbands, nl_d, psi_a, lda_a,
                                                               psi_a_packed.data(), npw);
    CHECK(cudaGetLastError());

    // 如果 psi_b != psi_a，也需要打包 psi_b
    if (psi_b != psi_a) {
        pack_wavefunction_kernel<<<blocks_a, threads, 0, stream>>>(npw, nbands, nl_d, psi_b, lda_b,
                                                                   psi_b_packed.data(), npw);
        CHECK(cudaGetLastError());
    }

    // [3] 执行 DGEMM（紧凑数组上，lda=npw）
    int npw2 = 2 * npw;
    int lda_packed2 = 2 * npw;  // 紧凑数组的 leading dimension

    const double* psi_a_real = reinterpret_cast<const double*>(psi_a_packed.data());
    const double* psi_b_real = (psi_b == psi_a)
                                   ? psi_a_real  // S_sub: psi_b = psi_a
                                   : reinterpret_cast<const double*>(psi_b_packed.data());

    double alpha = 2.0;  // 利用 Hermitian 对称性，乘以 2
    double beta = 0.0;

    // 调用实数 DGEMM: matrix_out = alpha * psi_a^T * psi_b + beta * matrix_out
    CUBLAS_SAFE_CALL(cublasDgemm(cb_handle,
                                 CUBLAS_OP_T,  // psi_a 转置
                                 CUBLAS_OP_N,  // psi_b 不转置
                                 nbands,       // M: matrix_out 的行数
                                 nbands,       // N: matrix_out 的列数
                                 npw2,         // K: 求和维度 (2*npw)
                                 &alpha, psi_a_real, lda_packed2, psi_b_real, lda_packed2, &beta,
                                 matrix_out, ldr));

    // [4] G=0 修正：如果包含 G=0，需要减去虚部的重复计数
    if (gstart == 2) {
        // 注意：使用紧凑数组，lda=npw
        correct_g0_term_kernel<<<nbands, nbands, 0, stream>>>(
            npw, nbands, gstart, psi_a_packed.data(), npw,
            (psi_b == psi_a) ? psi_a_packed.data() : psi_b_packed.data(), npw, matrix_out, ldr);
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
                              int ldh, const int* nl_d, cudaStream_t stream) {
    /*
     * H_sub[i,j] = <psi_i | H | psi_j> = <psi_i | hpsi_j>
     */
    compute_subspace_matrix_gamma(npw, nbands, gstart, psi, lda_psi, hpsi, lda_hpsi, h_sub, ldh,
                                  nl_d, stream);
}

void compute_s_subspace_gamma(int npw, int nbands, int gstart, const gpufftComplex* psi,
                              int lda_psi, double* s_sub, int lds, const int* nl_d,
                              cudaStream_t stream) {
    /*
     * S_sub[i,j] = <psi_i | psi_j>
     */
    compute_subspace_matrix_gamma(npw, nbands, gstart, psi, lda_psi, psi, lda_psi, s_sub, lds, nl_d,
                                  stream);
}

}  // namespace dftcu
