#include "fft/gamma_fft_solver.cuh"

namespace dftcu {

namespace {

// Scale kernel for FFT normalization
__global__ void scale_complex_kernel(size_t n, gpufftComplex* data, double scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

/**
 * @brief 展开 Hermitian 对称性：ψ(-G) = ψ*(G)
 *
 * 输入: psi_half 只包含 G >= 0 的系数（npw 个）
 * 输出: psi_full 完整 FFT 网格（nnr 个），满足 Hermitian 对称性
 */
__global__ void expand_hermitian_kernel(int npw, int nx, int ny, int nz, const int* miller_h,
                                        const int* miller_k, const int* miller_l,
                                        const gpufftComplex* psi_half, gpufftComplex* psi_full) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
        return;

    int h = miller_h[ig];
    int k = miller_k[ig];
    int l = miller_l[ig];

    // 正频率 (nl_d) - 正确处理负 Miller 指数
    int h_pos = (h % nx + nx) % nx;
    int k_pos = (k % ny + ny) % ny;
    int l_pos = (l % nz + nz) % nz;
    int idx_pos = (h_pos * ny + k_pos) * nz + l_pos;
    psi_full[idx_pos] = psi_half[ig];

    // 负频率 (nlm_d) - Hermitian 共轭
    if (!(h == 0 && k == 0 && l == 0)) {  // G=0 不需要镜像
        int h_neg = (-h + nx) % nx;
        int k_neg = (-k + ny) % ny;
        int l_neg = (-l + nz) % nz;
        int idx_neg = (h_neg * ny + k_neg) * nz + l_neg;

        psi_full[idx_neg].x = psi_half[ig].x;   // Re(conj) = Re
        psi_full[idx_neg].y = -psi_half[ig].y;  // Im(conj) = -Im
    }
}

/**
 * @brief 从完整 FFT 网格提取 Miller 指数对应的系数
 */
__global__ void extract_miller_kernel(int npw, int nx, int ny, int nz, const int* miller_h,
                                      const int* miller_k, const int* miller_l,
                                      const gpufftComplex* psi_full, gpufftComplex* psi_half) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
        return;

    int h = miller_h[ig];
    int k = miller_k[ig];
    int l = miller_l[ig];

    // 正确处理负 Miller 指数
    int h_mod = (h % nx + nx) % nx;
    int k_mod = (k % ny + ny) % ny;
    int l_mod = (l % nz + nz) % nz;
    int idx = (h_mod * ny + k_mod) * nz + l_mod;
    psi_half[ig] = psi_full[idx];
}

/**
 * @brief 打包两个波函数（QE 策略）
 *
 * ψ_packed(G) = ψ₁(G) + i·ψ₂(G)
 * ψ_packed(-G) = conj(ψ₁(G)) + i·conj(ψ₂(G))
 */
__global__ void pack_two_kernel(int npw, int nx, int ny, int nz, const int* miller_h,
                                const int* miller_k, const int* miller_l, const gpufftComplex* psi1,
                                const gpufftComplex* psi2, gpufftComplex* packed) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
        return;

    // Debug: print first few
    if (ig < 5) {
        printf("[pack_two_kernel] ig=%d: psi1=(%.6f,%.6f) psi2=(%.6f,%.6f)\n", ig, psi1[ig].x,
               psi1[ig].y, psi2[ig].x, psi2[ig].y);
    }

    int h = miller_h[ig];
    int k = miller_k[ig];
    int l = miller_l[ig];

    // 正频率: packed = psi1 + i*psi2 - 正确处理负 Miller 指数
    int h_pos = (h % nx + nx) % nx;
    int k_pos = (k % ny + ny) % ny;
    int l_pos = (l % nz + nz) % nz;
    int idx_pos = (h_pos * ny + k_pos) * nz + l_pos;
    packed[idx_pos].x = psi1[ig].x - psi2[ig].y;  // Re(psi1) - Im(psi2)
    packed[idx_pos].y = psi1[ig].y + psi2[ig].x;  // Im(psi1) + Re(psi2)

    // 负频率: packed(-G) = conj(psi1) + i*conj(psi2)
    if (!(h == 0 && k == 0 && l == 0)) {
        int h_neg = (-h + nx) % nx;
        int k_neg = (-k + ny) % ny;
        int l_neg = (-l + nz) % nz;
        int idx_neg = (h_neg * ny + k_neg) * nz + l_neg;

        // conj(psi1) + i*conj(psi2) = (Re1 - i*Im1) + i*(Re2 - i*Im2)
        //                            = (Re1 + Im2) + i*(-Im1 + Re2)
        packed[idx_neg].x = psi1[ig].x + psi2[ig].y;
        packed[idx_neg].y = -psi1[ig].y + psi2[ig].x;

        // Debug: print a few negative G
        if (ig >= 1 && ig <= 2) {
            printf("[pack_two_kernel] ig=%d (h,k,l)=(%d,%d,%d): idx_pos=%d packed_pos=(%.6f,%.6f), "
                   "idx_neg=%d packed_neg=(%.6f,%.6f)\n",
                   ig, h, k, l, idx_pos, packed[idx_pos].x, packed[idx_pos].y, idx_neg,
                   packed[idx_neg].x, packed[idx_neg].y);
        }
    }
}

/**
 * @brief 解包两个波函数（QE fftx_psi2c_gamma）
 *
 * fp = [packed(G) + packed(-G)]
 * fm = [packed(G) - packed(-G)]
 * ψ₁(G) = Re(fp) + i·Im(fm)
 * ψ₂(G) = Im(fp) - i·Re(fm)
 */
__global__ void unpack_two_kernel(int npw, int nx, int ny, int nz, const int* miller_h,
                                  const int* miller_k, const int* miller_l,
                                  const gpufftComplex* packed, gpufftComplex* psi1,
                                  gpufftComplex* psi2) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
        return;

    int h = miller_h[ig];
    int k = miller_k[ig];
    int l = miller_l[ig];

    // 正频率索引 - 正确处理负 Miller 指数
    int h_pos = (h % nx + nx) % nx;
    int k_pos = (k % ny + ny) % ny;
    int l_pos = (l % nz + nz) % nz;
    int idx_pos = (h_pos * ny + k_pos) * nz + l_pos;

    gpufftComplex p_pos = packed[idx_pos];
    gpufftComplex p_neg;

    if (h == 0 && k == 0 && l == 0) {
        // G=0: 只有正频率
        p_neg = p_pos;
    } else {
        int h_neg = (-h + nx) % nx;
        int k_neg = (-k + ny) % ny;
        int l_neg = (-l + nz) % nz;
        int idx_neg = (h_neg * ny + k_neg) * nz + l_neg;
        p_neg = packed[idx_neg];
    }

    // ✅ QE 解包公式（无 × 0.5 因子）
    // QE fftx_psi2c_gamma (FFTXlib/src/fft_helper_subroutines.f90:56-60):
    //   fp = vin(nl_d(ig)) + vin(nlm_d(ig))
    //   fm = vin(nl_d(ig)) - vin(nlm_d(ig))
    //   vout1 = CMPLX(DBLE(fp), AIMAG(fm))
    //   vout2 = CMPLX(AIMAG(fp), -DBLE(fm))
    double fp_re = p_pos.x + p_neg.x;
    double fp_im = p_pos.y + p_neg.y;
    double fm_re = p_pos.x - p_neg.x;
    double fm_im = p_pos.y - p_neg.y;

    // psi1 = Re(fp) + i*Im(fm)
    psi1[ig].x = fp_re;
    psi1[ig].y = fm_im;

    // psi2 = Im(fp) - i*Re(fm)
    psi2[ig].x = fp_im;
    psi2[ig].y = -fm_re;
}

}  // namespace

// ============================================================================
// GammaFFTSolver 实现
// ============================================================================

GammaFFTSolver::GammaFFTSolver(Grid& grid)
    : grid_(grid),
      d_miller_h_(nullptr),
      d_miller_k_(nullptr),
      d_miller_l_(nullptr),
      cached_npw_(0) {
    int nr[3] = {grid.nr()[0], grid.nr()[1], grid.nr()[2]};
    CUFFT_CHECK(cufftPlan3d(&plan_z2z_, nr[0], nr[1], nr[2], CUFFT_Z2Z));
    CUFFT_CHECK(cufftSetStream(plan_z2z_, grid.stream()));
}

GammaFFTSolver::~GammaFFTSolver() {
    cufftDestroy(plan_z2z_);

    if (d_miller_h_)
        CHECK(cudaFree(d_miller_h_));
    if (d_miller_k_)
        CHECK(cudaFree(d_miller_k_));
    if (d_miller_l_)
        CHECK(cudaFree(d_miller_l_));
}

void GammaFFTSolver::cache_miller_indices(const std::vector<int>& h, const std::vector<int>& k,
                                          const std::vector<int>& l) {
    int npw = h.size();

    // 如果已缓存且大小匹配，不需要重新分配
    if (cached_npw_ == npw) {
        // 更新数据
        CHECK(cudaMemcpyAsync(d_miller_h_, h.data(), npw * sizeof(int), cudaMemcpyHostToDevice,
                              grid_.stream()));
        CHECK(cudaMemcpyAsync(d_miller_k_, k.data(), npw * sizeof(int), cudaMemcpyHostToDevice,
                              grid_.stream()));
        CHECK(cudaMemcpyAsync(d_miller_l_, l.data(), npw * sizeof(int), cudaMemcpyHostToDevice,
                              grid_.stream()));
        return;
    }

    // 释放旧的缓存
    if (d_miller_h_)
        CHECK(cudaFree(d_miller_h_));
    if (d_miller_k_)
        CHECK(cudaFree(d_miller_k_));
    if (d_miller_l_)
        CHECK(cudaFree(d_miller_l_));

    // 分配新的缓存
    CHECK(cudaMalloc(&d_miller_h_, npw * sizeof(int)));
    CHECK(cudaMalloc(&d_miller_k_, npw * sizeof(int)));
    CHECK(cudaMalloc(&d_miller_l_, npw * sizeof(int)));

    // 拷贝数据
    CHECK(cudaMemcpyAsync(d_miller_h_, h.data(), npw * sizeof(int), cudaMemcpyHostToDevice,
                          grid_.stream()));
    CHECK(cudaMemcpyAsync(d_miller_k_, k.data(), npw * sizeof(int), cudaMemcpyHostToDevice,
                          grid_.stream()));
    CHECK(cudaMemcpyAsync(d_miller_l_, l.data(), npw * sizeof(int), cudaMemcpyHostToDevice,
                          grid_.stream()));

    cached_npw_ = npw;
}

void GammaFFTSolver::wave_g2r_single(const ComplexField& psi_g_half,
                                     const std::vector<int>& miller_h,
                                     const std::vector<int>& miller_k,
                                     const std::vector<int>& miller_l, ComplexField& psi_r) {
    int npw = miller_h.size();
    int nnr = grid_.nnr();
    const int* nr = grid_.nr();

    // 缓存 Miller 指数
    cache_miller_indices(miller_h, miller_k, miller_l);

    // 1. 清零输出
    CHECK(cudaMemsetAsync(psi_r.data(), 0, nnr * sizeof(gpufftComplex), grid_.stream()));

    // 2. 展开 Hermitian 对称性
    const int block_size = 256;
    const int grid_size = (npw + block_size - 1) / block_size;

    expand_hermitian_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        npw, nr[0], nr[1], nr[2], d_miller_h_, d_miller_k_, d_miller_l_, psi_g_half.data(),
        psi_r.data());
    GPU_CHECK_KERNEL;
    grid_.synchronize();  // Ensure kernel completes before FFT

    // 3. IFFT (cuFFT INVERSE)
    CUFFT_CHECK(cufftExecZ2Z(plan_z2z_, (cufftDoubleComplex*)psi_r.data(),
                             (cufftDoubleComplex*)psi_r.data(), CUFFT_INVERSE));
    GPU_CHECK_KERNEL;

    // 4. Normalize by 1/N (QE convention: invfft includes 1/N normalization)
    // 注意: cuFFT 的 CUFFT_INVERSE 不包含归一化，需要手动除以 N
    const int bs_norm = 256;
    const int gs_norm = (nnr + bs_norm - 1) / bs_norm;
    scale_complex_kernel<<<gs_norm, bs_norm, 0, grid_.stream()>>>(nnr, psi_r.data(),
                                                                  1.0 / (double)nnr);
    GPU_CHECK_KERNEL;
    grid_.synchronize();
}

void GammaFFTSolver::wave_r2g_single(const ComplexField& psi_r, const std::vector<int>& miller_h,
                                     const std::vector<int>& miller_k,
                                     const std::vector<int>& miller_l, ComplexField& psi_g_half) {
    int npw = miller_h.size();
    int nnr = grid_.nnr();
    const int* nr = grid_.nr();

    // 缓存 Miller 指数
    cache_miller_indices(miller_h, miller_k, miller_l);

    // 1. 创建临时缓冲区
    ComplexField temp(grid_);
    CHECK(cudaMemcpyAsync(temp.data(), psi_r.data(), nnr * sizeof(gpufftComplex),
                          cudaMemcpyDeviceToDevice, grid_.stream()));

    // 2. FFT (cuFFT FORWARD, 不归一化 - 匹配 QE fwfft 约定)
    CUFFT_CHECK(cufftExecZ2Z(plan_z2z_, (cufftDoubleComplex*)temp.data(),
                             (cufftDoubleComplex*)temp.data(), CUFFT_FORWARD));
    GPU_CHECK_KERNEL;

    // 注意：根据 QE fft_scalar.f90 约定
    // - invfft (G→R): FFTW_BACKWARD + 归一化 1/N
    // - fwfft (R→G): FFTW_FORWARD，不归一化
    // 因此 wave_r2g 不需要归一化

    // 3. 提取 Miller 指数对应的系数
    const int bs_r2g = 256;
    const int grid_size = (npw + bs_r2g - 1) / bs_r2g;

    extract_miller_kernel<<<grid_size, bs_r2g, 0, grid_.stream()>>>(
        npw, nr[0], nr[1], nr[2], d_miller_h_, d_miller_k_, d_miller_l_, temp.data(),
        psi_g_half.data());
    GPU_CHECK_KERNEL;
}

void GammaFFTSolver::wave_g2r_pair(const ComplexField& psi1_g, const ComplexField& psi2_g,
                                   const std::vector<int>& miller_h,
                                   const std::vector<int>& miller_k,
                                   const std::vector<int>& miller_l, ComplexField& psi_r_packed) {
    int npw = miller_h.size();
    int nnr = grid_.nnr();
    const int* nr = grid_.nr();

    printf("[GammaFFTSolver::wave_g2r_pair] Called with npw=%d, nnr=%d\n", npw, nnr);
    fflush(stdout);

    // 缓存 Miller 指数
    cache_miller_indices(miller_h, miller_k, miller_l);

    // 1. 清零输出
    CHECK(cudaMemsetAsync(psi_r_packed.data(), 0, nnr * sizeof(gpufftComplex), grid_.stream()));

    // 2. 打包两个波函数
    const int block_size = 256;
    const int grid_size = (npw + block_size - 1) / block_size;

    pack_two_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        npw, nr[0], nr[1], nr[2], d_miller_h_, d_miller_k_, d_miller_l_, psi1_g.data(),
        psi2_g.data(), psi_r_packed.data());
    GPU_CHECK_KERNEL;
    grid_.synchronize();  // Ensure packing completes before FFT

    // Debug: check packed result before FFT
    printf("[wave_g2r_pair] After packing, checking first few packed G values...\n");
    gpufftComplex h_packed[10];
    CHECK(cudaMemcpy(h_packed, psi_r_packed.data(), 10 * sizeof(gpufftComplex),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("  packed[%d] = (%.6f, %.6f)\n", i, h_packed[i].x, h_packed[i].y);
    }
    fflush(stdout);

    // 3. IFFT (cuFFT INVERSE)
    CUFFT_CHECK(cufftExecZ2Z(plan_z2z_, (cufftDoubleComplex*)psi_r_packed.data(),
                             (cufftDoubleComplex*)psi_r_packed.data(), CUFFT_INVERSE));
    GPU_CHECK_KERNEL;

    // 4. Normalize by 1/N (QE convention: invfft includes 1/N normalization)
    // 注意: cuFFT 的 CUFFT_INVERSE 不包含归一化，需要手动除以 N
    const int bs_norm = 256;
    const int gs_norm = (nnr + bs_norm - 1) / bs_norm;
    scale_complex_kernel<<<gs_norm, bs_norm, 0, grid_.stream()>>>(nnr, psi_r_packed.data(),
                                                                  1.0 / (double)nnr);
    GPU_CHECK_KERNEL;
    grid_.synchronize();

    // Debug: check result after FFT
    printf("[wave_g2r_pair] After FFT, checking first few r-space values...\n");
    gpufftComplex h_r[10];
    CHECK(cudaMemcpy(h_r, psi_r_packed.data(), 10 * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("  psi_r[%d] = (%.6f, %.6f)\n", i, h_r[i].x, h_r[i].y);
    }
    fflush(stdout);
}

void GammaFFTSolver::wave_g2r_pair_compact(const gpufftComplex* psi1_smooth,
                                           const gpufftComplex* psi2_smooth,
                                           const std::vector<int>& miller_h,
                                           const std::vector<int>& miller_k,
                                           const std::vector<int>& miller_l,
                                           ComplexField& psi_r_packed) {
    int npw = miller_h.size();
    int nnr = grid_.nnr();
    const int* nr = grid_.nr();

    printf("[GammaFFTSolver::wave_g2r_pair_compact] Called with npw=%d, nnr=%d\n", npw, nnr);
    fflush(stdout);

    // 缓存 Miller 指数
    cache_miller_indices(miller_h, miller_k, miller_l);

    // 1. 清零输出
    CHECK(cudaMemsetAsync(psi_r_packed.data(), 0, nnr * sizeof(gpufftComplex), grid_.stream()));

    // 2. 打包两个波函数（输入是紧凑数组）
    const int block_size = 256;
    const int grid_size = (npw + block_size - 1) / block_size;

    pack_two_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        npw, nr[0], nr[1], nr[2], d_miller_h_, d_miller_k_, d_miller_l_, psi1_smooth, psi2_smooth,
        psi_r_packed.data());
    GPU_CHECK_KERNEL;
    grid_.synchronize();  // Ensure packing completes before FFT

    // Debug: check packed result before FFT
    printf("[wave_g2r_pair_compact] After packing, checking first few packed G values...\n");
    gpufftComplex h_packed[10];
    CHECK(cudaMemcpy(h_packed, psi_r_packed.data(), 10 * sizeof(gpufftComplex),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("  packed[%d] = (%.6f, %.6f)\n", i, h_packed[i].x, h_packed[i].y);
    }
    fflush(stdout);

    // 3. IFFT (cuFFT INVERSE)
    CUFFT_CHECK(cufftExecZ2Z(plan_z2z_, (cufftDoubleComplex*)psi_r_packed.data(),
                             (cufftDoubleComplex*)psi_r_packed.data(), CUFFT_INVERSE));
    GPU_CHECK_KERNEL;

    // 4. Normalize by 1/N (QE convention: invfft includes 1/N normalization)
    const int bs_norm = 256;
    const int gs_norm = (nnr + bs_norm - 1) / bs_norm;
    scale_complex_kernel<<<gs_norm, bs_norm, 0, grid_.stream()>>>(nnr, psi_r_packed.data(),
                                                                  1.0 / (double)nnr);
    GPU_CHECK_KERNEL;
    grid_.synchronize();

    // Debug: check result after FFT
    printf("[wave_g2r_pair_compact] After FFT, checking first few r-space values...\n");
    gpufftComplex h_r[10];
    CHECK(cudaMemcpy(h_r, psi_r_packed.data(), 10 * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("  psi_r[%d] = (%.6f, %.6f)\n", i, h_r[i].x, h_r[i].y);
    }
    fflush(stdout);
}

void GammaFFTSolver::wave_r2g_pair(const ComplexField& psi_r_packed,
                                   const std::vector<int>& miller_h,
                                   const std::vector<int>& miller_k,
                                   const std::vector<int>& miller_l, ComplexField& psi1_g,
                                   ComplexField& psi2_g) {
    int npw = miller_h.size();
    int nnr = grid_.nnr();
    const int* nr = grid_.nr();

    // 缓存 Miller 指数
    cache_miller_indices(miller_h, miller_k, miller_l);

    // 1. 创建临时缓冲区并 FFT
    ComplexField temp(grid_);
    CHECK(cudaMemcpyAsync(temp.data(), psi_r_packed.data(), nnr * sizeof(gpufftComplex),
                          cudaMemcpyDeviceToDevice, grid_.stream()));

    // 1. FFT (R→G, cuFFT FORWARD, 不归一化 - 匹配 QE fwfft 约定)
    CUFFT_CHECK(cufftExecZ2Z(plan_z2z_, (cufftDoubleComplex*)temp.data(),
                             (cufftDoubleComplex*)temp.data(), CUFFT_FORWARD));
    GPU_CHECK_KERNEL;

    // 注意：根据 QE fft_scalar.f90 约定
    // - invfft (G→R): FFTW_BACKWARD + 归一化 1/N
    // - fwfft (R→G): FFTW_FORWARD，不归一化
    // 因此 wave_r2g 不需要归一化

    // 2. 解包两个波函数
    const int block_size = 256;
    const int grid_size = (npw + block_size - 1) / block_size;

    unpack_two_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        npw, nr[0], nr[1], nr[2], d_miller_h_, d_miller_k_, d_miller_l_, temp.data(), psi1_g.data(),
        psi2_g.data());
    GPU_CHECK_KERNEL;
}

}  // namespace dftcu
