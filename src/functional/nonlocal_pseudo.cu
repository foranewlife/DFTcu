#include "functional/nonlocal_pseudo.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <cublas_v2.h>

namespace dftcu {

namespace {

// Kernel to apply D_i coupling constants to projections
__global__ void scale_projections_kernel(int num_proj, int num_bands, const double* d_coupling,
                                         gpufftComplex* projections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_proj * num_bands) {
        int proj_idx = i % num_proj;
        projections[i].x *= d_coupling[proj_idx];
        projections[i].y *= d_coupling[proj_idx];
    }
}

}  // namespace

NonLocalPseudo::NonLocalPseudo(Grid& grid) : grid_(grid) {}

void NonLocalPseudo::clear() {
    num_projectors_ = 0;
    d_projectors_.resize(0);
    d_coupling_.resize(0);
}

void NonLocalPseudo::add_projector(const std::vector<std::complex<double>>& beta_g,
                                   double coupling_constant) {
    size_t n = grid_.nnr();
    if (beta_g.size() != n) {
        throw std::runtime_error("Projector size mismatch with grid");
    }

    int old_num = num_projectors_;
    num_projectors_++;

    // Resize and keep old data
    GPU_Vector<gpufftComplex> next_projectors(num_projectors_ * n);
    if (old_num > 0) {
        CHECK(cudaMemcpy(next_projectors.data(), d_projectors_.data(),
                         old_num * n * sizeof(gpufftComplex), cudaMemcpyDeviceToDevice));
    }
    CHECK(cudaMemcpy(next_projectors.data() + old_num * n, beta_g.data(), n * sizeof(gpufftComplex),
                     cudaMemcpyHostToDevice));
    d_projectors_ = std::move(next_projectors);

    // Update coupling constants
    std::vector<double> h_coupling(num_projectors_);
    if (old_num > 0) {
        d_coupling_.copy_to_host(h_coupling.data());
    }
    h_coupling[old_num] = coupling_constant;
    d_coupling_.resize(num_projectors_);
    d_coupling_.copy_from_host(h_coupling.data());
}

void NonLocalPseudo::apply(Wavefunction& psi_in, Wavefunction& h_psi_out) {
    if (num_projectors_ == 0)
        return;

    size_t n = grid_.nnr();
    int nbands = psi_in.num_bands();

    // Ensure projection buffer is ready
    d_projections_.resize(num_projectors_ * nbands);

    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));

    // 1. Calculate projections: P = B^H * Psi
    // B is [nbands, n], we need [num_proj, n]
    // cublasZgemm(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // Here: M=num_proj, N=nbands, K=n
    cuDoubleComplex alpha = {1.0, 0.0};
    cuDoubleComplex beta = {0.0, 0.0};

    // Note: Projectors are stored row-major in our head [proj][G],
    // so in cuBLAS (column-major) it's [G][proj].
    // Psi is stored [band][G], in cuBLAS it's [G][band].
    // Result P should be [num_proj][nbands].
    CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, num_projectors_, nbands, n,
                                 &alpha, (const cuDoubleComplex*)d_projectors_.data(), n,
                                 (const cuDoubleComplex*)psi_in.data(), n, &beta,
                                 (cuDoubleComplex*)d_projections_.data(), num_projectors_));

    // 2. Scale by coupling constants D_i
    const int block_size = 256;
    const int grid_size = (num_projectors_ * nbands + block_size - 1) / block_size;
    scale_projections_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        num_projectors_, nbands, d_coupling_.data(), d_projections_.data());
    GPU_CHECK_KERNEL;

    // 3. Accumulate back: h_psi_out += B * P_scaled
    // M=n, N=nbands, K=num_proj
    // alpha = 1.0 (addition)
    cuDoubleComplex alpha_add = {1.0, 0.0};
    cuDoubleComplex beta_keep = {1.0, 0.0};  // Add to existing data in h_psi_out

    CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, nbands, num_projectors_,
                                 &alpha_add, (const cuDoubleComplex*)d_projectors_.data(), n,
                                 (const cuDoubleComplex*)d_projections_.data(), num_projectors_,
                                 &beta_keep, (cuDoubleComplex*)h_psi_out.data(), n));

    grid_.synchronize();
}

}  // namespace dftcu
