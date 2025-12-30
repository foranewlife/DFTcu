#include "solver/davidson.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dftcu {

namespace {

__global__ void manual_overlap_kernel(int nbands, int n, const gpufftComplex* psi,
                                      gpufftComplex* s_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbands * nbands) {
        int i = idx % nbands;
        int j = idx / nbands;
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int k = 0; k < n; ++k) {
            gpufftComplex psi_i = psi[i * n + k];
            gpufftComplex psi_j = psi[j * n + k];
            sum_x += psi_i.x * psi_j.x + psi_i.y * psi_j.y;
            sum_y += psi_i.x * psi_j.y - psi_i.y * psi_j.x;
        }
        s_matrix[idx].x = sum_x;
        s_matrix[idx].y = sum_y;
    }
}

}  // namespace

DavidsonSolver::DavidsonSolver(Grid& grid, int max_iter, double tol)
    : grid_(grid), max_iter_(max_iter), tol_(tol) {
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle_));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle_, grid_.stream()));
}

DavidsonSolver::~DavidsonSolver() {
    if (cusolver_handle_) {
        cusolverDnDestroy(cusolver_handle_);
    }
}

void DavidsonSolver::rotate_subspace(Wavefunction& psi,
                                     const GPU_Vector<gpufftComplex>& eigenvectors) {
    int nbands = psi.num_bands();
    size_t n = grid_.nnr();

    GPU_Vector<gpufftComplex> tmp_psi(n * nbands);
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));

    cuDoubleComplex alpha = {1.0, 0.0};
    cuDoubleComplex beta = {0.0, 0.0};

    {
        CublasPointerModeGuard guard(handle, CUBLAS_POINTER_MODE_HOST);
        CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, nbands, nbands,
                                     &alpha, (const cuDoubleComplex*)psi.data(), (int)n,
                                     (const cuDoubleComplex*)eigenvectors.data(), nbands, &beta,
                                     (cuDoubleComplex*)tmp_psi.data(), (int)n));
    }

    CHECK(cudaMemcpyAsync(psi.data(), tmp_psi.data(), n * nbands * sizeof(gpufftComplex),
                          cudaMemcpyDeviceToDevice, grid_.stream()));
}

void DavidsonSolver::orthogonalize(Wavefunction& psi) {
    int nbands = psi.num_bands();
    size_t n = grid_.nnr();
    if (nbands == 0)
        return;

    GPU_Vector<gpufftComplex> s_matrix(nbands * nbands);

    const int block_size_s = 64;
    const int grid_size_s = (nbands * nbands + block_size_s - 1) / block_size_s;
    manual_overlap_kernel<<<grid_size_s, block_size_s, 0, grid_.stream()>>>(
        nbands, (int)n, psi.data(), s_matrix.data());

    CHECK(cudaStreamSynchronize(grid_.stream()));

    // Check CUDA errors before proceeding
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("    ERROR: CUDA error before Cholesky: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA error before Cholesky decomposition");
    }

    if (nbands == 1) {
        gpufftComplex h_s;
        CHECK(cudaMemcpy(&h_s, s_matrix.data(), sizeof(gpufftComplex), cudaMemcpyDeviceToHost));
        double norm = std::sqrt(h_s.x);
        if (norm > 1e-18) {
            double inv_norm = 1.0 / norm;
            v_scale(n * 2, inv_norm, (double*)psi.data(), (double*)psi.data(), grid_.stream());
        }
        return;
    }

    int lwork = 0;
    cusolverStatus_t status =
        cusolverDnZpotrf_bufferSize(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, nbands,
                                    (cuDoubleComplex*)s_matrix.data(), nbands, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("    ERROR: cusolverDnZpotrf_bufferSize failed with status %d\n", status);
        throw std::runtime_error("cusolverDnZpotrf_bufferSize failed");
    }

    if (lwork <= 0) {
        printf("    ERROR: Invalid lwork=%d\n", lwork);
        throw std::runtime_error("Invalid workspace size from cusolverDnZpotrf_bufferSize");
    }

    GPU_Vector<gpufftComplex> work(lwork);
    GPU_Vector<int> dev_info(1);

    // Ensure cuSOLVER uses the correct stream
    CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle_, grid_.stream()));

    CHECK_CUSOLVER(cusolverDnZpotrf(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, nbands,
                                    (cuDoubleComplex*)s_matrix.data(), nbands,
                                    (cuDoubleComplex*)work.data(), lwork, dev_info.data()));

    // CRITICAL: Synchronize stream before using Cholesky result
    CHECK(cudaStreamSynchronize(grid_.stream()));

    // Check if Cholesky decomposition succeeded
    int h_info = 0;
    CHECK(cudaMemcpy(&h_info, dev_info.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        printf("    ERROR: Cholesky decomposition failed with info=%d\n", h_info);
        if (h_info > 0) {
            printf("    Matrix is not positive definite (leading minor %d is not positive)\n",
                   h_info);
        }
        throw std::runtime_error("Cholesky decomposition failed in orthogonalize");
    }

    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));
    cuDoubleComplex alpha = {1.0, 0.0};

    // Solve psi * L = psi_in  =>  psi = psi_in * L^{-1}
    // where S = L * L^* from Cholesky decomposition
    // Use HOST pointer mode since alpha is a host variable
    {
        CublasPointerModeGuard guard(handle, CUBLAS_POINTER_MODE_HOST);
        cublasStatus_t cublas_status = cublasZtrsm(
            handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            (int)n, nbands, &alpha, (const cuDoubleComplex*)s_matrix.data(), nbands,
            (cuDoubleComplex*)psi.data(), (int)n);

        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("    ERROR: cublasZtrsm failed with status %d\n", cublas_status);
            throw std::runtime_error("cublasZtrsm failed");
        }
    }

    // Synchronize and check for errors
    CHECK(cudaStreamSynchronize(grid_.stream()));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("    ERROR: CUDA error after cublasZtrsm: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA error after cublasZtrsm");
    }
}

std::vector<double> DavidsonSolver::solve(Hamiltonian& ham, Wavefunction& psi) {
    int nbands = psi.num_bands();
    size_t n = grid_.nnr();
    Wavefunction h_psi(grid_, nbands, psi.encut());
    std::vector<double> eigenvalues(nbands, 0.0);

    for (int iter = 0; iter < max_iter_; ++iter) {
        orthogonalize(psi);
        ham.apply(psi, h_psi);

        h_matrix_.resize(nbands * nbands);
        cublasHandle_t handle = CublasManager::instance().handle();
        CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));

        cuDoubleComplex alpha = {1.0, 0.0};
        cuDoubleComplex beta = {0.0, 0.0};

        {
            CublasPointerModeGuard guard(handle, CUBLAS_POINTER_MODE_HOST);
            CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, (int)n,
                                         &alpha, (const cuDoubleComplex*)psi.data(), (int)n,
                                         (const cuDoubleComplex*)h_psi.data(), (int)n, &beta,
                                         (cuDoubleComplex*)h_matrix_.data(), nbands));
        }

        eval_buffer_.resize(nbands);
        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnZheevd_bufferSize(
            cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, nbands,
            (cuDoubleComplex*)h_matrix_.data(), nbands, eval_buffer_.data(), &lwork));
        GPU_Vector<gpufftComplex> work(lwork);
        GPU_Vector<int> dev_info(1);

        CHECK_CUSOLVER(cusolverDnZheevd(
            cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, nbands,
            (cuDoubleComplex*)h_matrix_.data(), nbands, eval_buffer_.data(),
            (cuDoubleComplex*)work.data(), lwork, dev_info.data()));

        rotate_subspace(psi, h_matrix_);
        eval_buffer_.copy_to_host(eigenvalues.data());
    }

    return eigenvalues;
}

}  // namespace dftcu
