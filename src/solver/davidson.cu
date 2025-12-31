#include "solver/davidson.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dftcu {

namespace {

// Preconditioned Residual Update
__global__ void apply_preconditioned_residual_kernel(size_t n, int nbands,
                                                     const gpufftComplex* h_psi,
                                                     const double* eigenvalues,
                                                     const gpufftComplex* psi, const double* gg,
                                                     gpufftComplex* out, double step_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * nbands) {
        int ig = idx % n;
        int iband = idx / n;
        double eps = eigenvalues[iband];

        double r_re = h_psi[idx].x - eps * psi[idx].x;
        double r_im = h_psi[idx].y - eps * psi[idx].y;

        const double BOHR_TO_ANGSTROM = 0.529177210903;
        double g2 = gg[ig] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

        // Simple Preconditioner
        double precond = 1.0 / (1.0 + g2);

        out[idx].x = psi[idx].x - step_size * precond * r_re;
        out[idx].y = psi[idx].y - step_size * precond * r_im;
    }
}

}  // namespace

DavidsonSolver::DavidsonSolver(Grid& grid, int max_iter, double tol)
    : grid_(grid), max_iter_(max_iter), tol_(tol) {
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle_));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle_, grid_.stream()));
}

DavidsonSolver::~DavidsonSolver() {
    if (cusolver_handle_)
        cusolverDnDestroy(cusolver_handle_);
}

void DavidsonSolver::rotate_subspace(Wavefunction& psi,
                                     const GPU_Vector<gpufftComplex>& eigenvectors) {
    int nbands = psi.num_bands();
    size_t n = grid_.nnr();
    GPU_Vector<gpufftComplex> tmp_psi(n * nbands);
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));
    cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
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
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));
    cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    {
        CublasPointerModeGuard guard(handle, CUBLAS_POINTER_MODE_HOST);
        CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, (int)n,
                                     &alpha, (const cuDoubleComplex*)psi.data(), (int)n,
                                     (const cuDoubleComplex*)psi.data(), (int)n, &beta,
                                     (cuDoubleComplex*)s_matrix.data(), nbands));
    }
    if (nbands == 1) {
        gpufftComplex h_s;
        CHECK(cudaMemcpyAsync(&h_s, s_matrix.data(), sizeof(gpufftComplex), cudaMemcpyDeviceToHost,
                              grid_.stream()));
        grid_.synchronize();
        double norm = std::sqrt(h_s.x);
        if (norm > 1e-18) {
            double inv_norm = 1.0 / norm;
            v_scale(n * 2, inv_norm, (double*)psi.data(), (double*)psi.data(), grid_.stream());
        }
        return;
    }
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnZpotrf_bufferSize(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, nbands,
                                               (cuDoubleComplex*)s_matrix.data(), nbands, &lwork));
    GPU_Vector<gpufftComplex> work(lwork);
    GPU_Vector<int> dev_info(1);
    CHECK_CUSOLVER(cusolverDnZpotrf(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, nbands,
                                    (cuDoubleComplex*)s_matrix.data(), nbands,
                                    (cuDoubleComplex*)work.data(), lwork, dev_info.data()));
    {
        CublasPointerModeGuard guard(handle, CUBLAS_POINTER_MODE_HOST);
        CUBLAS_SAFE_CALL(cublasZtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C,
                                     CUBLAS_DIAG_NON_UNIT, (int)n, nbands, &alpha,
                                     (const cuDoubleComplex*)s_matrix.data(), nbands,
                                     (cuDoubleComplex*)psi.data(), (int)n));
    }
}

std::vector<double> DavidsonSolver::solve(Hamiltonian& ham, Wavefunction& psi) {
    int nbands = psi.num_bands();
    size_t n = grid_.nnr();
    Wavefunction h_psi(grid_, nbands, psi.encut());
    std::vector<double> eigenvalues(nbands, 0.0);
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));
    for (int iter = 0; iter < max_iter_; ++iter) {
        orthogonalize(psi);
        ham.apply(psi, h_psi);
        h_matrix_.resize(nbands * nbands);
        cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
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
        if (iter < max_iter_ - 1) {
            ham.apply(psi, h_psi);
            apply_preconditioned_residual_kernel<<<(n * nbands + 255) / 256, 256, 0,
                                                   grid_.stream()>>>(
                n, nbands, h_psi.data(), eval_buffer_.data(), psi.data(), grid_.gg(), psi.data(),
                0.2);
            GPU_CHECK_KERNEL;
        }
    }
    return eigenvalues;
}

}  // namespace dftcu
