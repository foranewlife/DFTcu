#include "solver/davidson.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dftcu {

DavidsonSolver::DavidsonSolver(Grid& grid, int max_iter, double tol)
    : grid_(grid), max_iter_(max_iter), tol_(tol) {
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle_));
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

    // ZGEMM: C = alpha*A*B + beta*C
    // A: psi [n, nbands], B: eigenvectors [nbands, nbands]
    CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, nbands, nbands, &alpha,
                                 (const cuDoubleComplex*)psi.data(), n,
                                 (const cuDoubleComplex*)eigenvectors.data(), nbands, &beta,
                                 (cuDoubleComplex*)tmp_psi.data(), n));

    CHECK(cudaMemcpyAsync(psi.data(), tmp_psi.data(), n * nbands * sizeof(gpufftComplex),
                          cudaMemcpyDeviceToDevice, grid_.stream()));
}

void DavidsonSolver::orthogonalize(Wavefunction& psi) {
    int nbands = psi.num_bands();
    size_t n = grid_.nnr();
    if (nbands == 0)
        return;

    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));

    // 1. Compute overlap matrix S = Psi^H * Psi
    GPU_Vector<gpufftComplex> s_matrix(nbands * nbands);
    cuDoubleComplex alpha = {1.0, 0.0};
    cuDoubleComplex beta = {0.0, 0.0};

    CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, n, &alpha,
                                 (const cuDoubleComplex*)psi.data(), n,
                                 (const cuDoubleComplex*)psi.data(), n, &beta,
                                 (cuDoubleComplex*)s_matrix.data(), nbands));

    // 2. For single band, just normalize
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

    // 3. Multi-band: Cholesky-based Lowdin
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnZpotrf_bufferSize(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, nbands,
                                               (cuDoubleComplex*)s_matrix.data(), nbands, &lwork));
    GPU_Vector<gpufftComplex> work(lwork);
    GPU_Vector<int> dev_info(1);

    CHECK_CUSOLVER(cusolverDnZpotrf(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, nbands,
                                    (cuDoubleComplex*)s_matrix.data(), nbands,
                                    (cuDoubleComplex*)work.data(), lwork, dev_info.data()));

    CUBLAS_SAFE_CALL(cublasZtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, nbands, &alpha,
                                 (const cuDoubleComplex*)s_matrix.data(), nbands,
                                 (cuDoubleComplex*)psi.data(), n));
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

        CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, n, &alpha,
                                     (const cuDoubleComplex*)psi.data(), n,
                                     (const cuDoubleComplex*)h_psi.data(), n, &beta,
                                     (cuDoubleComplex*)h_matrix_.data(), nbands));

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
