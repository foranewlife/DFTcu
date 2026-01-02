#include "solver/subspace_solver.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

SubspaceSolver::SubspaceSolver(Grid& grid) : grid_(grid) {
    CHECK_CUSOLVER(cusolverDnCreate(&handle_));
    CHECK_CUSOLVER(cusolverDnSetStream(handle_, grid_.stream()));
}

SubspaceSolver::~SubspaceSolver() {
    if (handle_)
        cusolverDnDestroy(handle_);
}

void SubspaceSolver::solve_generalized(int nbands, gpufftComplex* h_matrix, gpufftComplex* s_matrix,
                                       double* eigenvalues, gpufftComplex* eigenvectors) {
    if (nbands <= 0)
        return;

    // We use cusolverDnZhegvd which solves the generalized Hermitian-definite
    // eigenvalue problem: A*x = lambda*B*x
    // It is more robust than manual Cholesky + Zheevd.

    int lwork = 0;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    // ITYPE = 1: A*x = lambda*B*x
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;

    CHECK_CUSOLVER(cusolverDnZhegvd_bufferSize(
        handle_, itype, jobz, uplo, nbands, (cuDoubleComplex*)h_matrix, nbands,
        (cuDoubleComplex*)s_matrix, nbands, eigenvalues, &lwork));

    GPU_Vector<gpufftComplex> work(lwork);
    GPU_Vector<int> dev_info(1);

    // Note: h_matrix will be overwritten by eigenvectors if jobz == CUSOLVER_EIG_MODE_VECTOR
    CHECK_CUSOLVER(cusolverDnZhegvd(handle_, itype, jobz, uplo, nbands, (cuDoubleComplex*)h_matrix,
                                    nbands, (cuDoubleComplex*)s_matrix, nbands, eigenvalues,
                                    (cuDoubleComplex*)work.data(), lwork, dev_info.data()));

    int h_info = 0;
    dev_info.copy_to_host(&h_info);
    if (h_info != 0) {
        if (h_info < 0) {
            std::cerr << "Error: " << -h_info << "-th parameter is wrong" << std::endl;
        } else if (h_info <= nbands) {
            std::cerr << "Error: Leading minor of order " << h_info
                      << " of B is not positive definite" << std::endl;
        } else {
            std::cerr << "Error: Zhegvd failed to converge (info=" << h_info << ")" << std::endl;
        }
    }

    // Copy eigenvectors to output if provided and different from h_matrix
    if (eigenvectors && eigenvectors != h_matrix) {
        CHECK(cudaMemcpyAsync(eigenvectors, h_matrix,
                              (size_t)nbands * nbands * sizeof(gpufftComplex),
                              cudaMemcpyDeviceToDevice, grid_.stream()));
    }
}

}  // namespace dftcu
