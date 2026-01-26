#include "solver/gamma_utils.cuh"
#include "solver/hamiltonian.cuh"
#include "solver/subspace_solver.cuh"
#include "utilities/cublas_manager.cuh"
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

void SubspaceSolver::solve_generalized_gamma(int nbands, double* h_matrix, double* s_matrix,
                                             double* eigenvalues, double* eigenvectors) {
    /*
     * Solve the generalized eigenvalue problem for Gamma-only (real symmetric matrices):
     *   H * c = epsilon * S * c
     *
     * This uses cusolverDnDsygvd which matches QE's DSYGVD call in regterg.f90:234.
     *
     * QE reference:
     *   CALL DSYGVD(1, 'V', 'U', nbase, hr, nvecx, sr, nvecx, e, work, lwork, iwork, liwork, info)
     *
     * Key differences from complex version:
     * - Uses DSYGVD (real symmetric) instead of ZHEGVD (complex Hermitian)
     * - Matrix storage is half the size (real vs complex)
     * - Numerical precision should be identical to QE at 1e-15 level
     */

    if (nbands <= 0)
        return;

    int lwork = 0;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;  // A*x = lambda*B*x

    // Query workspace size
    CHECK_CUSOLVER(cusolverDnDsygvd_bufferSize(handle_, itype, jobz, uplo, nbands, h_matrix, nbands,
                                               s_matrix, nbands, eigenvalues, &lwork));

    GPU_Vector<double> work(lwork);
    GPU_Vector<int> dev_info(1);

    // Solve: H * c = epsilon * S * c
    // Note: h_matrix will be overwritten by eigenvectors
    CHECK_CUSOLVER(cusolverDnDsygvd(handle_, itype, jobz, uplo, nbands, h_matrix, nbands, s_matrix,
                                    nbands, eigenvalues, work.data(), lwork, dev_info.data()));

    // Check for errors
    int h_info = 0;
    dev_info.copy_to_host(&h_info);
    if (h_info != 0) {
        if (h_info < 0) {
            std::cerr << "Error: " << -h_info << "-th parameter is wrong in Dsygvd" << std::endl;
        } else if (h_info <= nbands) {
            std::cerr << "Error: Leading minor of order " << h_info
                      << " of B is not positive definite (Dsygvd)" << std::endl;
        } else {
            std::cerr << "Error: Dsygvd failed to converge (info=" << h_info << ")" << std::endl;
        }
    }

    // Copy eigenvectors to output if provided and different from h_matrix
    if (eigenvectors && eigenvectors != h_matrix) {
        CHECK(cudaMemcpyAsync(eigenvectors, h_matrix, (size_t)nbands * nbands * sizeof(double),
                              cudaMemcpyDeviceToDevice, grid_.stream()));
    }
}

std::vector<double> SubspaceSolver::solve_direct(Hamiltonian& ham, Wavefunction& psi) {
    int nbands = psi.num_bands();
    int npw = psi.num_pw();  // Smooth grid size (e.g., 85 for Si Gamma)

    printf("DEBUG solve_direct: nbands=%d, npw=%d, encut=%.6f\n", nbands, npw, psi.encut());

    // 1. Compute H|psi>
    Wavefunction h_psi(grid_, nbands, psi.encut());
    printf("DEBUG solve_direct: applying ham...\n");
    ham.apply(psi, h_psi);
    grid_.synchronize();
    printf("DEBUG solve_direct: ham applied.\n");

    // 2. Build subspace matrices using Gamma-only optimized functions
    // These use real symmetric matrices and include factor 2 + G=0 correction
    GPU_Vector<double> h_matrix(nbands * nbands);
    GPU_Vector<double> s_matrix(nbands * nbands);
    GPU_Vector<double> eigenvalues_gpu(nbands);

    int lda = grid_.nnr();  // leading dimension = nnr (FFT grid size)
    int gstart = 2;         // Gamma-only, G=0 exists (Fortran 1-based indexing)

    // 获取 nl_d 映射（从 FFT grid 提取有效 G-vectors）
    const int* nl_d = grid_.nl_d();
    printf("DEBUG solve_direct: computing subspaces...\n");

    // Compute H_sub = <psi|H|psi> using Gamma-only optimization
    // Reference: QE regterg.f90:241-257
    compute_h_subspace_gamma(npw, nbands, gstart, psi.data(), lda, h_psi.data(), lda,
                             h_matrix.data(), nbands, nl_d, grid_.stream());

    // Compute S_sub = <psi|psi> using Gamma-only optimization
    compute_s_subspace_gamma(npw, nbands, gstart, psi.data(), lda, s_matrix.data(), nbands, nl_d,
                             grid_.stream());

    printf("DEBUG solve_direct: subspaces computed. solving...\n");

    // 3. Solve generalized eigenvalue problem using real symmetric solver
    // This matches QE's DSYGVD call in regterg.f90:234
    solve_generalized_gamma(nbands, h_matrix.data(), s_matrix.data(), eigenvalues_gpu.data(),
                            nullptr);
    grid_.synchronize();
    printf("DEBUG solve_direct: generalized problem solved.\n");

    // 4. Return results
    std::vector<double> eigenvalues(nbands);
    eigenvalues_gpu.copy_to_host(eigenvalues.data());

    // 5. Apply scalar G=0 shift (QE style)
    double v0 = ham.get_v_of_0();
    for (auto& e : eigenvalues) {
        e += v0;
    }

    return eigenvalues;
}

}  // namespace dftcu
