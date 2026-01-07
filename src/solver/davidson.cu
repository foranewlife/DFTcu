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

        const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
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
    subspace_solver_ = std::make_unique<SubspaceSolver>(grid_);
}

DavidsonSolver::~DavidsonSolver() {
    if (cusolver_handle_)
        cusolverDnDestroy(cusolver_handle_);
}

void DavidsonSolver::rotate_subspace(Wavefunction& psi,
                                     const GPU_Vector<gpufftComplex>& eigenvectors) {
    int nbands = (int)psi.num_bands();
    int n = (int)grid_.nnr();
    if (nbands <= 0)
        return;

    GPU_Vector<gpufftComplex> tmp_psi(n * nbands);
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));
    cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};

    {
        CUBLAS_SAFE_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, nbands, nbands, &alpha,
                                     (const cuDoubleComplex*)psi.data(), n,
                                     (const cuDoubleComplex*)eigenvectors.data(), nbands, &beta,
                                     (cuDoubleComplex*)tmp_psi.data(), n));
        CUBLAS_SAFE_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    }
    CHECK(cudaMemcpyAsync(psi.data(), tmp_psi.data(), n * nbands * sizeof(gpufftComplex),
                          cudaMemcpyDeviceToDevice, grid_.stream()));
}

void DavidsonSolver::orthogonalize(Wavefunction& psi) {
    // This is now handled by the generalized solver in the subspace during solve()
    // Or we could implement it using SubspaceSolver if needed for standalone use.
}

std::vector<double> DavidsonSolver::solve(Hamiltonian& ham, Wavefunction& psi) {
    int nbands = (int)psi.num_bands();
    int n = (int)grid_.nnr();
    if (nbands <= 0)
        return {};

    Wavefunction h_psi(grid_, nbands, psi.encut());
    std::vector<double> eigenvalues(nbands, 0.0);
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, grid_.stream()));

    GPU_Vector<gpufftComplex> h_sub(nbands * nbands);
    GPU_Vector<gpufftComplex> s_sub(nbands * nbands);
    GPU_Vector<double> d_evals(nbands);

    for (int iter = 0; iter < max_iter_; ++iter) {
        ham.apply(psi, h_psi);

        cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};

        // Form H_sub = Psi^H * H * Psi
        {
            CUBLAS_SAFE_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
            CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, n,
                                         &alpha, (const cuDoubleComplex*)psi.data(), n,
                                         (const cuDoubleComplex*)h_psi.data(), n, &beta,
                                         (cuDoubleComplex*)h_sub.data(), nbands));

            // Form S_sub = Psi^H * Psi
            CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, nbands, nbands, n,
                                         &alpha, (const cuDoubleComplex*)psi.data(), n,
                                         (const cuDoubleComplex*)psi.data(), n, &beta,
                                         (cuDoubleComplex*)s_sub.data(), nbands));
            CUBLAS_SAFE_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        }

        // Solve H_sub * c = epsilon * S_sub * c
        subspace_solver_->solve_generalized(nbands, h_sub.data(), s_sub.data(), d_evals.data(),
                                            h_sub.data());

        // Rotate wavefunctions
        rotate_subspace(psi, h_sub);

        d_evals.copy_to_host(eigenvalues.data());
        double v0 = ham.get_v_of_0();
        for (auto& e : eigenvalues) {
            e += v0;
        }

        // For now, this is just a subspace diagonalization (e.g. for initial wfcs).
        // Real Davidson would expand the subspace here.
        if (iter == 0 && max_iter_ == 1)
            break;

        if (iter < max_iter_ - 1) {
            ham.apply(psi, h_psi);
            apply_preconditioned_residual_kernel<<<(n * nbands + 255) / 256, 256, 0,
                                                   grid_.stream()>>>(
                n, nbands, h_psi.data(), d_evals.data(), psi.data(), grid_.gg(), psi.data(), 0.2);
            GPU_CHECK_KERNEL;
        }
    }
    return eigenvalues;
}

}  // namespace dftcu
