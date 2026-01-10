#include "solver/davidson.cuh"
#include "solver/gamma_utils.cuh"
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

        // gg is already in Bohr^-2 (no conversion needed)
        double g2 = gg[ig];

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

    // Use REAL symmetric matrices for Gamma-only subspace
    GPU_Vector<double> h_sub(nbands * nbands);
    GPU_Vector<double> s_sub(nbands * nbands);
    GPU_Vector<double> d_evals(nbands);

    for (int iter = 0; iter < max_iter_; ++iter) {
        // Ensure psi is clean (mask + gamma constraint)
        psi.apply_mask();
        psi.force_gamma_constraint();

        ham.apply(psi, h_psi);

        // Form H_sub and S_sub using Gamma-only optimized functions (REAL symmetric)
        // These handle the G=0 correction and 2.0 factor correctly.
        compute_h_subspace_gamma(n, nbands, 2, psi.data(), n, h_psi.data(), n, h_sub.data(), nbands,
                                 grid_.stream());
        compute_s_subspace_gamma(n, nbands, 2, psi.data(), n, s_sub.data(), nbands, grid_.stream());

        // Solve H_sub * c = epsilon * S_sub * c (REAL symmetric)
        // eigenvectors are returned in h_sub

        // Debug print H_sub diagonal
        std::vector<double> h_diag(nbands * nbands);
        h_sub.copy_to_host(h_diag.data(), grid_.stream());
        grid_.synchronize();
        printf("DEBUG Davidson: iter=%d, H_sub diag[0]=%f, [1]=%f, [2]=%f\n", iter, h_diag[0],
               h_diag[nbands + 1], h_diag[2 * nbands + 2]);

        subspace_solver_->solve_generalized_gamma(nbands, h_sub.data(), s_sub.data(),
                                                  d_evals.data(), h_sub.data());

        // Subspace Rotation: psi' = psi * c
        // Since c is REAL, we need a real-complex GEMM or promote c to complex.
        // For simplicity, we promote c (in h_sub) to complex.
        GPU_Vector<gpufftComplex> c_complex(nbands * nbands);
        std::vector<double> h_sub_host(nbands * nbands);
        h_sub.copy_to_host(h_sub_host.data(), grid_.stream());
        std::vector<gpufftComplex> c_host(nbands * nbands);
        for (int i = 0; i < nbands * nbands; ++i)
            c_host[i] = {h_sub_host[i], 0.0};
        c_complex.copy_from_host(c_host.data(), grid_.stream());

        rotate_subspace(psi, c_complex);

        d_evals.copy_to_host(eigenvalues.data());
        double v0 = ham.get_v_of_0();
        for (auto& e : eigenvalues) {
            e += v0;
        }

        if (iter == 0 && max_iter_ == 1)
            break;

        if (iter < max_iter_ - 1) {
            ham.apply(psi, h_psi);
            apply_preconditioned_residual_kernel<<<(n * nbands + 255) / 256, 256, 0,
                                                   grid_.stream()>>>(
                n, nbands, h_psi.data(), d_evals.data(), psi.data(), grid_.gg(), psi.data(), 0.2);
            GPU_CHECK_KERNEL;

            // IMPORTANT: Orthonormalize after preconditioned update
            psi.orthonormalize();
        }
    }
    return eigenvalues;
}

}  // namespace dftcu
