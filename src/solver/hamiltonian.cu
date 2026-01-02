#include "solver/hamiltonian.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

__global__ void apply_kinetic_kernel(size_t n, int num_bands, const double* gg,
                                     const gpufftComplex* psi, gpufftComplex* h_psi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = n * num_bands;
    if (i < total_size) {
        int grid_idx = i % n;
        const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
        double g2_bohr = gg[grid_idx] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
        double t = 0.5 * g2_bohr;
        h_psi[i].x = t * psi[i].x;
        h_psi[i].y = t * psi[i].y;
    }
}

__global__ void apply_vloc_kernel(size_t n, const double* v_loc, gpufftComplex* psi_r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        psi_r[i].x *= v_loc[i];
        psi_r[i].y *= v_loc[i];
    }
}

__global__ void accumulate_hpsi_kernel(size_t n, const gpufftComplex* tmp, gpufftComplex* h_psi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        h_psi[i].x += tmp[i].x;
        h_psi[i].y += tmp[i].y;
    }
}

}  // namespace

Hamiltonian::Hamiltonian(Grid& grid, Evaluator& evaluator,
                         std::shared_ptr<NonLocalPseudo> nl_pseudo)
    : grid_(grid), evaluator_(evaluator), nonlocal_(nl_pseudo), v_loc_tot_(grid) {
    v_loc_tot_.fill(0.0);
}

void Hamiltonian::update_potentials(const RealField& rho) {
    evaluator_.compute(rho, v_loc_tot_);
    grid_.synchronize();
}

void Hamiltonian::set_ecutrho(double ecutrho) {}

void Hamiltonian::apply(Wavefunction& psi, Wavefunction& h_psi) {
    size_t n = grid_.nnr();
    int nbands = psi.num_bands();

    cudaMemsetAsync(h_psi.data(), 0, n * nbands * sizeof(gpufftComplex), grid_.stream());

    // 1. Kinetic Term (Reciprocal Space)
    const int block_size = 256;
    const int grid_size_k = (n * nbands + block_size - 1) / block_size;
    apply_kinetic_kernel<<<grid_size_k, block_size, 0, grid_.stream()>>>(n, nbands, grid_.gg(),
                                                                         psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;

    // 2. Local Potential Term (Real Space)
    FFTSolver fft(grid_);
    ComplexField tmp_g(grid_);
    const int grid_size_v = (n + block_size - 1) / block_size;

    for (int nb = 0; nb < nbands; ++nb) {
        CHECK(cudaMemcpyAsync(tmp_g.data(), psi.band_data(nb), n * sizeof(gpufftComplex),
                              cudaMemcpyDeviceToDevice, grid_.stream()));

        // G -> R (scaled by N by cufftExecZ2Z CUFFT_INVERSE)
        fft.backward(tmp_g);

        // psi(r)*N * V(r)_phys
        apply_vloc_kernel<<<grid_size_v, block_size, 0, grid_.stream()>>>(n, v_loc_tot_.data(),
                                                                          tmp_g.data());
        GPU_CHECK_KERNEL;

        // R -> G (scaled by 1/N by FFTSolver::forward)
        // 1/N * FFT(V_phys * psi_r * N) = (V*psi)_G
        fft.forward(tmp_g);

        accumulate_hpsi_kernel<<<grid_size_v, block_size, 0, grid_.stream()>>>(n, tmp_g.data(),
                                                                               h_psi.band_data(nb));
        GPU_CHECK_KERNEL;
    }

    // 3. Non-local Potential Term
    if (nonlocal_) {
        nonlocal_->apply(psi, h_psi);
    }

    h_psi.apply_mask();
    grid_.synchronize();
}

}  // namespace dftcu
