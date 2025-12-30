#include "solver/hamiltonian.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

__global__ void shift_potential_kernel(size_t n, double* v, double shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] -= shift;
    }
}

__global__ void apply_kinetic_kernel(size_t n, int num_bands, const double* gg,
                                     const gpufftComplex* psi, gpufftComplex* h_psi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = n * num_bands;
    if (i < total_size) {
        int grid_idx = i % n;
        double t = 0.5 * gg[grid_idx];
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
    // Compute the local potential from all functionals
    double energy = evaluator_.compute(rho, v_loc_tot_);

    // Adjust potential zero-point to match QE convention
    // QE sets the vacuum level (or average potential) as the reference zero
    // This is crucial for correct eigenvalue alignment
    double v_avg = v_loc_tot_.integral() / grid_.volume();

    printf("DEBUG: v_avg = %.10f Ha\n", v_avg);

    // Shift potential to set average to zero
    // This makes eigenvalues relative to the vacuum level
    size_t n = v_loc_tot_.size();
    double* d_v = v_loc_tot_.data();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    shift_potential_kernel<<<blocks, threads>>>(n, d_v, v_avg);
    CHECK(cudaGetLastError());
    grid_.synchronize();

    printf("DEBUG: Potential shifted by %.10f Ha\n", v_avg);
}

void Hamiltonian::apply(Wavefunction& psi, Wavefunction& h_psi) {
    size_t n = grid_.nnr();
    int nbands = psi.num_bands();

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
        fft.backward(tmp_g);

        apply_vloc_kernel<<<grid_size_v, block_size, 0, grid_.stream()>>>(n, v_loc_tot_.data(),
                                                                          tmp_g.data());
        GPU_CHECK_KERNEL;

        fft.forward(tmp_g);
        // Normalize the FFT output to match physical conventions
        v_scale(2 * n, 1.0 / (double)n, (const double*)tmp_g.data(), (double*)tmp_g.data(),
                grid_.stream());

        accumulate_hpsi_kernel<<<grid_size_v, block_size, 0, grid_.stream()>>>(n, tmp_g.data(),
                                                                               h_psi.band_data(nb));
        GPU_CHECK_KERNEL;
    }

    // 3. Non-local Potential Term
    if (nonlocal_) {
        nonlocal_->apply(psi, h_psi);
    }

    // 4. Project onto valid PW sphere
    h_psi.apply_mask();
    grid_.synchronize();
}

}  // namespace dftcu
