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

}  // namespace

Hamiltonian::Hamiltonian(Grid& grid, Evaluator& evaluator)
    : grid_(grid), evaluator_(evaluator), v_loc_tot_(grid) {}

void Hamiltonian::update_potentials(const RealField& rho) {
    // Collect all local potential components from the evaluator
    // Evaluator::compute resets v_loc_tot_ to zero and sums Hartree, XC, etc.
    evaluator_.compute(rho, v_loc_tot_);
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
    ComplexField tmp_r(grid_);
    ComplexField tmp_g(grid_);

    const int grid_size_v = (n + block_size - 1) / block_size;

    for (int nb = 0; nb < nbands; ++nb) {
        // psi(G) -> psi(r)
        CHECK(cudaMemcpyAsync(tmp_g.data(), psi.band_data(nb), n * sizeof(gpufftComplex),
                              cudaMemcpyDeviceToDevice, grid_.stream()));
        fft.backward(tmp_g);

        // psi(r) * V_loc(r)
        apply_vloc_kernel<<<grid_size_v, block_size, 0, grid_.stream()>>>(n, v_loc_tot_.data(),
                                                                          tmp_g.data());
        GPU_CHECK_KERNEL;

        // V_loc * psi -> (G-space)
        fft.forward(tmp_g);

        // Accumulate into h_psi: h_psi(G) += FFT(V_loc * psi)
        v_axpy(n * 2, 1.0, (double*)tmp_g.data(), (double*)h_psi.band_data(nb), grid_.stream());
    }

    // 3. Project onto valid PW sphere
    h_psi.apply_mask();
    grid_.synchronize();
}

}  // namespace dftcu
