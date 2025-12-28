#include "model/wavefunction.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <curand_kernel.h>

namespace dftcu {

namespace {

__global__ void initialize_mask_kernel(size_t n, const double* gg, double encut, int* mask,
                                       int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Plane wave condition: G^2/2 <= ENCUT
        if (0.5 * gg[i] <= encut) {
            mask[i] = 1;
            atomicAdd(count, 1);
        } else {
            mask[i] = 0;
        }
    }
}

__global__ void apply_mask_kernel(size_t n, int num_bands, const int* mask, gpufftComplex* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = n * num_bands;
    if (i < total_size) {
        int grid_idx = i % n;
        if (mask[grid_idx] == 0) {
            data[i].x = 0.0;
            data[i].y = 0.0;
        }
    }
}

__global__ void randomize_wavefunction_kernel(size_t total_size, gpufftComplex* data,
                                              unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size) {
        curandState state;
        curand_init(seed, i, 0, &state);
        data[i].x = curand_uniform(&state) - 0.5;
        data[i].y = curand_uniform(&state) - 0.5;
    }
}

__global__ void accumulate_density_kernel(size_t n, const gpufftComplex* psi_r, double occupation,
                                          double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double magnitude_sq = psi_r[i].x * psi_r[i].x + psi_r[i].y * psi_r[i].y;
        rho[i] += occupation * magnitude_sq;
    }
}

}  // namespace

Wavefunction::Wavefunction(Grid& grid, int num_bands, double encut)
    : grid_(grid),
      num_bands_(num_bands),
      encut_(encut),
      data_(grid.nnr() * num_bands),
      pw_mask_(grid.nnr()) {
    initialize_mask();
}

void Wavefunction::initialize_mask() {
    size_t n = grid_.nnr();
    int* d_count;
    CHECK(cudaMalloc(&d_count, sizeof(int)));
    CHECK(cudaMemset(d_count, 0, sizeof(int)));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    initialize_mask_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(n, grid_.gg(), encut_,
                                                                         pw_mask_.data(), d_count);
    GPU_CHECK_KERNEL;

    CHECK(cudaMemcpy(&num_pw_, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_count));
}

void Wavefunction::randomize(unsigned int seed) {
    size_t total_size = data_.size();
    const int block_size = 256;
    const int grid_size = (total_size + block_size - 1) / block_size;

    randomize_wavefunction_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(total_size,
                                                                                data_.data(), seed);
    GPU_CHECK_KERNEL;

    apply_mask();
    // Note: Normalization would ideally happen here via cuBLAS dot products
}

void Wavefunction::apply_mask() {
    size_t n = grid_.nnr();
    const int block_size = 256;
    const int grid_size = (n * num_bands_ + block_size - 1) / block_size;

    apply_mask_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(n, num_bands_, pw_mask_.data(),
                                                                    data_.data());
    GPU_CHECK_KERNEL;
}

void Wavefunction::compute_density(const std::vector<double>& occupations, RealField& rho) {
    size_t n = grid_.nnr();
    rho.fill(0.0);

    FFTSolver fft(grid_);
    ComplexField psi_r(grid_);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    for (int nb = 0; nb < num_bands_; ++nb) {
        if (occupations[nb] < 1e-12)
            continue;

        // 1. Copy band to temp buffer
        CHECK(cudaMemcpyAsync(psi_r.data(), band_data(nb), n * sizeof(gpufftComplex),
                              cudaMemcpyDeviceToDevice, grid_.stream()));

        // 2. Transform to real space
        fft.backward(psi_r);

        // 3. Accumulate: rho += f_n * |psi_n|^2
        accumulate_density_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
            n, psi_r.data(), occupations[nb], rho.data());
        GPU_CHECK_KERNEL;
    }

    grid_.synchronize();
}

}  // namespace dftcu
