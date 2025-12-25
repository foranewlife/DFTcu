#include <cmath>

#include "tf.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

namespace dftcu {

// CUDA kernel for TF energy density and potential
// Energy density: e = C_TF * ρ^(5/3)
// Potential: v = (5/3) * C_TF * ρ^(2/3)
namespace {
__global__ void compute_tf_kernel(const int n, const double c_tf, const double c_tf_pot,
                                  const double* rho, double* v_kedf, double* energy_density) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double rho_val = rho[i];
        if (rho_val > 1e-30) {  // Avoid numerical issues at zero density
            // Compute ρ^(2/3)
            double rho_2_3 = cbrt(rho_val * rho_val);

            // Potential: V = (5/3) * C_TF * ρ^(2/3)
            v_kedf[i] = c_tf_pot * rho_2_3;

            // Energy density: e = C_TF * ρ^(5/3) = C_TF * ρ * ρ^(2/3)
            energy_density[i] = c_tf * rho_val * rho_2_3;
        } else {
            v_kedf[i] = 0.0;
            energy_density[i] = 0.0;
        }
    }
}

// Reduction kernel for summing energy density
__global__ void reduce_sum_kernel(const int n, const double* data, double* partial_sums,
                                  const int block_size) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * block_size + tid;

    // Load data into shared memory
    sdata[tid] = (i < n) ? data[i] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[tid];
    }
}
}  // anonymous namespace

ThomasFermi::ThomasFermi(double coeff) : coeff_(coeff) {
    // Thomas-Fermi constant: C_TF = (3/10) * (3π²)^(2/3)
    const double pi = 3.14159265358979323846;
    c_tf_ = (3.0 / 10.0) * pow(3.0 * pi * pi, 2.0 / 3.0);

    // Potential prefactor: (5/3) * C_TF
    c_tf_pot_ = (5.0 / 3.0) * c_tf_;
}

double ThomasFermi::compute(const RealField& rho, RealField& v_kedf) {
    const int n = rho.size();
    const double dV = rho.grid().dv();

    // Apply coefficient
    const double c_tf = coeff_ * c_tf_;
    const double c_tf_pot = coeff_ * c_tf_pot_;

    // Allocate temporary array for energy density
    GPU_Vector<double> energy_density(n);

    // Launch kernel to compute potential and energy density
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    compute_tf_kernel<<<grid_size, block_size>>>(n, c_tf, c_tf_pot, rho.data(), v_kedf.data(),
                                                 energy_density.data());
    GPU_CHECK_KERNEL;

    // Reduce energy density to get total energy
    // Two-step reduction: first reduce within blocks, then reduce block results
    GPU_Vector<double> partial_sums(grid_size);

    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(double)>>>(
        n, energy_density.data(), partial_sums.data(), block_size);
    GPU_CHECK_KERNEL;

    // Final reduction on CPU (small array)
    std::vector<double> h_partial_sums(grid_size);
    partial_sums.copy_to_host(h_partial_sums.data());

    double total_energy_density = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        total_energy_density += h_partial_sums[i];
    }

    // Multiply by volume element
    double energy = total_energy_density * dV;

    return energy;
}

}  // namespace dftcu
