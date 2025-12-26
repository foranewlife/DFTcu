#include <cmath>

#include "tf.cuh"
#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"

namespace dftcu {

// CUDA kernel for TF energy density and potential
namespace {
__global__ void compute_tf_kernel(const int n, const double c_tf, const double c_tf_pot,
                                  const double* rho, double* v_kedf, double* energy_density,
                                  ThomasFermi::Parameters params) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double rho_val = rho[i];
        if (rho_val > params.rho_threshold) {
            double rho_2_3 = cbrt(rho_val * rho_val);
            v_kedf[i] = c_tf_pot * rho_2_3;
            energy_density[i] = c_tf * rho_val * rho_2_3;
        } else {
            v_kedf[i] = 0.0;
            energy_density[i] = 0.0;
        }
    }
}

__global__ void reduce_sum_kernel(const int n, const double* data, double* partial_sums,
                                  const int block_size) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * block_size + tid;
    sdata[tid] = (i < n) ? data[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[tid];
    }
}
}  // anonymous namespace

ThomasFermi::ThomasFermi(double coeff) : coeff_(coeff) {
    c_tf_ = constants::C_TF_BASE;
    c_tf_pot_ = (5.0 / 3.0) * c_tf_;
}

double ThomasFermi::compute(const RealField& rho, RealField& v_kedf) {
    const int n = rho.size();
    const double dV = rho.grid().dv();

    const double c_tf = coeff_ * c_tf_;
    const double c_tf_pot = coeff_ * c_tf_pot_;

    GPU_Vector<double> energy_density(n);
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    compute_tf_kernel<<<grid_size, block_size>>>(n, c_tf, c_tf_pot, rho.data(), v_kedf.data(),
                                                 energy_density.data(), params_);
    GPU_CHECK_KERNEL;

    GPU_Vector<double> partial_sums(grid_size);
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(double)>>>(
        n, energy_density.data(), partial_sums.data(), block_size);
    GPU_CHECK_KERNEL;

    std::vector<double> h_partial_sums(grid_size);
    partial_sums.copy_to_host(h_partial_sums.data());

    double total_energy_density = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        total_energy_density += h_partial_sums[i];
    }

    return total_energy_density * dV;
}

}  // namespace dftcu
