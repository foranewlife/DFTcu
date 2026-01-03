#include <cmath>

#include "lda_pz.cuh"
#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void lda_pz_kernel(int n, const double* rho, double* v_xc, double* energy_density,
                              LDA_PZ::Parameters params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        // Use abs(rho) to handle negative density fluctuations (like QE)
        double r_abs = fabs(r);

        if (r_abs < params.rho_threshold) {
            v_xc[i] = 0.0;
            energy_density[i] = 0.0;
            return;
        }

        double rho_cbrt = cbrt(r_abs);
        const double pi = constants::D_PI;
        double rs = pow(3.0 / (4.0 * pi), 1.0 / 3.0) / rho_cbrt;

        // Exchange
        double ex = constants::EX_LDA_COEFF * rho_cbrt;
        double vx = (4.0 / 3.0) * ex;

        // Correlation (PZ)
        double ec, vc;
        if (rs < 1.0) {
            double log_rs = log(rs);
            ec = params.a * log_rs + params.b + params.c * rs * log_rs + params.d * rs;
            vc = log_rs * (params.a + (2.0 / 3.0) * params.c * rs) + params.b -
                 (1.0 / 3.0) * params.a + (1.0 / 3.0) * (2.0 * params.d - params.c) * rs;
        } else {
            double rs_sqrt = sqrt(rs);
            double denom = 1.0 + params.beta1 * rs_sqrt + params.beta2 * rs;
            ec = params.gamma / denom;
            vc = (params.gamma + (7.0 / 6.0 * params.gamma * params.beta1) * rs_sqrt +
                  (4.0 / 3.0 * params.gamma * params.beta2) * rs) /
                 (denom * denom);
        }

        v_xc[i] = vx + vc;
        energy_density[i] = (ex + ec) * r_abs;
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
}  // namespace

double LDA_PZ::compute(const RealField& rho, RealField& v_xc) {
    int n = rho.size();
    Grid& grid = rho.grid();
    double dv = grid.dv();

    GPU_Vector<double> energy_density(n);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    lda_pz_kernel<<<grid_size, block_size, 0, grid.stream()>>>(n, rho.data(), v_xc.data(),
                                                               energy_density.data(), params_);
    GPU_CHECK_KERNEL;

    // Reduce energy density
    GPU_Vector<double> partial_sums(grid_size);
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(double), grid.stream()>>>(
        n, energy_density.data(), partial_sums.data(), block_size);
    GPU_CHECK_KERNEL;

    std::vector<double> h_partial_sums(grid_size);
    partial_sums.copy_to_host(h_partial_sums.data(), grid.stream());
    grid.synchronize();

    double total_energy = 0.0;
    for (double s : h_partial_sums)
        total_energy += s;

    return total_energy * grid.dv_bohr();
}

}  // namespace dftcu
