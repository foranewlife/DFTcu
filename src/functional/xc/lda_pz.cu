#include <cmath>

#include "lda_pz.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void lda_pz_kernel(int n, const double* rho, double* v_xc, double* energy_density) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double r = rho[i];
        if (r < 1e-30) {
            v_xc[i] = 0.0;
            energy_density[i] = 0.0;
            return;
        }

        double rho_cbrt = cbrt(r);
        double rs = 0.6203504908994001 / rho_cbrt;  // (3 / (4*pi))^(1/3) / rho^(1/3)

        // Exchange
        double ex = -0.7385587663820223 * rho_cbrt;  // -3/4 * (3/pi)^(1/3) * rho^(1/3)
        double vx = -0.9847450218426964 * rho_cbrt;  // - (3/pi)^(1/3) * rho^(1/3)

        // Correlation (PZ)
        double ec, vc;
        if (rs < 1.0) {
            double log_rs = log(rs);
            // Constants from DFTpy (semilocal_xc.py)
            const double a = 0.0311;
            const double b = -0.048;
            const double c = 0.0020;
            const double d = -0.0116;

            ec = a * log_rs + b + c * rs * log_rs + d * rs;
            vc = log_rs * (a + (2.0 / 3.0) * c * rs) + b - (1.0 / 3.0) * a +
                 (1.0 / 3.0) * (2.0 * d - c) * rs;
        } else {
            double rs_sqrt = sqrt(rs);
            const double gamma = -0.1423;
            const double beta1 = 1.0529;
            const double beta2 = 0.3334;

            double denom = 1.0 + beta1 * rs_sqrt + beta2 * rs;
            ec = gamma / denom;
            vc =
                (gamma + (7.0 / 6.0 * gamma * beta1) * rs_sqrt + (4.0 / 3.0 * gamma * beta2) * rs) /
                (denom * denom);
        }

        v_xc[i] = vx + vc;
        energy_density[i] = (ex + ec) * r;
    }
}

// Reduction kernel (borrowed from tf.cu)
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
    double dv = rho.grid().dv();

    GPU_Vector<double> energy_density(n);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    lda_pz_kernel<<<grid_size, block_size>>>(n, rho.data(), v_xc.data(), energy_density.data());
    GPU_CHECK_KERNEL

    // Reduce energy density
    GPU_Vector<double> partial_sums(grid_size);
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(double)>>>(
        n, energy_density.data(), partial_sums.data(), block_size);
    GPU_CHECK_KERNEL

    std::vector<double> h_partial_sums(grid_size);
    partial_sums.copy_to_host(h_partial_sums.data());
    double total_energy = 0.0;
    for (double s : h_partial_sums)
        total_energy += s;

    return total_energy * dv;
}

}  // namespace dftcu
