#include "model/wavefunction.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <curand_kernel.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace dftcu {

namespace {

using ThrustComplex = thrust::complex<double>;
using WaveDotTuple = thrust::tuple<ThrustComplex, ThrustComplex>;

struct HermitianProductOp {
    __host__ __device__ ThrustComplex operator()(const WaveDotTuple& tpl) const {
        const ThrustComplex& a = thrust::get<0>(tpl);
        const ThrustComplex& b = thrust::get<1>(tpl);
        return thrust::conj(a) * b;
    }
};

__global__ void initialize_mask_kernel(size_t n, const double* gg, double encut, int* mask,
                                       int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double BOHR_TO_ANGSTROM = 0.529177210903;
        double g2_bohr = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
        // Plane wave condition: G^2/2 <= ENCUT
        if (0.5 * g2_bohr <= encut) {
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

__global__ void randomize_wavefunction_kernel(size_t n, int num_bands, const double* gg,
                                              gpufftComplex* data, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = n * num_bands;
    if (i < total_size) {
        int grid_idx = i % n;
        curandState state;
        curand_init(seed, i, 0, &state);

        // QE Scaling: rr1 = randy() / (G^2 + 1.0)
        // G^2 is in Rydberg (same as Bohr^-2)
        const double BOHR_TO_ANGSTROM = 0.529177210903;
        double g2_ry = gg[grid_idx] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

        double r1 = curand_uniform(&state);
        double r2 = curand_uniform(&state);

        double amp = r1 / (g2_ry + 1.0);
        double phase = r2 * 2.0 * constants::D_PI;

        double s, c;
        sincos(phase, &s, &c);
        data[i].x = amp * c;
        data[i].y = amp * s;
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

struct KineticEnergyOp {
    const double* gg;
    const double BOHR_TO_ANGSTROM = 0.529177210903;

    KineticEnergyOp(const double* g) : gg(g) {}

    __host__ __device__ double operator()(const thrust::tuple<gpufftComplex, size_t>& tpl) const {
        const gpufftComplex& psi = thrust::get<0>(tpl);
        size_t i = thrust::get<1>(tpl);
        double g2_bohr = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
        return 0.5 * g2_bohr * (psi.x * psi.x + psi.y * psi.y);
    }
};

}  // namespace

Wavefunction::Wavefunction(Grid& grid, int num_bands, double encut)
    : grid_(grid),
      num_bands_(num_bands),
      encut_(encut),
      data_(grid.nnr() * num_bands),
      pw_mask_(grid.nnr()) {
    data_.fill({0.0, 0.0}, grid_.stream());
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

std::vector<std::complex<double>> Wavefunction::get_coefficients(int band) const {
    size_t nnr = grid_.nnr();
    std::vector<std::complex<double>> host_data(nnr);
    CHECK(cudaMemcpy(host_data.data(), data_.data() + band * nnr, nnr * sizeof(gpufftComplex),
                     cudaMemcpyDeviceToHost));
    return host_data;
}

void Wavefunction::set_coefficients(const std::vector<std::complex<double>>& coeffs, int band) {
    size_t nnr = grid_.nnr();
    if (coeffs.size() != nnr) {
        throw std::runtime_error("Wavefunction::set_coefficients: size mismatch");
    }
    CHECK(cudaMemcpy(data_.data() + band * nnr, coeffs.data(), nnr * sizeof(gpufftComplex),
                     cudaMemcpyHostToDevice));
}

void Wavefunction::randomize(unsigned int seed) {
    size_t n = grid_.nnr();
    const int block_size = 256;
    const int grid_size = (n * num_bands_ + block_size - 1) / block_size;

    randomize_wavefunction_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        n, num_bands_, grid_.gg(), data_.data(), seed);
    GPU_CHECK_KERNEL;

    apply_mask();
    // Normalize to orthonormal set (Gram-Schmidt for now)
    orthonormalize();
}

void Wavefunction::apply_mask() {
    size_t n = grid_.nnr();
    const int block_size = 256;
    const int grid_size = (n * num_bands_ + block_size - 1) / block_size;

    apply_mask_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(n, num_bands_, pw_mask_.data(),
                                                                    data_.data());
    GPU_CHECK_KERNEL;
}

void Wavefunction::copy_from_host(const std::complex<double>* data) {
    size_t total_size = grid_.nnr() * num_bands_;
    data_.copy_from_host(reinterpret_cast<const gpufftComplex*>(data), grid_.stream());
}

void Wavefunction::copy_to_host(std::complex<double>* data) const {
    size_t total_size = grid_.nnr() * num_bands_;
    data_.copy_to_host(reinterpret_cast<gpufftComplex*>(data), grid_.stream());
}

void Wavefunction::compute_density(const std::vector<double>& occupations, RealField& rho) {
    size_t n = grid_.nnr();
    rho.fill(0.0);

    FFTSolver fft(grid_);
    ComplexField psi_r(grid_);

    const int block_size = 256;
    const int grid_size_v = (n + block_size - 1) / block_size;
    // Physical Normalization:
    // rho(r) = 1/Volume * sum( f_n * |IFFT_unnorm(C_G)|^2 )
    double inv_vol = 1.0 / grid_.volume_bohr();

    for (int nb = 0; nb < num_bands_; ++nb) {
        if (occupations[nb] < 1e-12)
            continue;

        // 1. Copy band to temp buffer
        CHECK(cudaMemcpyAsync(psi_r.data(), band_data(nb), n * sizeof(gpufftComplex),
                              cudaMemcpyDeviceToDevice, grid_.stream()));

        // 2. Transform to real space (Raw IFFT: no scaling)
        fft.backward(psi_r);

        // 3. Accumulate: rho += (f_n / Volume) * |psi_r|^2
        // Volume is grid_.volume_bohr() for atomic units
        accumulate_density_kernel<<<grid_size_v, block_size, 0, grid_.stream()>>>(
            n, psi_r.data(), occupations[nb] * inv_vol, rho.data());
        GPU_CHECK_KERNEL;
    }

    grid_.synchronize();
}

void Wavefunction::compute_occupations(const std::vector<double>& eigenvalues, double nelectrons,
                                       double sigma, std::vector<double>& occupations,
                                       double& fermi_energy) {
    int nband = eigenvalues.size();
    occupations.resize(nband);

    auto get_ne = [&](double mu) {
        double ne = 0;
        for (double e : eigenvalues) {
            double arg = (e - mu) / sigma;
            if (arg > 30.0)
                continue;
            if (arg < -30.0) {
                ne += 2.0;  // Spin degenerate
                continue;
            }
            ne += 2.0 / (exp(arg) + 1.0);
        }
        return ne;
    };

    // Bisection for Fermi Level
    double low = eigenvalues.front() - 5.0 * sigma;
    double high = eigenvalues.back() + 5.0 * sigma;

    for (int iter = 0; iter < 50; ++iter) {
        double mid = (low + high) / 2.0;
        if (get_ne(mid) > nelectrons) {
            low = mid;
        } else {
            high = mid;
        }
    }

    fermi_energy = (low + high) / 2.0;
    for (int i = 0; i < nband; ++i) {
        double arg = (eigenvalues[i] - fermi_energy) / sigma;
        if (arg > 30.0)
            occupations[i] = 0.0;
        else if (arg < -30.0)
            occupations[i] = 2.0;
        else
            occupations[i] = 2.0 / (exp(arg) + 1.0);
    }
}

void Wavefunction::orthonormalize() {
    // Gram-Schmidt process: psi_n = psi_n - sum_{m < n} <psi_m | psi_n> psi_m
    // and then psi_n = psi_n / sqrt(<psi_n | psi_n>)
    for (int n = 0; n < num_bands_; ++n) {
        for (int m = 0; m < n; ++m) {
            std::complex<double> overlap = dot(m, n);
            // psi_n = psi_n - overlap * psi_m
            v_axpy(grid_.nnr(), -overlap, band_data(m), band_data(n), grid_.stream());
        }
        std::complex<double> self_dot = dot(n, n);
        double norm = std::sqrt(self_dot.real());
        if (norm > 1e-15) {
            v_scale(grid_.nnr(), 1.0 / norm, band_data(n), band_data(n), grid_.stream());
        }
    }
}

double Wavefunction::compute_kinetic_energy(const std::vector<double>& occupations) {
    size_t n = grid_.nnr();
    double total_ek = 0.0;

    auto counting_begin = thrust::make_counting_iterator<size_t>(0);
    auto counting_end = counting_begin + n;

    for (int nb = 0; nb < num_bands_; ++nb) {
        if (occupations[nb] < 1e-12)
            continue;

        gpufftComplex* psi_ptr = band_data(nb);
        auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(psi_ptr, counting_begin));
        auto zipped_end = zipped_begin + n;

        double band_ek =
            thrust::transform_reduce(thrust::cuda::par.on(grid_.stream()), zipped_begin, zipped_end,
                                     KineticEnergyOp(grid_.gg()), 0.0, thrust::plus<double>());

        total_ek += occupations[nb] * band_ek;
    }

    return total_ek;
}

std::complex<double> Wavefunction::dot(int band_a, int band_b) {
    size_t n = grid_.nnr();
    using Complex = thrust::complex<double>;

    Complex* a_ptr = reinterpret_cast<Complex*>(band_data(band_a));
    Complex* b_ptr = reinterpret_cast<Complex*>(band_data(band_b));

    auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(a_ptr, b_ptr));
    auto zipped_end = zipped_begin + n;

    Complex sum =
        thrust::transform_reduce(thrust::cuda::par.on(grid_.stream()), zipped_begin, zipped_end,
                                 HermitianProductOp(), Complex(0.0, 0.0), thrust::plus<Complex>());

    grid_.synchronize();
    return {sum.real(), sum.imag()};
}

std::vector<int> Wavefunction::get_pw_indices() {
    size_t n = grid_.nnr();
    std::vector<int> host_mask(n);
    pw_mask_.copy_to_host(host_mask.data());

    std::vector<int> indices;
    indices.reserve(num_pw_);
    for (int i = 0; i < n; ++i) {
        if (host_mask[i] != 0) {
            indices.push_back(i);
        }
    }
    return indices;
}

}  // namespace dftcu
