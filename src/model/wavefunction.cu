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
        const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
        double g2_bohr = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
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
    size_t total_size = (size_t)n * num_bands;
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
    size_t total_size = (size_t)n * num_bands;
    if (i < total_size) {
        int grid_idx = i % n;
        curandState state;
        curand_init(seed, i, 0, &state);

        const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
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
    const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
    KineticEnergyOp(const double* g) : gg(g) {}
    __host__ __device__ double operator()(const thrust::tuple<gpufftComplex, size_t>& tpl) const {
        const gpufftComplex& psi = thrust::get<0>(tpl);
        size_t i = thrust::get<1>(tpl);
        double g2_bohr = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
        return 0.5 * g2_bohr * (psi.x * psi.x + psi.y * psi.y);
    }
};

__global__ void set_coefficients_miller_kernel(int nr0, int nr1, int nr2, int npw, const int* h,
                                               const int* k, const int* l,
                                               const gpufftComplex* values, int num_bands,
                                               gpufftComplex* data, bool expand_hermitian) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw) {
        int h_val = h[ig];
        int k_val = k[ig];
        int l_val = l[ig];

        int n0 = (h_val % nr0 + nr0) % nr0;
        int n1 = (k_val % nr1 + nr1) % nr1;
        int n2 = (l_val % nr2 + nr2) % nr2;
        size_t nnr_v = (size_t)nr0 * nr1 * nr2;
        size_t idx = (size_t)n0 * nr1 * nr2 + n1 * nr2 + n2;

        for (int b = 0; b < num_bands; ++b) {
            gpufftComplex val = values[b * npw + ig];
            const double inv_sqrt2 = 0.7071067811865475;

            if (expand_hermitian && (h_val != 0 || k_val != 0 || l_val != 0)) {
                // QE Gamma-only 数据带 sqrt(2) 因子，需要归一化
                // 展开到全球：G 位置和 -G 位置都是 val/sqrt(2)
                val.x *= inv_sqrt2;
                val.y *= inv_sqrt2;

                // 填充 -G 位置（复共轭）
                int n0_i = (-h_val % nr0 + nr0) % nr0;
                int n1_i = (-k_val % nr1 + nr1) % nr1;
                int n2_i = (-l_val % nr2 + nr2) % nr2;
                size_t idx_i = (size_t)n0_i * nr1 * nr2 + n1_i * nr2 + n2_i;
                data[b * nnr_v + idx_i].x = val.x;
                data[b * nnr_v + idx_i].y = -val.y;
            }

            // 填充 G 位置（expand_hermitian 时已经缩放过）
            data[b * nnr_v + idx] = val;
        }
    }
}

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
    if (coeffs.size() != nnr)
        throw std::runtime_error("Wavefunction::set_coefficients mismatch");
    CHECK(cudaMemcpy(data_.data() + band * nnr, coeffs.data(), nnr * sizeof(gpufftComplex),
                     cudaMemcpyHostToDevice));
}

void Wavefunction::set_coefficients_miller(const std::vector<int>& h, const std::vector<int>& k,
                                           const std::vector<int>& l,
                                           const std::vector<std::complex<double>>& values,
                                           bool expand_hermitian) {
    int npw_input = (int)h.size();
    if (k.size() != (size_t)npw_input || l.size() != (size_t)npw_input)
        throw std::runtime_error("Miller size mismatch");
    if (values.size() != (size_t)num_bands_ * npw_input)
        throw std::runtime_error("Values size mismatch");

    GPU_Vector<int> d_h(npw_input), d_k(npw_input), d_l(npw_input);
    GPU_Vector<gpufftComplex> d_values(values.size());
    CHECK(cudaMemcpy(d_h.data(), h.data(), npw_input * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k.data(), k.data(), npw_input * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l.data(), l.data(), npw_input * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(gpufftComplex),
                     cudaMemcpyHostToDevice));

    data_.fill({0.0, 0.0}, grid_.stream());
    const int block_size = 256;
    const int grid_size = (npw_input + block_size - 1) / block_size;

    int nr0 = grid_.nr()[0], nr1 = grid_.nr()[1], nr2 = grid_.nr()[2];

    set_coefficients_miller_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        nr0, nr1, nr2, npw_input, d_h.data(), d_k.data(), d_l.data(), d_values.data(), num_bands_,
        data_.data(), expand_hermitian);
    GPU_CHECK_KERNEL;
    cudaStreamSynchronize(grid_.stream());
}

void Wavefunction::randomize(unsigned int seed) {
    size_t n = grid_.nnr();
    const int block_size = 256;
    const int grid_size = (n * num_bands_ + block_size - 1) / block_size;
    randomize_wavefunction_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        n, num_bands_, grid_.gg(), data_.data(), seed);
    GPU_CHECK_KERNEL;
    apply_mask();
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
    data_.copy_from_host(reinterpret_cast<const gpufftComplex*>(data), grid_.stream());
}

void Wavefunction::copy_to_host(std::complex<double>* data) const {
    data_.copy_to_host(reinterpret_cast<gpufftComplex*>(data), grid_.stream());
}

void Wavefunction::compute_density(const std::vector<double>& occupations, RealField& rho) {
    size_t n = grid_.nnr();
    rho.fill(0.0);
    FFTSolver fft(grid_);
    ComplexField psi_r(grid_);
    const int block_size = 256;
    const int grid_size_v = (n + block_size - 1) / block_size;
    double inv_vol = 1.0 / grid_.volume_bohr();
    for (int nb = 0; nb < num_bands_; ++nb) {
        if (occupations[nb] < 1e-12)
            continue;
        CHECK(cudaMemcpyAsync(psi_r.data(), band_data(nb), n * sizeof(gpufftComplex),
                              cudaMemcpyDeviceToDevice, grid_.stream()));
        fft.backward(psi_r);
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
                ne += 2.0;
                continue;
            }
            ne += 2.0 / (exp(arg) + 1.0);
        }
        return ne;
    };
    double low = eigenvalues.front() - 5.0 * sigma, high = eigenvalues.back() + 5.0 * sigma;
    for (int iter = 0; iter < 50; ++iter) {
        double mid = (low + high) / 2.0;
        if (get_ne(mid) > nelectrons)
            low = mid;
        else
            high = mid;
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
    for (int n = 0; n < num_bands_; ++n) {
        for (int m = 0; m < n; ++m) {
            std::complex<double> overlap = dot(m, n);
            v_axpy(grid_.nnr(), -overlap, band_data(m), band_data(n), grid_.stream());
        }
        std::complex<double> self_dot = dot(n, n);
        double norm = std::sqrt(self_dot.real());
        if (norm > 1e-15)
            v_scale(grid_.nnr(), 1.0 / norm, band_data(n), band_data(n), grid_.stream());
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
    for (int i = 0; i < n; ++i)
        if (host_mask[i] != 0)
            indices.push_back(i);
    return indices;
}

std::vector<double> Wavefunction::get_g2kin() {
    std::vector<int> indices = get_pw_indices();
    size_t n = grid_.nnr();
    std::vector<double> h_gg(n);
    CHECK(cudaMemcpy(h_gg.data(), grid_.gg(), n * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> g2kin;
    g2kin.reserve(indices.size());
    const double b2 = constants::BOHR_TO_ANGSTROM * constants::BOHR_TO_ANGSTROM;
    for (int idx : indices)
        g2kin.push_back(h_gg[idx] * b2);
    return g2kin;
}

// ====================================================================================
// Gamma-point constraint
// ====================================================================================

__global__ void force_gamma_constraint_kernel(gpufftComplex* data, int nbands, size_t nnr) {
    /*
     * Force Im[ψ(G=0)] = 0 for all bands
     *
     * G=0 corresponds to index 0 in the FFT grid layout.
     * Set data[band * nnr + 0].y = 0 for all bands.
     */
    int band = blockIdx.x * blockDim.x + threadIdx.x;
    if (band < nbands) {
        data[band * nnr].y = 0.0;  // Im[ψ(G=0, band)] = 0
    }
}

void Wavefunction::force_gamma_constraint() {
    /*
     * Enforce Gamma-point constraint: Im[ψ(G=0)] = 0
     *
     * QE reference: regterg.f90:172, 375
     *   IF (gstart == 2) psi(1,k) = CMPLX( DBLE( psi(1,k) ), 0.D0 ,kind=DP)
     */
    if (!has_g0()) {
        return;  // No G=0 in this process, nothing to do
    }

    int block_size = 256;
    int num_blocks = (num_bands_ + block_size - 1) / block_size;

    force_gamma_constraint_kernel<<<num_blocks, block_size, 0, grid_.stream()>>>(
        data_.data(), num_bands_, grid_.nnr());

    CHECK(cudaGetLastError());
}

bool Wavefunction::has_g0() const {
    /*
     * Check if G=0 is included in this grid
     *
     * In QE terminology: gstart = 2 means G=0 is included (Fortran 1-based indexing)
     * In C++: G=0 is at index 0 if the grid includes the origin
     *
     * For now, assume G=0 is always included in a single-process Gamma-only calculation.
     * TODO: In MPI-parallel code, need to check if this process owns G=0.
     */
    return true;  // Gamma-only, single process
}

}  // namespace dftcu
