#include "fft/fft_solver.cuh"
#include "model/density_builder.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

// Reuse constant memory for atom data
static __device__ __constant__ double c_atom_x[constants::MAX_ATOMS_PSEUDO];
static __device__ __constant__ double c_atom_y[constants::MAX_ATOMS_PSEUDO];
static __device__ __constant__ double c_atom_z[constants::MAX_ATOMS_PSEUDO];
static __device__ __constant__ int c_atom_type[constants::MAX_ATOMS_PSEUDO];

__global__ void interpolate_density_kernel(size_t nnr, const double* gg, int n_radial,
                                           const double* q_radial, const double* a, const double* b,
                                           const double* c, const double* d, double* rho_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        double q = sqrt(gg[i]);

        int low = 0, high = n_radial - 2;
        int idx = 0;
        if (q <= q_radial[0]) {
            idx = 0;
        } else if (q >= q_radial[n_radial - 1]) {
            rho_out[i] = 0.0;
            return;
        } else {
            while (low <= high) {
                int mid = (low + high) / 2;
                if (q_radial[mid] <= q && q < q_radial[mid + 1]) {
                    idx = mid;
                    break;
                } else if (q_radial[mid] < q) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }

        double dx = q - q_radial[idx];
        rho_out[i] = a[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx;
    }
}

__global__ void density_sum_kernel(int nnr, const double* gx, const double* gy, const double* gz,
                                   const double* gg, const double* rho_types, int nat,
                                   gpufftComplex* rho_g) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnr) {
        double cur_gx = gx[i];
        double cur_gy = gy[i];
        double cur_gz = gz[i];

        double sum_re = 0.0;
        double sum_im = 0.0;

        for (int j = 0; j < nat; ++j) {
            int type = c_atom_type[j];
            double phase = (cur_gx * c_atom_x[j] + cur_gy * c_atom_y[j] + cur_gz * c_atom_z[j]);
            double s, c;
            sincos(phase, &s, &c);

            double rho_val = rho_types[type * nnr + i];
            sum_re += rho_val * c;
            sum_im -= rho_val * s;
        }
        rho_g[i].x = sum_re;
        rho_g[i].y = sum_im;
    }
}

// Cubic spline solver (simplified from pseudo.cu)
void solve_spline_local(const std::vector<double>& x, const std::vector<double>& y,
                        std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
                        std::vector<double>& d) {
    int n = static_cast<int>(x.size()) - 1;
    a = y;
    std::vector<double> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = x[i + 1] - x[i];
    std::vector<double> alpha(n);
    for (int i = 1; i < n; ++i)
        alpha[i] = 3.0 / h[i] * (a[i + 1] - a[i]) - 3.0 / h[i - 1] * (a[i] - a[i - 1]);
    std::vector<double> l(n + 1), mu(n + 1), z(n + 1);
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n] = 1.0;
    z[n] = 0.0;
    c.resize(n + 1);
    c[n] = 0.0;
    b.resize(n);
    d.resize(n);
    for (int j = n - 1; j >= 0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }
}

}  // namespace

DensityBuilder::DensityBuilder(Grid& grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms) {}

void DensityBuilder::set_atomic_rho_g(int type, const std::vector<double>& q,
                                      const std::vector<double>& rho_q) {
    if (type >= num_types_) {
        int new_num_types = type + 1;
        GPU_Vector<double> new_rho(grid_.nnr() * new_num_types);
        if (num_types_ > 0) {
            CHECK(cudaMemcpyAsync(new_rho.data(), rho_types_g_.data(),
                                  grid_.nnr() * num_types_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, grid_.stream()));
        }
        rho_types_g_ = std::move(new_rho);
        num_types_ = new_num_types;
    }

    std::vector<double> a, b, c, d;
    solve_spline_local(q, rho_q, a, b, c, d);

    GPU_Vector<double> d_q(q.size());
    GPU_Vector<double> d_a(a.size());
    GPU_Vector<double> d_b(b.size());
    GPU_Vector<double> d_c(c.size());
    GPU_Vector<double> d_d(d.size());

    d_q.copy_from_host(q.data(), grid_.stream());
    d_a.copy_from_host(a.data(), grid_.stream());
    d_b.copy_from_host(b.data(), grid_.stream());
    d_c.copy_from_host(c.data(), grid_.stream());
    d_d.copy_from_host(d.data(), grid_.stream());

    const int block_size = 256;
    const int grid_size = (static_cast<int>(grid_.nnr()) + block_size - 1) / block_size;

    interpolate_density_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        grid_.nnr(), grid_.gg(), static_cast<int>(q.size()), d_q.data(), d_a.data(), d_b.data(),
        d_c.data(), d_d.data(), rho_types_g_.data() + type * grid_.nnr());

    GPU_CHECK_KERNEL;
}

void DensityBuilder::build_density(RealField& rho) {
    size_t nnr = grid_.nnr();
    FFTSolver solver(grid_);
    ComplexField rho_g(grid_);

    if (atoms_->nat() > constants::MAX_ATOMS_PSEUDO) {
        throw std::runtime_error("Too many atoms for DensityBuilder");
    }

    CHECK(cudaMemcpyToSymbolAsync(c_atom_x, atoms_->h_pos_x().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_atom_y, atoms_->h_pos_y().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_atom_z, atoms_->h_pos_z().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_atom_type, atoms_->h_type().data(), atoms_->nat() * sizeof(int),
                                  0, cudaMemcpyHostToDevice, grid_.stream()));

    const int block_size = 256;
    const int grid_size = (static_cast<int>(nnr) + block_size - 1) / block_size;

    density_sum_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        static_cast<int>(nnr), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), rho_types_g_.data(),
        static_cast<int>(atoms_->nat()), rho_g.data());

    GPU_CHECK_KERNEL;
    solver.backward(rho_g);
    complex_to_real(nnr, rho_g.data(), rho.data(), grid_.stream());
    // v_scale(nnr, 1.0 / (double)nnr, rho.data(), rho.data(), grid_.stream()); // Remove redundant
    // scaling
    grid_.synchronize();
}

}  // namespace dftcu
