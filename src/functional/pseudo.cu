#include "fft/fft_solver.cuh"
#include "pseudo.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void pseudo_rec_kernel(int nnr, const double* gx, const double* gy, const double* gz,
                                  const double* gg, const double* vloc_types, int nat,
                                  const double* pos_x, const double* pos_y, const double* pos_z,
                                  const int* types, gpufftComplex* v_g, double g2max) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnr) {
        if (gg[i] > g2max) {
            v_g[i].x = 0.0;
            v_g[i].y = 0.0;
            return;
        }

        double cur_gx = gx[i];
        double cur_gy = gy[i];
        double cur_gz = gz[i];

        double sum_re = 0.0;
        double sum_im = 0.0;

        for (int j = 0; j < nat; ++j) {
            int type = types[j];
            double phase = (cur_gx * pos_x[j] + cur_gy * pos_y[j] + cur_gz * pos_z[j]);
            double s, c;
            sincos(phase, &s, &c);

            double v_val = vloc_types[type * nnr + i];
            sum_re += v_val * c;
            sum_im -= v_val * s;
        }

        v_g[i].x = sum_re;
        v_g[i].y = sum_im;
    }
}

__global__ void interpolate_radial_kernel(size_t nnr, const double* gg, int n_radial,
                                          const double* q_radial, const double* a, const double* b,
                                          const double* c, const double* d, double* v_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        double q = sqrt(gg[i]);

        int low = 0, high = n_radial - 2;
        int idx = 0;
        if (q <= q_radial[0]) {
            idx = 0;
        } else if (q >= q_radial[n_radial - 1]) {
            v_out[i] = 0.0;
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
        v_out[i] = a[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx;
    }
}

void solve_spline(const std::vector<double>& x, const std::vector<double>& y,
                  std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
                  std::vector<double>& d) {
    int n = static_cast<int>(x.size()) - 1;
    a = y;
    std::vector<double> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = x[i + 1] - x[i];

    std::vector<double> alpha(n);
    for (int i = 1; i < n; ++i)
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);

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

LocalPseudo::LocalPseudo(std::shared_ptr<Grid> grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms) {}

void LocalPseudo::set_vloc(int type, const std::vector<double>& vloc_g) {
    if (type >= num_types_) {
        int new_num_types = type + 1;
        GPU_Vector<double> new_vloc(grid_->nnr() * new_num_types);
        if (num_types_ > 0) {
            CHECK(cudaMemcpy(new_vloc.data(), vloc_types_.data(),
                             grid_->nnr() * num_types_ * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        vloc_types_ = std::move(new_vloc);
        num_types_ = new_num_types;
    }
    CHECK(cudaMemcpy(vloc_types_.data() + type * grid_->nnr(), vloc_g.data(),
                     vloc_g.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void LocalPseudo::set_vloc_radial(int type, const std::vector<double>& q,
                                  const std::vector<double>& v_q) {
    if (type >= num_types_) {
        int new_num_types = type + 1;
        GPU_Vector<double> new_vloc(grid_->nnr() * new_num_types);
        if (num_types_ > 0) {
            CHECK(cudaMemcpy(new_vloc.data(), vloc_types_.data(),
                             grid_->nnr() * num_types_ * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        vloc_types_ = std::move(new_vloc);
        num_types_ = new_num_types;
    }

    std::vector<double> a, b, c, d;
    solve_spline(q, v_q, a, b, c, d);

    GPU_Vector<double> d_q(q.size());
    GPU_Vector<double> d_a(a.size());
    GPU_Vector<double> d_b(b.size());
    GPU_Vector<double> d_c(c.size());
    GPU_Vector<double> d_d(d.size());

    d_q.copy_from_host(q.data());
    d_a.copy_from_host(a.data());
    d_b.copy_from_host(b.data());
    d_c.copy_from_host(c.data());
    d_d.copy_from_host(d.data());

    const int block_size = 256;
    const int grid_size = (static_cast<int>(grid_->nnr()) + block_size - 1) / block_size;

    interpolate_radial_kernel<<<grid_size, block_size>>>(
        grid_->nnr(), grid_->gg(), static_cast<int>(q.size()), d_q.data(), d_a.data(), d_b.data(),
        d_c.data(), d_d.data(), vloc_types_.data() + type * grid_->nnr());

    CHECK(cudaDeviceSynchronize());
    GPU_CHECK_KERNEL
}

void LocalPseudo::compute(RealField& v) {
    size_t nnr = grid_->nnr();
    ComplexField v_g(grid_);
    FFTSolver solver(grid_);

    const int block_size = 256;
    const int grid_size = (static_cast<int>(nnr) + block_size - 1) / block_size;

    // Use the maximum G2 from the grid instead of the inscribed sphere
    double g2max = grid_->g2max() + 1e-6;

    pseudo_rec_kernel<<<grid_size, block_size>>>(
        static_cast<int>(nnr), grid_->gx(), grid_->gy(), grid_->gz(), grid_->gg(),
        vloc_types_.data(), static_cast<int>(atoms_->nat()), atoms_->pos_x(), atoms_->pos_y(),
        atoms_->pos_z(), atoms_->type(), v_g.data(), g2max);

    CHECK(cudaDeviceSynchronize());
    solver.backward(v_g);
    complex_to_real(nnr, v_g.data(), v.data());
    CHECK(cudaDeviceSynchronize());
}

double LocalPseudo::compute(const RealField& rho, RealField& v_out) {
    size_t n = grid_->nnr();
    RealField v_ps(grid_);
    compute(v_ps);

    double energy = rho.dot(v_ps) * grid_->dv();
    v_add(n, v_out.data(), v_ps.data(), v_out.data());
    return energy;
}

}  // namespace dftcu
