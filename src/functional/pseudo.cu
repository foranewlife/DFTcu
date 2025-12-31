#include "fft/fft_solver.cuh"
#include "pseudo.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {

__constant__ double c_pseudo_atom_x[constants::MAX_ATOMS_PSEUDO];
__constant__ double c_pseudo_atom_y[constants::MAX_ATOMS_PSEUDO];
__constant__ double c_pseudo_atom_z[constants::MAX_ATOMS_PSEUDO];
__constant__ int c_pseudo_atom_type[constants::MAX_ATOMS_PSEUDO];

__global__ void pseudo_rec_kernel(int nnr, const double* gx, const double* gy, const double* gz,
                                  const double* gg, const double* vloc_types, int nat,
                                  gpufftComplex* v_g, double g2max, double omega,
                                  const double* zv) {
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
            int type = c_pseudo_atom_type[j];
            double phase = (cur_gx * c_pseudo_atom_x[j] + cur_gy * c_pseudo_atom_y[j] +
                            cur_gz * c_pseudo_atom_z[j]);
            double s, c;
            sincos(phase, &s, &c);

            // Get vloc_short(G) and multiply by nnr to compensate for 1/nnr FFT normalization
            // that will be applied later in v_scale(nnr, 1.0/nnr, ...)
            double v_val = vloc_types[type * nnr + i] * (double)nnr;

            // Add back the analytical FT of -zv*erf(r)/r
            // QE formula: vloc(G) = vloc_short(G) - (4π*zv*e²/(Ω*G²)) * exp(-G²/4)
            // where e² = 1 Ha·Bohr in Hartree atomic units
            //
            // CRITICAL: Grid stores gg in Å^-2 and omega in Å³, but the erf formula
            // requires Bohr units. Must convert!
            //
            // CRITICAL: Both vloc_short and erf term are multiplied by nnr to compensate
            // for the 1/nnr FFT normalization applied later in v_scale(...)
            //
            // NOTE: vloc_short from set_vloc_radial must be computed correctly:
            //   integral = simpson(aux, x=r_grid)  # NOT simpson(aux*rab, x=r_grid)!
            if (gg[i] > 1e-12) {
                const double fpi = 4.0 * constants::D_PI;
                const double BOHR_TO_ANGSTROM = 0.529177;

                // Convert gg from Å^-2 to Bohr^-2
                const double gg_bohr2 = gg[i] / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

                // Convert omega from Å³ to Bohr³
                const double omega_bohr =
                    omega / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

                // In Hartree atomic units: e² = 1 Ha·Bohr
                const double fac = fpi * zv[type] / omega_bohr;  // Ha

                // CRITICAL: Both vloc_short and erf term are multiplied by nnr to compensate
                // for the 1/nnr FFT normalization applied later in v_scale(...)
                v_val -= fac * exp(-0.25 * gg_bohr2) / gg_bohr2 * (double)nnr;
            }

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

LocalPseudo::LocalPseudo(Grid& grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms), v_g_(grid) {}

void LocalPseudo::set_vloc(int type, const std::vector<double>& vloc_g) {
    if (type >= num_types_) {
        int new_num_types = type + 1;
        GPU_Vector<double> new_vloc(grid_.nnr() * new_num_types);
        if (num_types_ > 0) {
            CHECK(cudaMemcpyAsync(new_vloc.data(), vloc_types_.data(),
                                  grid_.nnr() * num_types_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, grid_.stream()));
        }
        vloc_types_ = std::move(new_vloc);
        num_types_ = new_num_types;
    }
    CHECK(cudaMemcpyAsync(vloc_types_.data() + type * grid_.nnr(), vloc_g.data(),
                          vloc_g.size() * sizeof(double), cudaMemcpyHostToDevice, grid_.stream()));
}

void LocalPseudo::set_vloc_radial(int type, const std::vector<double>& q,
                                  const std::vector<double>& v_q) {
    if (type >= num_types_) {
        int new_num_types = type + 1;
        GPU_Vector<double> new_vloc(grid_.nnr() * new_num_types);
        if (num_types_ > 0) {
            CHECK(cudaMemcpyAsync(new_vloc.data(), vloc_types_.data(),
                                  grid_.nnr() * num_types_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, grid_.stream()));
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

    d_q.copy_from_host(q.data(), grid_.stream());
    d_a.copy_from_host(a.data(), grid_.stream());
    d_b.copy_from_host(b.data(), grid_.stream());
    d_c.copy_from_host(c.data(), grid_.stream());
    d_d.copy_from_host(d.data(), grid_.stream());

    const int block_size = 256;
    const int grid_size = (static_cast<int>(grid_.nnr()) + block_size - 1) / block_size;

    interpolate_radial_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        grid_.nnr(), grid_.gg(), static_cast<int>(q.size()), d_q.data(), d_a.data(), d_b.data(),
        d_c.data(), d_d.data(), vloc_types_.data() + type * grid_.nnr());

    GPU_CHECK_KERNEL;
}

void LocalPseudo::set_valence_charge(int type, double zv) {
    // Resize zv_ if needed
    int new_num_types = type + 1;
    if (new_num_types > static_cast<int>(zv_.size())) {
        GPU_Vector<double> new_zv(new_num_types);
        std::vector<double> host_zv(new_num_types, 0.0);

        // Copy old values if any
        if (zv_.size() > 0) {
            std::vector<double> old_vals(zv_.size());
            CHECK(cudaMemcpy(old_vals.data(), zv_.data(), zv_.size() * sizeof(double),
                             cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < zv_.size(); ++i) {
                host_zv[i] = old_vals[i];
            }
        }

        // Set new value
        host_zv[type] = zv;

        // Copy to GPU
        CHECK(cudaMemcpy(new_zv.data(), host_zv.data(), new_num_types * sizeof(double),
                         cudaMemcpyHostToDevice));

        zv_ = std::move(new_zv);
    } else {
        // Just update the single value
        CHECK(cudaMemcpy(zv_.data() + type, &zv, sizeof(double), cudaMemcpyHostToDevice));
    }
}

void LocalPseudo::compute(RealField& v) {
    size_t nnr = grid_.nnr();
    FFTSolver solver(grid_);

    if (atoms_->nat() > constants::MAX_ATOMS_PSEUDO) {
        throw std::runtime_error(
            "Number of atoms exceeds MAX_ATOMS_PSEUDO for constant memory optimization in "
            "LocalPseudo.");
    }
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_x, atoms_->h_pos_x().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_y, atoms_->h_pos_y().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_z, atoms_->h_pos_z().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));
    CHECK(cudaMemcpyToSymbolAsync(c_pseudo_atom_type, atoms_->h_type().data(),
                                  atoms_->nat() * sizeof(int), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));

    const int block_size = 256;
    const int grid_size = (static_cast<int>(nnr) + block_size - 1) / block_size;

    // Apply QE's G² cutoff from ecutrho = 120 Ry = 60 Ha
    // In atomic units: G²_max = 2 * ecutrho = 120 Bohr⁻²
    // Converted to Å⁻²: 120 / (0.529177²) = 428.528121 Å⁻²
    // This ensures DFTcu uses the same G-space range as QE
    const double ECUTRHO_HA = 60.0;  // Ha
    const double BOHR_TO_ANG = 0.529177;
    double g2max = 2.0 * ECUTRHO_HA / (BOHR_TO_ANG * BOHR_TO_ANG);  // ≈ 428.53 Å⁻²

    double omega = grid_.volume();

    pseudo_rec_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        static_cast<int>(nnr), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), vloc_types_.data(),
        static_cast<int>(atoms_->nat()), v_g_.data(), g2max, omega, zv_.data());

    GPU_CHECK_KERNEL;
    solver.backward(v_g_);
    complex_to_real(nnr, v_g_.data(), v.data(), grid_.stream());
    v_scale(nnr, 1.0 / (double)nnr, v.data(), v.data(), grid_.stream());
}

double LocalPseudo::compute(const RealField& rho, RealField& v_out) {
    size_t n = grid_.nnr();
    RealField v_ps(grid_);
    compute(v_ps);

    grid_.synchronize();
    double energy = rho.dot(v_ps) * grid_.dv();
    v_add(n, v_out.data(), v_ps.data(), v_out.data());
    return energy;
}
}  // namespace dftcu
