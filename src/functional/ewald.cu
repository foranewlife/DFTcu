#include <algorithm>
#include <cmath>
#include <numeric>

#include "ewald.cuh"
#include "fft/fft_solver.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void ewald_recip_kernel(size_t nnr, size_t nat, const double* gx, const double* gy,
                                   const double* gz, const double* pos_x, const double* pos_y,
                                   const double* pos_z, const double* charges, double eta,
                                   double* energy_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        double cur_gx = gx[i];
        double cur_gy = gy[i];
        double cur_gz = gz[i];
        double g2 = cur_gx * cur_gx + cur_gy * cur_gy + cur_gz * cur_gz;

        if (g2 < 1e-12)
            return;

        double sum_re = 0.0;
        double sum_im = 0.0;

        for (int j = 0; j < nat; ++j) {
            double phase = -(cur_gx * pos_x[j] + cur_gy * pos_y[j] + cur_gz * pos_z[j]);
            double cos_p, sin_p;
            sincos(phase, &sin_p, &cos_p);
            sum_re += charges[j] * cos_p;
            sum_im += charges[j] * sin_p;
        }

        double strf_sq = sum_re * sum_re + sum_im * sum_im;
        double term = strf_sq * exp(-g2 / (4.0 * eta)) / g2;

        atomicAdd(energy_out, term);
    }
}

__global__ void ewald_real_kernel(size_t nat, const double* pos_x, const double* pos_y,
                                  const double* pos_z, const double* charges, double eta,
                                  double a1x, double a1y, double a1z, double a2x, double a2y,
                                  double a2z, double a3x, double a3y, double a3z, int n_images,
                                  double* energy_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nat) {
        double ei = 0.0;
        double xi = pos_x[i];
        double yi = pos_y[i];
        double zi = pos_z[i];
        double zi_charge = charges[i];
        double sqrt_eta = sqrt(eta);

        for (int j = 0; j < nat; ++j) {
            double xj = pos_x[j];
            double yj = pos_y[j];
            double zj = pos_z[j];
            double zj_charge = charges[j];

            for (int ix = -n_images; ix <= n_images; ++ix) {
                for (int iy = -n_images; iy <= n_images; ++iy) {
                    for (int iz = -n_images; iz <= n_images; ++iz) {
                        if (i == j && ix == 0 && iy == 0 && iz == 0)
                            continue;

                        double rx = xj - xi + ix * a1x + iy * a2x + iz * a3x;
                        double ry = yj - yi + ix * a1y + iy * a2y + iz * a3y;
                        double rz = zj - zi + ix * a1z + iy * a2z + iz * a3z;

                        double r2 = rx * rx + ry * ry + rz * rz;
                        double r = sqrt(r2);

                        if (r > 1e-10) {
                            ei += zi_charge * zj_charge * erfc(sqrt_eta * r) / r;
                        }
                    }
                }
            }
        }
        atomicAdd(energy_out, ei);
    }
}

__device__ void calc_Mn_pme(double x, int order, double* Mn) {
    for (int i = 0; i <= order; ++i)
        Mn[i] = 0.0;
    Mn[1] = x;
    Mn[2] = 1.0 - x;
    for (int i = 3; i <= order; ++i) {
        for (int j = 0; j < i; ++j) {
            int n = i - j;
            Mn[n] = ((x + n - 1) * Mn[n] + (j + 1 - x) * Mn[n - 1]) / (i - 1);
        }
    }
}

__global__ void spread_charges_pme_kernel(size_t nat, const double* pos_x, const double* pos_y,
                                          const double* pos_z, const double* charges, int nr0,
                                          int nr1, int nr2, double b00, double b01, double b02,
                                          double b10, double b11, double b12, double b20,
                                          double b21, double b22, int order, double dv, double* Q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nat) {
        double px = pos_x[i];
        double py = pos_y[i];
        double pz = pos_z[i];
        double zi_density = charges[i] / dv;

        const double PI_INV = 1.0 / (2.0 * PI);
        double u0_raw = (px * b00 + py * b01 + pz * b02) * PI_INV;
        double u1_raw = (px * b10 + py * b11 + pz * b12) * PI_INV;
        double u2_raw = (px * b20 + py * b21 + pz * b22) * PI_INV;

        double u0 = (u0_raw - floor(u0_raw)) * nr0;
        double u1 = (u1_raw - floor(u1_raw)) * nr1;
        double u2 = (u2_raw - floor(u2_raw)) * nr2;

        double Mn0[20], Mn1[20], Mn2[20];
        calc_Mn_pme(u0 - floor(u0), order, Mn0);
        calc_Mn_pme(u1 - floor(u1), order, Mn1);
        calc_Mn_pme(u2 - floor(u2), order, Mn2);

        int i0_base = static_cast<int>(floor(u0));
        int i1_base = static_cast<int>(floor(u1));
        int i2_base = static_cast<int>(floor(u2));

        for (int k0 = 0; k0 < order; ++k0) {
            int idx0 = (i0_base - k0 + 1 + nr0 * 10) % nr0;
            double w0 = Mn0[k0 + 1];
            for (int k1 = 0; k1 < order; ++k1) {
                int idx1 = (i1_base - k1 + 1 + nr1 * 10) % nr1;
                double w1 = Mn1[k1 + 1];
                for (int k2 = 0; k2 < order; ++k2) {
                    int idx2 = (i2_base - k2 + 1 + nr2 * 10) % nr2;
                    double w2 = Mn2[k2 + 1];

                    int grid_idx = (idx2 * nr1 + idx1) * nr0 + idx0;
                    atomicAdd(&Q[grid_idx], zi_density * w0 * w1 * w2);
                }
            }
        }
    }
}

__global__ void apply_pme_correction_kernel(size_t nnr, int nr0, int nr1, int nr2, const double* gg,
                                            int order, double eta, gpufftComplex* Q_g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        double g2 = gg[i];
        if (g2 < 1e-12) {
            Q_g[i].x = 0;
            Q_g[i].y = 0;
            return;
        }

        int n2 = i / (nr0 * nr1);
        int n1 = (i % (nr0 * nr1)) / nr0;
        int n0 = i % nr0;

        auto get_freq = [](int n, int nr) {
            return (n < (nr + 1) / 2) ? (double)n : (double)(n - nr);
        };

        double f0 = get_freq(n0, nr0);
        double f1 = get_freq(n1, nr1);
        double f2 = get_freq(n2, nr2);

        auto get_bm = [order](double f, int nr) {
            double q = 2.0 * PI * f / nr;
            // Precomputed Mn(k) for order 10 at x=1.0 (from DFTpy calc_Mn(1.0))
            double M[11] = {0.0,
                            2.7557319223985893e-06,
                            0.0013833774250440916,
                            0.040255731922398584,
                            0.24314925044091706,
                            0.4304177689594356,
                            0.24314925044091706,
                            0.040255731922398584,
                            0.0013833774250440916,
                            2.7557319223985893e-06,
                            0.0};

            if (order != 10) {
                for (int i = 0; i < 11; ++i)
                    M[i] = 0;
                M[1] = 1.0;
                for (int k = 3; k <= order; ++k) {
                    for (int j = k - 1; j >= 1; --j) {
                        M[j] = (j * M[j] + (k - j) * M[j - 1]) / (k - 1);
                    }
                }
            }

            gpufftComplex factor = {0, 0};
            for (int k = 1; k < order; ++k) {
                double phase = k * q;
                factor.x += (float)(M[k] * cos(phase));
                factor.y += (float)(M[k] * sin(phase));
            }

            double phase_top = (order - 1.0) * q;
            gpufftComplex top = {(float)cos(phase_top), (float)-sin(phase_top)};
            double den = factor.x * factor.x + factor.y * factor.y;
            gpufftComplex res;
            res.x = (top.x * factor.x + top.y * factor.y) / (float)den;
            res.y = (top.y * factor.x - top.x * factor.y) / (float)den;
            return res;
        };

        gpufftComplex bm0 = get_bm(f0, nr0);
        gpufftComplex bm1 = get_bm(f1, nr1);
        gpufftComplex bm2 = get_bm(f2, nr2);

        gpufftComplex BG;
        BG.x = bm0.x * (bm1.x * bm2.x - bm1.y * bm2.y) - bm0.y * (bm1.x * bm2.y + bm1.y * bm2.x);
        BG.y = bm0.x * (bm1.x * bm2.y + bm1.y * bm2.x) + bm0.y * (bm1.x * bm2.x - bm1.y * bm2.y);

        gpufftComplex strf;
        strf.x = Q_g[i].x * BG.x - Q_g[i].y * BG.y;
        strf.y = Q_g[i].x * BG.y + Q_g[i].y * BG.x;

        double strf_sq = strf.x * strf.x + strf.y * strf.y;
        double term = strf_sq * exp(-g2 / (4.0 * eta)) / g2;

        Q_g[i].x = (float)term;
        Q_g[i].y = 0;
    }
}

__global__ void sum_energy_kernel(size_t n, const gpufftComplex* data, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(out, (double)data[i].x);
}
}  // namespace

Ewald::Ewald(const Grid& grid, const Atoms& atoms, double precision, int bspline_order)
    : grid_(grid), atoms_(atoms), precision_(precision), bspline_order_(bspline_order) {
    eta_ = get_best_eta();
}

double Ewald::get_best_eta() {
    double charge_sq_sum = 0.0;
    std::vector<double> h_charge(atoms_.nat());
    cudaMemcpy(h_charge.data(), atoms_.charge(), atoms_.nat() * sizeof(double),
               cudaMemcpyDeviceToHost);
    for (double c : h_charge)
        charge_sq_sum += c * c;

    double gmax = sqrt(grid_.g2max());
    double eta = 1.6;
    bool not_good = true;
    while (not_good && eta > 0.01) {
        double upbound = 4.0 * PI * atoms_.nat() * charge_sq_sum * sqrt(eta / PI) *
                         erfc(gmax / 2.0 * sqrt(1.0 / eta));
        if (upbound < precision_) {
            not_good = false;
        } else {
            eta -= 0.01;
        }
    }
    return eta;
}

double Ewald::compute_recip_exact() {
    double* d_energy;
    gpuMalloc(&d_energy, sizeof(double));
    gpuMemset(d_energy, 0, sizeof(double));

    size_t nnr = grid_.nnr();
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    ewald_recip_kernel<<<grid_size, block_size>>>(nnr, atoms_.nat(), grid_.gx(), grid_.gy(),
                                                  grid_.gz(), atoms_.pos_x(), atoms_.pos_y(),
                                                  atoms_.pos_z(), atoms_.charge(), eta_, d_energy);
    GPU_CHECK_KERNEL

    double h_energy;
    gpuMemcpy(&h_energy, d_energy, sizeof(double), gpuMemcpyDeviceToHost);
    gpuFree(d_energy);

    return 2.0 * PI * h_energy / grid_.volume();
}

double Ewald::compute_real() {
    auto lattice = grid_.lattice();
    double e_tol = precision_ / 10.0;
    double rmax = sqrt(-log(e_tol)) / sqrt(eta_);

    double l[3];
    for (int i = 0; i < 3; ++i) {
        l[i] = sqrt(lattice[i][0] * lattice[i][0] + lattice[i][1] * lattice[i][1] +
                    lattice[i][2] * lattice[i][2]);
    }

    int n1 = static_cast<int>(ceil(rmax / l[0]));
    int n2 = static_cast<int>(ceil(rmax / l[1]));
    int n3 = static_cast<int>(ceil(rmax / l[2]));
    int n_images = std::max({n1, n2, n3, 1});

    double* d_energy;
    gpuMalloc(&d_energy, sizeof(double));
    gpuMemset(d_energy, 0, sizeof(double));

    const int block_size = 64;
    const int grid_size = (atoms_.nat() + block_size - 1) / block_size;

    ewald_real_kernel<<<grid_size, block_size>>>(
        atoms_.nat(), atoms_.pos_x(), atoms_.pos_y(), atoms_.pos_z(), atoms_.charge(), eta_,
        lattice[0][0], lattice[0][1], lattice[0][2], lattice[1][0], lattice[1][1], lattice[1][2],
        lattice[2][0], lattice[2][1], lattice[2][2], n_images, d_energy);
    GPU_CHECK_KERNEL

    double h_energy;
    gpuMemcpy(&h_energy, d_energy, sizeof(double), gpuMemcpyDeviceToHost);
    gpuFree(d_energy);

    return 0.5 * h_energy;
}

double Ewald::compute_corr() {
    double charge_sq_sum = 0.0;
    double charge_sum = 0.0;
    std::vector<double> h_charge(atoms_.nat());
    cudaMemcpy(h_charge.data(), atoms_.charge(), atoms_.nat() * sizeof(double),
               cudaMemcpyDeviceToHost);
    for (double c : h_charge) {
        charge_sq_sum += c * c;
        charge_sum += c;
    }

    double e_self = -sqrt(eta_ / PI) * charge_sq_sum;
    double e_bg = -0.5 * PI * charge_sum * charge_sum / (eta_ * grid_.volume());

    return e_self + e_bg;
}

double Ewald::compute_recip_pme() {
    size_t nnr = grid_.nnr();
    RealField Q(grid_);
    Q.fill(0.0);

    const int spread_block = 64;
    const int spread_grid = (atoms_.nat() + spread_block - 1) / spread_block;

    auto b = grid_.rec_lattice();

    spread_charges_pme_kernel<<<spread_grid, spread_block>>>(
        atoms_.nat(), atoms_.pos_x(), atoms_.pos_y(), atoms_.pos_z(), atoms_.charge(),
        grid_.nr()[0], grid_.nr()[1], grid_.nr()[2], b[0][0], b[0][1], b[0][2], b[1][0], b[1][1],
        b[1][2], b[2][0], b[2][1], b[2][2], bspline_order_, grid_.dv(), Q.data());
    GPU_CHECK_KERNEL

    ComplexField Q_g(grid_);
    real_to_complex(nnr, Q.data(), Q_g.data());

    FFTSolver solver(grid_);
    solver.forward(Q_g);

    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;
    apply_pme_correction_kernel<<<grid_size, block_size>>>(nnr, grid_.nr()[0], grid_.nr()[1],
                                                           grid_.nr()[2], grid_.gg(),
                                                           bspline_order_, eta_, Q_g.data());
    GPU_CHECK_KERNEL

    double* d_sum;
    gpuMalloc(&d_sum, sizeof(double));
    gpuMemset(d_sum, 0, sizeof(double));

    sum_energy_kernel<<<grid_size, block_size>>>(nnr, Q_g.data(), d_sum);

    double h_energy;
    gpuMemcpy(&h_energy, d_sum, sizeof(double), gpuMemcpyDeviceToHost);
    gpuFree(d_sum);

    return 2.0 * PI * h_energy / grid_.volume();
}

double Ewald::compute(bool use_pme) {
    double real = compute_real();
    double recip = use_pme ? compute_recip_pme() : compute_recip_exact();
    double corr = compute_corr();

    // printf("Ewald Debug (%s): Real=%.6f, Recip=%.6f, Corr=%.6f, Total=%.6f\n",
    //        use_pme ? "PME" : "Exact", real, recip, corr, real + recip + corr);

    return real + recip + corr;
}

}  // namespace dftcu
