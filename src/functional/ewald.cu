#include <cmath>
#include <vector>

#include "ewald.cuh"
#include "fft/fft_solver.cuh"
#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void ewald_recip_kernel(size_t nnr, int nat, const double* gx, const double* gy,
                                   const double* gz, const double* pos_x, const double* pos_y,
                                   const double* pos_z, const double* charge, double eta,
                                   double* energy_out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnr) {
        double cur_gx = gx[i];
        double cur_gy = gy[i];
        double cur_gz = gz[i];
        double g2 = cur_gx * cur_gx + cur_gy * cur_gy + cur_gz * cur_gz;

        if (g2 < 1e-12)
            return;

        double struc_re = 0.0;
        double struc_im = 0.0;
        for (int j = 0; j < nat; ++j) {
            double gr = cur_gx * pos_x[j] + cur_gy * pos_y[j] + cur_gz * pos_z[j];
            double s, c;
            sincos(gr, &s, &c);
            struc_re += charge[j] * c;
            struc_im += charge[j] * s;
        }

        double term = (struc_re * struc_re + struc_im * struc_im) * exp(-g2 / (4.0 * eta)) / g2;
        atomicAdd(energy_out, term);
    }
}

__global__ void ewald_real_kernel(int nat, const double* pos_x, const double* pos_y,
                                  const double* pos_z, const double* charge, double eta,
                                  double rmax, double L0, double L1, double L2,
                                  double* energy_out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nat) {
        double ei = 0.0;
        double xi = pos_x[i];
        double yi = pos_y[i];
        double zi = pos_z[i];
        double qi = charge[i];

        for (int j = 0; j < nat; ++j) {
            double qj = charge[j];
            double xj = pos_x[j];
            double yj = pos_y[j];
            double zj = pos_z[j];

            for (int nx = -1; nx <= 1; ++nx) {
                for (int ny = -1; ny <= 1; ++ny) {
                    for (int nz = -1; nz <= 1; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;

                        double dx = xi - (xj + nx * L0);
                        double dy = yi - (yj + ny * L1);
                        double dz = zi - (zj + nz * L2);
                        double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 < rmax * rmax) {
                            double r = sqrt(r2);
                            ei += qi * qj * erfc(sqrt(eta) * r) / r;
                        }
                    }
                }
            }
        }
        atomicAdd(energy_out, 0.5 * ei);
    }
}
}  // namespace

Ewald::Ewald(std::shared_ptr<Grid> grid, std::shared_ptr<Atoms> atoms, double precision,
             int bspline_order)
    : grid_(grid), atoms_(atoms), precision_(precision), bspline_order_(bspline_order) {
    eta_ = get_best_eta();
}

double Ewald::get_best_eta() {
    double vol = grid_->volume();
    double gmax = sqrt(grid_->g2max());
    double eta = (gmax * gmax) / (-4.0 * log(precision_));
    return eta;
}

double Ewald::compute_recip_exact() {
    GPU_Vector<double> energy(1);
    energy.fill(0.0);

    size_t nnr = grid_->nnr();
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    ewald_recip_kernel<<<grid_size, block_size>>>(nnr, static_cast<int>(atoms_->nat()), grid_->gx(),
                                                  grid_->gy(), grid_->gz(), atoms_->pos_x(),
                                                  atoms_->pos_y(), atoms_->pos_z(),
                                                  atoms_->charge_data(), eta_, energy.data());

    double h_energy;
    energy.copy_to_host(&h_energy);
    return 2.0 * constants::D_PI / grid_->volume() * h_energy;
}

double Ewald::compute_real() {
    auto lattice = grid_->lattice();
    double e_tol = precision_ / 10.0;
    double rmax = sqrt(-log(e_tol)) / sqrt(eta_);

    GPU_Vector<double> energy(1);
    energy.fill(0.0);

    const int block_size = 128;
    const int grid_size = (static_cast<int>(atoms_->nat()) + block_size - 1) / block_size;

    ewald_real_kernel<<<grid_size, block_size>>>(static_cast<int>(atoms_->nat()), atoms_->pos_x(),
                                                 atoms_->pos_y(), atoms_->pos_z(),
                                                 atoms_->charge_data(), eta_, rmax, lattice[0][0],
                                                 lattice[1][1], lattice[2][2], energy.data());

    double h_energy;
    energy.copy_to_host(&h_energy);
    return h_energy;
}

double Ewald::compute_corr() {
    double charge_sum = 0;
    double charge_sq_sum = 0;
    for (double c : atoms_->h_charge()) {
        charge_sum += c;
        charge_sq_sum += c * c;
    }

    double e_self = -sqrt(eta_ / constants::D_PI) * charge_sq_sum;
    double e_bg = -0.5 * constants::D_PI * charge_sum * charge_sum / (eta_ * grid_->volume());

    return e_self + e_bg;
}

double Ewald::compute_recip_pme() {
    return compute_recip_exact();
}

double Ewald::compute(bool use_pme) {
    double real = compute_real();
    double recip = use_pme ? compute_recip_pme() : compute_recip_exact();
    double corr = compute_corr();
    return real + recip + corr;
}

}  // namespace dftcu
