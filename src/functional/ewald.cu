#include <cmath>
#include <vector>

#include "ewald.cuh"
#include "fft/fft_solver.cuh"
#include "utilities/common.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

const int MAX_ATOMS = 512;
__constant__ double c_atom_x[MAX_ATOMS];
__constant__ double c_atom_y[MAX_ATOMS];
__constant__ double c_atom_z[MAX_ATOMS];
__constant__ double c_atom_q[MAX_ATOMS];

namespace {
__global__ void ewald_recip_kernel(size_t nnr, int nat, const double* gx, const double* gy,
                                   const double* gz, double eta, double* energy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnr) {
        double g2 = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
        if (g2 > 1e-12) {
            double str_fac_real = 0.0;
            double str_fac_imag = 0.0;
            for (int ia = 0; ia < nat; ++ia) {
                double gr = gx[i] * c_atom_x[ia] + gy[i] * c_atom_y[ia] + gz[i] * c_atom_z[ia];
                str_fac_real += c_atom_q[ia] * cos(gr);
                str_fac_imag -= c_atom_q[ia] * sin(gr);
            }
            double str_fac_sq = str_fac_real * str_fac_real + str_fac_imag * str_fac_imag;
            double val = exp(-g2 / (4.0 * eta)) / g2 * str_fac_sq;
            atomicAdd(energy, val);
        }
    }
}

__global__ void ewald_real_kernel(int nat, const double* pos_x, const double* pos_y,
                                  const double* pos_z, const double* charge, double eta,
                                  double rmax, double a00, double a01, double a02, double a10,
                                  double a11, double a12, double a20, double a21, double a22,
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

            for (int nx = -3; nx <= 3; ++nx) {
                for (int ny = -3; ny <= 3; ++ny) {
                    for (int nz = -3; nz <= 3; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;

                        double dx = xi - (xj + nx * a00 + ny * a10 + nz * a20);
                        double dy = yi - (yj + nx * a01 + ny * a11 + nz * a21);
                        double dz = zi - (zj + nx * a02 + ny * a12 + nz * a22);
                        double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 < rmax * rmax && r2 > 1e-14) {
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

Ewald::Ewald(Grid& grid, std::shared_ptr<Atoms> atoms, double precision, int bspline_order)
    : grid_(grid), atoms_(atoms), precision_(precision), bspline_order_(bspline_order) {
    eta_ = get_best_eta();
}

double Ewald::get_best_eta() {
    double vol = grid_.volume();
    double gmax = sqrt(grid_.g2max());
    double eta = (gmax * gmax) / (-4.0 * log(precision_));
    return eta;
}

double Ewald::compute_recip_exact() {
    GPU_Vector<double> energy(1);
    energy.fill(0.0, grid_.stream());

    if (atoms_->nat() > MAX_ATOMS) {
        throw std::runtime_error(
            "Number of atoms exceeds MAX_ATOMS for constant memory optimization in Ewald.");
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
    CHECK(cudaMemcpyToSymbolAsync(c_atom_q, atoms_->h_charge().data(),
                                  atoms_->nat() * sizeof(double), 0, cudaMemcpyHostToDevice,
                                  grid_.stream()));

    size_t nnr = grid_.nnr();
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    ewald_recip_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        nnr, static_cast<int>(atoms_->nat()), grid_.gx(), grid_.gy(), grid_.gz(), eta_,
        energy.data());
    GPU_CHECK_KERNEL;

    double h_energy;
    energy.copy_to_host(&h_energy, grid_.stream());
    grid_.synchronize();
    return 2.0 * constants::D_PI / grid_.volume() * h_energy;
}

double Ewald::compute_real() {
    auto lattice = grid_.lattice();
    double e_tol = precision_ / 10.0;
    double rmax = sqrt(-log(e_tol)) / sqrt(eta_);

    GPU_Vector<double> energy(1);
    energy.fill(0.0, grid_.stream());

    const int block_size = 128;
    const int grid_size = (static_cast<int>(atoms_->nat()) + block_size - 1) / block_size;

    ewald_real_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        static_cast<int>(atoms_->nat()), atoms_->pos_x(), atoms_->pos_y(), atoms_->pos_z(),
        atoms_->charge_data(), eta_, rmax, lattice[0][0], lattice[0][1], lattice[0][2],
        lattice[1][0], lattice[1][1], lattice[1][2], lattice[2][0], lattice[2][1], lattice[2][2],
        energy.data());

    double h_energy;
    energy.copy_to_host(&h_energy, grid_.stream());
    grid_.synchronize();
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
    double e_bg = -0.5 * constants::D_PI * charge_sum * charge_sum / (eta_ * grid_.volume());

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
