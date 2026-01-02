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
                                   const double* gz, double eta, double gcut, double* energy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnr) {
        const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
        double cur_gx = gx[i] * BOHR_TO_ANGSTROM;
        double cur_gy = gy[i] * BOHR_TO_ANGSTROM;
        double cur_gz = gz[i] * BOHR_TO_ANGSTROM;
        double g2 = cur_gx * cur_gx + cur_gy * cur_gy + cur_gz * cur_gz;

        if (g2 > 1e-12 && g2 <= gcut) {
            double str_fac_real = 0.0;
            double str_fac_imag = 0.0;
            for (int ia = 0; ia < nat; ++ia) {
                double ax = c_atom_x[ia] / BOHR_TO_ANGSTROM;
                double ay = c_atom_y[ia] / BOHR_TO_ANGSTROM;
                double az = c_atom_z[ia] / BOHR_TO_ANGSTROM;
                double gr = cur_gx * ax + cur_gy * ay + cur_gz * az;
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
        const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
        double ei = 0.0;
        double xi = pos_x[i] / BOHR_TO_ANGSTROM;
        double yi = pos_y[i] / BOHR_TO_ANGSTROM;
        double zi = pos_z[i] / BOHR_TO_ANGSTROM;
        double qi = charge[i];

        for (int j = 0; j < nat; ++j) {
            double qj = charge[j];
            double xj = pos_x[j] / BOHR_TO_ANGSTROM;
            double yj = pos_y[j] / BOHR_TO_ANGSTROM;
            double zj = pos_z[j] / BOHR_TO_ANGSTROM;

            for (int nx = -3; nx <= 3; ++nx) {
                for (int ny = -3; ny <= 3; ++ny) {
                    for (int nz = -3; nz <= 3; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;

                        double dx = xi - (xj + (nx * a00 + ny * a10 + nz * a20) / BOHR_TO_ANGSTROM);
                        double dy = yi - (yj + (nx * a01 + ny * a11 + nz * a21) / BOHR_TO_ANGSTROM);
                        double dz = zi - (zj + (nx * a02 + ny * a12 + nz * a22) / BOHR_TO_ANGSTROM);
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

Ewald::Ewald(Grid& grid, std::shared_ptr<Atoms> atoms, double precision, double gcut_hint)
    : grid_(grid), atoms_(atoms), precision_(precision) {
    double charge_sum = 0;
    for (double c : atoms_->h_charge())
        charge_sum += c;

    double gcut = (gcut_hint > 0) ? gcut_hint : grid_.g2max();

    constexpr double INITIAL_ALPHA = 2.9;
    constexpr double ALPHA_DECREMENT = 0.1;

    double alpha = INITIAL_ALPHA;
    for (int i = 0; i < 100; ++i) {
        double upperbound = 2.0 * charge_sum * charge_sum * sqrt(2.0 * alpha / constants::D_PI) *
                            erfc(sqrt(gcut / 4.0 / alpha));
        if (upperbound < precision_)
            break;
        alpha -= ALPHA_DECREMENT;
        if (alpha <= 0.05) {
            alpha = 0.05;
            break;
        }
    }
    eta_ = alpha;
}

double Ewald::get_best_eta() {
    double gmax = sqrt(grid_.g2max());
    double eta = (gmax * gmax) / (-4.0 * log(precision_));
    return eta;
}

double Ewald::compute_legacy() {
    double saved_eta = eta_;
    eta_ = get_best_eta();
    double real = compute_real();
    double recip = compute_recip_exact();
    double corr = compute_corr();
    eta_ = saved_eta;
    return real + recip + corr;
}

double Ewald::compute_recip_exact() {
    return compute_recip_exact(grid_.g2max());
}

double Ewald::compute_recip_exact(double gcut) {
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
        nnr, static_cast<int>(atoms_->nat()), grid_.gx(), grid_.gy(), grid_.gz(), eta_, gcut,
        energy.data());
    GPU_CHECK_KERNEL;

    double h_energy;
    energy.copy_to_host(&h_energy, grid_.stream());
    grid_.synchronize();

    const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
    double vol_bohr = grid_.volume() / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

    return 2.0 * constants::D_PI / vol_bohr * h_energy;
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

    const double BOHR_TO_ANGSTROM = constants::BOHR_TO_ANGSTROM;
    double vol_bohr = grid_.volume() / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

    double e_self = -charge_sq_sum * sqrt(eta_ / constants::D_PI);
    double e_bg = -0.5 * constants::D_PI * charge_sum * charge_sum / (eta_ * vol_bohr);

    return e_self + e_bg;
}

double Ewald::compute_recip_pme() {
    return compute_recip_exact();
}

double Ewald::compute(bool use_pme, double gcut) {
    if (gcut <= 0)
        gcut = grid_.g2max();
    double real = compute_real();
    double recip = use_pme ? compute_recip_pme() : compute_recip_exact(gcut);
    double corr = compute_corr();
    return real + recip + corr;
}

}  // namespace dftcu
