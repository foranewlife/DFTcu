#include <cmath>

#include "fft/fft_solver.cuh"
#include "model/density_builder.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"
#include "utilities/math_utils.cuh"

namespace dftcu {

namespace {

// Reuse constant memory for atom data
static __device__ __constant__ double c_atom_x[constants::MAX_ATOMS_PSEUDO];
static __device__ __constant__ double c_atom_y[constants::MAX_ATOMS_PSEUDO];
static __device__ __constant__ double c_atom_z[constants::MAX_ATOMS_PSEUDO];
static __device__ __constant__ int c_atom_type[constants::MAX_ATOMS_PSEUDO];

__global__ void density_sum_kernel(int nnr, const double* gx, const double* gy, const double* gz,
                                   const double* gg, const double* tab_rho, int nat, int nqx,
                                   double dq, double gcut, gpufftComplex* rho_g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        const double BOHR_TO_ANGSTROM = 0.529177210903;
        double g2 = gg[i] * (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);
        double gmod = sqrt(g2);

        if (gcut > 0 && g2 > gcut) {
            rho_g[i].x = 0.0;
            rho_g[i].y = 0.0;
            return;
        }

        double sum_re = 0.0;
        double sum_im = 0.0;

        for (int j = 0; j < nat; ++j) {
            int type = c_atom_type[j];
            double rho_at_g = 0.0;

            // 4-point cubic Lagrange interpolation (QE Style)
            // tab_rho[type * nqx + 1] is q=0
            int i0 = (int)(gmod / dq) + 1;
            i0 = min(max(i0, 1), nqx - 4);
            double px = gmod / dq - (double)(i0 - 1);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;

            rho_at_g = tab_rho[type * nqx + i0] * ux * vx * wx / 6.0 +
                       tab_rho[type * nqx + i0 + 1] * px * vx * wx / 2.0 -
                       tab_rho[type * nqx + i0 + 2] * px * ux * wx / 2.0 +
                       tab_rho[type * nqx + i0 + 3] * px * ux * vx / 6.0;

            double phase = (gx[i] * c_atom_x[j] + gy[i] * c_atom_y[j] + gz[i] * c_atom_z[j]);
            double s, c;
            sincos(phase, &s, &c);
            sum_re += rho_at_g * c;
            sum_im -= rho_at_g * s;
        }
        rho_g[i].x = sum_re;
        rho_g[i].y = sum_im;
    }
}

}  // namespace

DensityBuilder::DensityBuilder(Grid& grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms) {}

void DensityBuilder::set_atomic_rho_g(int type, const std::vector<double>& q,
                                      const std::vector<double>& rho_q) {
    if (type >= num_types_) {
        num_types_ = type + 1;
        tab_rho_g_.resize(num_types_);
    }
    // We expect q to be (0, dq, 2dq...), but for internal 1-based logic
    // we pad it so tab[1] is q=0
    tab_rho_g_[type].resize(rho_q.size() + 1);
    std::copy(rho_q.begin(), rho_q.end(), tab_rho_g_[type].begin() + 1);
    nqx_ = static_cast<int>(tab_rho_g_[type].size());
}

void DensityBuilder::set_atomic_rho_r(int type, const std::vector<double>& r,
                                      const std::vector<double>& rho_r,
                                      const std::vector<double>& rab) {
    const double BOHR_TO_ANGSTROM = 0.529177210903;
    double qmax = sqrt(grid_.g2max() * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM) * 1.5;
    nqx_ = static_cast<int>(qmax / dq_) + 5;

    if (type >= num_types_) {
        num_types_ = type + 1;
        tab_rho_g_.resize(num_types_);
    }
    tab_rho_g_[type].resize(nqx_);

    double omega_bohr = grid_.volume_bohr();
    int msh = r.size();
    std::vector<double> aux(msh);

    // iq=1 corresponds to q=0
    for (int iq = 1; iq < nqx_; ++iq) {
        double q = (iq - 1) * dq_;
        if (iq == 1) {
            for (int ir = 0; ir < msh; ++ir)
                aux[ir] = rho_r[ir];
        } else {
            for (int ir = 0; ir < msh; ++ir) {
                if (r[ir] < 1e-12)
                    aux[ir] = rho_r[ir];
                else
                    aux[ir] = rho_r[ir] * sin(q * r[ir]) / (q * r[ir]);
            }
        }
        tab_rho_g_[type][iq] = simpson_integrate(aux, rab) / omega_bohr;
    }
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

    // Prepare flat table for GPU
    if (d_tab_.size() != num_types_ * nqx_) {
        std::vector<double> flat_table(num_types_ * nqx_, 0.0);
        for (int t = 0; t < num_types_; ++t) {
            if (tab_rho_g_[t].size() >= nqx_) {
                std::copy(tab_rho_g_[t].begin(), tab_rho_g_[t].begin() + nqx_,
                          flat_table.begin() + t * nqx_);
            }
        }
        d_tab_.resize(flat_table.size());
        d_tab_.copy_from_host(flat_table.data(), grid_.stream());
    }

    const int block_size = 256;
    const int grid_size = (static_cast<int>(nnr) + block_size - 1) / block_size;

    density_sum_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        static_cast<int>(nnr), grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), d_tab_.data(),
        static_cast<int>(atoms_->nat()), nqx_, dq_, gcut_, rho_g.data());

    GPU_CHECK_KERNEL;
    solver.backward(rho_g);
    complex_to_real(nnr, rho_g.data(), rho.data(), grid_.stream());
    grid_.synchronize();
}

}  // namespace dftcu
