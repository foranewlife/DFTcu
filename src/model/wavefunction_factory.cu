#include <algorithm>
#include <cmath>

#include "math/bessel.cuh"
#include "math/ylm.cuh"
#include "model/wavefunction_factory.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/math_utils.cuh"

#include <curand_kernel.h>

namespace dftcu {

namespace {

__global__ void build_atomic_band_kernel(int nnr, const double* atom_x, const double* atom_y,
                                         const double* atom_z, int iat, int l, int m_idx,
                                         const double* gx, const double* gy, const double* gz,
                                         const double* gg, const double* tab_chi, int nqx,
                                         double dq, double omega_bohr, double encut_hartree,
                                         gpufftComplex* band_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        // gg is already in Bohr^-2, convert to Rydberg: multiply by 2
        double g2_ry = gg[i] * 2.0;

        if (g2_ry > encut_hartree * 2.0 + 1e-5)
            return;

        double gmod_phys = sqrt(gg[i]) * 2.0 * constants::D_PI;  // |G| in Bohr^-1
        double chi_g = 0.0;

        int i0 = (int)(gmod_phys / dq) + 1;
        if (i0 < nqx - 4) {
            i0 = max(i0, 1);
            double px = gmod_phys / dq - (double)(i0 - 1);
            double ux = 1.0 - px;
            double vx = 2.0 - px;
            double wx = 3.0 - px;

            chi_g = tab_chi[i0] * ux * vx * wx / 6.0 + tab_chi[i0 + 1] * px * vx * wx / 2.0 -
                    tab_chi[i0 + 2] * px * ux * wx / 2.0 + tab_chi[i0 + 3] * px * ux * vx / 6.0;
        }

        double ylm = get_ylm(l, m_idx, gx[i], gy[i], gz[i], sqrt(gg[i]));
        double phase = -2.0 * constants::D_PI *
                       (gx[i] * atom_x[iat] + gy[i] * atom_y[iat] + gz[i] * atom_z[iat]);
        double s, c;
        sincos(phase, &s, &c);

        double re_pre = 0, im_pre = 0;
        if (l == 0) {
            re_pre = 1.0;
        } else if (l == 1) {
            im_pre = -1.0;  // (-i)^1 = -i
        } else if (l == 2) {
            re_pre = -1.0;  // (-i)^2 = -1
        } else if (l == 3) {
            im_pre = 1.0;  // (-i)^3 = i
        }

        // Normalization: (1/sqrt(Omega)) * integral * Ylm * (-i)^l
        double norm = chi_g * ylm / sqrt(omega_bohr);
        band_out[i].x = norm * (re_pre * c - im_pre * s);
        band_out[i].y = norm * (re_pre * s + im_pre * c);
    }
}

__global__ void apply_random_phase_kernel(int nnr, gpufftComplex* band, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        curandState state;
        curand_init(seed, i, 0, &state);
        double phase = curand_uniform(&state) * 2.0 * constants::D_PI;
        double s, c;
        sincos(phase, &s, &c);
        double re = band[i].x;
        double im = band[i].y;
        band[i].x = re * c - im * s;
        band[i].y = re * s + im * c;
    }
}

}  // namespace

WavefunctionFactory::WavefunctionFactory(Grid& grid, std::shared_ptr<Atoms> atoms)
    : grid_(grid), atoms_(atoms) {}

void WavefunctionFactory::add_atomic_orbital(int type, int l, const std::vector<double>& r,
                                             const std::vector<double>& chi,
                                             const std::vector<double>& rab) {
    if (type >= (int)orbital_tables_.size())
        orbital_tables_.resize(type + 1);

    // qmax should be in physical units (Bohr^-1)
    double qmax = sqrt(grid_.g2max()) * 2.0 * constants::D_PI * 1.1;
    int nqx = static_cast<int>(qmax / dq_) + 10;

    std::vector<double> chi_q(nqx + 1, 0.0);
    std::vector<double> aux(r.size());
    const double fpi = 4.0 * constants::D_PI;

    for (int iq = 1; iq <= nqx; ++iq) {
        double q = (iq - 1) * dq_;
        if (q < 1e-12) {
            if (l == 0) {
                for (int ir = 0; ir < (int)r.size(); ++ir)
                    aux[ir] = r[ir] * chi[ir];
            } else {
                for (int ir = 0; ir < (int)r.size(); ++ir)
                    aux[ir] = 0.0;
            }
        } else {
            for (int ir = 0; ir < (int)r.size(); ++ir) {
                aux[ir] = r[ir] * chi[ir] * spherical_bessel_jl(l, q * r[ir]);
            }
        }
        chi_q[iq] = simpson_integrate(aux, rab) * fpi;
    }
    printf("DEBUG WavefunctionFactory: Added orbital type=%d, l=%d, chi_q[1](q=0)=%.6f\n", type, l,
           chi_q[1]);
    orbital_tables_[type].push_back({l, chi_q});
}

void WavefunctionFactory::build_atomic_wavefunctions(Wavefunction& psi, bool randomize_phase) {
    int n_bands = psi.num_bands();
    int nnr = grid_.nnr();
    double omega_bohr = grid_.volume_bohr();

    printf("DEBUG WavefunctionFactory: Building %d bands for %zu atoms...\n", n_bands,
           atoms_->nat());

    grid_.synchronize();
    cudaMemset(psi.data(), 0, n_bands * nnr * sizeof(gpufftComplex));

    // Combine all tables into a flat contiguous buffer
    std::vector<double> h_total_table;
    std::vector<std::vector<size_t>> table_offsets(orbital_tables_.size());

    for (int t = 0; t < (int)orbital_tables_.size(); ++t) {
        for (const auto& orb : orbital_tables_[t]) {
            table_offsets[t].push_back(h_total_table.size());
            h_total_table.insert(h_total_table.end(), orb.chi_q.begin(), orb.chi_q.end());
        }
    }

    d_tab_.resize(h_total_table.size());
    d_tab_.copy_from_host(h_total_table.data());
    grid_.synchronize();

    int current_band = 0;
    // Iterate over orbital indices (e.g. first orbital of all atoms, then second...)
    // This ensures that even with few bands, we get a symmetric starting point.
    int max_orbs = 0;
    for (int t = 0; t < (int)orbital_tables_.size(); ++t) {
        max_orbs = std::max(max_orbs, (int)orbital_tables_[t].size());
    }

    for (int i_orb = 0; i_orb < max_orbs; ++i_orb) {
        for (int iat = 0; iat < (int)atoms_->nat(); ++iat) {
            int type = atoms_->h_type()[iat];
            if (i_orb >= (int)orbital_tables_[type].size())
                continue;

            const auto& orb = orbital_tables_[type][i_orb];
            size_t offset = table_offsets[type][i_orb];

            for (int m = 0; m < 2 * orb.l + 1; ++m) {
                if (current_band >= n_bands)
                    break;

                const int block_size = 256;
                const int grid_size = (nnr + block_size - 1) / block_size;

                build_atomic_band_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
                    nnr, atoms_->pos_x(), atoms_->pos_y(), atoms_->pos_z(), iat, orb.l, m,
                    grid_.gx(), grid_.gy(), grid_.gz(), grid_.gg(), d_tab_.data() + offset,
                    (int)orb.chi_q.size(), dq_, omega_bohr, psi.encut(),
                    psi.band_data(current_band));

                if (randomize_phase) {
                    unsigned int seed = 42 + current_band;
                    apply_random_phase_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
                        nnr, psi.band_data(current_band), seed);
                }

                current_band++;
            }
            if (current_band >= n_bands)
                break;
        }
        if (current_band >= n_bands)
            break;
    }

    grid_.synchronize();
}

}  // namespace dftcu
