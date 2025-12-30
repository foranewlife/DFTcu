#include "functional/nonlocal_pseudo.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <cublas_v2.h>

namespace dftcu {

namespace {

// Kernel to apply D_i coupling constants to projections
__global__ void scale_projections_kernel(int num_proj, int num_bands, const double* d_coupling,
                                         gpufftComplex* projections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_proj * num_bands) {
        int proj_idx = i % num_proj;
        projections[i].x *= d_coupling[proj_idx];
        projections[i].y *= d_coupling[proj_idx];
    }
}

__global__ void manual_projection_kernel(int num_proj, int nbands, int n,
                                         const gpufftComplex* projectors, const gpufftComplex* psi,
                                         gpufftComplex* projections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_proj * nbands) {
        int iproj = idx % num_proj;
        int iband = idx / num_proj;
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int i = 0; i < n; ++i) {
            gpufftComplex p = projectors[iproj * n + i];
            gpufftComplex w = psi[iband * n + i];
            sum_x += p.x * w.x + p.y * w.y;
            sum_y += p.x * w.y - p.y * w.x;
        }
        projections[idx].x = sum_x;
        projections[idx].y = sum_y;
    }
}

__global__ void manual_apply_nl_kernel(int num_proj, int nbands, int n, double omega,
                                       const gpufftComplex* projectors,
                                       const gpufftComplex* projections, gpufftComplex* hpsi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * nbands) {
        int ig = idx % n;
        int iband = idx / n;
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int iproj = 0; iproj < num_proj; ++iproj) {
            gpufftComplex p = projectors[iproj * n + ig];
            gpufftComplex proj_val = projections[iband * num_proj + iproj];
            sum_x += p.x * proj_val.x - p.y * proj_val.y;
            sum_y += p.x * proj_val.y + p.y * proj_val.x;
        }
        hpsi[idx].x += omega * sum_x;
        hpsi[idx].y += omega * sum_y;
    }
}

}  // namespace

NonLocalPseudo::NonLocalPseudo(Grid& grid) : grid_(grid) {}

void NonLocalPseudo::clear() {
    num_projectors_ = 0;
    d_projectors_.resize(0);
    d_coupling_.resize(0);
}

void NonLocalPseudo::add_projector(const std::vector<std::complex<double>>& beta_g,
                                   double coupling_constant) {
    size_t n = grid_.nnr();
    if (beta_g.size() != n) {
        throw std::runtime_error("Projector size mismatch with grid");
    }

    int old_num = num_projectors_;
    num_projectors_++;

    GPU_Vector<gpufftComplex> next_projectors(num_projectors_ * n);
    if (old_num > 0) {
        CHECK(cudaMemcpy(next_projectors.data(), d_projectors_.data(),
                         old_num * n * sizeof(gpufftComplex), cudaMemcpyDeviceToDevice));
    }
    CHECK(cudaMemcpy(next_projectors.data() + old_num * n, beta_g.data(), n * sizeof(gpufftComplex),
                     cudaMemcpyHostToDevice));
    d_projectors_ = std::move(next_projectors);

    std::vector<double> h_coupling(num_projectors_);
    if (old_num > 0) {
        d_coupling_.copy_to_host(h_coupling.data());
    }
    h_coupling[old_num] = coupling_constant;
    d_coupling_.resize(num_projectors_);
    d_coupling_.copy_from_host(h_coupling.data());
}

void NonLocalPseudo::apply(Wavefunction& psi_in, Wavefunction& h_psi_out) {
    if (num_projectors_ == 0)
        return;

    size_t n = grid_.nnr();
    int nbands = psi_in.num_bands();

    d_projections_.resize(num_projectors_ * nbands);

    const int block_size_p = 64;
    const int grid_size_p = (num_projectors_ * nbands + block_size_p - 1) / block_size_p;

    manual_projection_kernel<<<grid_size_p, block_size_p, 0, grid_.stream()>>>(
        num_projectors_, nbands, (int)n, d_projectors_.data(), psi_in.data(),
        d_projections_.data());

    const int block_size_s = 256;
    const int grid_size_s = (num_projectors_ * nbands + block_size_s - 1) / block_size_s;
    scale_projections_kernel<<<grid_size_s, block_size_s, 0, grid_.stream()>>>(
        num_projectors_, nbands, d_coupling_.data(), d_projections_.data());

    const int block_size_a = 256;
    const int grid_size_a = (int)(n * nbands + block_size_a - 1) / block_size_a;

    manual_apply_nl_kernel<<<grid_size_a, block_size_a, 0, grid_.stream()>>>(
        num_projectors_, nbands, (int)n, grid_.volume(), d_projectors_.data(),
        d_projections_.data(), h_psi_out.data());

    grid_.synchronize();
}

double NonLocalPseudo::calculate_energy(const Wavefunction& psi,
                                        const std::vector<double>& occupations) {
    if (num_projectors_ == 0)
        return 0.0;

    size_t n = grid_.nnr();
    int nbands = psi.num_bands();

    if (d_projections_.size() < (size_t)num_projectors_ * nbands) {
        d_projections_.resize(num_projectors_ * nbands);
    }

    const int block_size = 64;
    const int grid_size = (num_projectors_ * nbands + block_size - 1) / block_size;

    manual_projection_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(
        num_projectors_, nbands, (int)n, d_projectors_.data(), psi.data(), d_projections_.data());

    std::vector<gpufftComplex> h_projections(num_projectors_ * nbands);
    d_projections_.copy_to_host(h_projections.data(), grid_.stream());

    std::vector<double> h_coupling(num_projectors_);
    d_coupling_.copy_to_host(h_coupling.data(), grid_.stream());

    grid_.synchronize();

    double energy = 0.0;
    double omega = grid_.volume();
    double omega2 = omega * omega;

    for (int n_idx = 0; n_idx < nbands; ++n_idx) {
        double band_energy = 0.0;
        for (int i = 0; i < num_projectors_; ++i) {
            gpufftComplex p = h_projections[n_idx * num_projectors_ + i];
            double p2 = p.x * p.x + p.y * p.y;
            band_energy += h_coupling[i] * p2;
        }
        energy += occupations[n_idx] * band_energy;
    }

    // TEST: Use omega instead of omega2
    return energy * omega;
}

}  // namespace dftcu
