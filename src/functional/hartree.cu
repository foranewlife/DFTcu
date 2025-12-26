#include "hartree.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
void __global__ hartree_kernel(size_t size, gpufftComplex* rho_g, const double* gg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (gg[i] > 1e-12) {
            double factor = 4.0 * constants::D_PI / gg[i];
            rho_g[i].x *= factor;
            rho_g[i].y *= factor;
        } else {
            rho_g[i].x = 0.0;
            rho_g[i].y = 0.0;
        }
    }
}
}  // namespace

Hartree::Hartree(std::shared_ptr<Grid> grid) : grid_(grid) {}

void Hartree::compute(const RealField& rho, RealField& vh, double& energy) {
    size_t nnr = grid_->nnr();
    ComplexField rho_g(grid_);
    FFTSolver solver(grid_);

    // 1. Copy Real to Complex
    real_to_complex(nnr, rho.data(), rho_g.data());

    // 2. Forward FFT: rho(r) -> rho(G)
    solver.forward(rho_g);

    // 3. Multiply by 4*D_PI/G^2 in reciprocal space
    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;
    hartree_kernel<<<grid_size, block_size>>>(nnr, rho_g.data(), grid_->gg());
    CHECK(cudaDeviceSynchronize());

    // 4. Backward FFT: V_h(G) -> V_h(r)
    solver.backward(rho_g);

    // 5. Copy Complex back to Real
    complex_to_real(nnr, rho_g.data(), vh.data());

    // 6. Hartree energy: E_H = 0.5 * integral( rho(r) * v_h(r) ) dV
    energy = 0.5 * dot_product(nnr, rho.data(), vh.data()) * grid_->dv();
}

}  // namespace dftcu
