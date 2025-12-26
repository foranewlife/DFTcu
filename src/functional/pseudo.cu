#include "fft/fft_solver.cuh"
#include "pseudo.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void pseudo_rec_kernel(size_t nnr, int nat, const double* gx, const double* gy,
                                  const double* gz, const double* pos_x, const double* pos_y,
                                  const double* pos_z, const int* types, const double* vloc_types,
                                  int num_types, gpufftComplex* v_g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnr) {
        double cur_gx = gx[i];
        double cur_gy = gy[i];
        double cur_gz = gz[i];

        double sum_re = 0.0;
        double sum_im = 0.0;

        for (int j = 0; j < nat; ++j) {
            int type = types[j];
            if (type < 0 || type >= num_types)
                continue;

            double phase = -(cur_gx * pos_x[j] + cur_gy * pos_y[j] + cur_gz * pos_z[j]);
            double s, c;
            sincos(phase, &s, &c);

            double v_val = vloc_types[type * nnr + i];
            sum_re += v_val * c;
            sum_im += v_val * s;
        }

        // No additional factor needed - FFT normalization is handled in backward transform
        v_g[i].x = sum_re;
        v_g[i].y = sum_im;
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

void LocalPseudo::compute(RealField& v) {
    size_t nnr = grid_->nnr();
    ComplexField v_g(grid_);
    FFTSolver solver(grid_);

    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    pseudo_rec_kernel<<<grid_size, block_size>>>(nnr, static_cast<int>(atoms_->nat()), grid_->gx(),
                                                 grid_->gy(), grid_->gz(), atoms_->pos_x(),
                                                 atoms_->pos_y(), atoms_->pos_z(), atoms_->type(),
                                                 vloc_types_.data(), num_types_, v_g.data());
    CHECK(cudaDeviceSynchronize());

    solver.backward(v_g);
    complex_to_real(nnr, v_g.data(), v.data());
}

}  // namespace dftcu
