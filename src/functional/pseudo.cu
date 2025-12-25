#include "pseudo.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

void __global__ pseudo_rec_kernel(size_t nnr, size_t nat, const double* gx, const double* gy,
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
            double phase = -(cur_gx * pos_x[j] + cur_gy * pos_y[j] + cur_gz * pos_z[j]);
            double cos_p = cos(phase);
            double sin_p = sin(phase);

            // For now, assume vloc is same for all types or just use type 0
            double v_val = vloc_types[types[j] * nnr + i];
            sum_re += v_val * cos_p;
            sum_im += v_val * sin_p;
        }

        v_g[i].x = sum_re;
        v_g[i].y = sum_im;
    }
}

void LocalPseudo::set_vloc(int type, const std::vector<double>& vloc_g) {
    vlines_[type].resize(vloc_g.size());
    vlines_[type].copy_from_host(vloc_g.data());
}

void LocalPseudo::compute(RealField& v) {
    size_t nnr = grid_.nnr();
    ComplexField v_g(grid_);

    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    pseudo_rec_kernel<<<grid_size, block_size>>>(
        nnr, atoms_.nat(), grid_.gx(), grid_.gy(), grid_.gz(), atoms_.pos_x(), atoms_.pos_y(),
        atoms_.pos_z(), atoms_.type(), vlines_[0].data(), 1, v_g.data());
    GPU_CHECK_KERNEL

    solver_.backward(v_g);

    complex_to_real(nnr, v_g.data(), v.data());
}

}  // namespace dftcu
