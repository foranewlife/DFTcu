#include "pseudo.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

namespace {
__global__ void pseudo_rec_kernel(size_t nnr, size_t nat, const double* gx, const double* gy,
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
            if (type < 0 || type >= num_types) continue;

            double phase = -(cur_gx * pos_x[j] + cur_gy * pos_y[j] + cur_gz * pos_z[j]);
            double cos_p, sin_p;
            sincos(phase, &sin_p, &cos_p);

            double v_val = vloc_types[type * nnr + i];
            sum_re += v_val * cos_p;
            sum_im += v_val * sin_p;
        }

        v_g[i].x = sum_re;
        v_g[i].y = sum_im;
    }
}
} // namespace

LocalPseudo::LocalPseudo(const Grid& grid, const Atoms& atoms) 
    : grid_(grid), atoms_(atoms), solver_(grid), num_types_(0) {
    // Initial allocation for up to 4 types, will resize if needed
    vloc_types_.resize(grid_.nnr() * 4);
    num_types_ = 4;
}

void LocalPseudo::set_vloc(int type, const std::vector<double>& vloc_g) {
    if (type >= num_types_) {
        // Resize buffer if needed
        int new_num_types = type + 1;
        GPU_Vector<double> new_vloc(grid_.nnr() * new_num_types);
        // Copy old data
        cudaMemcpy(new_vloc.data(), vloc_types_.data(), grid_.nnr() * num_types_ * sizeof(double), cudaMemcpyDeviceToDevice);
        vloc_types_ = std::move(new_vloc);
        num_types_ = new_num_types;
    }
    
    // Copy vloc_g to the correct offset
    cudaMemcpy(vloc_types_.data() + type * grid_.nnr(), vloc_g.data(), vloc_g.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void LocalPseudo::compute(RealField& v) {
    size_t nnr = grid_.nnr();
    ComplexField v_g(grid_);

    const int block_size = 256;
    const int grid_size = (nnr + block_size - 1) / block_size;

    pseudo_rec_kernel<<<grid_size, block_size>>>(
        nnr, atoms_.nat(), grid_.gx(), grid_.gy(), grid_.gz(), atoms_.pos_x(), atoms_.pos_y(),
        atoms_.pos_z(), atoms_.type(), vloc_types_.data(), num_types_, v_g.data());
    GPU_CHECK_KERNEL

    solver_.backward(v_g);

    complex_to_real(nnr, v_g.data(), v.data());
}

}  // namespace dftcu
