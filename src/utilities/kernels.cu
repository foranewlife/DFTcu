#include "error.cuh"
#include "kernels.cuh"

namespace dftcu {

namespace {
void __global__ real_to_complex_kernel(size_t size, const double* real, gpufftComplex* complex) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        complex[i].x = real[i];
        complex[i].y = 0.0;
    }
}

void __global__ complex_to_real_kernel(size_t size, const gpufftComplex* complex, double* real) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        real[i] = complex[i].x;
    }
}

void __global__ dot_kernel(size_t size, const double* a, const double* b, double* result) {
    // This is a very simple reduction. For production, use cublas or a proper tree reduction.
    // For now, let's use atomicAdd (slow but correct).
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        atomicAdd(result, a[i] * b[i]);
    }
}
}  // namespace

void real_to_complex(size_t size, const double* real, gpufftComplex* complex) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    real_to_complex_kernel<<<grid_size, block_size>>>(size, real, complex);
    GPU_CHECK_KERNEL
}

void complex_to_real(size_t size, const gpufftComplex* complex, double* real) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    complex_to_real_kernel<<<grid_size, block_size>>>(size, complex, real);
    GPU_CHECK_KERNEL
}

double dot_product(size_t size, const double* a, const double* b) {
    double* d_result;
    gpuMalloc(&d_result, sizeof(double));
    gpuMemset(d_result, 0, sizeof(double));

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    dot_kernel<<<grid_size, block_size>>>(size, a, b, d_result);
    GPU_CHECK_KERNEL

    double h_result;
    gpuMemcpy(&h_result, d_result, sizeof(double), gpuMemcpyDeviceToHost);
    gpuFree(d_result);
    return h_result;
}

namespace {
__global__ void v_add_kernel(size_t n, const double* a, const double* b, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}
__global__ void v_sub_kernel(size_t n, const double* a, const double* b, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] - b[i];
}
__global__ void v_mul_kernel(size_t n, const double* a, const double* b, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}
__global__ void v_axpy_kernel(size_t n, double alpha, const double* x, double* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}
__global__ void v_scale_kernel(size_t n, double alpha, const double* x, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = alpha * x[i];
}
__global__ void v_sqrt_kernel(size_t n, const double* x, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = (x[i] > 1e-30) ? sqrt(x[i]) : 0.0;
}
}  // namespace

void v_add(size_t n, const double* a, const double* b, double* out) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_add_kernel<<<grid_size, block_size>>>(n, a, b, out);
    GPU_CHECK_KERNEL
}
void v_sub(size_t n, const double* a, const double* b, double* out) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_sub_kernel<<<grid_size, block_size>>>(n, a, b, out);
    GPU_CHECK_KERNEL
}
void v_mul(size_t n, const double* a, const double* b, double* out) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_mul_kernel<<<grid_size, block_size>>>(n, a, b, out);
    GPU_CHECK_KERNEL
}
void v_axpy(size_t n, double alpha, const double* x, double* y) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_axpy_kernel<<<grid_size, block_size>>>(n, alpha, x, y);
    GPU_CHECK_KERNEL
}
void v_scale(size_t n, double alpha, const double* x, double* out) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_scale_kernel<<<grid_size, block_size>>>(n, alpha, x, out);
    GPU_CHECK_KERNEL
}
void v_sqrt(size_t n, const double* x, double* out) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_sqrt_kernel<<<grid_size, block_size>>>(n, x, out);
    GPU_CHECK_KERNEL
}

}  // namespace dftcu
