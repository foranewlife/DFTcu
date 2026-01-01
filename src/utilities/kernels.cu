#include "cublas_manager.cuh"
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

void __global__ sum_kernel(size_t n, const double* x, double* result) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, x[i]);
    }
}

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

void real_to_complex(size_t size, const double* real, gpufftComplex* complex, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    real_to_complex_kernel<<<grid_size, block_size, 0, stream>>>(size, real, complex);
    GPU_CHECK_KERNEL;
}

void complex_to_real(size_t size, const gpufftComplex* complex, double* real, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    complex_to_real_kernel<<<grid_size, block_size, 0, stream>>>(size, complex, real);
    GPU_CHECK_KERNEL;
}

double dot_product(size_t size, const double* a, const double* b, cudaStream_t stream) {
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, stream));

    double dot_result;
    // Use RAII guard to safely set pointer mode to host and ensure it's reset
    CublasPointerModeGuard mode_guard(handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_SAFE_CALL(cublasDdot(handle, size, a, 1, b, 1, &dot_result));

    // Synchronize to ensure the host-side result is ready
    CHECK(cudaStreamSynchronize(stream));

    return dot_result;
}

void v_add(size_t n, const double* a, const double* b, double* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_add_kernel<<<grid_size, block_size, 0, stream>>>(n, a, b, out);
    GPU_CHECK_KERNEL;
}
void v_sub(size_t n, const double* a, const double* b, double* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_sub_kernel<<<grid_size, block_size, 0, stream>>>(n, a, b, out);
    GPU_CHECK_KERNEL;
}
void v_mul(size_t n, const double* a, const double* b, double* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_mul_kernel<<<grid_size, block_size, 0, stream>>>(n, a, b, out);
    GPU_CHECK_KERNEL;
}
void v_axpy(size_t n, double alpha, const double* x, double* y, cudaStream_t stream) {
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, stream));
    // Set pointer mode to host for the alpha scalar
    CublasPointerModeGuard mode_guard(handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_SAFE_CALL(cublasDaxpy(handle, (int)n, &alpha, x, 1, y, 1));
}

void v_axpy(size_t n, std::complex<double> alpha, const gpufftComplex* x, gpufftComplex* y,
            cudaStream_t stream) {
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, stream));
    cuDoubleComplex ca = {alpha.real(), alpha.imag()};
    CublasPointerModeGuard mode_guard(handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_SAFE_CALL(
        cublasZaxpy(handle, (int)n, &ca, (const cuDoubleComplex*)x, 1, (cuDoubleComplex*)y, 1));
}

void v_scale(size_t n, double alpha, const double* x, double* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_scale_kernel<<<grid_size, block_size, 0, stream>>>(n, alpha, x, out);
    GPU_CHECK_KERNEL;
}

void v_scale(size_t n, double alpha, const gpufftComplex* x, gpufftComplex* out,
             cudaStream_t stream) {
    cublasHandle_t handle = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(handle, stream));
    // cublasZdscal scales a complex vector by a real scalar
    CUBLAS_SAFE_CALL(cublasZdscal(handle, (int)n, &alpha, (cuDoubleComplex*)out, 1));
}
void v_sqrt(size_t n, const double* x, double* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    v_sqrt_kernel<<<grid_size, block_size, 0, stream>>>(n, x, out);
    GPU_CHECK_KERNEL;
}

double v_sum(size_t n, const double* x, cudaStream_t stream) {
    double* d_result;
    CHECK(cudaMallocAsync(&d_result, sizeof(double), stream));
    CHECK(cudaMemsetAsync(d_result, 0, sizeof(double), stream));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    sum_kernel<<<grid_size, block_size, 0, stream>>>(n, x, d_result);
    GPU_CHECK_KERNEL;

    double h_result;
    CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaFreeAsync(d_result, stream));
    return h_result;
}

}  // namespace dftcu
