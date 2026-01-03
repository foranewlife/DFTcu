#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace dftcu {

namespace {

__global__ void real_to_complex_kernel(size_t n, const double* real, gpufftComplex* complex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        complex[i].x = real[i];
        complex[i].y = 0.0;
    }
}

__global__ void complex_to_real_kernel(size_t n, const gpufftComplex* complex, double* real) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        real[i] = complex[i].x;
    }
}

__global__ void v_add_kernel(size_t n, const double* x, const double* y, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = x[i] + y[i];
}

__global__ void v_sub_kernel(size_t n, const double* x, const double* y, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = x[i] - y[i];
}

__global__ void v_mul_kernel(size_t n, const double* x, const double* y, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = x[i] * y[i];
}

__global__ void v_sqrt_kernel(size_t n, const double* x, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = sqrt(x[i]);
}

__global__ void swap_axes_kernel(const double* in, double* out, int nr0, int nr1, int nr2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nr0 * nr1 * nr2;
    if (idx < n) {
        int i = idx / (nr1 * nr2);
        int j = (idx / nr2) % nr1;
        int k = idx % nr2;
        int new_idx = j * nr0 * nr2 + i * nr2 + k;
        out[new_idx] = in[idx];
    }
}

__global__ void swap_axes_complex_kernel(const gpufftComplex* in, gpufftComplex* out, int nr0,
                                         int nr1, int nr2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nr0 * nr1 * nr2;
    if (idx < n) {
        int i = idx / (nr1 * nr2);
        int j = (idx / nr2) % nr1;
        int k = idx % nr2;
        int new_idx = j * nr0 * nr2 + i * nr2 + k;
        out[new_idx] = in[idx];
    }
}

__global__ void v_shift_kernel(size_t n, double shift, const double* in, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in[i] + shift;
}

}  // namespace

void real_to_complex(size_t size, const double* real, gpufftComplex* complex, cudaStream_t stream) {
    const int bs = 256;
    real_to_complex_kernel<<<(size + bs - 1) / bs, bs, 0, stream>>>(size, real, complex);
}

void complex_to_real(size_t size, const gpufftComplex* complex, double* real, cudaStream_t stream) {
    const int bs = 256;
    complex_to_real_kernel<<<(size + bs - 1) / bs, bs, 0, stream>>>(size, complex, real);
}

void v_add(size_t n, const double* x, const double* y, double* out, cudaStream_t stream) {
    const int bs = 256;
    v_add_kernel<<<(n + bs - 1) / bs, bs, 0, stream>>>(n, x, y, out);
}

void v_sub(size_t n, const double* x, const double* y, double* out, cudaStream_t stream) {
    const int bs = 256;
    v_sub_kernel<<<(n + bs - 1) / bs, bs, 0, stream>>>(n, x, y, out);
}

void v_mul(size_t n, const double* x, const double* y, double* out, cudaStream_t stream) {
    const int bs = 256;
    v_mul_kernel<<<(n + bs - 1) / bs, bs, 0, stream>>>(n, x, y, out);
}

void v_sqrt(size_t n, const double* x, double* out, cudaStream_t stream) {
    const int bs = 256;
    v_sqrt_kernel<<<(n + bs - 1) / bs, bs, 0, stream>>>(n, x, out);
}

void v_scale(size_t n, double alpha, const double* x, double* out, cudaStream_t stream) {
    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, stream));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);
    if (x != out)
        CHECK(cudaMemcpyAsync(out, x, n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUBLAS_SAFE_CALL(cublasDscal(h, (int)n, &alpha, out, 1));
}

void v_scale(size_t n, double alpha, const gpufftComplex* x, gpufftComplex* out,
             cudaStream_t stream) {
    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, stream));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);
    if (x != out)
        CHECK(cudaMemcpyAsync(out, x, n * sizeof(gpufftComplex), cudaMemcpyDeviceToDevice, stream));
    CUBLAS_SAFE_CALL(cublasZdscal(h, (int)n, &alpha, (cuDoubleComplex*)out, 1));
}

void v_axpy(size_t n, double alpha, const double* x, double* y, cudaStream_t stream) {
    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, stream));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_SAFE_CALL(cublasDaxpy(h, (int)n, &alpha, x, 1, y, 1));
}

void v_axpy(size_t n, std::complex<double> alpha, const gpufftComplex* x, gpufftComplex* y,
            cudaStream_t stream) {
    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, stream));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);
    cuDoubleComplex a = {alpha.real(), alpha.imag()};
    CUBLAS_SAFE_CALL(
        cublasZaxpy(h, (int)n, &a, (const cuDoubleComplex*)x, 1, (cuDoubleComplex*)y, 1));
}

double dot_product(size_t size, const double* a, const double* b, cudaStream_t stream) {
    cublasHandle_t h = CublasManager::instance().handle();
    CUBLAS_SAFE_CALL(cublasSetStream(h, stream));
    CublasPointerModeGuard guard(h, CUBLAS_POINTER_MODE_HOST);
    double res = 0;
    CUBLAS_SAFE_CALL(cublasDdot(h, (int)size, a, 1, b, 1, &res));
    return res;
}

double v_sum(size_t n, const double* x, cudaStream_t stream) {
    // Use thrust::reduce for proper summation (not abs sum like cublasDasum)
    thrust::device_ptr<const double> x_ptr(x);
    double res =
        thrust::reduce(thrust::cuda::par.on(stream), x_ptr, x_ptr + n, 0.0, thrust::plus<double>());
    return res;
}

void swap_xy(int nr0, int nr1, int nr2, const double* in, double* out, cudaStream_t stream) {
    const int bs = 256;
    swap_axes_kernel<<<(nr0 * nr1 * nr2 + bs - 1) / bs, bs, 0, stream>>>(in, out, nr0, nr1, nr2);
}

void swap_xy(int nr0, int nr1, int nr2, const gpufftComplex* in, gpufftComplex* out,
             cudaStream_t stream) {
    const int bs = 256;
    swap_axes_complex_kernel<<<(nr0 * nr1 * nr2 + bs - 1) / bs, bs, 0, stream>>>(in, out, nr0, nr1,
                                                                                 nr2);
}

void v_shift(size_t n, double shift, const double* in, double* out, cudaStream_t stream) {
    const int bs = 256;
    v_shift_kernel<<<(n + bs - 1) / bs, bs, 0, stream>>>(n, shift, in, out);
}

}  // namespace dftcu
