#pragma once
#include <memory>

#include "grid.cuh"
#include "math/field_expr.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

template <typename View>
__global__ void assignment_kernel(double* out, View expr_view, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expr_view[i];
    }
}

/** @brief Alias for real-valued scalar fields (standard density, potential) */
class RealField : public Expr<RealField> {
  public:
    struct RealFieldView {
        const double* ptr;

        __device__ __forceinline__ double operator[](size_t i) const { return ptr[i]; }
    };

    using View = RealFieldView;

    RealField(Grid& grid, int rank = 1) : grid_(grid), rank_(rank), data_(grid.nnr() * rank) {}

    __host__ __device__ __forceinline__ View view() const { return View{data_.data()}; }

    // Assignment from an expression template
    template <typename E>
    RealField& operator=(const Expr<E>& expr) {
        const int block_size = 256;
        const int grid_size = (size() + block_size - 1) / block_size;
        auto expr_view = expr.view();
        assignment_kernel<<<grid_size, block_size, 0, grid_.stream()>>>(data(), expr_view, size());
        GPU_CHECK_KERNEL;
        return *this;
    }

    // Explicit copy-assignment operator for RealField = RealField
    RealField& operator=(const RealField& other) {
        if (this != &other) {
            CHECK(cudaMemcpyAsync(data_.data(), other.data_.data(), data_.size() * sizeof(double),
                                  cudaMemcpyDeviceToDevice, grid_.stream()));
            GPU_CHECK_KERNEL;
        }
        return *this;
    }

    void fill(double value) { data_.fill(value, grid_.stream()); }
    void copy_from_host(const double* h_data) { data_.copy_from_host(h_data, grid_.stream()); }
    void copy_to_host(double* h_data) const { data_.copy_to_host(h_data, grid_.stream()); }

    /** @brief Copy data from another RealField (device to device) */
    void copy_from(const RealField& other) {
        CHECK(cudaMemcpyAsync(data_.data(), other.data_.data(), data_.size() * sizeof(double),
                              cudaMemcpyDeviceToDevice, grid_.stream()));
    }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    int rank() const { return rank_; }
    Grid& grid() const { return grid_; }

    double dot(const RealField& other) const {
        return dot_product(size(), data(), other.data(), grid_.stream());
    }

    double integral() const { return v_sum(size(), data(), grid_.stream()) * grid_.dv(); }

    __device__ __forceinline__ double operator[](size_t i) const { return data_.data()[i]; }

  private:
    Grid& grid_;
    int rank_;
    GPU_Vector<double> data_;

    RealField(const RealField&) = delete;
    RealField(RealField&&) = delete;
    RealField& operator=(RealField&&) = delete;
};

/** @brief Alias for complex-valued scalar fields (G-space representations) */
// Note: ComplexField does not yet support Expression Templates
class ComplexField {
  public:
    ComplexField(Grid& grid, int rank = 1) : grid_(grid), rank_(rank), data_(grid.nnr() * rank) {}

    void fill(gpufftComplex value) { data_.fill(value, grid_.stream()); }
    void copy_from_host(const gpufftComplex* h_data) {
        data_.copy_from_host(h_data, grid_.stream());
    }
    void copy_to_host(gpufftComplex* h_data) const { data_.copy_to_host(h_data, grid_.stream()); }

    gpufftComplex* data() { return data_.data(); }
    const gpufftComplex* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    int rank() const { return rank_; }
    Grid& grid() const { return grid_; }

  private:
    Grid& grid_;
    int rank_;
    GPU_Vector<gpufftComplex> data_;
};

}  // namespace dftcu
