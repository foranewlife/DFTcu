#pragma once
#include <memory>

#include "grid.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

/**
 * @brief Represents a physical field defined on a simulation grid.
 */
template <typename T>
class Field {
  public:
    Field(Grid& grid, int rank = 1) : grid_(grid), rank_(rank), data_(grid.nnr() * rank) {}

    void fill(T value) { data_.fill(value, grid_.stream()); }
    void copy_from_host(const T* h_data) { data_.copy_from_host(h_data, grid_.stream()); }
    void copy_to_host(T* h_data) const { data_.copy_to_host(h_data, grid_.stream()); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    int rank() const { return rank_; }
    Grid& grid() const { return grid_; }

    double dot(const Field<double>& other) const {
        return dot_product(size(), data(), other.data(), grid_.stream());
    }

    double integral() const { return v_sum(size(), data(), grid_.stream()) * grid_.dv(); }

  private:
    Grid& grid_;
    int rank_;
    GPU_Vector<T> data_;

    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;
    Field(Field&&) = delete;
    Field& operator=(Field&&) = delete;
};

/** @brief Alias for real-valued scalar fields (standard density, potential) */
using RealField = Field<double>;

/** @brief Alias for complex-valued scalar fields (G-space representations) */
using ComplexField = Field<gpufftComplex>;

}  // namespace dftcu
