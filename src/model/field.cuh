#pragma once
#include <memory>

#include "grid.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {
class Grid;

/**
 * @brief Computes the dot product of two device arrays.
 * @param size Number of elements.
 * @param a Pointer to first device array.
 * @param b Pointer to second device array.
 * @return The scalar dot product.
 */
double dot_product(size_t size, const double* a, const double* b);

/**
 * @brief Represents a physical field defined on a simulation grid.
 *
 * A Field maps grid points to values of type T. It can represent scalar fields
 * (rank=1) or vector/tensor fields (rank > 1). The data is stored on the GPU.
 *
 * @tparam T Data type of the field elements (e.g., double for real fields).
 */
template <typename T>
class Field {
  public:
    /**
     * @brief Constructs a field on a specific grid.
     * @param grid Shared pointer to the grid.
     * @param rank Dimensionality of the value at each grid point (default: 1 for scalar).
     */
    Field(std::shared_ptr<Grid> grid, int rank = 1)
        : grid_(grid), rank_(rank), data_(grid->nnr() * rank) {}

    /**
     * @brief Fills the field with a constant value on the GPU.
     * @param value The value to fill with.
     */
    void fill(T value) { data_.fill(value); }

    /**
     * @brief Copies data from host memory to the field's device memory.
     * @param h_data Pointer to host memory.
     */
    void copy_from_host(const T* h_data) { data_.copy_from_host(h_data); }

    /**
     * @brief Copies data from the field's device memory to host memory.
     * @param h_data Pointer to host memory.
     */
    void copy_to_host(T* h_data) { data_.copy_to_host(h_data); }

    /**
     * @brief Returns a pointer to the device memory.
     */
    T* data() { return data_.data(); }

    /**
     * @brief Returns a const pointer to the device memory.
     */
    const T* data() const { return data_.data(); }

    /**
     * @brief Total number of elements in the field (nnr * rank).
     */
    size_t size() const { return data_.size(); }

    /**
     * @brief Dimensionality of the values at each grid point.
     */
    int rank() const { return rank_; }

    /**
     * @brief Reference to the grid associated with this field.
     */
    const Grid& grid() const { return *grid_; }

    /**
     * @brief Shared pointer to the grid associated with this field.
     */
    std::shared_ptr<Grid> grid_ptr() const { return grid_; }

    /**
     * @brief Computes the L2 dot product with another field.
     * @param other The other field to compute the dot product with.
     * @return The dot product value.
     */
    double dot(const Field<double>& other) const {
        return dot_product(size(), data(), other.data());
    }

    /**
     * @brief Computes the integral of the field over the unit cell volume.
     * @return The integral value.
     */
    double integral() const {
        Field<double> ones(grid_);
        ones.fill(1.0);
        return dot(ones) * grid_->dv();
    }

  private:
    std::shared_ptr<Grid> grid_; /**< The underlying grid */
    int rank_;                   /**< Field rank (scalar=1, vector=3, etc.) */
    GPU_Vector<T> data_;         /**< Device memory container */

    // Prevent copying
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
