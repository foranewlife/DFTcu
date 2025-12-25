#pragma once
#include "grid.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {
class Grid;

double dot_product(size_t size, const double* a, const double* b);

template <typename T>
class Field {
  public:
    Field(const Grid& grid, int rank = 1) : grid_(grid), rank_(rank), data_(grid.nnr() * rank) {}

    void fill(T value) { data_.fill(value); }

    void copy_from_host(const T* h_data) { data_.copy_from_host(h_data); }

    void copy_to_host(T* h_data) { data_.copy_to_host(h_data); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    int rank() const { return rank_; }
    const Grid& grid() const { return grid_; }

    double dot(const Field<double>& other) const {
        return dot_product(size(), data(), other.data());
    }

    double integral() const {
        Field<double> ones(grid_);
        ones.fill(1.0);
        return dot(ones) * grid_.dv();
    }

  private:
    const Grid& grid_;
    int rank_;
    GPU_Vector<T> data_;
};

using RealField = Field<double>;
using ComplexField = Field<gpufftComplex>;

}  // namespace dftcu
