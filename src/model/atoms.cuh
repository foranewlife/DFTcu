#pragma once
#include <string>
#include <vector>

#include "utilities/gpu_vector.cuh"

namespace dftcu {

struct Atom {
    double x, y, z;
    double charge;
    int type;
};

class Atoms {
  public:
    Atoms(const std::vector<Atom>& atoms) : nat_(atoms.size()) {
        h_pos_x_.resize(nat_);
        h_pos_y_.resize(nat_);
        h_pos_z_.resize(nat_);
        h_charge_.resize(nat_);
        h_type_.resize(nat_);

        for (size_t i = 0; i < nat_; ++i) {
            h_pos_x_[i] = atoms[i].x;
            h_pos_y_[i] = atoms[i].y;
            h_pos_z_[i] = atoms[i].z;
            h_charge_[i] = atoms[i].charge;
            h_type_[i] = atoms[i].type;
        }

        d_pos_x_.resize(nat_);
        d_pos_y_.resize(nat_);
        d_pos_z_.resize(nat_);
        d_charge_.resize(nat_);
        d_type_.resize(nat_);

        d_pos_x_.copy_from_host(h_pos_x_.data());
        d_pos_y_.copy_from_host(h_pos_y_.data());
        d_pos_z_.copy_from_host(h_pos_z_.data());
        d_charge_.copy_from_host(h_charge_.data());
        d_type_.copy_from_host(h_type_.data());
    }

    size_t nat() const { return nat_; }
    const double* pos_x() const { return d_pos_x_.data(); }
    const double* pos_y() const { return d_pos_y_.data(); }
    const double* pos_z() const { return d_pos_z_.data(); }
    const double* charge() const { return d_charge_.data(); }
    const int* type() const { return d_type_.data(); }

  private:
    size_t nat_;
    std::vector<double> h_pos_x_, h_pos_y_, h_pos_z_, h_charge_;
    std::vector<int> h_type_;
    GPU_Vector<double> d_pos_x_, d_pos_y_, d_pos_z_, d_charge_;
    GPU_Vector<int> d_type_;
};

}  // namespace dftcu
