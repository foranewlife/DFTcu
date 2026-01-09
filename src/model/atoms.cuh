#pragma once
#include <string>
#include <vector>

#include "utilities/constants.cuh"
#include "utilities/gpu_vector.cuh"

namespace dftcu {

struct Atom {
    double x, y, z;  // Positions (units specified by user)
    double charge;
    int type;
};

/**
 * @brief Atoms class - manages atomic positions and types
 *
 * UNIT CONVENTION (Hartree Atomic Units):
 *   - Internal storage: Bohr (positions), e⁻ (charge)
 *   - NO unit conversion inside this class
 *   - Use factory functions (atoms_factory.cuh) for unit conversion
 *
 * IMPORTANT: This constructor expects positions in BOHR.
 * For Angstrom input, use create_atoms_from_angstrom() factory function.
 */
class Atoms {
  public:
    /**
     * @brief Construct Atoms from a list of Atom structs
     * @param atoms Vector of Atom structs with positions in BOHR (atomic units)
     *
     * No unit conversion is performed. Positions must be in Bohr.
     */
    Atoms(const std::vector<Atom>& atoms) : nat_(atoms.size()) {
        h_pos_x_.resize(nat_);
        h_pos_y_.resize(nat_);
        h_pos_z_.resize(nat_);
        h_charge_.resize(nat_);
        h_type_.resize(nat_);

        for (size_t i = 0; i < nat_; ++i) {
            // Store positions directly in Bohr (no conversion)
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

    /** @brief Get x positions [Bohr] (GPU) */
    const double* pos_x() const { return d_pos_x_.data(); }
    /** @brief Get y positions [Bohr] (GPU) */
    const double* pos_y() const { return d_pos_y_.data(); }
    /** @brief Get z positions [Bohr] (GPU) */
    const double* pos_z() const { return d_pos_z_.data(); }
    const double* charge_data() const { return d_charge_.data(); }
    const int* type() const { return d_type_.data(); }

    /** @brief Get x positions [Bohr] (CPU) */
    const std::vector<double>& h_pos_x() const { return h_pos_x_; }
    /** @brief Get y positions [Bohr] (CPU) */
    const std::vector<double>& h_pos_y() const { return h_pos_y_; }
    /** @brief Get z positions [Bohr] (CPU) */
    const std::vector<double>& h_pos_z() const { return h_pos_z_; }
    const std::vector<double>& h_charge() const { return h_charge_; }
    const std::vector<int>& h_type() const { return h_type_; }

    int n_type(int type) const {
        int count = 0;
        for (int t : h_type_)
            if (t == type)
                count++;
        return count;
    }

    const GPU_Vector<double>& d_charge() const { return d_charge_; }

  private:
    size_t nat_;
    std::vector<double> h_pos_x_, h_pos_y_, h_pos_z_, h_charge_;  // Positions in Bohr
    std::vector<int> h_type_;
    GPU_Vector<double> d_pos_x_, d_pos_y_, d_pos_z_, d_charge_;
    GPU_Vector<int> d_type_;

    // Prevent copying
    Atoms(const Atoms&) = delete;
    Atoms& operator=(const Atoms&) = delete;
    Atoms(Atoms&&) = delete;
    Atoms& operator=(Atoms&&) = delete;
};

}  // namespace dftcu
