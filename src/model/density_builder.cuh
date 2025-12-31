#pragma once
#include <memory>
#include <vector>

#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Utility to build initial charge density from atomic contributions.
 */
class DensityBuilder {
  public:
    DensityBuilder(Grid& grid, std::shared_ptr<Atoms> atoms);
    ~DensityBuilder() = default;

    /**
     * @brief Set the radial charge density for an atom type in G-space.
     * @param type Atom type index.
     * @param q Radial G-grid points (Bohr^-1).
     * @param rho_q Radial charge density values rho(q).
     *
     * Note: rho(q) should be the Fourier transform of the atomic charge density.
     */
    void set_atomic_rho_g(int type, const std::vector<double>& q, const std::vector<double>& rho_q);

    /**
     * @brief Build the total charge density on the real space grid.
     * @param rho Output real space density field.
     */
    void build_density(RealField& rho);

  private:
    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;

    // Store radial rho(G) for each type
    int num_types_ = 0;
    GPU_Vector<double> rho_types_g_;
};

}  // namespace dftcu
