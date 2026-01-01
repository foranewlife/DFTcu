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
     * @brief Set atomic radial density in G-space.
     */
    void set_atomic_rho_g(int type, const std::vector<double>& q, const std::vector<double>& rho_q);

    /**
     * @brief Set atomic radial density in real-space (will be transformed to G-space).
     * @param type Atom type index.
     * @param r Radial grid points.
     * @param rho_r Radial charge density (4*pi*r^2 * rho(r)).
     * @param rab Integration weights dr/di.
     */
    void set_atomic_rho_r(int type, const std::vector<double>& r, const std::vector<double>& rho_r,
                          const std::vector<double>& rab);

    /**
     * @brief Build initial charge density from atomic superposition.
     * @param rho Output real space density field.
     */
    void build_density(RealField& rho);

    /**
     * @brief Set G-vector cutoff for the density (in Rydberg).
     * Set to -1 to use the full FFT grid.
     */
    void set_gcut(double gcut) { gcut_ = gcut; }

  private:
    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;

    int num_types_ = 0;
    double gcut_ = -1.0;
    static constexpr double dq_ = 0.01;  // Matches QE's dq
    int nqx_ = 0;
    std::vector<std::vector<double>> tab_rho_g_;
};

}  // namespace dftcu
