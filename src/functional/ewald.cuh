#pragma once
#include "model/atoms.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Ewald summation for ion-ion interaction energy.
 */
class Ewald {
  public:
    Ewald(const Grid& grid, const Atoms& atoms, double precision = 1e-8, int bspline_order = 10);
    ~Ewald() = default;

    /**
     * @brief Compute total Ewald energy
     * @param use_pme If true, use Particle Mesh Ewald for reciprocal sum
     */
    double compute(bool use_pme = false);

  private:
    const Grid& grid_;
    const Atoms& atoms_;
    double precision_;
    double eta_;
    int bspline_order_;

    double compute_real();
    double compute_recip_exact();
    double compute_recip_pme();
    double compute_corr();

    double get_best_eta();
};

}  // namespace dftcu
