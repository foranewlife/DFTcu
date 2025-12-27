#pragma once

#include <memory>

#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Ewald summation for ion-ion interaction energy.
 * Supports both exact summation and Particle Mesh Ewald (PME).
 */
class Ewald {
  public:
    Ewald(Grid& grid, std::shared_ptr<Atoms> atoms, double precision = 1e-8,
          int bspline_order = 10);
    ~Ewald() = default;

    /**
     * @brief Compute the total Ewald energy.
     * @param use_pme Use Particle Mesh Ewald method.
     */
    double compute(bool use_pme = false);

    /**
     * @brief Unified interface for Evaluator.
     */
    double compute(const RealField& rho, RealField& v_out) {
        v_out.fill(0.0);
        return compute(use_pme_);
    }

    void set_pme(bool use_pme) { use_pme_ = use_pme; }
    void set_eta(double eta) { eta_ = eta; }

  private:
    Grid& grid_;
    std::shared_ptr<Atoms> atoms_;
    double precision_;
    int bspline_order_;
    double eta_;
    bool use_pme_ = false;

    // Helper methods
    double get_best_eta();
    double compute_real();
    double compute_recip_exact();
    double compute_recip_pme();
    double compute_corr();
};

}  // namespace dftcu
