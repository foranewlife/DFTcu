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
    /**
     * @brief Constructs Ewald solver.
     * @param grid Simulation grid.
     * @param atoms Atomic system.
     * @param precision Target precision (Ha). Used to determine alpha if gcut is not provided.
     * @param gcut_hint Optional reciprocal cutoff hint. If > 0, alpha is optimized for this cutoff.
     */
    Ewald(Grid& grid, std::shared_ptr<Atoms> atoms, double precision = 1e-8,
          double gcut_hint = -1.0);
    ~Ewald() = default;

    /**
     * @brief Compute the total Ewald energy using QE-style algorithm.
     * Matches Quantum ESPRESSO's numerical result.
     * @param use_pme Use Particle Mesh Ewald method.
     * @param gcut Reciprocal space cutoff (Hartree). If <= 0, uses grid.g2max().
     */
    double compute(bool use_pme = false, double gcut = -1.0);

    /**
     * @brief Legacy Ewald computation (DFTpy-compatible).
     */
    double compute_legacy();

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
    double compute_recip_exact(double gcut);
    double compute_recip_pme();
    double compute_corr();
};

}  // namespace dftcu
