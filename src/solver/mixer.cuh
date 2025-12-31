#pragma once
#include <memory>
#include <vector>

#include "model/field.cuh"

namespace dftcu {

/**
 * @brief Abstract base class for density mixers.
 */
class Mixer {
  public:
    virtual ~Mixer() = default;

    /**
     * @brief Perform density mixing.
     * @param rho_in Input density (previous mixed density).
     * @param rho_out Output density (calculated from wavefunctions).
     * @param rho_next Resulting mixed density for next iteration.
     */
    virtual void mix(const RealField& rho_in, const RealField& rho_out, RealField& rho_next) = 0;

    /** @brief Reset mixer state (history) */
    virtual void reset() = 0;
};

/**
 * @brief Linear Mixer: rho_next = (1-beta)*rho_in + beta*rho_out
 */
class LinearMixer : public Mixer {
  public:
    LinearMixer(Grid& grid, double beta) : grid_(grid), beta_(beta) {}
    void mix(const RealField& rho_in, const RealField& rho_out, RealField& rho_next) override;
    void reset() override {}

  private:
    Grid& grid_;
    double beta_;
};

/**
 * @brief Broyden (Anderson/Pulay) Mixer for accelerated SCF convergence.
 *
 * Uses history of past densities and residuals to estimate the Jacobian.
 */
class BroydenMixer : public Mixer {
  public:
    /**
     * @param grid Simulation grid.
     * @param beta Mixing parameter for the linear part.
     * @param history_size Number of previous iterations to store.
     */
    BroydenMixer(Grid& grid, double beta, int history_size = 8);
    virtual ~BroydenMixer() = default;

    void mix(const RealField& rho_in, const RealField& rho_out, RealField& rho_next) override;
    void reset() override;

  private:
    Grid& grid_;
    double beta_;
    int history_size_;
    int current_history_ = 0;
    int iter_count_ = 0;

    // History of densities: x_i = rho_in_i
    std::vector<std::unique_ptr<RealField>> x_history_;
    // History of residuals: f_i = rho_out_i - rho_in_i
    std::vector<std::unique_ptr<RealField>> f_history_;

    // Work buffers
    std::vector<double> weights_;
};

}  // namespace dftcu
