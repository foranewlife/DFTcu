#pragma once
#include "evaluator.cuh"
#include "optimizer.cuh"  // For OptimizationOptions

namespace dftcu {

class TNOptimizerLegacy {
  public:
    TNOptimizerLegacy(Grid& grid, OptimizationOptions options = {});
    ~TNOptimizerLegacy() = default;

    /**
     * @brief Optimize density using Truncated Newton on sqrt(rho)
     */
    void solve(RealField& rho, Evaluator& evaluator);

  private:
    Grid& grid_;
    OptimizationOptions options_;
};

}  // namespace dftcu
