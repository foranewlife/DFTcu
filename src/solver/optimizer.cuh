#pragma once
#include "evaluator.cuh"

namespace dftcu {

struct OptimizationOptions {
    int max_iter = 100;
    double econv = 1e-6;
    int ncheck = 2;
    double step_size = 0.05;
};

class SimpleOptimizer {
  public:
    SimpleOptimizer(const Grid& grid, OptimizationOptions options = {});
    ~SimpleOptimizer() = default;

    /**
     * @brief Optimize density using Steepest Descent on sqrt(rho)
     */
    void solve(RealField& rho, Evaluator& evaluator);

  private:
    const Grid& grid_;
    OptimizationOptions options_;
};

class CGOptimizer {
  public:
    CGOptimizer(const Grid& grid, OptimizationOptions options = {});
    ~CGOptimizer() = default;

    /**
     * @brief Optimize density using Conjugate Gradient (HS) on sqrt(rho)
     */
    void solve(RealField& rho, Evaluator& evaluator);

  private:
    const Grid& grid_;
    OptimizationOptions options_;
};

}  // namespace dftcu
