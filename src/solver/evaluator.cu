#include "evaluator.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

Evaluator::Evaluator(std::shared_ptr<Grid> grid) : grid_(grid) {}

double Evaluator::compute(const RealField& rho, RealField& v_tot) {
    v_tot.fill(0.0);
    double energy = 0.0;
    size_t n = rho.size();

    // Use a single temporary field to accumulate contributions
    // and avoid repeated allocations.
    RealField v_tmp(grid_);

    for (const auto& comp : components_) {
        v_tmp.fill(0.0);
        energy += comp.compute(rho, v_tmp);
        v_add(n, v_tot.data(), v_tmp.data(), v_tot.data());
    }

    return energy;
}

}  // namespace dftcu
