#include "evaluator.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

Evaluator::Evaluator(Grid& grid) : grid_(grid), v_tmp_(grid) {}

double Evaluator::compute(const RealField& rho, RealField& v_tot) {
    v_tot.fill(0.0);
    double energy = 0.0;
    size_t n = rho.size();

    for (const auto& comp : components_) {
        v_tmp_.fill(0.0);
        energy += comp.compute(rho, v_tmp_);
        v_add(n, v_tot.data(), v_tmp_.data(), v_tot.data());
    }

    return energy;
}

}  // namespace dftcu
