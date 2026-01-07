#include "density_functional_potential.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

DensityFunctionalPotential::DensityFunctionalPotential(Grid& grid) : grid_(grid), v_tmp_(grid) {}

double DensityFunctionalPotential::compute(const RealField& rho, RealField& v_tot) {
    v_tot.fill(0.0);
    double energy = 0.0;
    size_t n = rho.size();

    for (const auto& comp : components_) {
        v_tmp_.fill(0.0);
        energy += comp.compute(rho, v_tmp_);
        v_tot = v_tot + v_tmp_;
    }

    return energy;
}

double DensityFunctionalPotential::get_v0() const {
    double v0 = 0.0;
    for (const auto& comp : components_) {
        v0 += comp.get_v0();
    }
    return v0;
}

}  // namespace dftcu
