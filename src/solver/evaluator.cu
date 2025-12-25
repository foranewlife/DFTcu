#include "evaluator.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

Evaluator::Evaluator(const Grid& grid) : grid_(grid) {}

double Evaluator::compute(const RealField& rho, RealField& v_tot) {
    v_tot.fill(0.0);
    double energy = 0.0;
    size_t n = rho.size();

    // 1. KEDF
    if (kedf_) {
        RealField v_tmp(grid_);
        energy += kedf_->compute(rho, v_tmp);
        v_add(n, v_tot.data(), v_tmp.data(), v_tot.data());
    }

    // 2. Hartree
    if (hartree_) {
        RealField v_tmp(grid_);
        double e_h = 0.0;
        hartree_->compute(rho, v_tmp, e_h);
        energy += e_h;
        v_add(n, v_tot.data(), v_tmp.data(), v_tot.data());
    }

    // 3. XC
    if (xc_) {
        RealField v_tmp(grid_);
        energy += xc_->compute(rho, v_tmp);
        v_add(n, v_tot.data(), v_tmp.data(), v_tot.data());
    }

    // 4. Local Pseudo
    if (pseudo_) {
        RealField v_tmp(grid_);
        pseudo_->compute(v_tmp);
        // Energy E_ext = integral( rho * v_ext )
        energy += dot_product(n, rho.data(), v_tmp.data()) * grid_.dv();
        v_add(n, v_tot.data(), v_tmp.data(), v_tot.data());
    }

    return energy;
}

}  // namespace dftcu
