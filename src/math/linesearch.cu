#include "dcsrch.cuh"
#include "linesearch.cuh"

namespace dftcu {

bool scalar_search_wolfe1(std::function<double(double)> phi, std::function<double(double)> derphi,
                          double phi0, double derphi0, double phi0_old, double c1, double c2,
                          double amax, double amin, double xtol, double& alpha_star,
                          double& phi_star) {
    double alpha1;
    if (phi0_old < 1e9 && derphi0 != 0.0) {
        alpha1 = std::min(1.0, 1.01 * 2.0 * (phi0 - phi0_old) / derphi0);
        if (alpha1 < 0.0)
            alpha1 = 1.0;
    } else {
        alpha1 = 1.0;
    }

    DCSRCH dcsrch(c1, c2, xtol, amin, amax);
    auto res = dcsrch(phi, derphi, alpha1, phi0, derphi0);

    alpha_star = res.alpha;
    phi_star = res.f;

    return res.task.substr(0, 4) == "CONV";
}

}  // namespace dftcu
