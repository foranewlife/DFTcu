#pragma once
#include <functional>

namespace dftcu {

/**
 * @brief Scalar line search satisfying strong Wolfe conditions
 *
 * This implements scipy's scalar_search_wolfe1 (DCSRCH algorithm from MINPACK).
 *
 * @param phi Function to minimize: phi(alpha)
 * @param derphi Derivative: phi'(alpha)
 * @param phi0 Value at alpha=0
 * @param derphi0 Derivative at alpha=0
 * @param phi0_old Previous value (for initial step estimation), use 1e10 if unavailable
 * @param c1 Armijo condition parameter (typically 1e-4)
 * @param c2 Curvature condition parameter (typically 0.2 for CG, 0.9 for Newton)
 * @param amax Maximum step size
 * @param amin Minimum step size
 * @param xtol Relative tolerance for acceptable step
 * @param alpha_star Output: optimal step size
 * @param phi_star Output: value at optimal step
 * @return true if converged, false otherwise
 */
bool scalar_search_wolfe1(std::function<double(double)> phi, std::function<double(double)> derphi,
                          double phi0, double derphi0, double phi0_old, double c1, double c2,
                          double amax, double amin, double xtol, double& alpha_star,
                          double& phi_star);

}  // namespace dftcu
