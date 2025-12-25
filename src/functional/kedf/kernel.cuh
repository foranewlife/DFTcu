#pragma once

namespace dftcu {

/**
 * @brief Inverse Lindhard function used in non-local KEDFs
 * 
 * F(eta) = 1 / (0.5 + 0.25 * (1 - eta^2) / eta * log|(1+eta)/(1-eta)|) - 3*mu*eta^2 - lbda
 * 
 * @param eta q / (2 * k_F)
 * @param lbda Weight for TF part (usually 1.0)
 * @param mu Weight for vW part (usually 1.0)
 * @return Value of the inverse Lindhard function
 */
__device__ inline double lindhard_function(double eta, double lbda, double mu) {
    const double tol = 1e-8;
    if (eta < tol) {
        // Limit eta -> 0: 1 - lbda + eta^2 * (1/3 - 3*mu)
        return 1.0 - lbda + eta * eta * (1.0 / 3.0 - 3.0 * mu);
    } else if (std::abs(eta - 1.0) < tol) {
        // Limit eta -> 1: 2 - lbda - 3*mu + 20*(eta-1)
        // Note: DFTpy uses 20 for eta-1 term
        return 2.0 - lbda - 3.0 * mu + 20.0 * (eta - 1.0);
    } else if (eta > 10.0) {
        // Large eta expansion to avoid log issues and match behavior
        // Lindhard -> 3*eta^2 for large eta
        // So 1/Lindhard -> 1/(3*eta^2) -> 0
        // The term -3*mu*eta^2 will dominate if mu=1
        // However, usually we want to damp the kernel at high q
        return 3.0 * (1.0 - mu) * eta * eta - lbda - 0.6;
    } else {
        double log_term = std::log(std::abs((1.0 + eta) / (1.0 - eta)));
        double chi = 0.5 + 0.25 * (1.0 - eta * eta) / eta * log_term;
        return 1.0 / chi - 3.0 * mu * eta * eta - lbda;
    }
}

} // namespace dftcu
