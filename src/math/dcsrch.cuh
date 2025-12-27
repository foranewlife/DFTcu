#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

namespace dftcu {

/**
 * @brief Port of Scipy's DCSRCH (MINPACK-2) for line search.
 */
class DCSRCH {
  public:
    DCSRCH(double ftol = 1e-4, double gtol = 0.9, double xtol = 1e-11, double stpmin = 1e-20,
           double stpmax = 1e20)
        : ftol_(ftol), gtol_(gtol), xtol_(xtol), stpmin_(stpmin), stpmax_(stpmax) {}

    struct Result {
        double alpha;
        double f;
        double g;
        std::string task;
    };

    /**
     * @brief Performs the line search.
     * @param phi Function evaluating energy at alpha.
     * @param derphi Function evaluating derivative at alpha.
     * @param alpha1 Initial step estimate.
     * @param phi0 Initial energy (at alpha=0).
     * @param derphi0 Initial derivative (at alpha=0).
     * @param maxiter Maximum iterations.
     */
    Result operator()(std::function<double(double)> phi, std::function<double(double)> derphi,
                      double alpha1, double phi0, double derphi0, int maxiter = 100);

  private:
    double ftol_, gtol_, xtol_, stpmin_, stpmax_;

    // Internal state
    int stage;
    double ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1;
    bool brackt;

    std::string iterate(double& stp, double f, double g, std::string task);
    void dcstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy,
                double& stp, double fp, double dp, bool& brackt, double stpmin, double stpmax);
};

}  // namespace dftcu
