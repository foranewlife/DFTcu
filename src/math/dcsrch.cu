#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>

#include "dcsrch.cuh"

namespace dftcu {

DCSRCH::Result DCSRCH::operator()(std::function<double(double)> phi,
                                  std::function<double(double)> derphi, double alpha1, double phi0,
                                  double derphi0, int maxiter) {
    double f = phi0;
    double g = derphi0;
    double stp = alpha1;
    std::string task = "START";

    // SciPy's DCSRCH loop
    for (int i = 0; i < maxiter; ++i) {
        task = iterate(stp, f, g, task);

        if (task.compare(0, 2, "FG") == 0) {
            f = phi(stp);
            g = derphi(stp);
        } else {
            break;
        }
    }

    return {stp, f, g, task};
}

std::string DCSRCH::iterate(double& stp, double f, double g, std::string task) {
    const double p5 = 0.5;
    const double p66 = 0.66;
    const double xtrapl = 1.1;
    const double xtrapu = 4.0;

    if (task.substr(0, 5) == "START") {
        if (stp < stpmin_)
            return "ERROR: STP < STPMIN";
        if (stp > stpmax_)
            return "ERROR: STP > STPMAX";
        if (g >= 0)
            return "ERROR: INITIAL G >= ZERO";
        if (ftol_ < 0)
            return "ERROR: FTOL < ZERO";
        if (gtol_ < 0)
            return "ERROR: GTOL < ZERO";
        if (xtol_ < 0)
            return "ERROR: XTOL < ZERO";
        if (stpmin_ < 0)
            return "ERROR: STPMIN < ZERO";
        if (stpmax_ < stpmin_)
            return "ERROR: STPMAX < STPMIN";

        brackt = false;
        stage = 1;
        finit = f;
        ginit = g;
        gtest = ftol_ * ginit;
        width = stpmax_ - stpmin_;
        width1 = width / p5;

        stx = 0.0;
        fx = finit;
        gx = ginit;
        sty = 0.0;
        fy = finit;
        gy = ginit;
        stmin = 0.0;
        stmax = stp + xtrapu * stp;
        task = "FG";
        return task;
    }

    double ftest = finit + stp * gtest;
    if (stage == 1 && f <= ftest && g >= 0)
        stage = 2;

    if (brackt && (stp <= stmin || stp >= stmax))
        task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS";
    if (brackt && stmax - stmin <= xtol_ * stmax)
        task = "WARNING: XTOL TEST SATISFIED";
    if (stp == stpmax_ && f <= ftest && g <= gtest)
        task = "WARNING: STP = STPMAX";
    if (stp == stpmin_ && (f > ftest || g >= gtest))
        task = "WARNING: STP = STPMIN";

    if (f <= ftest && std::abs(g) <= gtol_ * (-ginit))
        task = "CONVERGENCE";

    if (task.substr(0, 4) == "WARN" || task.substr(0, 4) == "CONV")
        return task;

    if (stage == 1 && f <= fx && f > ftest) {
        double fm = f - stp * gtest;
        double fxm = fx - stx * gtest;
        double fym = fy - sty * gtest;
        double gm = g - gtest;
        double gxm = gx - gtest;
        double gym = gy - gtest;

        dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);

        fx = fxm + stx * gtest;
        fy = fym + sty * gtest;
        gx = gxm + gtest;
        gy = gym + gtest;
    } else {
        dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
    }

    if (brackt) {
        if (std::abs(sty - stx) >= p66 * width1)
            stp = stx + p5 * (sty - stx);
        width1 = width;
        width = std::abs(sty - stx);
    }

    if (brackt) {
        stmin = std::min(stx, sty);
        stmax = std::max(stx, sty);
    } else {
        stmin = stp + xtrapl * (stp - stx);
        stmax = stp + xtrapu * (stp - stx);
    }

    stp = std::max(stpmin_, std::min(stpmax_, stp));

    if ((brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol_ * stmax)) {
        stp = stx;
    }

    task = "FG";
    return task;
}

void DCSRCH::dcstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy,
                    double& stp, double fp, double dp, bool& brackt, double stpmin, double stpmax) {
    double sgnd = dp * (dx / std::abs(dx));
    double stpf;

    if (fp > fx) {
        double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        double s = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
        double gamma =
            s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
        if (stp < stx)
            gamma = -gamma;
        double p = (gamma - dx) + theta;
        double q = ((gamma - dx) + gamma) + dp;
        double r = p / q;
        double stpc = stx + r * (stp - stx);
        double stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
        if (std::abs(stpc - stx) <= std::abs(stpq - stx))
            stpf = stpc;
        else
            stpf = stpc + (stpq - stpc) / 2.0;
        brackt = true;
    } else if (sgnd < 0.0) {
        double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        double s = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
        double gamma =
            s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
        if (stp > stx)
            gamma = -gamma;
        double p = (gamma - dp) + theta;
        double q = ((gamma - dp) + gamma) + dx;
        double r = p / q;
        double stpc = stp + r * (stx - stp);
        double stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (std::abs(stpc - stp) > std::abs(stpq - stp))
            stpf = stpc;
        else
            stpf = stpq;
        brackt = true;
    } else if (std::abs(dp) < std::abs(dx)) {
        double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        double s = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
        double gamma =
            s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
        if (stp > stx)
            gamma = -gamma;
        double p = (gamma - dp) + theta;
        double q = (gamma + (dx - dp)) + gamma;
        double r = p / q;
        double stpc;
        if (r < 0.0 && gamma != 0.0)
            stpc = stp + r * (stx - stp);
        else if (stp > stx)
            stpc = stpmax;
        else
            stpc = stpmin;
        double stpq = stp + (dp / (dp - dx)) * (stx - stp);

        if (brackt) {
            if (std::abs(stpc - stp) < std::abs(stpq - stp))
                stpf = stpc;
            else
                stpf = stpq;
            if (stp > stx)
                stpf = std::min(stp + 0.66 * (sty - stp), stpf);
            else
                stpf = std::max(stp + 0.66 * (sty - stp), stpf);
        } else {
            if (std::abs(stpc - stp) > std::abs(stpq - stp))
                stpf = stpc;
            else
                stpf = stpq;
            stpf = std::max(stpmin, std::min(stpmax, stpf));
        }
    } else {
        if (brackt) {
            double theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
            double s = std::max({std::abs(theta), std::abs(dy), std::abs(dp)});
            double gamma =
                s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dy / s) * (dp / s)));
            if (stp > sty)
                gamma = -gamma;
            double p = (gamma - dp) + theta;
            double q = ((gamma - dp) + gamma) + dy;
            double r = p / q;
            double stpc = stp + r * (sty - stp);
            stpf = stpc;
        } else if (stp > stx)
            stpf = stpmax;
        else
            stpf = stpmin;
    }

    if (fp > fx) {
        sty = stp;
        fy = fp;
        dy = dp;
    } else {
        if (sgnd < 0.0) {
            sty = stx;
            fy = fx;
            dy = dx;
        }
        stx = stp;
        fx = fp;
        dx = dp;
    }

    stp = stpf;
}

}  // namespace dftcu
