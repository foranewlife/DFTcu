#include <cmath>
#include <iomanip>
#include <iostream>

#include "../src/math/dcsrch.cuh"
#include "../src/math/linesearch.cuh"

void test_dcsrch_simple() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test 3: DCSRCH on Simple Quadratic f(x) = (x-2)^2" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    auto phi = [](double x) { return (x - 2.0) * (x - 2.0); };
    auto derphi = [](double x) { return 2.0 * (x - 2.0); };

    double phi0 = phi(0.0);
    double derphi0 = derphi(0.0);

    dftcu::DCSRCH search(1e-4, 0.2);
    auto res = search(phi, derphi, 1.0, phi0, derphi0);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result:" << std::endl;
    std::cout << "  Task:  " << res.task << std::endl;
    std::cout << "  alpha: " << res.alpha << std::endl;
    std::cout << "  f:     " << res.f << std::endl;
    std::cout << "  g:     " << res.g << std::endl;

    double error = std::abs(res.alpha - 2.0);
    std::cout << "  Error: " << std::scientific << error << std::endl;
    std::cout << "  PASS:  " << (error < 1e-10 ? "YES" : "NO") << std::endl;
}

void test_simple_quadratic() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Test 1: Simple Quadratic f(x) = (x-2)^2" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // f(x) = (x-2)^2, f'(x) = 2(x-2)
    auto phi = [](double x) { return (x - 2.0) * (x - 2.0); };
    auto derphi = [](double x) { return 2.0 * (x - 2.0); };

    double phi0 = phi(0.0);
    double derphi0 = derphi(0.0);

    std::cout << "phi(0) = " << phi0 << std::endl;
    std::cout << "derphi(0) = " << derphi0 << std::endl;
    std::cout << "Expected minimum at x=2.0" << std::endl;

    double alpha, phi_result;
    bool converged = dftcu::scalar_search_wolfe1(phi, derphi, phi0, derphi0, 1e10, 1e-4, 0.2, 10.0,
                                                 0.0, 1e-14, alpha, phi_result);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\nResult:" << std::endl;
    std::cout << "  Converged: " << (converged ? "Yes" : "No") << std::endl;
    std::cout << "  alpha = " << alpha << std::endl;
    std::cout << "  phi   = " << phi_result << std::endl;
    std::cout << "  derphi = " << derphi(alpha) << std::endl;

    // Compare with scipy target: alpha=2.0, phi=0.0
    double error = std::abs(alpha - 2.0);
    std::cout << "\n  Error vs scipy: " << std::scientific << error << std::endl;
    std::cout << "  PASS: " << (error < 1e-6 ? "YES" : "NO") << std::endl;
}

void test_rosenbrock() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test 2: Rosenbrock along search direction" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Starting point and search direction
    double x0 = -1.0, y0 = 1.0;
    double px = 1.0, py = 0.0;

    // Rosenbrock: f(x,y) = 100(y-x^2)^2 + (1-x)^2
    auto phi = [x0, y0, px, py](double alpha) {
        double x = x0 + alpha * px;
        double y = y0 + alpha * py;
        return 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x);
    };

    auto derphi = [x0, y0, px, py](double alpha) {
        double x = x0 + alpha * px;
        double y = y0 + alpha * py;
        double dfdx = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x);
        double dfdy = 200.0 * (y - x * x);
        return dfdx * px + dfdy * py;
    };

    double phi0 = phi(0.0);
    double derphi0 = derphi(0.0);

    std::cout << "phi(0) = " << phi0 << std::endl;
    std::cout << "derphi(0) = " << derphi0 << std::endl;

    double alpha, phi_result;
    bool converged = dftcu::scalar_search_wolfe1(phi, derphi, phi0, derphi0, 1e10, 1e-4, 0.2, 10.0,
                                                 0.0, 1e-14, alpha, phi_result);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\nResult:" << std::endl;
    std::cout << "  Converged: " << (converged ? "Yes" : "No") << std::endl;
    std::cout << "  alpha = " << alpha << std::endl;
    std::cout << "  phi   = " << phi_result << std::endl;
    std::cout << "  derphi = " << derphi(alpha) << std::endl;
}

int main() {
    test_simple_quadratic();
    test_rosenbrock();
    test_dcsrch_simple();

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Line search unit tests completed" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}
