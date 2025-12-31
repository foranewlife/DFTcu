#include <iostream>
#include <vector>

#include "solver/mixer.cuh"
#include "utilities/error.cuh"
#include "utilities/kernels.cuh"

namespace dftcu {

// Simple LU decomposition solver for small matrices on CPU
void solve_linear_system(int n, double* A, double* b, double* x) {
    std::vector<double> matrix(n * n);
    for (int i = 0; i < n * n; ++i)
        matrix[i] = A[i];

    // Partial pivoting
    std::vector<int> p(n);
    for (int i = 0; i < n; i++)
        p[i] = i;

    for (int i = 0; i < n; i++) {
        double max_val = 0;
        int max_row = i;
        for (int k = i; k < n; k++) {
            if (std::abs(matrix[k * n + i]) > max_val) {
                max_val = std::abs(matrix[k * n + i]);
                max_row = k;
            }
        }
        std::swap(p[i], p[max_row]);
        for (int k = 0; k < n; k++)
            std::swap(matrix[i * n + k], matrix[max_row * n + k]);

        for (int j = i + 1; j < n; j++) {
            matrix[j * n + i] /= matrix[i * n + i];
            for (int k = i + 1; k < n; k++) {
                matrix[j * n + k] -= matrix[j * n + i] * matrix[i * n + k];
            }
        }
    }

    // Forward substitution
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        y[i] = b[p[i]];
        for (int j = 0; j < i; j++) {
            y[i] -= matrix[i * n + j] * y[j];
        }
    }

    // Backward substitution
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= matrix[i * n + j] * x[j];
        }
        x[i] /= matrix[i * n + i];
    }
}

// Linear Mixer Implementation
void LinearMixer::mix(const RealField& rho_in, const RealField& rho_out, RealField& rho_next) {
    size_t n = rho_in.size();

    // rho_next = (1-beta)*rho_in + beta*rho_out
    v_scale(n, 1.0 - beta_, rho_in.data(), rho_next.data(), grid_.stream());

    // Use a temporary for the second part or just use an axpy kernel if we had one.
    // field.cuh might have better operators. Let's check field.cuh.
    // Assuming RealField has data() and we have v_add and v_scale in kernels.cuh
    RealField v_tmp(grid_);
    v_scale(n, beta_, rho_out.data(), v_tmp.data(), grid_.stream());
    v_add(n, rho_next.data(), v_tmp.data(), rho_next.data(), grid_.stream());

    grid_.synchronize();
}

// Broyden (Anderson) Mixer Implementation
BroydenMixer::BroydenMixer(Grid& grid, double beta, int history_size)
    : grid_(grid), beta_(beta), history_size_(history_size) {
    x_history_.reserve(history_size);
    f_history_.reserve(history_size);
}

void BroydenMixer::reset() {
    x_history_.clear();
    f_history_.clear();
    current_history_ = 0;
    iter_count_ = 0;
}

void BroydenMixer::mix(const RealField& rho_in, const RealField& rho_out, RealField& rho_next) {
    size_t n = rho_in.size();

    // 1. Calculate current residual f = rho_out - rho_in
    auto f_curr = std::make_unique<RealField>(grid_);
    v_sub(n, rho_out.data(), rho_in.data(), f_curr->data(), grid_.stream());
    grid_.synchronize();

    // Store current rho_in and residual
    if (x_history_.size() < (size_t)history_size_) {
        x_history_.push_back(std::make_unique<RealField>(grid_));
        x_history_.back()->copy_from(rho_in);
        f_history_.push_back(std::move(f_curr));
    } else {
        // Recycle oldest history
        int oldest = iter_count_ % history_size_;
        x_history_[oldest]->copy_from(rho_in);
        f_history_[oldest] = std::move(f_curr);
    }

    int m = x_history_.size();
    iter_count_++;

    if (m == 1) {
        // First iteration, fallback to linear mixing
        LinearMixer(grid_, beta_).mix(rho_in, rho_out, rho_next);
        return;
    }

    // 2. Build Anderson matrix A_ij = <f_i | f_j> on CPU
    std::vector<double> A((m + 1) * (m + 1), 0.0);
    std::vector<double> b(m + 1, 0.0);

    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double dot = f_history_[i]->dot(*f_history_[j]) * grid_.dv_bohr();
            A[i * (m + 1) + j] = dot;
            A[j * (m + 1) + i] = dot;
        }
        A[i * (m + 1) + m] = 1.0;
        A[m * (m + 1) + i] = 1.0;
        b[i] = 0.0;
    }
    A[m * (m + 1) + m] = 0.0;
    b[m] = 1.0;

    // 3. Solve A * w = b for weights
    std::vector<double> w(m + 1);
    solve_linear_system(m + 1, A.data(), b.data(), w.data());

    // 4. Calculate next density: rho_next = sum w_i * (x_i + beta * f_i)
    rho_next.fill(0.0);
    RealField work(grid_);

    for (int i = 0; i < m; ++i) {
        double weight = w[i];
        // rho_next += weight * x_i
        v_scale(n, weight, x_history_[i]->data(), work.data(), grid_.stream());
        v_add(n, rho_next.data(), work.data(), rho_next.data(), grid_.stream());

        // rho_next += weight * beta * f_i
        v_scale(n, weight * beta_, f_history_[i]->data(), work.data(), grid_.stream());
        v_add(n, rho_next.data(), work.data(), rho_next.data(), grid_.stream());
    }

    grid_.synchronize();
}

}  // namespace dftcu
