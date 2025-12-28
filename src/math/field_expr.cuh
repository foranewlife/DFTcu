#pragma once

#include "utilities/common.cuh"

namespace dftcu {

// Forward declaration
class RealField;

// CRTP Base class for all expression objects. Every expression exposes a View
// type that is trivially copyable to device memory and implements operator[].
template <typename E>
struct Expr {
    __host__ __device__ __forceinline__ const E& self() const {
        return static_cast<const E&>(*this);
    }

    __host__ __device__ __forceinline__ auto view() const { return self().view(); }
};

// --- Operator Structs ---
struct AddOp {
    static __device__ __forceinline__ double apply(double a, double b) { return a + b; }
};

struct SubOp {
    static __device__ __forceinline__ double apply(double a, double b) { return a - b; }
};

struct MulOp {
    static __device__ __forceinline__ double apply(double a, double b) { return a * b; }
};

struct DivOp {
    static __device__ __forceinline__ double apply(double a, double b) {
        return (b != 0) ? a / b : 0.0;
    }
};

// --- Binary Expression Class ---
template <typename Op, typename LHS, typename RHS>
class BinaryExpr : public Expr<BinaryExpr<Op, LHS, RHS>> {
  public:
    using View = BinaryExpr;

    BinaryExpr(const LHS& lhs, const RHS& rhs) : lhs_view_(lhs.view()), rhs_view_(rhs.view()) {}

    __device__ __forceinline__ double operator[](size_t i) const {
        return Op::apply(lhs_view_[i], rhs_view_[i]);
    }

    __host__ __device__ __forceinline__ View view() const { return *this; }

  private:
    typename LHS::View lhs_view_;
    typename RHS::View rhs_view_;
};

// --- Operator Overloads ---
template <typename E1, typename E2>
BinaryExpr<AddOp, E1, E2> operator+(const Expr<E1>& u, const Expr<E2>& v) {
    return BinaryExpr<AddOp, E1, E2>(u.self(), v.self());
}

template <typename E1, typename E2>
BinaryExpr<SubOp, E1, E2> operator-(const Expr<E1>& u, const Expr<E2>& v) {
    return BinaryExpr<SubOp, E1, E2>(u.self(), v.self());
}

template <typename E1, typename E2>
BinaryExpr<MulOp, E1, E2> operator*(const Expr<E1>& u, const Expr<E2>& v) {
    return BinaryExpr<MulOp, E1, E2>(u.self(), v.self());
}

// Overload for scalar multiplication
template <typename E>
class ScalarMulExpr : public Expr<ScalarMulExpr<E>> {
  public:
    using View = ScalarMulExpr;

    ScalarMulExpr(double scalar, const Expr<E>& expr)
        : scalar_(scalar), expr_view_(expr.self().view()) {}

    __device__ __forceinline__ double operator[](size_t i) const { return scalar_ * expr_view_[i]; }

    __host__ __device__ __forceinline__ View view() const { return *this; }

  private:
    double scalar_;
    typename E::View expr_view_;
};

template <typename E>
ScalarMulExpr<E> operator*(double scalar, const Expr<E>& expr) {
    return ScalarMulExpr<E>(scalar, expr);
}

template <typename E>
ScalarMulExpr<E> operator*(const Expr<E>& expr, double scalar) {
    return ScalarMulExpr<E>(scalar, expr);
}

// --- Field-Scalar Operations ---
template <typename E>
class ScalarSubExpr : public Expr<ScalarSubExpr<E>> {
  public:
    using View = ScalarSubExpr;

    ScalarSubExpr(const Expr<E>& expr, double scalar)
        : scalar_(scalar), expr_view_(expr.self().view()) {}

    __device__ __forceinline__ double operator[](size_t i) const { return expr_view_[i] - scalar_; }

    __host__ __device__ __forceinline__ View view() const { return *this; }

  private:
    double scalar_;
    typename E::View expr_view_;
};

template <typename E>
ScalarSubExpr<E> operator-(const Expr<E>& expr, double scalar) {
    return ScalarSubExpr<E>(expr, scalar);
}

}  // namespace dftcu
