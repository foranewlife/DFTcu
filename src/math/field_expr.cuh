#pragma once

#include "utilities/common.cuh"

namespace dftcu {

// Forward declaration
class RealField;

// CRTP Base class for all expression objects
template <typename E>
struct Expr {
    __host__ __device__ __forceinline__ const E& self() const {
        return static_cast<const E&>(*this);
    }
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
    const LHS& lhs_;
    const RHS& rhs_;

  public:
    BinaryExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}

    // Recursively applies the operator to the operands' elements
    __device__ __forceinline__ double operator[](size_t i) const {
        return Op::apply(lhs_.self()[i], rhs_.self()[i]);
    }
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
    const double scalar_;
    const E& expr_;

  public:
    ScalarMulExpr(double scalar, const E& expr) : scalar_(scalar), expr_(expr) {}

    __device__ __forceinline__ double operator[](size_t i) const {
        return scalar_ * expr_.self()[i];
    }
};

template <typename E>
ScalarMulExpr<E> operator*(double scalar, const Expr<E>& expr) {
    return ScalarMulExpr<E>(scalar, expr.self());
}

template <typename E>
ScalarMulExpr<E> operator*(const Expr<E>& expr, double scalar) {
    return ScalarMulExpr<E>(scalar, expr.self());
}

// --- Field-Scalar Operations ---

template <typename E>
class ScalarSubExpr : public Expr<ScalarSubExpr<E>> {
    const double scalar_;
    const E& expr_;

  public:
    ScalarSubExpr(const E& expr, double scalar) : scalar_(scalar), expr_(expr) {}

    __device__ __forceinline__ double operator[](size_t i) const {
        return expr_.self()[i] - scalar_;
    }
};

template <typename E>
ScalarSubExpr<E> operator-(const Expr<E>& expr, double scalar) {
    return ScalarSubExpr<E>(expr.self(), scalar);
}

}  // namespace dftcu
