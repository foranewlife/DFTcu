# Technical Note: Computation Graph via Expression Templates

**TN-ID**: TN_COMPUTATION_GRAPH
**Date**: 2025-12-27
**Author**: Gemini
**Status**: In Progress

## 1. Motivation

The current computational backend executes operations in an "eager" or "imperative" manner. Each mathematical operation (e.g., `v_add`, `v_mul`) triggers an immediate CUDA kernel launch. This approach is simple but suboptimal, leading to:

1.  **High Kernel Launch Overhead**: Frequent, small kernel launches introduce significant latency.
2.  **Excessive Memory Bandwidth Usage**: Intermediate results from chained operations (e.g., `A + B * C`) are written to and read back from global memory, creating unnecessary data movement.
3.  **Lack of Global Optimization**: The execution model cannot perform optimizations like operator fusion, as it lacks a global view of the entire computation.

To address these issues, we propose implementing a lightweight computation graph using **Expression Templates (ET)**. This technique allows us to defer computation, build expression trees at compile time, and generate a single, fused CUDA kernel for a complex mathematical expression, thereby significantly improving performance.

## 2. Core Concept: Expression Templates

Expression Templates are a C++ template metaprogramming technique that overloads operators (like `+`, `*`) to return a lightweight proxy object—an "expression object"—instead of immediately computing the result.

Consider the expression:
```cpp
Field D = A * B + C;
```

Instead of executing this in two steps (`tmp = A * B`, `D = tmp + C`), the ET approach does the following:

1.  `A * B` returns an object of type `MulExpr<Field, Field>`. This object does **not** store the result; it only stores references to its operands, `A` and `B`.
2.  The result of that, `MulExpr<...>`, is then added to `C`. This returns another expression object, `AddExpr<MulExpr<Field, Field>, Field>`, which again only stores references to its operands.
3.  Finally, the assignment operator `Field::operator=(const Expression&)` is called. This is the trigger point. This operator traverses the expression tree represented by the nested template types, generates a single CUDA kernel string (e.g., `out[i] = a[i] * b[i] + c[i];`), compiles it on-the-fly using **NVRTC**, and launches the fused kernel.

## 3. Proposed Design & Implementation Plan

We will adopt a phased approach to manage complexity.

### Phase 1: Expression Templates with Pre-defined Fused Kernels

This phase avoids the complexity of runtime compilation (NVRTC) while still providing significant gains for common operation chains.

#### 3.1. `Expr` Base Class

We will define a CRTP (Curiously Recurring Template Pattern) base class for all expression objects.

```cpp
template <typename E>
struct Expr {
    // Allows us to access the actual expression type (e.g., AddExpr)
    const E& self() const { return static_cast<const E&>(*this); }
};
```

#### 3.2. `Field` as an Expression

`RealField` will inherit from `Expr<RealField>` and implement a simple `operator()` to access its own data.

```cpp
class RealField : public Expr<RealField> {
public:
    // ... existing methods ...

    __device__ double operator[](size_t i) const { return data_[i]; }
};
```

#### 3.3. Binary Expression Class

A generic class will represent binary operations.

```cpp
template <typename Op, typename LHS, typename RHS>
class BinaryExpr : public Expr<BinaryExpr<Op, LHS, RHS>> {
    const LHS& lhs_;
    const RHS& rhs_;

public:
    BinaryExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}

    // Recursively applies the operator to the operands' elements
    __device__ double operator[](size_t i) const {
        return Op::apply(lhs_.self()[i], rhs_.self()[i]);
    }
};
```
The `Op` template parameter will be a struct defining the operation, e.g., `struct AddOp { static __device__ double apply(double a, double b) { return a + b; } };`.

#### 3.4. Operator Overloads

Global operator overloads will construct the `BinaryExpr` objects.

```cpp
template <typename E1, typename E2>
BinaryExpr<AddOp, E1, E2> operator+(const Expr<E1>& u, const Expr<E2>& v) {
    return BinaryExpr<AddOp, E1, E2>(u.self(), v.self());
}
// Overloads for *, -, / will follow the same pattern.
```

#### 3.5. Fused Kernel and Assignment

The `RealField::operator=` will be overloaded to accept an `Expr`. We will use template specialization to detect specific patterns we want to fuse.

```cpp
// Generic assignment via a templated kernel
template <typename E>
__global__ void assignment_kernel(double* out, const E expr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expr[i];
    }
}

// In RealField class
template <typename E>
void RealField::operator=(const Expr<E>& expr) {
    const int block_size = 256;
    const int grid_size = (size() + block_size - 1) / block_size;
    assignment_kernel<<<grid_size, block_size, 0, stream()>>>(data(), expr.self(), size());
    GPU_CHECK_KERNEL;
}
```
This generic assignment already achieves fusion. For a `D = A*B+C` expression, the `expr[i]` call within `assignment_kernel` will be inlined by the compiler into `Op::apply(Op::apply(A[i], B[i]), C[i])`, which becomes `A[i] * B[i] + C[i]`. This happens within a single kernel, reading A, B, and C once and writing to D once.

### Phase 2: Dynamic Kernel Compilation with NVRTC (Future Work)

If Phase 1 proves successful but insufficient, we can extend the framework to support arbitrary expressions.

1.  **Kernel Generation**: The `operator=` will traverse the expression tree and generate a CUDA source string.
2.  **Runtime Compilation**: Use the NVRTC library to compile this string into PTX.
3.  **Dynamic Loading**: Use the CUDA Driver API (`cuModuleLoadData`, `cuModuleGetFunction`) to load the PTX and get a kernel handle.
4.  **Execution**: Launch the dynamically compiled kernel using `cuLaunchKernel`.
5.  **Caching**: Implement a caching mechanism where generated kernels are stored in a map, with the expression's type string (`typeid(E).name()`) acting as the key, to avoid recompiling the same expression repeatedly.

## 4. Initial Target for Implementation

The primary target for this refactoring is the `Evaluator::compute` method and the various optimizer implementations (`SimpleOptimizer`, `CGOptimizer`, `TNOptimizer`). These areas involve numerous chained, element-wise vector operations (e.g., gradient calculation, `phi` updates) that are ideal candidates for fusion.

By replacing calls like `v_add`, `v_mul` with overloaded operators, we expect to see a significant reduction in kernel launch overhead and an increase in performance due to improved data locality.
