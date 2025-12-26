#pragma once
#include <memory>

#include "model/field.cuh"

namespace dftcu {

/**
 * @brief A type-erased wrapper for any density functional component.
 *
 * Uses a Concept/Model pattern (Type Erasure) to provide a uniform interface for
 * disparate functional types (TF, vW, Hartree, XC, etc.) without requiring a
 * common base class. This allows the Evaluator to store a heterogeneous list of
 * functionals.
 */
class Functional {
  public:
    /**
     * @brief Constructs a Functional wrapper from any object providing a 'compute' method.
     *
     * The wrapped object must have a method with signature:
     * `double compute(const RealField& rho, RealField& v_out)`
     *
     * @tparam T The concrete functional type.
     * @param obj A shared pointer to the concrete functional object.
     */
    template <typename T>
    Functional(std::shared_ptr<T> obj) : self_(std::make_shared<Model<T>>(std::move(obj))) {}

    /**
     * @brief Executes the wrapped functional's computation.
     *
     * @param rho Input real-space density field.
     * @param v_out Output field to which the functional's potential contribution is added.
     * @return Energy contribution of this functional.
     */
    double compute(const RealField& rho, RealField& v_out) const {
        return self_->compute_impl(rho, v_out);
    }

  private:
    /** @brief Internal interface for type erasure. */
    struct Concept {
        virtual ~Concept() = default;
        virtual double compute_impl(const RealField& rho, RealField& v_out) const = 0;
    };

    /** @brief Internal template implementation for type erasure. */
    template <typename T>
    struct Model : Concept {
        Model(std::shared_ptr<T> obj) : data(std::move(obj)) {}
        double compute_impl(const RealField& rho, RealField& v_out) const override {
            return data->compute(rho, v_out);
        }
        std::shared_ptr<T> data;
    };

    std::shared_ptr<const Concept> self_; /**< Shared reference to the type-erased object */
};

}  // namespace dftcu
