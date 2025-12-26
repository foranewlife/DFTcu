#pragma once
#include <memory>

#include "model/field.cuh"

namespace dftcu {

/**
 * @brief A type-erased wrapper for any functional component.
 * Mimics Rust's Arc<dyn Trait> behavior. Stores a shared reference
 * to the underlying functional object.
 */
class Functional {
  public:
    /**
     * @brief Construct from a shared pointer to any class that has a compatible compute method.
     */
    template <typename T>
    Functional(std::shared_ptr<T> obj) : self_(std::make_shared<Model<T>>(std::move(obj))) {}

    /**
     * @brief Execute the functional computation.
     */
    double compute(const RealField& rho, RealField& v_out) const {
        return self_->compute_impl(rho, v_out);
    }

  private:
    struct Concept {
        virtual ~Concept() = default;
        virtual double compute_impl(const RealField& rho, RealField& v_out) const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(std::shared_ptr<T> obj) : data(std::move(obj)) {}
        double compute_impl(const RealField& rho, RealField& v_out) const override {
            return data->compute(rho, v_out);
        }
        std::shared_ptr<T> data;
    };

    std::shared_ptr<const Concept> self_;
};

}  // namespace dftcu
