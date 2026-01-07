#pragma once
#include <memory>
#include <type_traits>

#include "model/field.cuh"

namespace dftcu {

// C++14 compatible void_t
template <typename... T>
using void_t = void;

// Type trait to detect if a functional class has get_v_of_0() method
template <typename T, typename = void>
struct has_get_v_of_0 : std::false_type {};

template <typename T>
struct has_get_v_of_0<T, void_t<decltype(std::declval<T>().get_v_of_0())>> : std::true_type {};

// Helper to call get_v_of_0 if it exists (SFINAE)
template <typename T>
typename std::enable_if<has_get_v_of_0<T>::value, double>::type call_get_v_of_0(const T& obj) {
    return obj.get_v_of_0();
}

template <typename T>
typename std::enable_if<!has_get_v_of_0<T>::value, double>::type call_get_v_of_0(const T& obj) {
    return 0.0;
}

/**
 * @brief A type-erased wrapper for any density functional component.
 */
class Functional {
  public:
    template <typename T>
    Functional(std::shared_ptr<T> obj) : self_(std::make_shared<Model<T>>(std::move(obj))) {}

    double compute(const RealField& rho, RealField& v_out) const {
        return self_->compute_impl(rho, v_out);
    }

    /** @brief Returns G=0 component of the potential (Alpha term etc.) */
    double get_v0() const { return self_->get_v0_impl(); }

    std::shared_ptr<void> underlying() const { return self_->get_data(); }

  private:
    struct Concept {
        virtual ~Concept() = default;
        virtual double compute_impl(const RealField& rho, RealField& v_out) const = 0;
        virtual double get_v0_impl() const = 0;
        virtual std::shared_ptr<void> get_data() const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(std::shared_ptr<T> obj) : data(std::move(obj)) {}
        double compute_impl(const RealField& rho, RealField& v_out) const override {
            return data->compute(rho, v_out);
        }
        double get_v0_impl() const override { return call_get_v_of_0(*data); }
        std::shared_ptr<void> get_data() const override { return data; }
        std::shared_ptr<T> data;
    };

    std::shared_ptr<const Concept> self_;
};

}  // namespace dftcu
