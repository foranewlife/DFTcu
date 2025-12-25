#pragma once
#include "model/field.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief Abstract base class for Kinetic Energy Density Functionals (KEDF)
 *
 * All KEDF implementations should inherit from this class and implement
 * the compute() method.
 */
class KEDF_Base {
  public:
    virtual ~KEDF_Base() = default;

    /**
     * @brief Compute kinetic energy and potential
     * @param rho Input density field
     * @param v_kedf Output potential field (functional derivative δE/δρ)
     * @return Kinetic energy value
     */
    virtual double compute(const RealField& rho, RealField& v_kedf) = 0;

    /**
     * @brief Get the name of this functional
     */
    virtual const char* name() const = 0;
};

}  // namespace dftcu
