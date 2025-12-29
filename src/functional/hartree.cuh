#pragma once
#include "fft/fft_solver.cuh"
#include "model/field.cuh"

namespace dftcu {

class Hartree {
  public:
    /**
     * @brief Constructs a Hartree functional.
     */
    Hartree();

    /** @brief Default destructor. */
    ~Hartree() = default;

    /**
     * @brief Computes Hartree potential and energy.
     * @param rho Input real-space density field.
     * @param vh Output real-space Hartree potential field.
     * @param energy Output Hartree energy.
     */
    void compute(const RealField& rho, RealField& vh, double& energy);

    /**
     * @brief Unified interface for Evaluator.
     */
    double compute(const RealField& rho, RealField& v_out) {
        double energy = 0.0;
        compute(rho, v_out, energy);
        return energy;
    }

    const char* name() const { return "Hartree"; }

  private:
    void initialize_buffers(Grid& grid);

    Grid* grid_ = nullptr;
    std::unique_ptr<FFTSolver> fft_;
    std::unique_ptr<ComplexField> rho_g_;
    std::unique_ptr<RealField> v_tmp_;
};

}  // namespace dftcu
