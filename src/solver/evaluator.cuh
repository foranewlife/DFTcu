#pragma once
#include "functional/hartree.cuh"
#include "functional/kedf/kedf_base.cuh"
#include "functional/pseudo.cuh"
#include "functional/xc/lda_pz.cuh"

namespace dftcu {

/**
 * @brief Aggregates different energy and potential terms.
 */
class Evaluator {
  public:
    Evaluator(const Grid& grid);
    ~Evaluator() = default;

    void set_kedf(KEDF_Base* kedf) { kedf_ = kedf; }
    void set_xc(LDA_PZ* xc) { xc_ = xc; }
    void set_hartree(Hartree* hartree) { hartree_ = hartree; }
    void set_pseudo(LocalPseudo* pseudo) { pseudo_ = pseudo; }

    /**
     * @brief Compute total energy and potential
     * @param rho Input density
     * @param v_tot Output total potential field
     * @return Total energy
     */
    double compute(const RealField& rho, RealField& v_tot);

  private:
    const Grid& grid_;
    KEDF_Base* kedf_ = nullptr;
    LDA_PZ* xc_ = nullptr;
    Hartree* hartree_ = nullptr;
    LocalPseudo* pseudo_ = nullptr;
};

}  // namespace dftcu
