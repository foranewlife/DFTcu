#pragma once
#include "functional/ewald.cuh"
#include "functional/hartree.cuh"
#include "functional/kedf/tf.cuh"
#include "functional/kedf/vw.cuh"
#include "functional/kedf/wt.cuh"
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

    void set_tf(ThomasFermi* tf) { tf_ = tf; }
    void set_vw(vonWeizsacker* vw) { vw_ = vw; }
    void set_wt(WangTeter* wt) { wt_ = wt; }
    void set_xc(LDA_PZ* xc) { xc_ = xc; }
    void set_hartree(Hartree* hartree) { hartree_ = hartree; }
    void set_pseudo(LocalPseudo* pseudo) { pseudo_ = pseudo; }
    void set_ewald(Ewald* ewald) { ewald_ = ewald; }

    /**
     * @brief Compute total energy and potential
     * @param rho Input density
     * @param v_tot Output total potential field
     * @return Total energy
     */
    double compute(const RealField& rho, RealField& v_tot);

  private:
    const Grid& grid_;
    ThomasFermi* tf_ = nullptr;
    vonWeizsacker* vw_ = nullptr;
    WangTeter* wt_ = nullptr;
    LDA_PZ* xc_ = nullptr;
    Hartree* hartree_ = nullptr;
    LocalPseudo* pseudo_ = nullptr;
    Ewald* ewald_ = nullptr;
};

}  // namespace dftcu
