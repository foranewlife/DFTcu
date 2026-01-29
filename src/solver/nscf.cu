#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "functional/wavefunction_builder.cuh"
#include "model/pseudopotential_data.cuh"
#include "solver/nscf.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"

#include <sys/stat.h>
#include <sys/types.h>

namespace dftcu {

NonSCFSolver::NonSCFSolver(Grid& grid) : grid_(grid), subspace_solver_(grid) {}

EnergyBreakdown NonSCFSolver::solve(Hamiltonian& ham, Wavefunction& psi, double nelec,
                                    std::shared_ptr<Atoms> atoms, double ecutrho,
                                    const std::vector<PseudopotentialData>* pseudo_data,
                                    const RealField* rho_scf, const RealField* rho_core,
                                    double alpha_energy) {
    std::cout << "     TRACE: entering NonSCFSolver::solve" << std::endl;
    std::cout << "     Band Structure Calculation" << std::endl;

    // ════════════════════════════════════════════════════════════════════════════════
    // NSCF 完整流程 (基于 QE non_scf.f90)
    // ════════════════════════════════════════════════════════════════════════════════
    //
    // QE NSCF 流程:
    //   non_scf() [PW/src/non_scf.f90]
    //     ├─> c_bands_nscf()  [对角化所有 k-points]
    //     ├─> poolrecover()   [收集本征值]
    //     ├─> weights()       [计算权重/占据数]
    //     └─> print_ks_energies_nonscf()
    //
    // DFTcu 实现:
    //   - 假设单个 k-point (Gamma-only)
    //   - 使用 SubspaceSolver 直接对角化
    //   - 计算权重和能量
    //
    // ════════════════════════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 0: 波函数初始化 (对齐 QE: c_bands_nscf -> init_wfc)
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   c_bands_nscf() [PW/src/c_bands.f90]
    //     └─> init_wfc(ik) [PW/src/wfcinit.f90]
    //           └─> atomic_wfc() [PW/src/atomic_wfc.f90]
    //
    // DFTcu:
    //   如果 psi.num_bands() == 0，从 pseudo_data 构建原子波函数
    //   否则使用已提供的波函数
    //
    // ────────────────────────────────────────────────────────────────────────────────

    if (psi.num_bands() == 0) {
        // 需要初始化波函数
        if (pseudo_data == nullptr || pseudo_data->empty()) {
            throw std::invalid_argument(
                "NonSCFSolver::solve: psi is empty but pseudo_data is not provided. "
                "Cannot initialize wavefunctions.");
        }

        std::cout << "     TRACE: initializing wavefunctions from atomic orbitals (QE: init_wfc)"
                  << std::endl;

        // 使用 WavefunctionBuilder 从原子轨道构建初始波函数
        WavefunctionBuilder wfc_builder(grid_, atoms);

        // 添加所有原子轨道
        for (size_t itype = 0; itype < pseudo_data->size(); ++itype) {
            const auto& pseudo = (*pseudo_data)[itype];
            const auto& mesh = pseudo.mesh();
            const auto& wfcs = pseudo.atomic_wfc().wavefunctions;

            for (const auto& wfc : wfcs) {
                wfc_builder.add_atomic_orbital(itype, wfc.l, mesh.r, wfc.chi, mesh.rab);
            }
        }

        // 构建原子波函数（n_starting_wfc 个非正交波函数）
        auto psi_atomic = wfc_builder.build(false);  // randomize_phase = false

        // 重新构造 psi 为正确的大小并复制数据
        int n_starting_wfc = psi_atomic->num_bands();
        std::cout << "     TRACE: built " << n_starting_wfc << " atomic wavefunctions" << std::endl;

        // 销毁旧的 psi 并重新构造
        psi.~Wavefunction();
        new (&psi) Wavefunction(grid_, n_starting_wfc, grid_.ecutwfc());

        // 复制数据
        psi.copy_from(*psi_atomic);

        std::cout << "     TRACE: wavefunction initialization complete" << std::endl;
    } else {
        std::cout << "     TRACE: using provided wavefunctions (" << psi.num_bands() << " bands)"
                  << std::endl;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 1: c_bands_nscf - 对角化
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   c_bands_nscf() [PW/src/c_bands.f90:1171]
    //     └─> diag_bands() → regterg() → Davidson 迭代
    //
    // DFTcu:
    //   SubspaceSolver::solve_direct() 封装了完整的 Davidson 迭代
    //   (或单次子空间对角化，取决于实现)
    //
    // CRITICAL:
    //   - Hamiltonian 已经在调用 solve() 之前构建好 (potinit 阶段)
    //   - vrs = V_ps + V_H[ρ_SCF] + V_xc[ρ_SCF] 已固定
    //   - 整个 NSCF 过程中 vrs 不变
    //
    // ────────────────────────────────────────────────────────────────────────────────

    std::cout << "     TRACE: calling subspace_solver_.solve_direct (Davidson iteration)"
              << std::endl;
    std::vector<double> eigenvalues_all = subspace_solver_.solve_direct(ham, psi);

    // ✅ FIX (2026-01-26): 如果 psi 包含 n_starting_wfc > nbands 个原子波函数，
    // 只保留前 nbands 个本征值（对应占据态）
    int n_starting_wfc = psi.num_bands();
    int nbands_out = (int)(nelec / 2.0 + 0.5);  // 假设自旋简并，nelec/2 个占据态

    std::vector<double> eigenvalues;
    if (n_starting_wfc > nbands_out) {
        std::cout << "     TRACE: Subspace diagonalization: " << n_starting_wfc
                  << " atomic wfcs -> " << nbands_out << " eigenstates" << std::endl;
        eigenvalues.assign(eigenvalues_all.begin(), eigenvalues_all.begin() + nbands_out);

        // TODO: 需要用本征矢量旋转波函数，将 psi 从 n_starting_wfc bands 降维到 nbands_out bands
        // 当前暂时使用所有 n_starting_wfc 个本征值进行后续计算
        std::cout << "     WARNING: Wavefunction rotation not implemented yet. "
                  << "Using all " << n_starting_wfc << " bands for now." << std::endl;
        eigenvalues = eigenvalues_all;  // 临时：使用所有本征值
    } else {
        eigenvalues = eigenvalues_all;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 2: weights - 计算占据数和费米能级
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   weights() or weights_only() [PW/src/weights.f90]
    //
    // 绝缘体 (insulator):
    //   - 占据能带: occupation = 2.0 (自旋简并)
    //   - 空带: occupation = 0.0
    //   - 费米能级 ef = HOMO 能量
    //
    // ────────────────────────────────────────────────────────────────────────────────

    int nbands = psi.num_bands();
    std::vector<double> occupations(nbands, 0.0);
    double ef = 0.0;

    std::cout << "     TRACE: calling compute_weights_insulator" << std::endl;
    compute_weights_insulator(nbands, nelec, eigenvalues, occupations, ef);

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 3: 选择用于能量计算的密度
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // QE non_scf.f90:91-94:
    //   IF (lxdm .OR. ANY(...)) THEN
    //     ! 混合泛函: 从 NSCF 波函数重算密度
    //     CALL sum_band()
    //   ELSE
    //     ! 标准 DFT (LDA/GGA): 使用 SCF 密度
    //     ! (不需要重算)
    //   END IF
    //
    // DFTcu 实现:
    //   - 如果提供了 rho_scf (标准DFT): 使用 SCF 密度
    //   - 否则 (混合泛函): 从 NSCF 波函数计算密度
    //
    // ────────────────────────────────────────────────────────────────────────────────

    RealField rho_for_energy(grid_, 1);

    if (rho_scf != nullptr) {
        // 标准 DFT 路径: LDA, GGA, etc.
        std::cout << "     TRACE: using SCF density for energy calculation" << std::endl;
        rho_for_energy.copy_from(*rho_scf);
    } else {
        // 混合泛函路径: 从 NSCF 波函数重算密度
        std::cout
            << "     TRACE: computing density from NSCF wavefunctions (hybrid functional mode)"
            << std::endl;
        psi.compute_density(occupations, rho_for_energy);
        grid_.synchronize();
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 4: 能量计算和输出
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   print_ks_energies_nonscf() [PW/src/output.f90]
    //     输出本征值、费米能级、能量分解
    //
    // DFTcu:
    //   使用 SCFSolver 的 compute_energy_breakdown() 计算能量
    //
    // 能量公式 (QE electrons.f90:1060-1080):
    //   E_tot = E_band + deband + E_H + E_XC + E_Ewald + alpha
    //
    //   其中:
    //     E_band  = Σ_i f_i * ε_i  (能带能量)
    //     deband  = -∫ ρ * V_eff dr (双重计数修正)
    //     E_H     = Hartree 能量
    //     E_XC    = 交换关联能量
    //     E_Ewald = Ewald 能量 (离子-离子库仑能)
    //     alpha   = G=0 修正项 = Nelec * v_of_0
    //
    // ────────────────────────────────────────────────────────────────────────────────

    SCFSolver::Options options;
    options.verbose = false;
    SCFSolver scf_helper(grid_, options);

    // Alpha 能量修正 (QE style): Nelec * v_of_0
    // v_of_0 = G=0 处的总局域势 (V_ps + V_H + V_xc)
    // 注意: SubspaceSolver::solve_direct() 已经在本征值中包含了 v_of_0
    double v0 = ham.get_v_of_0();
    double total_alpha_energy = nelec * v0;

    scf_helper.set_alpha_energy(total_alpha_energy);
    scf_helper.set_atoms(atoms);
    scf_helper.set_ecutrho(ecutrho);

    std::cout << "     End of band structure calculation" << std::endl;

    // 计算完整的能量分解
    EnergyBreakdown breakdown = scf_helper.compute_energy_breakdown(eigenvalues, occupations, ham,
                                                                    psi, rho_for_energy, rho_core);

    // 存储本征值方便访问
    breakdown.eigenvalues = eigenvalues;

    // ────────────────────────────────────────────────────────────────────────────────
    // Diagnostic: 导出能量分解 (Phase 4 结果)
    // ────────────────────────────────────────────────────────────────────────────────

    return breakdown;
}

void NonSCFSolver::compute_weights_insulator(int nbands, double nelec,
                                             const std::vector<double>& eigenvalues,
                                             std::vector<double>& occupations, double& ef) {
    double degspin = 2.0;  // Assume nspin=1 and not noncollinear
    int n_filled = static_cast<int>(std::round(nelec / degspin));

    for (int i = 0; i < nbands; ++i) {
        if (i < n_filled) {
            occupations[i] = degspin;
        } else {
            occupations[i] = 0.0;
        }
    }

    if (n_filled > 0) {
        ef = eigenvalues[n_filled - 1];
    } else {
        ef = -1e10;
    }
}

}  // namespace dftcu
