#pragma once
#include <string>
#include <vector>

#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "solver/hamiltonian.cuh"
#include "solver/scf.cuh"  // For EnergyBreakdown
#include "solver/subspace_solver.cuh"

namespace dftcu {

/**
 * @brief Non-Self-Consistent Field (NSCF) solver for Kohn-Sham DFT
 *
 * NSCF 计算流程 (基于 QE non_scf.f90):
 * ─────────────────────────────────────────────────────────────────────
 *
 * 1. 前置条件 (potinit 阶段，调用 solve() 之前完成):
 *    - 输入密度 ρ (可以来自 SCF, 原子密度叠加, 或任意来源)
 *    - 从 ρ 计算势能:
 *      • V_ps: 局域赝势 (从 UPF 文件)
 *      • V_H: Hartree 势 (从 ρ 计算)
 *      • V_xc: 交换关联势 (从 ρ 计算)
 *      • vrs = V_ps + V_H + V_xc (总局域势, 固定不变)
 *    - 初始化非局域投影仪 β_lm(G)
 *
 * 2. NSCF 对角化 (solve() 执行):
 *    - 使用固定的 vrs 进行 Davidson 迭代
 *    - 求解本征值问题: H|ψ> = ε|ψ>
 *    - 计算占据数 (weights)
 *    - 计算能量分解
 *
 * 3. 与 SCF 的区别:
 *    - SCF: 每次迭代重新计算 V_H, V_xc (密度自洽)
 *    - NSCF: V_H, V_xc 只计算一次 (密度固定)
 *    - 哈密顿量形式完全相同: H = T + V_ps + V_H + V_xc + V_NL
 *
 * ─────────────────────────────────────────────────────────────────────
 *
 * 使用示例:
 *
 * @code
 * // Step 1: 准备输入密度和哈密顿量 (potinit 阶段)
 * RealField rho_scf(grid, 1);
 * // ... 从 SCF 读取或计算初始密度
 *
 * // Step 2: 创建 DensityFunctionalPotential 并计算势能
 * auto dfp = std::make_shared<DensityFunctionalPotential>(grid);
 * dfp->add_functional(std::make_shared<HartreeFunctional>(grid));
 * dfp->add_functional(std::make_shared<LDA_PZ>(grid));
 * dfp->add_functional(std::make_shared<LocalPseudo>(grid, atoms));
 *
 * // Step 3: 创建 Hamiltonian 并更新势能 (只调用一次!)
 * Hamiltonian ham(grid);
 * ham.set_density_functional_potential(dfp);
 * ham.update_potentials_inplace(rho_scf);  // vrs = V_ps + V_H[ρ] + V_xc[ρ]
 * ham.set_nonlocal(nl_pseudo);     // 设置非局域赝势
 *
 * // Step 4: 运行 NSCF 计算
 * NonSCFSolver nscf(grid);
 * EnergyBreakdown result = nscf.solve(ham, psi, nelec, atoms, ecutrho, &rho_scf);
 *
 * // Step 5: 输出结果
 * std::cout << "Eigenvalues: ";
 * for (double e : result.eigenvalues) std::cout << e << " ";
 * std::cout << "\nTotal energy: " << result.etot << " Ha\n";
 * @endcode
 *
 * ─────────────────────────────────────────────────────────────────────
 */
class NonSCFSolver {
  public:
    NonSCFSolver(Grid& grid);
    ~NonSCFSolver() = default;

    /**
     * @brief Run NSCF calculation (equivalent to QE's non_scf + c_bands_nscf)
     *
     * @param ham Hamiltonian with potentials already updated by ham.update_potentials_inplace(rho)
     *            CRITICAL: vrs must be pre-computed and fixed before calling this function!
     * @param psi Initial wavefunctions (random or atomic orbitals), updated to converged
     * eigenstates
     * @param nelec Total number of electrons (for occupation and alpha energy)
     * @param atoms Atomic positions (needed for Ewald energy calculation)
     * @param ecutrho Dense grid energy cutoff in Rydberg (e.g., 400.0 Ry = 200.0 Ha)
     * @param rho_scf Input density used to construct vrs (optional, for energy calculation)
     *                - If provided: use SCF density for energy (standard DFT: LDA, GGA)
     *                - If nullptr: compute density from NSCF wavefunctions (hybrid functionals)
     * @param rho_core Core charge density (optional, for XC energy calculation)
     * @param alpha_energy Alpha term correction (optional, usually computed internally)
     *
     * @return EnergyBreakdown structure containing:
     *         - eigenvalues:本征值 (Ha)
     *         - etot: 总能量 (Ha)
     *         - eband: 能带能量 Σ f_i * ε_i (Ha)
     *         - deband: 双重计数修正 -∫ ρ * V_eff dr (Ha)
     *         - ehart: Hartree 能量 (Ha)
     *         - etxc: XC 能量 (Ha)
     *         - eewld: Ewald 能量 (Ha)
     *         - alpha: G=0 修正项 = Nelec * v_of_0 (Ha)
     *
     * @note 整个 NSCF 过程中 vrs 保持不变 (与 SCF 的关键区别)
     * @note 本函数只执行对角化和能量计算, 不更新密度或势能
     * @note 如果启用了诊断模式，会在计算过程中导出中间变量
     */
    EnergyBreakdown solve(Hamiltonian& ham, Wavefunction& psi, double nelec,
                          std::shared_ptr<Atoms> atoms, double ecutrho,
                          const RealField* rho_scf = nullptr, const RealField* rho_core = nullptr,
                          double alpha_energy = 0.0);

  private:
    Grid& grid_;
    SubspaceSolver subspace_solver_;

    /**
     * @brief Compute occupation numbers for insulator (equivalent to QE weights.f90)
     *
     * For insulators:
     *   - Occupied bands: f_i = 2.0 (spin degeneracy)
     *   - Empty bands: f_i = 0.0
     *   - Fermi level: ef = ε_HOMO (energy of highest occupied band)
     *
     * @param nbands Number of bands
     * @param nelec Total number of electrons
     * @param eigenvalues Eigenvalues sorted in ascending order
     * @param occupations Output: occupation numbers (size = nbands)
     * @param ef Output: Fermi energy (Ha)
     */
    void compute_weights_insulator(int nbands, double nelec, const std::vector<double>& eigenvalues,
                                   std::vector<double>& occupations, double& ef);
};

}  // namespace dftcu
