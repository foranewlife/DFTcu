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
 * @brief Diagnostic options for NonSCFSolver
 *
 * 诊断模式可以导出 NSCF 计算过程中的中间变量，用于与 QE 对比验证。
 * 所有导出数据格式与 QE 的 dftcu_debug_*.txt 文件兼容。
 */
struct NonSCFDiagnostics {
    bool enabled = false;          ///< 是否启用诊断模式
    std::string output_dir = ".";  ///< 输出目录

    // Phase 1: Davidson 对角化
    bool dump_psi_initial = false;  ///< 导出初始波函数 ψ(G)
    bool dump_hpsi = false;         ///< 导出 H|ψ> (完整哈密顿量)
    bool dump_h_subspace = false;   ///< 导出 H_sub 矩阵
    bool dump_s_subspace = false;   ///< 导出 S_sub 矩阵
    bool dump_eigenvalues = false;  ///< 导出本征值

    // Phase 2: 占据数
    bool dump_occupations = false;  ///< 导出占据数

    // Phase 3: 密度
    bool dump_density = false;  ///< 导出密度 ρ(r)

    // Phase 4: 能量
    bool dump_energy_breakdown = false;  ///< 导出能量分解

    // Phase 5: 势能
    bool dump_v_eff = false;  ///< 导出有效势能 V_eff(r)

    /**
     * @brief 启用所有诊断输出（用于完整验证）
     */
    void enable_all() {
        enabled = true;
        dump_psi_initial = true;
        dump_hpsi = true;
        dump_h_subspace = true;
        dump_s_subspace = true;
        dump_eigenvalues = true;
        dump_occupations = true;
        dump_density = true;
        dump_energy_breakdown = true;
        dump_v_eff = true;
    }

    /**
     * @brief 启用关键诊断输出（用于快速验证）
     */
    void enable_essential() {
        enabled = true;
        dump_psi_initial = true;
        dump_eigenvalues = true;
        dump_energy_breakdown = true;
    }
};

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
 * ham.update_potentials(rho_scf);  // vrs = V_ps + V_H[ρ] + V_xc[ρ]
 * ham.set_nonlocal(nl_pseudo);     // 设置非局域赝势
 *
 * // Step 4: 运行 NSCF 计算 (启用诊断模式)
 * NonSCFSolver nscf(grid);
 * NonSCFDiagnostics diag;
 * diag.enable_all();  // 导出所有中间变量
 * diag.output_dir = "nscf_diagnostics";
 * nscf.enable_diagnostics(diag);
 *
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
     * @brief Enable diagnostic mode for NSCF calculation
     *
     * 启用诊断模式后，solve() 会在关键点导出中间变量到文件，
     * 格式与 QE 的 dftcu_debug_*.txt 兼容，用于逐步验证。
     *
     * @param diagnostics Diagnostic options
     */
    void enable_diagnostics(const NonSCFDiagnostics& diagnostics);

    /**
     * @brief Run NSCF calculation (equivalent to QE's non_scf + c_bands_nscf)
     *
     * @param ham Hamiltonian with potentials already updated by ham.update_potentials(rho)
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
    NonSCFDiagnostics diagnostics_;  ///< Diagnostic options

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

    // ════════════════════════════════════════════════════════════════════════
    // Diagnostic helper functions (导出中间变量到文件)
    // ════════════════════════════════════════════════════════════════════════

    /**
     * @brief Dump wavefunction to file (QE format)
     * 格式: 每行一个 G-vector，实部和虚部，所有能带
     */
    void dump_wavefunction(const std::string& filename, const Wavefunction& psi);

    /**
     * @brief Dump eigenvalues to file (QE format)
     * 格式: 每行一个本征值 (Hartree)
     */
    void dump_eigenvalues_to_file(const std::string& filename, const std::vector<double>& evals);

    /**
     * @brief Dump real matrix to file (QE format)
     * 格式: 行优先，每行所有列
     */
    void dump_real_matrix(const std::string& filename, int n, const double* matrix);

    /**
     * @brief Dump occupations to file
     * 格式: 每行一个占据数
     */
    void dump_occupations_to_file(const std::string& filename, const std::vector<double>& occs);

    /**
     * @brief Dump energy breakdown to file
     * 格式: 键值对
     */
    void dump_energy(const std::string& filename, const EnergyBreakdown& energy);

    /**
     * @brief Dump effective potential to file (QE format)
     * 格式: 第一行是点数，后面每行一个值 (Hartree)
     */
    void dump_v_eff(const std::string& filename, const RealField& v_eff);
};

}  // namespace dftcu
