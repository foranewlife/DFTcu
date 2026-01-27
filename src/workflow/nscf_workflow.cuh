#pragma once
#include <memory>
#include <string>
#include <vector>

#include "functional/density_functional_potential.cuh"
#include "functional/local_pseudo_builder.cuh"
#include "functional/nonlocal_pseudo_builder.cuh"
#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "model/pseudopotential_data.cuh"
#include "model/wavefunction.cuh"
#include "solver/hamiltonian.cuh"
#include "solver/nscf.cuh"
#include "solver/scf.cuh"  // For EnergyBreakdown

namespace dftcu {

/**
 * @brief Configuration for NSCF workflow
 *
 * 封装所有 NSCF 计算所需的配置参数。
 * 在构造 NSCFWorkflow 时传入，避免在 execute() 时传递大量参数。
 */
struct NSCFWorkflowConfig {
    int nbands;               ///< 能带数量
    double nelec;             ///< 电子数
    bool enable_diagnostics;  ///< 启用诊断模式
    std::string output_dir;   ///< 输出目录（诊断模式）

    NSCFWorkflowConfig() : nbands(0), nelec(0.0), enable_diagnostics(false), output_dir(".") {}

    /**
     * @brief 验证配置的物理合理性
     * @param grid Grid 对象（用于检查 nbands <= ngw）
     * @throws std::invalid_argument 如果配置不合理
     */
    void validate(const Grid& grid) const;
};

/**
 * @brief NSCF Workflow - 封装完整的 NSCF 计算流程
 *
 * 设计理念：
 * ─────────────────────────────────────────────────────────────────────
 * 1. **构造时初始化**：所有配置和数据在构造时传入
 * 2. **execute() 无参数**：一键执行完整流程
 * 3. **错误处理集中**：所有物理验证在 C++ 端完成
 * 4. **生命周期管理**：内部管理所有对象的生命周期
 *
 * 使用示例：
 * ─────────────────────────────────────────────────────────────────────
 * @code
 * // 1. 准备配置
 * NSCFWorkflowConfig config;
 * config.nbands = 4;
 * config.nelec = 8.0;
 * config.enable_diagnostics = true;
 * config.output_dir = "nscf_output";
 *
 * // 2. 创建 Workflow（构造时完成所有初始化）
 * NSCFWorkflow workflow(
 *     grid,
 *     atoms,
 *     pseudo_data,
 *     rho_data,
 *     config
 * );
 *
 * // 3. 一键执行
 * EnergyBreakdown result = workflow.execute();
 *
 * // 4. 获取结果
 * std::cout << "Total energy: " << result.etot << " Ha\n";
 * @endcode
 *
 * 内部流程：
 * ─────────────────────────────────────────────────────────────────────
 * 构造阶段：
 *   1. 验证配置（nbands, nelec, ecutrho >= 4*ecutwfc）
 *   2. 创建 LocalPseudoOperator 和 NonlocalPseudo
 *   3. 创建 Hamiltonian 并设置 DensityFunctionalPotential
 *   4. 加载密度并调用 initialize_potentials（ham.update_potentials_inplace）
 *   5. 初始化波函数
 *
 * execute() 阶段：
 *   1. 创建 NonSCFSolver
 *   2. 配置诊断模式（如果启用）
 *   3. 调用 solver.solve()
 *   4. 返回 EnergyBreakdown
 */
class NSCFWorkflow {
  public:
    /**
     * @brief 构造 NSCF Workflow
     *
     * @param grid Grid 对象（引用，外部管理生命周期）
     * @param atoms Atoms 对象（共享指针）
     * @param pseudo_data 赝势数据（所有原子类型）
     * @param config NSCF 配置
     *
     * @throws ConfigurationError 如果配置不合理
     * @throws FileIOError 如果赝势数据无效
     *
     * @note 构造函数会执行以下操作：
     *       1. 验证配置
     *       2. 创建赝势对象
     *       3. 创建哈密顿量
     *       4. 初始化密度（原子电荷叠加）
     *       5. 执行 initialize_potentials（计算势能）
     *       6. 初始化波函数（原子波函数叠加）
     */
    NSCFWorkflow(Grid& grid, std::shared_ptr<Atoms> atoms,
                 const std::vector<PseudopotentialData>& pseudo_data,
                 const NSCFWorkflowConfig& config);

    /**
     * @brief 构造 NSCF Workflow (接收已组装好的哈密顿量和波函数)
     *
     * @param grid Grid 对象
     * @param atoms Atoms 对象
     * @param ham 已配置好的哈密顿量
     * @param psi 初始波函数
     * @param rho_data 输入密度数据
     * @param config NSCF 配置
     */
    NSCFWorkflow(Grid& grid, std::shared_ptr<Atoms> atoms, const Hamiltonian& ham,
                 const Wavefunction& psi, const std::vector<double>& rho_data,
                 const NSCFWorkflowConfig& config);

    /**
     * @brief 执行 NSCF 计算
     *
     * @return EnergyBreakdown 包含本征值、能量分解等结果
     *
     * @note 此函数无参数，所有配置在构造时已传入
     * @note 如果启用诊断模式，会在 output_dir 下生成调试文件
     */
    EnergyBreakdown execute();

    /**
     * @brief 获取收敛后的波函数
     * @return 波函数的常量引用
     * @note 只有在 execute() 调用后才有效
     */
    const Wavefunction& get_wavefunction() const { return psi_; }

    /**
     * @brief 获取哈密顿量
     * @return 哈密顿量的常量引用
     */
    const Hamiltonian& get_hamiltonian() const { return ham_; }

  private:
    // ════════════════════════════════════════════════════════════════════════
    // 引用和配置（外部管理）
    // ════════════════════════════════════════════════════════════════════════
    Grid& grid_;                    ///< Grid 引用（外部管理生命周期）
    std::shared_ptr<Atoms> atoms_;  ///< Atoms 共享指针
    NSCFWorkflowConfig config_;     ///< NSCF 配置

    // ════════════════════════════════════════════════════════════════════════
    // 内部管理的对象（生命周期由 Workflow 管理）
    // ════════════════════════════════════════════════════════════════════════
    std::vector<std::shared_ptr<LocalPseudoOperator>> local_pseudos_;        ///< 局域赝势
    std::vector<std::shared_ptr<NonLocalPseudoOperator>> nonlocal_pseudos_;  ///< 非局域赝势
    std::shared_ptr<DensityFunctionalPotential> dfp_;                        ///< 密度泛函势
    Hamiltonian ham_;                                                        ///< 哈密顿量
    RealField rho_;                                                          ///< 输入密度
    Wavefunction psi_;                                                       ///< 波函数

    // ════════════════════════════════════════════════════════════════════════
    // 私有辅助方法
    // ════════════════════════════════════════════════════════════════════════

    /**
     * @brief 初始化赝势（局域 + 非局域）
     * @param pseudo_data 赝势数据（所有原子类型）
     */
    void initialize_pseudopotentials(const std::vector<PseudopotentialData>& pseudo_data);

    /**
     * @brief 初始化哈密顿量（设置 DFP 和非局域赝势）
     */
    void initialize_hamiltonian();

    /**
     * @brief 初始化密度（原子电荷叠加）
     * @param pseudo_data 赝势数据（包含原子密度）
     */
    void initialize_density(const std::vector<PseudopotentialData>& pseudo_data);

    /**
     * @brief 初始化势能（从密度计算势能）
     * [SIDE_EFFECT] Modifies Hamiltonian's internal potential fields
     *
     * 在 NSCF 计算中，此方法只调用一次，之后势能保持不变。
     * 等价于 QE 的 potinit 阶段。
     */
    void initialize_potentials();

    /**
     * @brief 初始化波函数（使用原子波函数叠加）
     * @param pseudo_data 赝势数据（包含原子波函数）
     */
    void initialize_wavefunction(const std::vector<PseudopotentialData>& pseudo_data);
};

}  // namespace dftcu
