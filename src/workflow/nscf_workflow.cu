#include "functional/hartree.cuh"
#include "functional/xc/lda_pz.cuh"
#include "utilities/error.cuh"
#include "workflow/nscf_workflow.cuh"

namespace dftcu {

// ════════════════════════════════════════════════════════════════════════
// NSCFWorkflowConfig 实现
// ════════════════════════════════════════════════════════════════════════

void NSCFWorkflowConfig::validate(const Grid& grid) const {
    // 1. 检查 nbands
    if (nbands <= 0) {
        throw ConfigurationError("nbands 必须 > 0");
    }

    if (nbands > grid.ngw()) {
        throw ConfigurationError("nbands (" + std::to_string(nbands) + ") 不能超过 ngw (" +
                                 std::to_string(grid.ngw()) + ")");
    }

    // 2. 检查 nelec
    if (nelec <= 0.0) {
        throw ConfigurationError("nelec 必须 > 0");
    }

    // 3. 检查 nelec 与 nbands 的关系（考虑自旋简并度 2）
    int max_electrons = nbands * 2;
    if (nelec > max_electrons) {
        throw ConfigurationError("nelec (" + std::to_string(nelec) + ") 超过最大容纳电子数 (" +
                                 std::to_string(max_electrons) + " = nbands * 2)");
    }

    // 4. 检查输出目录（如果启用诊断模式）
    if (enable_diagnostics && output_dir.empty()) {
        throw ConfigurationError("启用诊断模式时，output_dir 不能为空");
    }
}

// ════════════════════════════════════════════════════════════════════════
// NSCFWorkflow 实现
// ════════════════════════════════════════════════════════════════════════

NSCFWorkflow::NSCFWorkflow(Grid& grid, std::shared_ptr<Atoms> atoms,
                           const std::vector<PseudopotentialData>& pseudo_data,
                           const std::vector<double>& rho_data, const NSCFWorkflowConfig& config)
    : grid_(grid),
      atoms_(atoms),
      config_(config),
      ham_(grid),
      rho_(grid, 1),
      psi_(grid, config.nbands, grid.ecutwfc()) {
    // ════════════════════════════════════════════════════════════════════════
    // Step 1: 验证配置
    // ════════════════════════════════════════════════════════════════════════
    config_.validate(grid_);

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: 初始化赝势
    // ════════════════════════════════════════════════════════════════════════
    initialize_pseudopotentials(pseudo_data);

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: 初始化哈密顿量
    // ════════════════════════════════════════════════════════════════════════
    initialize_hamiltonian();

    // ════════════════════════════════════════════════════════════════════════
    // Step 4: potinit - 从密度计算势能（NSCF 只调用一次！）
    // ════════════════════════════════════════════════════════════════════════
    potinit(rho_data);

    // ════════════════════════════════════════════════════════════════════════
    // Step 5: 初始化波函数
    // ════════════════════════════════════════════════════════════════════════
    initialize_wavefunction();
}

NSCFWorkflow::NSCFWorkflow(Grid& grid, std::shared_ptr<Atoms> atoms, const Hamiltonian& ham,
                           const Wavefunction& psi, const std::vector<double>& rho_data,
                           const NSCFWorkflowConfig& config)
    : grid_(grid),
      atoms_(atoms),
      config_(config),
      ham_(grid),
      rho_(grid, 1),
      psi_(grid, config.nbands, grid.ecutwfc()) {
    // ════════════════════════════════════════════════════════════════════════
    // Step 1: 验证配置
    // ════════════════════════════════════════════════════════════════════════
    config_.validate(grid_);

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: 复制已组装好的哈密顿量和波函数
    // ════════════════════════════════════════════════════════════════════════
    ham_.copy_from(ham);
    psi_.copy_from(psi);

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: potinit - 从密度计算势能（确保 Hamiltonian 同步）
    // ════════════════════════════════════════════════════════════════════════
    potinit(rho_data);
}

EnergyBreakdown NSCFWorkflow::execute() {
    // ════════════════════════════════════════════════════════════════════════
    // Step 1: 创建 NonSCFSolver
    // ════════════════════════════════════════════════════════════════════════
    NonSCFSolver solver(grid_);

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: 配置诊断模式（如果启用）
    // ════════════════════════════════════════════════════════════════════════
    if (config_.enable_diagnostics) {
        NonSCFDiagnostics diag;
        diag.enable_all();
        diag.output_dir = config_.output_dir;
        solver.enable_diagnostics(diag);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: 执行 NSCF 计算
    // ════════════════════════════════════════════════════════════════════════
    // 注意：ham_ 的势能已经在构造函数中通过 potinit() 计算好了
    EnergyBreakdown result =
        solver.solve(ham_, psi_, config_.nelec, atoms_, grid_.ecutrho() * 2.0,  // Ha → Ry
                     &rho_);

    return result;
}

// ════════════════════════════════════════════════════════════════════════
// 私有辅助方法
// ════════════════════════════════════════════════════════════════════════

void NSCFWorkflow::initialize_pseudopotentials(
    const std::vector<PseudopotentialData>& pseudo_data) {
    if (pseudo_data.empty()) {
        throw ConfigurationError("pseudo_data 不能为空");
    }

    // 为每个原子类型创建赝势对象
    for (size_t itype = 0; itype < pseudo_data.size(); ++itype) {
        const auto& pseudo = pseudo_data[itype];

        // 验证赝势数据
        if (!pseudo.is_valid()) {
            throw FileIOError("赝势数据无效 (atom type " + std::to_string(itype) + ")");
        }

        // 创建局域赝势（使用工厂函数）
        auto local_ps = create_local_pseudo(grid_, atoms_, pseudo, itype);
        local_pseudos_.push_back(local_ps);

        // 创建非局域赝势（使用工厂函数）
        auto nonlocal_ps = create_nonlocal_pseudo(grid_, atoms_, pseudo, itype);
        nonlocal_pseudos_.push_back(nonlocal_ps);
    }
}

void NSCFWorkflow::initialize_hamiltonian() {
    // ════════════════════════════════════════════════════════════════════════
    // 创建 DensityFunctionalPotential
    // ════════════════════════════════════════════════════════════════════════
    dfp_ = std::make_shared<DensityFunctionalPotential>(grid_);

    // 添加 Hartree 泛函
    auto hartree = std::make_shared<Hartree>();
    dfp_->add_functional(hartree);

    // 添加 XC 泛函（LDA-PZ）
    auto lda = std::make_shared<LDA_PZ>();
    dfp_->add_functional(lda);

    // 添加局域赝势
    for (auto& local_ps : local_pseudos_) {
        dfp_->add_functional(local_ps);
    }

    // 设置 DFP 到哈密顿量
    ham_.set_density_functional_potential(dfp_);

    // ════════════════════════════════════════════════════════════════════════
    // 设置非局域赝势
    // ════════════════════════════════════════════════════════════════════════
    // 注意：目前只支持单一原子类型
    if (!nonlocal_pseudos_.empty()) {
        ham_.set_nonlocal(nonlocal_pseudos_[0]);
    }
}

void NSCFWorkflow::potinit(const std::vector<double>& rho_data) {
    // ════════════════════════════════════════════════════════════════════════
    // 验证密度数据尺寸
    // ════════════════════════════════════════════════════════════════════════
    if (rho_data.size() != grid_.nnr()) {
        throw ConfigurationError("rho_data 尺寸 (" + std::to_string(rho_data.size()) +
                                 ") 与 grid.nnr() (" + std::to_string(grid_.nnr()) + ") 不匹配");
    }

    // ════════════════════════════════════════════════════════════════════════
    // 加载密度到 GPU
    // ════════════════════════════════════════════════════════════════════════
    rho_.copy_from_host(rho_data.data());

    // ════════════════════════════════════════════════════════════════════════
    // 计算势能：vrs = V_ps + V_H[ρ] + V_xc[ρ]
    // ════════════════════════════════════════════════════════════════════════
    // 注意：NSCF 中此函数只调用一次，之后 vrs 保持不变
    ham_.update_potentials(rho_);
}

void NSCFWorkflow::initialize_wavefunction() {
    // ════════════════════════════════════════════════════════════════════════
    // 随机初始化波函数
    // ════════════════════════════════════════════════════════════════════════
    psi_.randomize();

    // ════════════════════════════════════════════════════════════════════════
    // 正交归一化
    // ════════════════════════════════════════════════════════════════════════
    psi_.orthonormalize();
}

}  // namespace dftcu
