#include "functional/hartree.cuh"
#include "functional/xc/lda_pz.cuh"
#include "model/density_factory.cuh"
#include "model/wavefunction_factory.cuh"
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
                           const NSCFWorkflowConfig& config)
    : grid_(grid),
      atoms_(atoms),
      config_(config),
      ham_(grid),
      rho_(grid, 1),
      psi_(grid, 1, grid.ecutwfc()) {  // ← 临时初始化为 1 band，稍后会重新赋值
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
    // Step 4: 初始化密度（原子电荷叠加）
    // ════════════════════════════════════════════════════════════════════════
    initialize_density(pseudo_data);

    // ════════════════════════════════════════════════════════════════════════
    // Step 5: potinit - 从密度计算势能（NSCF 只调用一次！）
    // ════════════════════════════════════════════════════════════════════════
    potinit();

    // ════════════════════════════════════════════════════════════════════════
    // Step 6: 初始化波函数（使用原子波函数叠加）
    // ════════════════════════════════════════════════════════════════════════
    initialize_wavefunction(pseudo_data);
}

NSCFWorkflow::NSCFWorkflow(Grid& grid, std::shared_ptr<Atoms> atoms, const Hamiltonian& ham,
                           const Wavefunction& psi, const std::vector<double>& rho_data,
                           const NSCFWorkflowConfig& config)
    : grid_(grid),
      atoms_(atoms),
      config_(config),
      ham_(grid),
      rho_(grid, 1),
      psi_(grid, psi.num_bands(), grid.ecutwfc()) {  // ✅ 使用输入 psi 的 band 数量
    // ════════════════════════════════════════════════════════════════════════
    // Step 1: 验证配置
    // ════════════════════════════════════════════════════════════════════════
    config_.validate(grid_);

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: 复制已组装好的哈密顿量和波函数
    // ════════════════════════════════════════════════════════════════════════
    // 注意：psi 可能包含 n_starting_wfc 个原子波函数（例如 8 个）
    //       而 config.nbands 是最终需要的本征态数量（例如 4 个）
    //       子空间对角化会在 NonSCFSolver::solve 中完成
    ham_.copy_from(ham);
    psi_.copy_from(psi);

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: 加载密度并执行 potinit
    // ════════════════════════════════════════════════════════════════════════
    if (rho_data.size() != grid_.nnr()) {
        throw ConfigurationError("rho_data 尺寸 (" + std::to_string(rho_data.size()) +
                                 ") 与 grid.nnr() (" + std::to_string(grid_.nnr()) + ") 不匹配");
    }
    rho_.copy_from_host(rho_data.data());
    potinit();
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

        // 验证赝势数据（基本字段）
        // 注意：暂时不强制要求 beta 函数完整性，因为 Python UPF 解析器可能未完全实现
        if (pseudo.header().element.empty()) {
            throw FileIOError("赝势数据无效 (atom type " + std::to_string(itype) +
                              "): element 为空");
        }
        if (pseudo.header().z_valence <= 0.0) {
            throw FileIOError("赝势数据无效 (atom type " + std::to_string(itype) +
                              "): z_valence <= 0");
        }
        if (pseudo.header().mesh_size <= 0) {
            throw FileIOError("赝势数据无效 (atom type " + std::to_string(itype) +
                              "): mesh_size <= 0");
        }
        if (pseudo.mesh().r.empty() || pseudo.mesh().rab.empty()) {
            throw FileIOError("赝势数据无效 (atom type " + std::to_string(itype) +
                              "): mesh.r 或 mesh.rab 为空");
        }
        if (pseudo.local().vloc_r.empty()) {
            throw FileIOError("赝势数据无效 (atom type " + std::to_string(itype) +
                              "): vloc_r 为空");
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

void NSCFWorkflow::initialize_density(const std::vector<PseudopotentialData>& pseudo_data) {
    // ════════════════════════════════════════════════════════════════════════
    // 使用 DensityFactory 从原子电荷叠加构建初始密度
    // ════════════════════════════════════════════════════════════════════════
    DensityFactory factory(grid_, atoms_);

    // 添加所有原子类型的原子密度
    for (size_t itype = 0; itype < pseudo_data.size(); ++itype) {
        const auto& pseudo = pseudo_data[itype];
        const auto& mesh = pseudo.mesh();
        const auto& rho_at = pseudo.atomic_density().rho_at;

        if (!rho_at.empty()) {
            factory.set_atomic_rho_r(itype, mesh.r, rho_at, mesh.rab);
        }
    }

    // 构建密度场
    factory.build_density(rho_);
}

void NSCFWorkflow::potinit() {
    // ════════════════════════════════════════════════════════════════════════
    // 计算势能：vrs = V_ps + V_H[ρ] + V_xc[ρ]
    // ════════════════════════════════════════════════════════════════════════
    // 注意：NSCF 中此函数只调用一次，之后 vrs 保持不变
    ham_.update_potentials(rho_);
}

void NSCFWorkflow::initialize_wavefunction(const std::vector<PseudopotentialData>& pseudo_data) {
    // ════════════════════════════════════════════════════════════════════════
    // 使用 WavefunctionFactory 从原子波函数构建初始波函数
    // ════════════════════════════════════════════════════════════════════════
    WavefunctionFactory factory(grid_, atoms_);

    // 添加所有原子轨道
    for (size_t itype = 0; itype < pseudo_data.size(); ++itype) {
        const auto& pseudo = pseudo_data[itype];
        const auto& mesh = pseudo.mesh();
        const auto& wfcs = pseudo.atomic_wfc().wavefunctions;

        for (const auto& wfc : wfcs) {
            factory.add_atomic_orbital(itype, wfc.l, mesh.r, wfc.chi, mesh.rab, mesh.msh);
        }
    }

    // 构建原子波函数（n_starting_wfc 个非正交波函数）
    auto psi_atomic = factory.build(false);  // randomize_phase = false

    // ════════════════════════════════════════════════════════════════════════
    // 重新构造 psi_ 为正确的大小并复制数据
    // ════════════════════════════════════════════════════════════════════════
    // 注意：psi_atomic 有 n_starting_wfc 个 band（例如 8 个）
    //       子空间对角化会在 NonSCFSolver::solve 中将其降维到 config_.nbands 个本征态

    // 销毁旧的 psi_ 并重新构造
    psi_.~Wavefunction();
    new (&psi_) Wavefunction(grid_, psi_atomic->num_bands(), grid_.ecutwfc());

    // 复制数据
    psi_.copy_from(*psi_atomic);
}

}  // namespace dftcu
