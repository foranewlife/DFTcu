#include "fft/fft_solver.cuh"
#include "functional/nonlocal_pseudo_builder.cuh"
#include "functional/nonlocal_pseudo_operator.cuh"
#include "utilities/constants.cuh"

namespace dftcu {

std::shared_ptr<NonLocalPseudoOperator> build_nonlocal_pseudo(
    Grid& grid, std::shared_ptr<Atoms> atoms, const PseudopotentialData& pseudo_data,
    int atom_type) {
    // ════════════════════════════════════════════════════════════════════════
    // 创建 NonLocalPseudoOperator 对象
    // ════════════════════════════════════════════════════════════════════════
    auto nl_pseudo = std::make_shared<NonLocalPseudoOperator>(grid);

    // ════════════════════════════════════════════════════════════════════════
    // 从单原子赝势模型提取 1D 径向数据
    // ════════════════════════════════════════════════════════════════════════
    const RadialMesh& mesh = pseudo_data.mesh();
    const NonlocalPotential& nl_pot = pseudo_data.nonlocal();

    // 提取 beta 函数（投影仪）
    std::vector<std::vector<double>> beta_r;
    std::vector<int> l_list;
    std::vector<int> kkbeta_list;

    for (const auto& beta : nl_pot.beta_functions) {
        beta_r.push_back(beta.beta_r);
        l_list.push_back(beta.angular_momentum);
        kkbeta_list.push_back(beta.cutoff_radius_index);
    }

    // ════════════════════════════════════════════════════════════════════════
    // 初始化 beta 插值表（1D → 3D 转换）
    // ════════════════════════════════════════════════════════════════════════
    // 输入：β_l(r) [无量纲]，1D 径向函数
    // 输出：β_l(G) 插值表 [无量纲]，3D 空间分布
    //
    // 关键：传递 Bohr³ 单位的体积（grid.volume_bohr()）
    nl_pseudo->init_tab_beta(atom_type, mesh.r, beta_r, mesh.rab, l_list, kkbeta_list,
                             grid.volume_bohr());

    // ════════════════════════════════════════════════════════════════════════
    // 初始化 D_ij 矩阵
    // ════════════════════════════════════════════════════════════════════════
    nl_pseudo->init_dij(atom_type, nl_pot.dij);

    // ════════════════════════════════════════════════════════════════════════
    // 更新投影仪（基于原子位置）
    // ════════════════════════════════════════════════════════════════════════
    // 注意：update_projectors_inplace 需要 const Atoms&
    nl_pseudo->update_projectors_inplace(*atoms);

    return nl_pseudo;
}

}  // namespace dftcu
