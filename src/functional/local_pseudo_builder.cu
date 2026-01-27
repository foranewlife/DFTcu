#include "fft/fft_solver.cuh"
#include "functional/local_pseudo_builder.cuh"
#include "functional/local_pseudo_operator.cuh"
#include "utilities/constants.cuh"

namespace dftcu {

std::shared_ptr<LocalPseudoOperator> build_local_pseudo(Grid& grid, std::shared_ptr<Atoms> atoms,
                                                        const PseudopotentialData& pseudo_data,
                                                        int atom_type) {
    // ════════════════════════════════════════════════════════════════════════
    // 创建 LocalPseudoOperator 对象
    // ════════════════════════════════════════════════════════════════════════
    auto local_pseudo = std::make_shared<LocalPseudoOperator>(grid, atoms);

    // ════════════════════════════════════════════════════════════════════════
    // 从单原子赝势模型提取 1D 径向数据
    // ════════════════════════════════════════════════════════════════════════
    const RadialMesh& mesh = pseudo_data.mesh();
    const LocalPotential& local_pot = pseudo_data.local();
    const PseudopotentialHeader& header = pseudo_data.header();

    // ════════════════════════════════════════════════════════════════════════
    // 计算网格截断（遵循 QE 约定）
    // ════════════════════════════════════════════════════════════════════════
    // QE 使用 rcut=10 Bohr 截断积分网格，避免大 r 处的数值噪声
    // 参考：QE read_pseudo.f90:179-186
    const double rcut = 10.0;  // Bohr, 与 QE 一致
    int msh = mesh.r.size();   // 默认：使用完整网格
    for (size_t ir = 0; ir < mesh.r.size(); ++ir) {
        if (mesh.r[ir] > rcut) {
            msh = ir;  // 第一个 r > rcut 的点
            break;
        }
    }
    // 强制 msh 为奇数（Simpson 积分要求，QE 约定）
    msh = 2 * ((msh + 1) / 2) - 1;

    // ════════════════════════════════════════════════════════════════════════
    // 初始化 V_loc 插值表（1D → 3D 转换）
    // ════════════════════════════════════════════════════════════════════════
    // 输入：V_loc(r) [Rydberg]，1D 径向函数
    // 输出：V_loc(G) 插值表 [Hartree]，3D 空间分布
    //
    // 注意：init_tab_vloc 内部会进行单位转换（Rydberg → Hartree）
    local_pseudo->init_tab_vloc(atom_type, mesh.r, local_pot.vloc_r, mesh.rab, header.z_valence,
                                grid.volume(), msh);

    return local_pseudo;
}

}  // namespace dftcu
