#pragma once
#include <memory>

#include "functional/nonlocal_pseudo_operator.cuh"
#include "model/atoms.cuh"
#include "model/grid.cuh"
#include "model/pseudopotential_data.cuh"

namespace dftcu {

/**
 * @brief 从单原子赝势模型创建 3D 空间非局域赝势
 * [BUILDER] Heavy construction: 1D radial → 3D spatial distribution with Bessel transform
 *
 * 三层赝势模型：
 * ─────────────────────────────────────────────────────────────────────
 * 1. UPF 文件格式 (*.UPF)
 *    - Python 层：UPFParser.parse()
 *    - 输出：PseudopotentialData
 *
 * 2. 单原子赝势模型 (PseudopotentialData)
 *    - 1D 径向函数：β_l(r), D_ij
 *    - 球对称，与空间网格无关
 *    - 可被多个原子共享
 *
 * 3. 3D 空间赝势分布 (NonLocalPseudoOperator)
 *    - 3D 数据：β_lm(G) 在 Smooth grid 上
 *    - 依赖 Grid 和 Atoms 位置
 *    - 用于实际计算
 * ─────────────────────────────────────────────────────────────────────
 *
 * 工作流程：
 * 1. 从 PseudopotentialData 提取 β(r) 和 D_ij
 * 2. 在 G 空间计算 β(G) = 4π/Ω ∫ β(r) j_l(Gr) r² dr
 * 3. 创建 NonLocalPseudoOperator 对象
 *
 * @param grid Grid 对象（使用 Smooth grid 的 G-shell）
 * @param atoms Atoms 对象（原子位置）
 * @param pseudo_data 单原子赝势模型（1D 径向函数）
 * @param atom_type 原子类型索引
 * @return NonLocalPseudoOperator 对象
 *
 * @example
 * ```cpp
 * // Python 层解析 UPF
 * auto upf_parser = UPFParser();
 * auto pseudo_data = upf_parser.parse("Si.pz-rrkj.UPF");
 *
 * // C++ 层创建 3D 空间非局域赝势
 * auto nonlocal_ps = build_nonlocal_pseudo(grid, atoms, *pseudo_data, 0);
 * ```
 */
std::shared_ptr<NonLocalPseudoOperator> build_nonlocal_pseudo(
    Grid& grid, std::shared_ptr<Atoms> atoms, const PseudopotentialData& pseudo_data,
    int atom_type = 0);

}  // namespace dftcu
