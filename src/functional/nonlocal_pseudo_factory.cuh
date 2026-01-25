#pragma once
#include <memory>

#include "functional/nonlocal_pseudo.cuh"
#include "functional/pseudopotential_data.cuh"
#include "model/atoms.cuh"
#include "model/grid.cuh"

namespace dftcu {

/**
 * @brief 从单原子赝势模型创建 3D 空间非局域赝势
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
 * 3. 3D 空间赝势分布 (NonLocalPseudo)
 *    - 3D 数据：β_lm(G) 在 Smooth grid 上
 *    - 依赖 Grid 和 Atoms 位置
 *    - 用于实际计算
 * ─────────────────────────────────────────────────────────────────────
 *
 * 工作流程：
 * 1. 从 PseudopotentialData 提取 β_l(r) 径向函数和 D_ij 矩阵
 * 2. 对 Smooth grid 的每个 |G| 进行插值，得到 β_l(G)
 * 3. 展开为球谐函数：β_lm(G) = β_l(|G|) * Y_lm(θ, φ)
 * 4. 创建 NonLocalPseudo 对象
 *
 * @param grid Grid 对象（使用 Smooth grid）
 * @param atoms Atoms 对象（原子位置）
 * @param pseudo_data 单原子赝势模型（1D 径向函数）
 * @param atom_type 原子类型索引
 * @return NonLocalPseudo 对象（3D 空间分布）
 *
 * @note 此函数执行从 1D → 3D 的转换：
 *       - 输入：β_l(r) [无量纲]
 *       - 输出：β_lm(G) [无量纲]
 *       - 网格：Smooth grid (ecutwfc)
 *
 * @example
 * ```cpp
 * // Python 层解析 UPF
 * auto upf_parser = UPFParser();
 * auto pseudo_data = upf_parser.parse("Si.pz-rrkj.UPF");
 *
 * // C++ 层创建 3D 空间赝势
 * auto nonlocal_ps = create_nonlocal_pseudo(grid, atoms, *pseudo_data, 0);
 * ```
 */
std::shared_ptr<NonLocalPseudo> create_nonlocal_pseudo(Grid& grid, std::shared_ptr<Atoms> atoms,
                                                       const PseudopotentialData& pseudo_data,
                                                       int atom_type = 0);

}  // namespace dftcu
