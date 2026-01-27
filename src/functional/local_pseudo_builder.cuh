#pragma once
#include <memory>

#include "functional/local_pseudo_operator.cuh"
#include "model/atoms.cuh"
#include "model/grid.cuh"
#include "model/pseudopotential_data.cuh"

namespace dftcu {

/**
 * @brief 从单原子赝势模型创建 3D 空间局域赝势
 *
 * 三层赝势模型：
 * ─────────────────────────────────────────────────────────────────────
 * 1. UPF 文件格式 (*.UPF)
 *    - Python 层：UPFParser.parse()
 *    - 输出：PseudopotentialData
 *
 * 2. 单原子赝势模型 (PseudopotentialData)
 *    - 1D 径向函数：V_loc(r), β_l(r), ρ_atom(r)
 *    - 球对称，与空间网格无关
 *    - 可被多个原子共享
 *
 * 3. 3D 空间赝势分布 (LocalPseudoOperator)
 *    - 3D 数据：V_loc(G) 在 Dense grid 上
 *    - 依赖 Grid 和 Atoms 位置
 *    - 用于实际计算
 * ─────────────────────────────────────────────────────────────────────
 *
 * 工作流程：
 * 1. 从 PseudopotentialData 提取 V_loc(r) 径向函数
 * 2. 对 Dense grid 的每个 |G| 进行插值，得到 V_loc(G)
 * 3. 计算 G=0 项（alpha 修正）
 * 4. 创建 LocalPseudoOperator 对象
 *
 * @param grid Grid 对象（使用 Dense grid 的 G-shell）
 * @param atoms Atoms 对象（原子位置）
 * @param pseudo_data 单原子赝势模型（1D 径向函数）
 * @param atom_type 原子类型索引
 * @return LocalPseudoOperator 对象（3D 空间分布）
 *
 * @note 此函数执行从 1D → 3D 的转换：
 *       - 输入：V_loc(r) [Rydberg]
 *       - 输出：V_loc(G) 插值表 [Hartree]
 *       - 单位转换：Rydberg → Hartree (× 0.5)
 *
 * @example
 * ```cpp
 * // Python 层解析 UPF
 * auto upf_parser = UPFParser();
 * auto pseudo_data = upf_parser.parse("Si.pz-rrkj.UPF");
 *
 * // C++ 层创建 3D 空间赝势
 * auto local_ps = build_local_pseudo(grid, atoms, *pseudo_data, 0);
 * ```
 */
std::shared_ptr<LocalPseudoOperator> build_local_pseudo(Grid& grid, std::shared_ptr<Atoms> atoms,
                                                        const PseudopotentialData& pseudo_data,
                                                        int atom_type = 0);

}  // namespace dftcu
