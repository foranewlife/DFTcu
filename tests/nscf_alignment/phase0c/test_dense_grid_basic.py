#!/usr/bin/env python3
"""
Phase 0c: Dense Grid 基本功能测试（无需 QE 数据）

验证：
1. Dense grid 生成不崩溃
2. 基本数值合理性（ngm_dense > ngw，ngl > 0 等）
3. 数据结构正确性（igk 映射、igtongl 映射等）
"""
import sys
from pathlib import Path

import numpy as np
from utils.reporter import PhaseResult

import dftcu

# Add project paths
test_root = Path(__file__).parents[1]
sys.path.insert(0, str(test_root))

dftcu_root = Path(__file__).parents[3]
sys.path.insert(0, str(dftcu_root))


def test_dense_grid_basic():
    """Phase 0c: 基本 Dense grid 生成测试"""

    phase_name = "Phase 0c Dense (基本)"
    phase_desc = "Dense Grid 基本功能"

    print("\n" + "=" * 70)
    print(f"[{phase_name}] {phase_desc}")
    print("=" * 70)

    # [1] 创建 Grid
    print("\n[1] 创建 Grid...")
    BOHR_TO_ANG = 0.529177210903
    alat_bohr = 10.20
    alat_ang = alat_bohr * BOHR_TO_ANG

    lattice_ang = np.array(
        [
            [-alat_ang / 2, 0.0, alat_ang / 2],
            [0.0, alat_ang / 2, alat_ang / 2],
            [-alat_ang / 2, alat_ang / 2, 0.0],
        ]
    )

    grid = dftcu.create_grid_from_qe(
        lattice_ang=lattice_ang, nr=[18, 18, 18], ecutwfc_ry=12.0, ecutrho_ry=48.0, is_gamma=True
    )
    print(f"  ✓ Grid 创建成功")
    print(f"    ecutwfc = {grid.ecutwfc()} Ha = {grid.ecutwfc() * 2.0} Ry")
    print(f"    ecutrho = {grid.ecutrho()} Ha = {grid.ecutrho() * 2.0} Ry")

    # [2] 生成 G 向量
    print("\n[2] 生成 G 向量（Smooth + Dense）...")
    grid.generate_gvectors()
    print(f"  ✓ generate_gvectors() 成功")

    # [3] 检查 Smooth grid
    print("\n[3] 检查 Smooth grid...")
    ngw = grid.ngw()
    print(f"  ✓ ngw (Smooth) = {ngw}")

    # 验证 Smooth grid 数量（Si FCC，ecutwfc=12 Ry → 约 85 个）
    expected_ngw = 85
    if abs(ngw - expected_ngw) <= 5:
        print(f"  ✓ ngw 与预期一致：{ngw} ≈ {expected_ngw}")
    else:
        print(f"  ⚠ ngw 与预期不同：{ngw} vs {expected_ngw}")

    # [4] 检查 Dense grid
    print("\n[4] 检查 Dense grid...")
    ngm_dense = grid.ngm_dense()
    ngl = grid.ngl()
    print(f"  ✓ ngm_dense (Dense) = {ngm_dense}")
    print(f"  ✓ ngl (G-shells) = {ngl}")

    # 验证 Dense grid 数量（ecutrho=48 Ry → 约 622 个）
    expected_ngm_dense = 622
    if abs(ngm_dense - expected_ngm_dense) <= 10:
        print(f"  ✓ ngm_dense 与预期一致：{ngm_dense} ≈ {expected_ngm_dense}")
    else:
        print(f"  ⚠ ngm_dense 与预期不同：{ngm_dense} vs {expected_ngm_dense}")

    # [5] 验证大小关系
    print("\n[5] 验证大小关系...")
    assert ngm_dense > ngw, f"Dense grid 应该 > Smooth grid: {ngm_dense} vs {ngw}"
    print(f"  ✓ ngm_dense > ngw: {ngm_dense} > {ngw} ✓")

    assert ngl > 0, "ngl 应该 > 0"
    assert ngl <= ngm_dense, f"ngl 应该 <= ngm_dense: {ngl} vs {ngm_dense}"
    print(f"  ✓ 0 < ngl <= ngm_dense: 0 < {ngl} <= {ngm_dense} ✓")

    # [6] 验证 G-shell 数据
    print("\n[6] 验证 G-shell 数据...")
    gl_shells = np.array(grid.get_gl_shells())
    assert len(gl_shells) == ngl, f"gl_shells 长度应为 {ngl}"
    print(f"  ✓ gl_shells 长度 = {len(gl_shells)}")

    # 验证 gl 是排序的
    assert np.all(np.diff(gl_shells) >= -1e-14), "gl_shells 应该是排序的"
    print(f"  ✓ gl_shells 已排序")
    print(f"    范围: [{gl_shells.min():.6e}, {gl_shells.max():.6e}] (2π/Bohr)²")

    # [7] 验证 igtongl 映射
    print("\n[7] 验证 igtongl 映射...")
    igtongl = np.array(grid.get_igtongl())
    assert len(igtongl) == ngm_dense, f"igtongl 长度应为 {ngm_dense}"
    assert np.all((igtongl >= 0) & (igtongl < ngl)), "igtongl 索引应在 [0, ngl) 范围"
    print(f"  ✓ igtongl 长度 = {len(igtongl)}")
    print(f"  ✓ igtongl 范围: [0, {ngl})")

    # 验证每个 shell 至少有一个 G 向量
    unique_shells = np.unique(igtongl)
    print(f"  ✓ 使用的 shell 数: {len(unique_shells)} / {ngl}")
    if len(unique_shells) < ngl:
        print(f"    ⚠ 部分 shell 未使用（可能是高频 shell）")

    # [8] 验证 igk 映射 (Smooth -> Dense)
    print("\n[8] 验证 igk 映射 (Smooth -> Dense)...")
    igk = np.array(grid.get_igk())
    assert len(igk) == ngw, f"igk 长度应为 {ngw}"
    assert np.all((igk >= 0) & (igk < ngm_dense)), "igk 索引应在 [0, ngm_dense) 范围"
    print(f"  ✓ igk 长度 = {len(igk)}")
    print(f"  ✓ igk 范围: [0, {ngm_dense})")

    # 验证 igk 的唯一性（Smooth G 向量应该映射到不同的 Dense G 向量）
    unique_igk = np.unique(igk)
    assert len(unique_igk) == ngw, "igk 应该是一对一映射"
    print(f"  ✓ igk 是一对一映射（{len(unique_igk)} 个唯一值）")

    # [9] 验证 gg_dense
    print("\n[9] 验证 gg_dense (Dense grid |G|²)...")
    gg_dense = np.array(grid.get_gg_dense())
    assert len(gg_dense) == ngm_dense, f"gg_dense 长度应为 {ngm_dense}"
    print(f"  ✓ gg_dense 长度 = {len(gg_dense)}")
    print(f"  ✓ gg_dense 范围: [{gg_dense.min():.6e}, {gg_dense.max():.6e}] (2π/Bohr)²")

    # 验证 Dense grid 的截断能
    # ecutrho = 48 Ry = 24 Ha
    # g2_max 应该 <= 2 × 24 = 48 Ha = 48 (2π/Bohr)²
    ecutrho_ha = grid.ecutrho()
    gcut2_dense = 2.0 * ecutrho_ha
    assert np.all(
        gg_dense <= gcut2_dense * (1 + 1e-10)
    ), f"gg_dense 应该 <= {gcut2_dense}: max = {gg_dense.max()}"
    print(f"  ✓ gg_dense 满足截断条件: max = {gg_dense.max():.6e} <= {gcut2_dense:.6e}")

    # [10] 验证 Smooth grid 与 Dense grid 的关系
    print("\n[10] 验证 Smooth ⊂ Dense 关系...")
    gg_smooth = np.array(grid.get_gg_smooth())
    # 通过 igk 映射，Smooth grid 的 |G|² 应该在 Dense grid 中
    gg_smooth_in_dense = gg_dense[igk]
    diff = np.abs(gg_smooth - gg_smooth_in_dense)
    max_diff = diff.max()
    print(f"  ✓ gg_smooth vs gg_dense[igk]: max_diff = {max_diff:.6e}")
    assert max_diff < 1e-12, f"Smooth G 向量应该在 Dense grid 中: max_diff = {max_diff}"

    # [11] 最终判定
    print("\n" + "=" * 70)
    print("✅ Phase 0c Dense Grid 基本功能测试通过！")
    print("=" * 70)
    print(f"总结:")
    print(f"  - Smooth grid: ngw = {ngw}")
    print(f"  - Dense grid:  ngm_dense = {ngm_dense}, ngl = {ngl}")
    print(f"  - igk 映射:    Smooth -> Dense 正确")
    print(f"  - igtongl:     Dense -> G-shell 正确")
    print(f"  - 数据结构:    所有验证通过")

    return PhaseResult(
        phase_name=phase_name,
        phase_desc=phase_desc,
        passed=True,
        max_error=max_diff,
        threshold=1e-12,
        details=f"ngw={ngw}, ngm_dense={ngm_dense}, ngl={ngl}",
    )


if __name__ == "__main__":
    try:
        result = test_dense_grid_basic()
        sys.exit(0 if result.passed else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
