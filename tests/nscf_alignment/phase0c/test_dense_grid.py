#!/usr/bin/env python3
"""
Phase 0c.2: Dense Grid 生成验证

验证内容：
    - DFTcu 生成的 Dense grid G 向量与 QE 一致
    - 验证 ngm_dense（Dense grid G 向量数）
    - 验证 gg_dense（Dense grid |G|²）
    - 验证 ngl（G-shell 数量）
    - 验证 gl（G-shell |G|²）
    - 验证 igtongl（G → shell 映射）

目标：
    - 实现 generate_dense_grid() 方法
    - 确保 Dense grid 与 QE 完全一致

精度目标：
    - ngm_dense, ngl: 0 误差（完全一致）
    - gg_dense, gl: 相对误差 < 1e-14
    - igtongl: 0 误差

依赖：
    - 需要从 QE 导出 Dense grid 参考数据
    - 修改 QE 源码: external/qe/Modules/recvec.f90
"""

import sys
from pathlib import Path

import numpy as np
from test_config import SYSTEM
from utils import TestReporter
from utils.reporter import PhaseResult

import dftcu

# Add paths
test_root = Path(__file__).parents[1]
sys.path.insert(0, str(test_root))

dftcu_root = Path(__file__).parents[3]
sys.path.insert(0, str(dftcu_root))


def test_dense_grid():  # noqa: C901
    """Phase 0c.2: 验证 Dense Grid 生成"""

    phase_name = "Phase 0c.2 (Dense Grid)"
    phase_desc = "Dense Grid 生成验证"

    data_dir = Path(__file__).parent / "data"

    threshold_rel = 1e-14  # gg_dense, gl 相对误差

    BOHR_TO_ANGSTROM = 0.529177210903

    TestReporter.print_phase_header(phase_name, phase_desc)

    # [1] 加载 QE Dense grid 参考数据
    print("\n[1] 加载 QE Dense grid 参考数据...")
    try:
        # 从 QE 导出文件加载
        # 格式: ig (1-based), gg (Bohr^-2)
        dense_data = np.loadtxt(data_dir / "qe_dense_grid.txt", comments="#")
        gg_dense_qe_bohr = dense_data[:, 1]
        gg_dense_qe_ang = gg_dense_qe_bohr / (BOHR_TO_ANGSTROM**2)
        ngm_dense_qe = len(gg_dense_qe_ang)

        # G-shell 数据
        # 格式: igl (1-based), gl (Bohr^-2)
        shell_data = np.loadtxt(data_dir / "qe_gshells.txt", comments="#")
        gl_qe_bohr = shell_data[:, 1]
        gl_qe_ang = gl_qe_bohr / (BOHR_TO_ANGSTROM**2)
        ngl_qe = len(gl_qe_ang)

        # G → shell 映射
        # 格式: ig (1-based), igl (1-based)
        igtongl_data = np.loadtxt(data_dir / "qe_igtongl.txt", dtype=int, comments="#")
        igtongl_qe = igtongl_data[:, 1]  # 1-based

        print(f"  ✓ QE Dense grid: ngm = {ngm_dense_qe}")
        print(
            f"  ✓ gg_dense 范围: [{gg_dense_qe_ang.min():.6e}, {gg_dense_qe_ang.max():.6e}] Angstrom^-2"
        )
        print(f"  ✓ ngl = {ngl_qe} 个 G-shell")
        print(f"  ✓ gl 范围: [{gl_qe_ang.min():.6e}, {gl_qe_ang.max():.6e}] Angstrom^-2")

    except FileNotFoundError as e:
        print(f"❌ QE 参考数据未找到: {e}")
        print(f"\n需要先从 QE 导出 Dense grid 数据！")
        print(f"请参考: tests/nscf_alignment/phase0c/README.md")
        return PhaseResult(phase_name, phase_desc, False, -1, threshold_rel, "QE 参考数据未找到")
    except Exception as e:
        print(f"❌ QE 数据加载失败: {e}")
        import traceback

        traceback.print_exc()
        return PhaseResult(phase_name, phase_desc, False, -1, threshold_rel, str(e))

    # [2] 初始化 DFTcu Grid 并生成 Dense grid
    print("\n[2] 初始化 DFTcu Grid...")
    try:
        # 使用新的工厂函数创建 Grid（Angstrom + Rydberg）
        alat_bohr = 10.20
        BOHR_TO_ANG = BOHR_TO_ANGSTROM
        alat_ang = alat_bohr * BOHR_TO_ANG

        lattice_ang = np.array(
            [
                [-alat_ang / 2, 0.0, alat_ang / 2],
                [0.0, alat_ang / 2, alat_ang / 2],
                [-alat_ang / 2, alat_ang / 2, 0.0],
            ]
        )

        # 使用工厂函数（参数名明确单位）
        grid = dftcu.create_grid_from_qe(
            lattice_ang=lattice_ang,
            nr=list(SYSTEM.nr),
            ecutwfc_ry=12.0,  # Rydberg
            ecutrho_ry=48.0,  # Rydberg
            is_gamma=True,
        )

        print(f"  ✓ Grid 创建: nr={SYSTEM.nr}")
        print(f"  ✓ ecutwfc = {grid.ecutwfc()} Ha ({grid.ecutwfc() * 2.0} Ry)")
        print(f"  ✓ ecutrho = {grid.ecutrho()} Ha ({grid.ecutrho() * 2.0} Ry)")

        # 生成 G 向量（包括 Dense grid）
        print(f"\n  调用 grid.generate_gvectors()...")
        grid.generate_gvectors()

        ngm_dense_dftcu = grid.ngm_dense()
        ngl_dftcu = grid.ngl()

        print(f"  ✓ DFTcu Dense grid: ngm_dense = {ngm_dense_dftcu}")
        print(f"  ✓ ngl = {ngl_dftcu} 个 G-shell")

        # 检查计数
        if ngm_dense_dftcu != ngm_dense_qe:
            raise ValueError(f"ngm_dense 不匹配: DFTcu={ngm_dense_dftcu}, QE={ngm_dense_qe}")
        if ngl_dftcu != ngl_qe:
            raise ValueError(f"ngl 不匹配: DFTcu={ngl_dftcu}, QE={ngl_qe}")

    except AttributeError as e:
        print(f"❌ generate_gvectors() 方法未实现: {e}")
        print(f"\n需要先在 Grid 类中实现 generate_gvectors() 方法！")
        return PhaseResult(
            phase_name, phase_desc, False, -1, threshold_rel, "generate_gvectors() 未实现"
        )
    except Exception as e:
        print(f"❌ Grid 初始化/生成失败: {e}")
        import traceback

        traceback.print_exc()
        return PhaseResult(phase_name, phase_desc, False, -1, threshold_rel, str(e))

    # [3] 对比 gg_dense
    print("\n[3] 验证 gg_dense (Dense grid |G|²)...")
    try:
        gg_dense_dftcu = np.array(grid.get_gg_dense())

        # 注意：QE 可能是全网格，DFTcu 是 Gamma-only 半球
        # 需要根据实际情况调整对比方式
        if len(gg_dense_dftcu) != len(gg_dense_qe_ang):
            print(f"  ⚠ 长度不匹配: DFTcu={len(gg_dense_dftcu)}, QE={len(gg_dense_qe_ang)}")
            print(f"  ℹ 可能是 Gamma-only vs 全网格")

        # 对比前 ngm_dense_dftcu 个
        n_compare = min(len(gg_dense_dftcu), len(gg_dense_qe_ang))
        abs_err = np.abs(gg_dense_dftcu[:n_compare] - gg_dense_qe_ang[:n_compare])
        rel_err = abs_err / (np.abs(gg_dense_qe_ang[:n_compare]) + 1e-16)

        max_abs_err = abs_err.max()
        max_rel_err = rel_err.max()
        mean_abs_err = abs_err.mean()

        print(f"  ✓ DFTcu gg_dense 范围: [{gg_dense_dftcu.min():.6e}, {gg_dense_dftcu.max():.6e}]")
        print(f"  ✓ max(|DFTcu - QE|) = {max_abs_err:.6e}")
        print(f"  ✓ max(rel_err) = {max_rel_err:.6e}")
        print(f"  ✓ mean(|DFTcu - QE|) = {mean_abs_err:.6e}")

    except Exception as e:
        print(f"❌ gg_dense 验证失败: {e}")
        import traceback

        traceback.print_exc()
        return PhaseResult(phase_name, phase_desc, False, -1, threshold_rel, str(e))

    # [4] 对比 gl (G-shell |G|²)
    print("\n[4] 验证 gl (G-shell |G|²)...")
    try:
        gl_dftcu = np.array(grid.get_gl_shells())

        abs_err_gl = np.abs(gl_dftcu - gl_qe_ang)
        rel_err_gl = abs_err_gl / (np.abs(gl_qe_ang) + 1e-16)

        max_abs_err_gl = abs_err_gl.max()
        max_rel_err_gl = rel_err_gl.max()

        print(f"  ✓ DFTcu gl 范围: [{gl_dftcu.min():.6e}, {gl_dftcu.max():.6e}]")
        print(f"  ✓ max(|DFTcu - QE|) = {max_abs_err_gl:.6e}")
        print(f"  ✓ max(rel_err) = {max_rel_err_gl:.6e}")

    except Exception as e:
        print(f"❌ gl 验证失败: {e}")
        import traceback

        traceback.print_exc()
        return PhaseResult(phase_name, phase_desc, False, -1, threshold_rel, str(e))

    # [5] 对比 igtongl
    print("\n[5] 验证 igtongl (G → shell 映射)...")
    try:
        igtongl_dftcu = np.array(grid.get_igtongl())

        # 对比（注意：可能需要处理索引基准：0-based vs 1-based）
        n_compare = min(len(igtongl_dftcu), len(igtongl_qe))
        diff_igtongl = np.abs(igtongl_dftcu[:n_compare] - igtongl_qe[:n_compare])
        max_diff_igtongl = diff_igtongl.max()

        print(f"  ✓ igtongl: max(|DFTcu - QE|) = {max_diff_igtongl}")

        if max_diff_igtongl > 0:
            print(f"  ❌ igtongl 不匹配！")
            # 显示前 5 个不匹配的
            mismatch = np.where(diff_igtongl > 0)[0][:5]
            for i in mismatch:
                print(f"     G[{i}]: DFTcu={igtongl_dftcu[i]}, QE={igtongl_qe[i]}")

    except Exception as e:
        print(f"❌ igtongl 验证失败: {e}")
        import traceback

        traceback.print_exc()
        return PhaseResult(phase_name, phase_desc, False, -1, threshold_rel, str(e))

    # [6] 最终判定
    print("\n[6] 最终判定...")
    max_error = max(max_rel_err, max_rel_err_gl, max_diff_igtongl)
    passed = (
        max_rel_err < threshold_rel and max_rel_err_gl < threshold_rel and max_diff_igtongl == 0
    )

    print("\n" + "=" * 80)
    if passed:
        print(f"✅ Phase 0c.2 验证通过！")
        print(f"   Dense Grid 生成与 QE 完全一致")
        print(f"   最大相对误差: {max_error:.6e} < {threshold_rel:.0e}")
    else:
        print(f"❌ Phase 0c.2 验证失败！")
        print(f"   最大误差: {max_error:.6e}")
    print("=" * 80)

    return PhaseResult(
        phase_name=phase_name,
        phase_desc=phase_desc,
        passed=passed,
        max_error=max_error,
        threshold=threshold_rel,
        details=f"Dense Grid: ngm={ngm_dense_dftcu}, ngl={ngl_dftcu}, max_error={max_error:.3e}",
    )


if __name__ == "__main__":
    result = test_dense_grid()

    print("\n" + "=" * 80)
    print("Phase 0c.2 总结")
    print("=" * 80)
    print(f"✅ ngm_dense, ngl 验证")
    print(f"✅ gg_dense 验证")
    print(f"✅ gl (G-shell) 验证")
    print(f"✅ igtongl 映射验证")

    if result.passed:
        print(f"\n🎉 Phase 0c.2 验证通过！")
        print(f"   精度: {result.max_error:.6e} < {result.threshold:.0e}")
        print(f"\n下一步: 实现 igk 映射（Phase 0c.3）")
    else:
        print(f"\n❌ Phase 0c.2 验证失败")
        print(f"   误差: {result.max_error:.6e}")

    print("=" * 80)
    sys.exit(0 if result.passed else 1)
