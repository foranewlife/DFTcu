#!/usr/bin/env python3
"""
DFTcu vs QE 结果对比分析

功能：
  1. 读取 DFTcu 诊断输出 (nscf_output/)
  2. 读取 QE 参考数据 (qe_run/)
  3. 对比本征值、波函数、能量分解
  4. 生成详细的对比报告

使用方法：
    python compare_qe.py

要求：
    - 已运行 run_nscf.py 生成 nscf_output/
    - QE 参考数据位于 qe_run/

⚠️ 重要提示 (2026-01-16 修复):
    DFTcu 内部使用 Fortran 列主序 (ix 最快变化) 存储数据，与 QE 一致。
    run_nscf.py 必须使用 reorder=False 加载密度数据，避免重新排序。
    详见：VLOC_DATA_LAYOUT_FIX.md
"""

import sys
from pathlib import Path

import numpy as np

# ════════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════════

DFTCU_OUTPUT_DIR = Path("nscf_output")
QE_DATA_DIR = Path("qe_run")

# QE 参考文件
QE_PSI_FILE = QE_DATA_DIR / "dftcu_debug_psi_iter0.txt"
QE_HPSI_FILE = QE_DATA_DIR / "dftcu_debug_fullhpsi_iter0.txt"
QE_TPSI_FILE = QE_DATA_DIR / "dftcu_debug_tpsi_iter0.txt"
QE_TVLOCPSI_FILE = QE_DATA_DIR / "dftcu_debug_tvlocpsi_iter0.txt"
QE_EIGENVALUES_FILE = QE_DATA_DIR / "si_nscf.out"
QE_G2KIN_FILE = QE_DATA_DIR / "dftcu_debug_g2kin.txt"

# QE V_loc 分量文件
QE_VPS_FILE = QE_DATA_DIR / "dftcu_debug_vps_r.txt"
QE_VH_FILE = QE_DATA_DIR / "dftcu_debug_v_hartree.txt"
QE_VXC_FILE = QE_DATA_DIR / "dftcu_debug_v_xc.txt"

# DFTcu 输出文件
DFTCU_PSI_FILE = DFTCU_OUTPUT_DIR / "dftcu_psi_initial.txt"
DFTCU_EIGENVALUES_FILE = DFTCU_OUTPUT_DIR / "dftcu_eigenvalues.txt"
DFTCU_OCCUPATIONS_FILE = DFTCU_OUTPUT_DIR / "dftcu_occupations.txt"
DFTCU_ENERGY_FILE = DFTCU_OUTPUT_DIR / "dftcu_energy_breakdown.txt"

# DFTcu Hamiltonian 分项文件
DFTCU_TPSI_FILE = DFTCU_OUTPUT_DIR / "dftcu_tpsi_iter0.txt"
DFTCU_TVLOCPSI_FILE = DFTCU_OUTPUT_DIR / "dftcu_tvlocpsi_iter0.txt"
DFTCU_VNLPSI_FILE = DFTCU_OUTPUT_DIR / "dftcu_vnlpsi_iter0.txt"
DFTCU_HPSI_FILE = DFTCU_OUTPUT_DIR / "dftcu_hpsi_iter0.txt"

# DFTcu V_loc 分量文件
DFTCU_VPS_FILE = DFTCU_OUTPUT_DIR / "dftcu_vps_r.txt"
DFTCU_VH_FILE = DFTCU_OUTPUT_DIR / "dftcu_vh_r.txt"
DFTCU_VXC_FILE = DFTCU_OUTPUT_DIR / "dftcu_vxc_r.txt"

# 单位转换
RY_TO_HA = 0.5
HA_TO_EV = 27.2114


# ════════════════════════════════════════════════════════════════════════════════
# 数据加载函数
# ════════════════════════════════════════════════════════════════════════════════


def load_qe_complex_gspace(filename, npw=85):
    """加载 QE 导出的 G 空间复数数组

    格式: ig Re(ψ) Im(ψ)  或  ig band Re(ψ) Im(ψ)
    注意: QE 使用 Fortran 1-based 索引
    """
    data = np.zeros(npw, dtype=np.complex128)
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()

            # 判断格式
            if len(parts) == 3:
                # 格式: ig Re Im
                ig = int(parts[0]) - 1
                re_val = float(parts[1])
                im_val = float(parts[2])
            elif len(parts) == 4:
                # 格式: ig band Re Im (只取 band=1)
                ig = int(parts[0]) - 1
                band = int(parts[1])
                if band != 1:
                    continue
                re_val = float(parts[2])
                im_val = float(parts[3])
            else:
                continue

            if ig < npw:
                data[ig] = complex(re_val, im_val)
    return data


def load_dftcu_complex_gspace(filename, npw=85):
    """加载 DFTcu 导出的 G 空间复数数组

    格式: ig Re(ψ) Im(ψ)
    注意: DFTcu 导出时使用 Fortran 1-based 索引（为了与 QE 对比）
    单位: Ha（DFTcu 内部单位）
    """
    data = np.zeros(npw, dtype=np.complex128)
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()

            if len(parts) >= 3:
                # 格式: ig Re Im
                ig = int(parts[0]) - 1  # Fortran 1-based -> 0-based
                re_val = float(parts[1])
                im_val = float(parts[2])

                if ig < npw:
                    data[ig] = complex(re_val, im_val)
    return data


def compare_complex_gspace(name, dftcu_arr, qe_arr_ry, units="Ha"):
    """对比 G 空间复数数组

    Args:
        name: 数组名称
        dftcu_arr: DFTcu 数据 (Ha)
        qe_arr_ry: QE 数据 (Ry)
        units: 单位
    """
    qe_arr = qe_arr_ry * RY_TO_HA  # Ry -> Ha

    diff = dftcu_arr - qe_arr
    rms = np.sqrt(np.mean(np.abs(diff) ** 2))
    max_diff = np.abs(diff).max()
    rel_rms = rms / (np.mean(np.abs(qe_arr)) + 1e-10)

    print(f"\n  {name}:")
    print(f"    |ψ| Mean (DFTcu): {np.abs(dftcu_arr).mean():.6e} {units}")
    print(f"    |ψ| Mean (QE):    {np.abs(qe_arr).mean():.6e} {units}")
    print(f"    RMS 误差:         {rms:.6e} {units}")
    print(f"    最大误差:         {max_diff:.6e} {units}")
    print(f"    相对 RMS:         {rel_rms:.6e}")

    # 前5个点对比
    print(f"    前5个点:")
    for i in range(min(5, len(dftcu_arr))):
        print(f"      [{i}] DFTcu: {dftcu_arr[i]:.6e}, QE: {qe_arr[i]:.6e}, Diff: {diff[i]:.6e}")

    threshold = 1e-10
    passed = rms < threshold
    status = "✅" if passed else "❌"
    print(f"    {status} {'通过' if passed else '失败'} (目标 RMS < {threshold:.0e})")

    return passed


def load_dftcu_eigenvalues(filename):
    """加载 DFTcu 本征值 (Ha)"""
    eigenvalues = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                eigenvalues.append(float(parts[1]))
    return np.array(eigenvalues)


def load_dftcu_energy_breakdown(filename):
    """加载 DFTcu 能量分解 (Ha)"""
    energy = {}
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = float(parts[1])
                energy[key] = value
    return energy


def load_qe_eigenvalues_from_output(filename):
    """从 QE 输出文件解析本征值 (eV -> Ha)

    查找类似如下的输出：
          k = 0.0000 0.0000 0.0000 (    85 PWs)   bands (ev):
            -4.9923   7.2642   7.2642   7.2642
    """
    eigenvalues = []
    with open(filename, "r") as f:
        found_bands = False
        for line in f:
            # 查找 "bands (ev):" 行
            if "bands (ev):" in line:
                found_bands = True
                continue

            # 读取本征值（eV）
            if found_bands:
                # 如果遇到 occupation numbers 或新的 k-point，停止
                if "occupation" in line or "k =" in line:
                    break

                # 跳过空行
                if line.strip() == "":
                    continue

                # 解析本征值
                parts = line.split()
                for part in parts:
                    try:
                        e_ev = float(part)
                        e_ha = e_ev / HA_TO_EV  # eV -> Ha
                        eigenvalues.append(e_ha)
                    except ValueError:
                        continue

    return np.array(eigenvalues)


def load_qe_g2kin(filename):
    """加载 QE g2kin 数据 (Ry -> Ha)"""
    g2kin = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                g2kin.append(float(parts[1]) * RY_TO_HA)  # Ry -> Ha
    return np.array(g2kin)


def load_wavefunction(filename, npw, nbands):
    """加载波函数数据

    DFTcu格式: ig  Re[ψ(:,1)]  Im[ψ(:,1)]  Re[ψ(:,2)]  Im[ψ(:,2)]  ...
    QE格式:    ig  band  Re[ψ]  Im[ψ]
    """
    psi = np.zeros((npw, nbands), dtype=np.complex128)

    with open(filename, "r") as f:
        lines = f.readlines()

    # 检测格式
    first_data_line = None
    for line in lines:
        if not line.startswith("#") and line.strip():
            first_data_line = line
            break

    if first_data_line is None:
        return psi

    parts = first_data_line.split()

    # QE 格式: ig band Re Im (4列)
    # DFTcu 格式: ig Re1 Im1 Re2 Im2 ... (1 + 2*nbands 列)
    if len(parts) == 4:
        # QE 格式
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            ig = int(parts[0]) - 1  # Fortran 1-based -> 0-based
            ib = int(parts[1]) - 1  # Fortran 1-based -> 0-based
            re = float(parts[2])
            im = float(parts[3])

            if ig < npw and ib < nbands:
                psi[ig, ib] = re + 1j * im
    else:
        # DFTcu 格式
        ig = 0
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 1 + 2 * nbands:
                continue

            # 跳过第一列（G-vector 索引）
            for ib in range(nbands):
                re = float(parts[1 + 2 * ib])
                im = float(parts[1 + 2 * ib + 1])
                psi[ig, ib] = re + 1j * im

            ig += 1
            if ig >= npw:
                break

    return psi


def load_qe_potential_r(filename, nnr=3375, nr=None, reorder=True):
    """加载 QE 导出的实空间势能 (Ry -> Ha)

    格式: ir value (Rydberg)
    注意: QE 使用 Fortran 1-based 索引 (ix 最快)
    """
    data_fortran = np.zeros(nnr)
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                ir = int(parts[0]) - 1  # Fortran 1-based -> 0-based
                val_ry = float(parts[1])
                if ir < nnr:
                    data_fortran[ir] = val_ry * RY_TO_HA  # Ry -> Ha

    if not reorder or nr is None:
        return data_fortran

    # 重新排列: Fortran (ix 最快) → C (iz 最快)
    nr1, nr2, nr3 = nr
    data_c = np.zeros(nnr)
    for iz in range(nr3):
        for iy in range(nr2):
            for ix in range(nr1):
                idx_fortran = ix + nr1 * (iy + nr2 * iz)
                idx_c = iz + nr3 * (iy + nr2 * ix)
                data_c[idx_c] = data_fortran[idx_fortran]
    return data_c


def load_dftcu_potential_r(filename, nnr=3375):
    """加载 DFTcu 导出的实空间势能 (已经是 Ha)

    格式: 每行一个值（无索引）
    """
    data = np.zeros(nnr)
    ir = 0
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            val_ha = float(line.strip())
            if ir < nnr:
                data[ir] = val_ha
                ir += 1
    return data


# ════════════════════════════════════════════════════════════════════════════════
# 对比函数
# ════════════════════════════════════════════════════════════════════════════════


def compare_eigenvalues(dftcu_evals, qe_evals):
    """对比本征值"""
    print("\n" + "=" * 70)
    print("  本征值对比")
    print("=" * 70)

    if len(qe_evals) == 0:
        print("\n  ⚠️  未找到 QE 参考本征值")
        print(f"     请检查 QE 输出文件: {QE_EIGENVALUES_FILE}")
        return None

    # 取最小长度
    n_compare = min(len(dftcu_evals), len(qe_evals))

    print(f"\n  对比能带数: {n_compare}")
    print(
        f"\n  {'Band':<6} {'DFTcu (Ha)':<18} {'QE (Ha)':<18} {'Diff (Ha)':<14} {'Diff (meV)':<12}"
    )
    print("  " + "-" * 66)

    max_diff_ha = 0.0
    for i in range(n_compare):
        e_dftcu = dftcu_evals[i]
        e_qe = qe_evals[i]
        diff_ha = abs(e_dftcu - e_qe)
        diff_mev = diff_ha * HA_TO_EV * 1000

        max_diff_ha = max(max_diff_ha, diff_ha)

        print(f"  {i+1:<6} {e_dftcu:16.10f}  {e_qe:16.10f}  {diff_ha:12.6e}  {diff_mev:10.4f}")

    print("\n  " + "=" * 66)
    print(f"  最大差异: {max_diff_ha:.6e} Ha ({max_diff_ha * HA_TO_EV * 1000:.4f} meV)")

    # 精度判断
    threshold_ha = 1e-8  # ~20 meV
    if max_diff_ha < threshold_ha:
        print(f"  ✅ 精度达标 (< {threshold_ha:.2e} Ha)")
        return True
    else:
        print(f"  ❌ 精度未达标 (目标 < {threshold_ha:.2e} Ha)")
        return False


def gamma_only_inner_product(psi_a, psi_b):
    """计算 Gamma-only 内积: ⟨a|b⟩ = a*(G=0) b(G=0) + 2 Σ_{G≠0} Re[a*(G) b(G)]

    Args:
        psi_a, psi_b: 复数数组，长度 npw

    Returns:
        内积（实数）
    """
    # G=0 项（只计数一次）
    inner = np.real(np.conj(psi_a[0]) * psi_b[0])

    # G≠0 项（乘以 2，Hermitian 对称）
    if len(psi_a) > 1:
        inner += 2.0 * np.sum(np.real(np.conj(psi_a[1:]) * psi_b[1:]))

    return inner


def compare_wavefunctions(dftcu_psi, qe_psi):
    """对比波函数"""
    print("\n" + "=" * 70)
    print("  波函数对比")
    print("=" * 70)

    npw_dftcu, nbands_dftcu = dftcu_psi.shape
    npw_qe, nbands_qe = qe_psi.shape

    print(f"\n  DFTcu: npw={npw_dftcu}, nbands={nbands_dftcu}")
    print(f"  QE:    npw={npw_qe}, nbands={nbands_qe}")

    if npw_dftcu != npw_qe:
        print(f"\n  ⚠️  警告: G-vector 数量不匹配 ({npw_dftcu} vs {npw_qe})")
        return None

    n_compare = min(nbands_dftcu, nbands_qe)

    print(f"\n  {'Band':<6} {'DFTcu Norm':<14} {'QE Norm':<14} {'Overlap':<12} {'Diff':<12}")
    print("  " + "-" * 60)

    for ib in range(n_compare):
        psi_dftcu = dftcu_psi[:, ib]
        psi_qe = qe_psi[:, ib]

        # 使用 Gamma-only 内积公式计算 norm
        norm_dftcu_sq = gamma_only_inner_product(psi_dftcu, psi_dftcu)
        norm_qe_sq = gamma_only_inner_product(psi_qe, psi_qe)

        norm_dftcu = np.sqrt(norm_dftcu_sq)
        norm_qe = np.sqrt(norm_qe_sq)

        # 归一化后计算重叠度（使用 Gamma-only 内积）
        overlap_inner = gamma_only_inner_product(psi_dftcu, psi_qe)
        overlap = abs(overlap_inner) / (norm_dftcu * norm_qe)

        # 计算差异（归一化后）
        psi_dftcu_norm = psi_dftcu / norm_dftcu
        psi_qe_norm = psi_qe / norm_qe
        diff_sq = gamma_only_inner_product(
            psi_dftcu_norm - psi_qe_norm, psi_dftcu_norm - psi_qe_norm
        )
        diff = np.sqrt(abs(diff_sq))

        print(f"  {ib+1:<6} {norm_dftcu:12.6f}  {norm_qe:12.6f}  {overlap:10.6f}  {diff:10.6f}")

    print("\n  说明:")
    print("    - Overlap = 1.0: 完全匹配")
    print("    - Overlap > 0.999: 高度一致")
    print("    - 使用 Gamma-only 内积公式: ⟨ψ|ψ⟩ = |ψ(G=0)|² + 2Σ_{G≠0}|ψ(G)|²")


def compare_energy_breakdown(dftcu_energy, qe_energy=None):
    """对比能量分解"""
    print("\n" + "=" * 70)
    print("  能量分解")
    print("=" * 70)

    print(f"\n  DFTcu 能量 (Ha):")
    for key in ["eband", "deband", "ehart", "etxc", "eewld", "alpha", "etot"]:
        if key in dftcu_energy:
            value_ha = dftcu_energy[key]
            value_ev = value_ha * HA_TO_EV
            print(f"    {key:<10} = {value_ha:16.10f} Ha  ({value_ev:12.6f} eV)")

    if qe_energy:
        print(f"\n  QE 能量 (Ha):")
        # TODO: 解析 QE 输出文件获取能量分解
        print("    (待实现)")


def compare_vloc_components(nnr=3375):
    """对比 V_loc 各分量：V_ps, V_H, V_xc

    Args:
        nnr: 实空间网格点数 (默认 15^3 = 3375)

    Returns:
        dict: 包含各分量误差统计的字典，如果文件缺失返回 None
    """
    print("\n" + "=" * 70)
    print("  V_loc 分量对比 (实空间)")
    print("=" * 70)

    # 检查文件是否存在
    required_files = [
        ("QE V_ps", QE_VPS_FILE),
        ("QE V_H", QE_VH_FILE),
        ("QE V_xc", QE_VXC_FILE),
        ("DFTcu V_ps", DFTCU_VPS_FILE),
        ("DFTcu V_H", DFTCU_VH_FILE),
        ("DFTcu V_xc", DFTCU_VXC_FILE),
    ]

    missing = []
    for name, fpath in required_files:
        if not fpath.exists():
            missing.append(f"{name} ({fpath})")

    if missing:
        print(f"\n  ⚠️  缺少文件:")
        for m in missing:
            print(f"     - {m}")
        print(f"\n  跳过 V_loc 分量对比")
        return None

    print(f"\n  加载数据 (nnr={nnr})...")

    # 网格维度
    nr = [15, 15, 15]

    # 加载 QE 数据
    v_ps_qe = load_qe_potential_r(QE_VPS_FILE, nnr, nr=nr, reorder=True)
    v_h_qe = load_qe_potential_r(QE_VH_FILE, nnr, nr=nr, reorder=True)
    v_xc_qe = load_qe_potential_r(QE_VXC_FILE, nnr, nr=nr, reorder=True)
    v_loc_qe = v_ps_qe + v_h_qe + v_xc_qe

    # 加载 DFTcu 数据
    v_ps_dftcu = load_dftcu_potential_r(DFTCU_VPS_FILE, nnr)
    v_h_dftcu = load_dftcu_potential_r(DFTCU_VH_FILE, nnr)
    v_xc_dftcu = load_dftcu_potential_r(DFTCU_VXC_FILE, nnr)
    v_loc_dftcu = v_ps_dftcu + v_h_dftcu + v_xc_dftcu

    print(f"  ✓ 已加载所有分量")

    # 对比函数
    def compare_component(name, dftcu_data, qe_data):
        diff = dftcu_data - qe_data
        rms = np.sqrt(np.mean(diff**2))
        max_diff = np.abs(diff).max()
        max_pos = np.argmax(np.abs(diff))
        rel_rms = rms / (np.abs(qe_data).mean() + 1e-16)

        return {
            "name": name,
            "rms": rms,
            "max_diff": max_diff,
            "max_pos": max_pos,
            "rel_rms": rel_rms,
            "qe_mean": qe_data.mean(),
            "dftcu_mean": dftcu_data.mean(),
        }

    # 对比各分量
    results = {
        "V_ps": compare_component("V_ps (局域赝势)", v_ps_dftcu, v_ps_qe),
        "V_H": compare_component("V_H  (Hartree势)", v_h_dftcu, v_h_qe),
        "V_xc": compare_component("V_xc (交换关联)", v_xc_dftcu, v_xc_qe),
        "V_loc": compare_component("V_loc (总和)", v_loc_dftcu, v_loc_qe),
    }

    # 打印结果
    print("\n" + "=" * 70)
    print("  分量对比结果")
    print("=" * 70)

    threshold = 1e-10

    for key in ["V_ps", "V_H", "V_xc", "V_loc"]:
        r = results[key]
        print(f"\n  [{r['name']}]")
        print(f"    QE mean:      {r['qe_mean']:16.6e} Ha")
        print(f"    DFTcu mean:   {r['dftcu_mean']:16.6e} Ha")
        print(f"    RMS 误差:     {r['rms']:16.6e} Ha")
        print(f"    最大误差:     {r['max_diff']:16.6e} Ha (位置 ir={r['max_pos']})")
        print(f"    相对 RMS:     {r['rel_rms']:16.6e}")

        if r["rms"] < threshold:
            status = "✅ 通过"
        elif r["rms"] < 1e-6:
            status = "⚠️  小误差"
        else:
            status = "❌ 失败"
        print(f"    状态: {status} (目标 RMS < {threshold:.0e})")

    # 总结
    print("\n" + "=" * 70)
    print("  误差来源分析")
    print("=" * 70)

    # 找到最大误差来源
    max_component = max(results.items(), key=lambda x: x[1]["rms"] if x[0] != "V_loc" else 0)
    print(f"\n  主要误差来源: {max_component[1]['name']}")
    print(f"    RMS 误差: {max_component[1]['rms']:.6e} Ha")

    # 理论预期
    rms_ps = results["V_ps"]["rms"]
    rms_h = results["V_H"]["rms"]
    rms_xc = results["V_xc"]["rms"]
    rms_total = results["V_loc"]["rms"]
    theoretical_rms = np.sqrt(rms_ps**2 + rms_h**2 + rms_xc**2)

    print(f"\n  理论预期（独立误差）:")
    print(f"    sqrt(RMS_ps² + RMS_H² + RMS_xc²) = {theoretical_rms:.6e} Ha")
    print(f"    实际 V_loc RMS:                    = {rms_total:.6e} Ha")
    print(f"    差异:                               = {abs(theoretical_rms - rms_total):.6e} Ha")

    if abs(theoretical_rms - rms_total) < 1e-12:
        print(f"    ✅ 误差基本独立，符合预期")
    else:
        print(f"    ⚠️  误差可能存在关联")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════════


def main():  # noqa: C901
    print("=" * 70)
    print("  DFTcu vs QE 结果对比分析")
    print("=" * 70)

    # ────────────────────────────────────────────────────────────────────────────
    # 检查文件是否存在
    # ────────────────────────────────────────────────────────────────────────────

    print("\n[检查文件]")

    required_files = [
        ("DFTcu 本征值", DFTCU_EIGENVALUES_FILE),
        ("DFTcu 能量分解", DFTCU_ENERGY_FILE),
    ]

    optional_files = [
        ("DFTcu 波函数", DFTCU_PSI_FILE),
        ("QE 输出", QE_EIGENVALUES_FILE),
        ("QE 波函数", QE_PSI_FILE),
    ]

    missing_required = False
    for name, fpath in required_files:
        if not fpath.exists():
            print(f"  ❌ 缺少: {name} ({fpath})")
            missing_required = True
        else:
            print(f"  ✓ {name}: {fpath}")

    for name, fpath in optional_files:
        if fpath.exists():
            print(f"  ✓ {name}: {fpath}")
        else:
            print(f"  ⚠️  可选: {name} ({fpath})")

    if missing_required:
        print("\n  ❌ 错误: 缺少必需文件")
        print("     请先运行 python run_nscf.py")
        return 1

    # ────────────────────────────────────────────────────────────────────────────
    # 加载 DFTcu 数据
    # ────────────────────────────────────────────────────────────────────────────

    print("\n[加载 DFTcu 数据]")

    dftcu_evals = load_dftcu_eigenvalues(DFTCU_EIGENVALUES_FILE)
    print(f"  ✓ 本征值: {len(dftcu_evals)} 个能带")

    dftcu_energy = load_dftcu_energy_breakdown(DFTCU_ENERGY_FILE)
    print(f"  ✓ 能量分解: {len(dftcu_energy)} 项")

    # ────────────────────────────────────────────────────────────────────────────
    # 加载 QE 参考数据
    # ────────────────────────────────────────────────────────────────────────────

    print("\n[加载 QE 参考数据]")

    qe_evals = []
    if QE_EIGENVALUES_FILE.exists():
        try:
            qe_evals = load_qe_eigenvalues_from_output(QE_EIGENVALUES_FILE)
            if len(qe_evals) > 0:
                print(f"  ✓ QE 本征值: {len(qe_evals)} 个能带")
            else:
                print(f"  ⚠️  QE 输出中未找到本征值")
        except Exception as e:
            print(f"  ⚠️  无法解析 QE 本征值: {e}")
    else:
        print(f"  ⚠️  QE 输出文件不存在: {QE_EIGENVALUES_FILE}")

    # ────────────────────────────────────────────────────────────────────────────
    # 对比分析
    # ────────────────────────────────────────────────────────────────────────────

    # 1. 本征值对比
    eigenvalue_passed = compare_eigenvalues(dftcu_evals, qe_evals)

    # 2. 能量分解
    compare_energy_breakdown(dftcu_energy)

    # 3. 波函数对比（可选）
    if DFTCU_PSI_FILE.exists() and QE_PSI_FILE.exists():
        try:
            npw = 85  # Si Gamma-only, ecutwfc=12 Ry
            nbands = len(dftcu_evals)

            print("\n[加载波函数数据]")
            print(f"  参数: npw={npw}, nbands={nbands}")

            dftcu_psi = load_wavefunction(DFTCU_PSI_FILE, npw, nbands)
            print(f"  ✓ DFTcu 波函数: shape={dftcu_psi.shape}")

            qe_psi = load_wavefunction(QE_PSI_FILE, npw, nbands)
            print(f"  ✓ QE 波函数: shape={qe_psi.shape}")

            compare_wavefunctions(dftcu_psi, qe_psi)
        except Exception as e:
            print(f"\n  ⚠️  波函数对比失败: {e}")

    # 4. Hamiltonian 分项诊断
    print("\n" + "=" * 70)
    print("  Hamiltonian 分项诊断: H|ψ> = T|ψ> + V_loc|ψ> + V_NL|ψ>")
    print("=" * 70)

    # DFTcu 数据文件路径（已在文件头部定义）

    # 检查是否有数据可以对比
    has_qe_data = QE_TPSI_FILE.exists() and QE_TVLOCPSI_FILE.exists() and QE_HPSI_FILE.exists()
    has_dftcu_data = (
        DFTCU_TPSI_FILE.exists()
        and DFTCU_TVLOCPSI_FILE.exists()
        and DFTCU_VNLPSI_FILE.exists()
        and DFTCU_HPSI_FILE.exists()
    )

    if not has_qe_data and not has_dftcu_data:
        print("\n  ⚠️  缺少 QE 和 DFTcu 中间数据文件，跳过 Hamiltonian 诊断")
    else:
        try:
            npw = 85

            # 加载 QE 数据（如果存在）
            tpsi_qe_ha = None
            vlocpsi_qe_ha = None
            vnlpsi_qe_ha = None
            hpsi_qe_ha = None

            if has_qe_data:
                print(f"\n  [1/2] 加载 QE Hamiltonian 中间数据 (npw={npw})...")
                tpsi_qe = load_qe_complex_gspace(QE_TPSI_FILE, npw)
                tvlocpsi_qe = load_qe_complex_gspace(QE_TVLOCPSI_FILE, npw)
                hpsi_qe = load_qe_complex_gspace(QE_HPSI_FILE, npw)

                # 计算各项贡献（QE 是 Ry）
                vlocpsi_qe = tvlocpsi_qe - tpsi_qe
                vnlpsi_qe = hpsi_qe - tvlocpsi_qe

                # 转换到 Ha
                tpsi_qe_ha = tpsi_qe * RY_TO_HA
                vlocpsi_qe_ha = vlocpsi_qe * RY_TO_HA
                vnlpsi_qe_ha = vnlpsi_qe * RY_TO_HA
                hpsi_qe_ha = hpsi_qe * RY_TO_HA

                print(f"  ✓ 已加载 QE 数据")

            # 加载 DFTcu 数据（如果存在）
            tpsi_dftcu_ha = None
            vlocpsi_dftcu_ha = None
            vnlpsi_dftcu_ha = None
            hpsi_dftcu_ha = None

            if has_dftcu_data:
                print(f"\n  [2/2] 加载 DFTcu Hamiltonian 中间数据 (npw={npw})...")
                tpsi_dftcu_ha = load_dftcu_complex_gspace(DFTCU_TPSI_FILE, npw)
                tvlocpsi_dftcu_ha = load_dftcu_complex_gspace(DFTCU_TVLOCPSI_FILE, npw)
                vnlpsi_dftcu_ha = load_dftcu_complex_gspace(DFTCU_VNLPSI_FILE, npw)
                hpsi_dftcu_ha = load_dftcu_complex_gspace(DFTCU_HPSI_FILE, npw)

                # 计算 V_loc|ψ>（DFTcu 已经是 Ha）
                vlocpsi_dftcu_ha = tvlocpsi_dftcu_ha - tpsi_dftcu_ha

                print(f"  ✓ 已加载 DFTcu 数据")

            # 显示对比结果
            print("\n" + "=" * 70)
            print("  Hamiltonian 分项对比")
            print("=" * 70)

            if has_qe_data and has_dftcu_data:
                # 完整对比
                print("\n[统计对比]")
                print(f"  {'项目':<12} {'QE |mean| (Ha)':<18} {'DFTcu |mean| (Ha)':<18} {'相对误差':<12}")
                print("  " + "-" * 70)

                def compare_component(name, qe_data, dftcu_data):
                    qe_mean = np.abs(qe_data).mean()
                    dftcu_mean = np.abs(dftcu_data).mean()
                    rel_err = abs(dftcu_mean - qe_mean) / (qe_mean + 1e-16)
                    status = "✅" if rel_err < 1e-6 else "⚠️" if rel_err < 1e-3 else "❌"
                    print(
                        f"  {name:<12} {qe_mean:<18.6e} {dftcu_mean:<18.6e} {rel_err:<12.6e} {status}"
                    )

                compare_component("T|ψ>", tpsi_qe_ha, tpsi_dftcu_ha)
                compare_component("V_loc|ψ>", vlocpsi_qe_ha, vlocpsi_dftcu_ha)
                compare_component("V_NL|ψ>", vnlpsi_qe_ha, vnlpsi_dftcu_ha)
                compare_component("H|ψ>", hpsi_qe_ha, hpsi_dftcu_ha)

                # 逐点对比前5个 G-vectors
                print("\n[前5个 G-vector 详细对比] (Ha)")
                print(f"  {'ig':<4} {'项目':<10} {'QE':<24} {'DFTcu':<24} {'误差':<12}")
                print("  " + "-" * 80)

                components = [
                    ("T|ψ>", tpsi_qe_ha, tpsi_dftcu_ha),
                    ("V_loc|ψ>", vlocpsi_qe_ha, vlocpsi_dftcu_ha),
                    ("V_NL|ψ>", vnlpsi_qe_ha, vnlpsi_dftcu_ha),
                    ("H|ψ>", hpsi_qe_ha, hpsi_dftcu_ha),
                ]

                for ig in range(min(5, npw)):
                    for name, qe_data, dftcu_data in components:
                        qe_val = qe_data[ig]
                        dftcu_val = dftcu_data[ig]
                        diff = abs(dftcu_val - qe_val)
                        print(
                            f"  {ig:<4} {name:<10} {qe_val:22.6e}  {dftcu_val:22.6e}  {diff:12.6e}"
                        )
                    if ig < 4:
                        print()  # 空行分隔

                # 最大误差统计
                print("\n[最大误差统计]")
                print(f"  {'项目':<12} {'max |DFTcu - QE| (Ha)':<25} {'位置':<8} {'精度目标':<15}")
                print("  " + "-" * 70)

                for name, qe_data, dftcu_data in components:
                    diff = np.abs(dftcu_data - qe_data)
                    max_diff = np.max(diff)
                    max_pos = np.argmax(diff)
                    target = 1e-10
                    status = "✅" if max_diff < target else "❌"
                    print(
                        f"  {name:<12} {max_diff:<25.6e} ig={max_pos:<6} < {target:.0e}  {status}"
                    )

            elif has_qe_data:
                # 只有 QE 数据
                print("\n[QE Hamiltonian 各项统计] (Ha)")
                print(f"  T|ψ>:      |mean|={np.abs(tpsi_qe_ha).mean():.6e}")
                print(f"  V_loc|ψ>:  |mean|={np.abs(vlocpsi_qe_ha).mean():.6e}")
                print(f"  V_NL|ψ>:   |mean|={np.abs(vnlpsi_qe_ha).mean():.6e}")
                print(f"  H|ψ>:      |mean|={np.abs(hpsi_qe_ha).mean():.6e}")
                print("\n  注: 仅显示 QE 参考数据（缺少 DFTcu 数据用于对比）")

            elif has_dftcu_data:
                # 只有 DFTcu 数据
                print("\n[DFTcu Hamiltonian 各项统计] (Ha)")
                print(f"  T|ψ>:      |mean|={np.abs(tpsi_dftcu_ha).mean():.6e}")
                print(f"  V_loc|ψ>:  |mean|={np.abs(vlocpsi_dftcu_ha).mean():.6e}")
                print(f"  V_NL|ψ>:   |mean|={np.abs(vnlpsi_dftcu_ha).mean():.6e}")
                print(f"  H|ψ>:      |mean|={np.abs(hpsi_dftcu_ha).mean():.6e}")
                print("\n  注: 仅显示 DFTcu 数据（缺少 QE 参考数据用于对比）")

        except Exception as e:
            print(f"\n  ⚠️  Hamiltonian 诊断失败: {e}")
            import traceback

            traceback.print_exc()

    # ────────────────────────────────────────────────────────────────────────────
    # 4.5. V_loc 分量对比 (V_ps, V_H, V_xc)
    # ────────────────────────────────────────────────────────────────────────────

    try:
        compare_vloc_components(nnr=3375)
    except Exception as e:
        print(f"\n  ⚠️  V_loc 分量对比失败: {e}")
        import traceback

        traceback.print_exc()

    # ────────────────────────────────────────────────────────────────────────────
    # 5. Gamma-only 内积验证（compute_s_subspace_gamma_packed）
    # ────────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Gamma-only 内积验证: compute_s_subspace_gamma_packed")
    print("=" * 70)

    if not QE_PSI_FILE.exists():
        print("\n  ⚠️  缺少 QE 波函数文件，跳过内积验证")
    else:
        try:
            import dftcu

            npw = 85
            nbands = 4
            print(f"\n  参数: npw={npw}, nbands={nbands}")

            # 加载 QE 波函数
            qe_psi = load_wavefunction(QE_PSI_FILE, npw, nbands)  # (npw, nbands)
            print(f"  ✓ 已加载 QE 波函数: shape={qe_psi.shape}")

            # 手动计算 Gamma-only 归一化（参考值）
            print("\n[手动计算 Gamma-only 归一化]")
            print(f"  {'Band':<6} {'<ψ|ψ>':<15} {'Status'}")
            print("  " + "-" * 40)
            manual_norms = []
            for band in range(nbands):
                psi_band = qe_psi[:, band]
                # Gamma-only formula: 2*Re(Σ ψ*·ψ) - ψ*(G=0)·ψ(G=0)
                norm_sq = 2.0 * np.sum(np.abs(psi_band) ** 2).real
                norm_sq -= np.abs(psi_band[0]) ** 2
                manual_norms.append(norm_sq)
                status = "✅" if np.abs(norm_sq - 1.0) < 1e-10 else "❌"
                print(f"  {band+1:<6} {norm_sq:<15.10f} {status}")

            # 使用 compute_s_subspace_gamma_packed 计算
            print("\n[使用 compute_s_subspace_gamma_packed 计算]")
            qe_psi_packed = qe_psi.T  # (nbands, npw) - packed 布局
            gstart = 2  # Gamma-only, G=0 exists
            s_sub = dftcu.compute_s_subspace_gamma_packed(qe_psi_packed, npw, nbands, gstart)

            print(f"  {'Band':<6} {'S_sub[i,i]':<15} {'vs Manual':<15} {'Status'}")
            print("  " + "-" * 50)
            all_correct = True
            for i in range(nbands):
                norm_dftcu = s_sub[i, i]
                norm_manual = manual_norms[i]
                diff = abs(norm_dftcu - norm_manual)
                status = "✅" if diff < 1e-13 else "❌"
                if diff >= 1e-13:
                    all_correct = False
                print(f"  {i+1:<6} {norm_dftcu:<15.10f} {norm_manual:<15.10f} {status}")

            # 检查对称性
            max_asymmetry = np.max(np.abs(s_sub - s_sub.T))
            print(f"\n  S_sub 对称性: max|S - S^T| = {max_asymmetry:.2e}", end="")
            if max_asymmetry < 1e-13:
                print("  ✅")
            else:
                print("  ❌")

            # 结论
            if all_correct and max_asymmetry < 1e-13:
                print("\n  ✅ Gamma-only 内积计算正确！")
                print("     compute_s_subspace_gamma_packed 已验证（精度 < 1e-13）")
            else:
                print("\n  ❌ Gamma-only 内积计算有误差")
                print("     请检查 compute_s_subspace_gamma 实现")

        except Exception as e:
            print(f"\n  ⚠️  Gamma-only 内积验证失败: {e}")
            import traceback

            traceback.print_exc()

    # ────────────────────────────────────────────────────────────────────────────
    # 总结
    # ────────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)

    if eigenvalue_passed is True:
        print("\n  ✅ 本征值验证通过")
    elif eigenvalue_passed is False:
        print("\n  ❌ 本征值验证失败")
    else:
        print("\n  ⚠️  本征值未对比（缺少 QE 参考数据）")

    print("\n  下一步:")
    print("  1. 如果精度未达标，检查 Hamiltonian 各项计算")
    print("  2. 验证 FFT 约定和 G-vector 映射")
    print("  3. 对比中间变量（T|ψ>, V_loc|ψ>, V_NL|ψ>）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
