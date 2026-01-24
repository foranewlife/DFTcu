#!/usr/bin/env python3
"""
DFTcu NSCF 完整计算流程

功能：
  1. 加载 QE 输入参数（晶格、原子位置、赝势等）
  2. 读取 SCF 电荷密度（可选，使用均匀密度作为简化）
  3. 构建完整 Hamiltonian (T + V_ps + V_H + V_xc + V_NL)
  4. 运行 NonSCFSolver 求解本征值和能量
  5. 导出诊断数据到 nscf_output/

使用方法：
    python run_nscf.py

输出：
    - nscf_output/dftcu_psi_initial.txt      : 初始波函数
    - nscf_output/dftcu_eigenvalues.txt      : 本征值
    - nscf_output/dftcu_occupations.txt      : 占据数
    - nscf_output/dftcu_energy_breakdown.txt : 能量分解
"""

import sys
from pathlib import Path

import numpy as np

import dftcu

# ════════════════════════════════════════════════════════════════════════════════
# 系统配置（Si 2原子，Gamma-only）
# ════════════════════════════════════════════════════════════════════════════════

# QE 输入参数（from qe_run/si_nscf.in）
ALAT = 10.20  # Bohr
LATTICE_BOHR = np.array(
    [[-ALAT / 2, 0, ALAT / 2], [0, ALAT / 2, ALAT / 2], [-ALAT / 2, ALAT / 2, 0]]
)

# Si 原子位置（晶体坐标）
ATOMIC_POSITIONS_CRYST = np.array([[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]])  # Si 1  # Si 2

# FFT 网格和截断能（必须与 QE 输入一致）
NR = [15, 15, 15]  # FFT grid (匹配 QE si_nscf.in: nr1=nr2=nr3=15)
ECUTWFC_RY = 12.0  # 波函数截断能 [Ry]
ECUTRHO_RY = 48.0  # 密度截断能 [Ry]

# 系统参数
NELEC = 8.0  # 电子数 (2 Si atoms)
NBANDS = 4  # 能带数
IS_GAMMA = True  # Gamma-only 计算

# 赝势文件
UPF_FILE = Path(__file__).parent / "qe_run" / "Si.pz-rrkj.UPF"

# 输出目录
OUTPUT_DIR = "nscf_output"

# QE 数据目录
QE_RHO_FILE = Path(__file__).parent / "qe_run" / "qe_rho_r.txt"
QE_PSI_FILE = Path(__file__).parent / "qe_run" / "dftcu_debug_psi_iter0.txt"


# ════════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════════════════════


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def cryst_to_cart(cryst_coords, lattice):
    """晶体坐标 -> 笛卡尔坐标（Bohr）"""
    cart_coords = []
    for c in cryst_coords:
        cart = c[0] * lattice[0] + c[1] * lattice[1] + c[2] * lattice[2]
        cart_coords.append(cart)
    return np.array(cart_coords)


def load_qe_density(filename, nnr, nr=None, reorder=True):
    """从 QE 导出的文本文件加载密度 ρ(r)

    格式:
        3375              # 网格点数
        2.694820e-03      # ρ(0) [e⁻/Bohr³]
        1.561438e-02      # ρ(1)
        ...

    QE 使用 Fortran 列主序 (ix 最快变化):
        idx = ix + nr1 * (iy + nr2 * iz)

    DFTcu 使用 C 行主序 (iz 最快变化):
        idx = iz + nr3 * (iy + nr2 * ix)

    Args:
        filename: QE 数据文件路径
        nnr: 期望的网格点数
        nr: [nr1, nr2, nr3] 网格维度（用于重排序）
        reorder: 是否重新排序 (Fortran → C)

    返回: numpy array shape=(nnr,), 单位 e⁻/Bohr³
    """
    # 读取 QE 数据 (Fortran 顺序)
    rho_fortran = np.zeros(nnr)
    with open(filename, "r") as f:
        # 第一行：网格点数
        n_read = int(f.readline().strip())
        if n_read != nnr:
            print(f"  ⚠️  警告: 文件中网格点数 {n_read} ≠ 期望值 {nnr}")

        # 读取密度值
        for i in range(min(n_read, nnr)):
            line = f.readline().strip()
            if not line:
                break
            rho_fortran[i] = float(line)

    if not reorder or nr is None:
        return rho_fortran

    # 重新排列: Fortran (ix 最快) → C (iz 最快)
    nr1, nr2, nr3 = nr
    rho_c = np.zeros(nnr)

    for iz in range(nr3):
        for iy in range(nr2):
            for ix in range(nr1):
                # Fortran 索引: ix 最快变化
                idx_fortran = ix + nr1 * (iy + nr2 * iz)
                # C 索引: iz 最快变化
                idx_c = iz + nr3 * (iy + nr2 * ix)
                rho_c[idx_c] = rho_fortran[idx_fortran]

    return rho_c


def load_qe_complex_gspace_multi(filename, npw=85, nbands=4):
    """加载 QE 导出的 G 空间复数数组（多 band）

    格式: ig band Re(ψ) Im(ψ)
    注意: QE 使用 Fortran 1-based 索引

    Args:
        filename: QE 数据文件路径
        npw: Smooth G-vectors 数量
        nbands: 能带数

    Returns:
        np.ndarray: shape=(npw, nbands), dtype=complex128
    """
    data = np.zeros((npw, nbands), dtype=np.complex128)

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()

            if len(parts) == 4:
                # 格式: ig band Re Im
                ig = int(parts[0]) - 1  # Fortran 1-based → 0-based
                band = int(parts[1]) - 1  # Fortran 1-based → 0-based
                re_val = float(parts[2])
                im_val = float(parts[3])

                if ig < npw and band < nbands:
                    # 波函数系数是无量纲的，不需要单位转换
                    data[ig, band] = complex(re_val, im_val)
            elif len(parts) == 3:
                # 格式: ig Re Im (默认 band=0)
                ig = int(parts[0]) - 1
                re_val = float(parts[1])
                im_val = float(parts[2])

                if ig < npw:
                    # 波函数系数是无量纲的，不需要单位转换
                    data[ig, 0] = complex(re_val, im_val)

    return data


# ════════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════════


def main():  # noqa: C901
    print_section("DFTcu NSCF 完整计算流程")
    print(f"\n  系统: Si 2原子 (Gamma-only)")
    print(f"  FFT 网格: {NR}")
    print(f"  ecutwfc: {ECUTWFC_RY} Ry ({ECUTWFC_RY/2:.1f} Ha)")
    print(f"  ecutrho: {ECUTRHO_RY} Ry ({ECUTRHO_RY/2:.1f} Ha)")
    print(f"  能带数: {NBANDS}")
    print(f"  电子数: {int(NELEC)}")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 1: 创建 Grid
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 1: 创建 Grid")

    lattice_ang = LATTICE_BOHR * dftcu.constants.BOHR_TO_ANGSTROM
    grid = dftcu.create_grid_from_qe(
        lattice_ang=lattice_ang,
        nr=NR,
        ecutwfc_ry=ECUTWFC_RY,
        ecutrho_ry=ECUTRHO_RY,
        is_gamma=IS_GAMMA,
    )

    print(f"  ✓ Grid 创建完成")
    print(f"    - FFT 网格: {grid.nr()}")
    print(f"    - 体积: {grid.volume():.2f} Bohr³")
    print(f"    - Smooth G-vectors: {grid.ngw()}")
    print(f"    - Dense G-vectors: {grid.ngm_dense()}")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 2: 创建 Atoms
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 2: 创建 Atoms")

    # 转换为笛卡尔坐标
    atomic_positions_cart = cryst_to_cart(ATOMIC_POSITIONS_CRYST, LATTICE_BOHR)

    atoms_list = [
        dftcu.Atom(pos[0], pos[1], pos[2], 14.0, 0)  # Z=14 (Si), type=0
        for pos in atomic_positions_cart
    ]
    atoms = dftcu.create_atoms_from_bohr(atoms_list)

    print(f"  ✓ Atoms 创建完成 ({len(atoms_list)} Si atoms)")
    for i, pos in enumerate(atomic_positions_cart):
        print(f"    - Si {i+1}: ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}) Bohr")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 3: 加载 UPF 赝势并构建 Hamiltonian
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 3: 加载 UPF 赝势并构建 Hamiltonian")

    if not UPF_FILE.exists():
        print(f"  ❌ 错误: UPF 文件不存在: {UPF_FILE}")
        return 1

    print(f"  加载赝势文件: {UPF_FILE.name}")

    # Parse UPF file
    parser = dftcu.UPFParser()
    upf_data = parser.parse(str(UPF_FILE))

    # Create LocalPseudo and NonLocalPseudo from UPF data
    local_pseudo = dftcu.LocalPseudo.from_upf(grid, atoms, upf_data, atom_type=0)
    nl_pseudo = dftcu.NonLocalPseudo.from_upf(grid, atoms, upf_data, atom_type=0)

    print(f"  ✓ 赝势加载完成")
    print(f"    - 局域赝势: LocalPseudo")
    print(f"    - 非局域赝势: NonLocalPseudo")
    print(f"    - Z_valence: {upf_data.z_valence()}")

    # CRITICAL: Must call update_projectors to compute beta projectors!
    # Otherwise num_projectors_ = 0 and V_NL is never applied
    if nl_pseudo is not None:
        print(f"\n  调用 update_projectors 计算 beta 投影算符...")
        nl_pseudo.update_projectors(atoms)
        print(f"  ✓ Beta projectors 已计算")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 4: 加载 SCF 自洽密度
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 4: 加载 SCF 自洽密度")

    if not QE_RHO_FILE.exists():
        print(f"  ❌ 错误: QE 密度文件不存在: {QE_RHO_FILE}")
        print(f"     请先运行 QE SCF 计算并导出密度")
        return 1

    print(f"  加载 QE SCF 密度: {QE_RHO_FILE.name}")

    # ⚠️ IMPORTANT: 必须重新排序！
    # DFTcu 内部的 FFT、Poisson 求解器、XC 泛函计算现在使用 C 行主序 (iz 最快变化)
    # 之前保持 Fortran 顺序是为了调试，但现在 Grid 映射已统一为行主序
    rho_data = load_qe_density(QE_RHO_FILE, grid.nnr(), nr=NR, reorder=True)
    print(f"  ✓ 密度已加载 (已转换为 C 行主序，匹配 DFTcu 内部布局)")

    # 创建 DFTcu RealField
    rho_input = dftcu.RealField(grid, 1)

    # 使用 copy_from_host 传输数据
    rho_input.copy_from_host(rho_data)

    # 验证密度
    rho_min = rho_data.min()
    rho_max = rho_data.max()
    rho_avg = rho_data.mean()
    rho_integral = rho_avg * grid.volume()

    print(f"  ✓ 密度加载完成")
    print(f"    - 网格点数: {grid.nnr()}")
    print(f"    - 密度范围: [{rho_min:.6e}, {rho_max:.6e}] e⁻/Bohr³")
    print(f"    - 平均密度: {rho_avg:.6e} e⁻/Bohr³")
    print(f"    - 积分电子数: {rho_integral:.4f} (期望: {NELEC})")

    if abs(rho_integral - NELEC) > 0.01:
        print(f"  ⚠️  警告: 积分电子数与期望值偏差较大")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 5: 构建 DensityFunctionalPotential
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 5: 构建 DensityFunctionalPotential")

    print("\n创建 DensityFunctionalPotential...")
    dfp = dftcu.DensityFunctionalPotential(grid)

    # Hartree 势
    print("  - 添加 Hartree 泛函")
    hartree = dftcu.Hartree()
    dfp.add_functional(hartree)

    # LDA-PZ 交换关联泛函
    print("  - 添加 LDA-PZ 泛函")
    lda = dftcu.LDA_PZ()
    dfp.add_functional(lda)

    # 局域赝势（已经创建）
    print("  - 添加局域赝势")
    dfp.add_functional(local_pseudo)

    print("\n✓ DensityFunctionalPotential 创建完成")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 6: 创建 Hamiltonian 并计算势能
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 6: 创建 Hamiltonian 并计算势能")

    print("\n创建 Hamiltonian...")
    ham = dftcu.Hamiltonian(grid)
    ham.set_density_functional_potential(dfp)

    # 导出输入密度用于诊断
    rho_host = np.zeros(grid.nnr())
    rho_input.copy_to_host(rho_host)
    np.savetxt(
        "nscf_output/dftcu_rho_r.txt",
        rho_host,
        fmt="%.16e",
        header=f"DFTcu input density rho(r) [e/Bohr^3], nnr={grid.nnr()}",
    )
    print(f"✓ 已导出输入密度 → nscf_output/dftcu_rho_r.txt")
    print(f"  ρ 统计: mean={rho_host.mean():.6e}, integral={rho_host.sum()*grid.dv():.6f} e")

    print("\n调用 ham.update_potentials(rho_input) ...")
    ham.update_potentials(rho_input)
    print("✓ vrs = V_ps + V_H[ρ] + V_xc[ρ] 已计算 (NSCF 中固定不变)")

    print("\n设置非局域赝势...")
    ham.set_nonlocal(nl_pseudo)
    print("✓ 非局域赝势已设置")

    # 获取 v_of_0 (用于 alpha 修正)
    v0 = ham.get_v_of_0()
    print(f"\nv_of_0 = {v0:.10f} Ha (G=0 处的总局域势)")

    # 导出 v_loc 及其各个分量用于诊断
    vloc = ham.v_loc()
    vloc_host = np.zeros(grid.nnr())
    vloc.copy_to_host(vloc_host)
    np.savetxt(
        "nscf_output/dftcu_vloc_r.txt",
        vloc_host,
        fmt="%.16e",
        header=f"DFTcu V_loc(r) [Ha], nnr={grid.nnr()}",
    )
    print(f"\n✓ 已导出 v_loc → nscf_output/dftcu_vloc_r.txt")
    print(
        f"  V_loc 统计: mean={vloc_host.mean():.6f}, min/max=[{vloc_host.min():.6f}, {vloc_host.max():.6f}] Ha"
    )

    # 导出 V_ps, V_H, V_xc 各个分量
    v_ps = ham.v_ps()
    v_ps_host = np.zeros(grid.nnr())
    v_ps.copy_to_host(v_ps_host)
    np.savetxt(
        "nscf_output/dftcu_vps_r.txt",
        v_ps_host,
        fmt="%.16e",
        header=f"DFTcu V_ps(r) [Ha], nnr={grid.nnr()}",
    )
    print(
        f"  V_ps 统计: mean={v_ps_host.mean():.6f}, min/max=[{v_ps_host.min():.6f}, {v_ps_host.max():.6f}] Ha"
    )

    v_h = ham.v_h()
    v_h_host = np.zeros(grid.nnr())
    v_h.copy_to_host(v_h_host)
    np.savetxt(
        "nscf_output/dftcu_vh_r.txt",
        v_h_host,
        fmt="%.16e",
        header=f"DFTcu V_H(r) [Ha], nnr={grid.nnr()}",
    )
    print(
        f"  V_H 统计:  mean={v_h_host.mean():.6f}, min/max=[{v_h_host.min():.6f}, {v_h_host.max():.6f}] Ha"
    )

    v_xc = ham.v_xc()
    v_xc_host = np.zeros(grid.nnr())
    v_xc.copy_to_host(v_xc_host)
    np.savetxt(
        "nscf_output/dftcu_vxc_r.txt",
        v_xc_host,
        fmt="%.16e",
        header=f"DFTcu V_xc(r) [Ha], nnr={grid.nnr()}",
    )
    print(
        f"  V_xc 统计: mean={v_xc_host.mean():.6f}, min/max=[{v_xc_host.min():.6f}, {v_xc_host.max():.6f}] Ha"
    )

    # ────────────────────────────────────────────────────────────────────────────
    # Step 7: 初始化波函数（从 QE 加载）
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 7: 初始化波函数（从 QE 加载）")

    if not QE_PSI_FILE.exists():
        print(f"  ❌ 错误: QE 波函数文件不存在: {QE_PSI_FILE}")
        print(f"     使用随机初始化作为 fallback")
        psi = dftcu.Wavefunction(grid, NBANDS, ECUTWFC_RY / 2)  # Ha
        psi.randomize()
        print(f"  ✓ 波函数随机初始化完成 ({NBANDS} bands)")
    else:
        print(f"  加载 QE 初始波函数: {QE_PSI_FILE.name}")

        # 加载 QE 波函数（npw=85, nbands=4）
        qe_psi = load_qe_complex_gspace_multi(QE_PSI_FILE, npw=grid.ngw(), nbands=NBANDS)
        print(f"  ✓ 已加载 QE 波函数 (npw={grid.ngw()}, nbands={NBANDS})")

        # 创建 DFTcu Wavefunction 并设置系数
        psi = dftcu.Wavefunction(grid, NBANDS, ECUTWFC_RY / 2)  # Ha

        # 获取网格参数
        npw = grid.ngw()

        # 将数据展平为 (nbands * npw,) 按 band 优先顺序
        # qe_psi shape: (npw, nbands) - 列存储
        # 需要转置为 (nbands, npw) 然后展平
        # values_flat = qe_psi.T.flatten()  # shape: (nbands*npw,)

        print(f"  ✓ 数据准备完成:")
        print(f"    - qe_psi shape: {qe_psi.shape}")
        print(f"    - Band 1, G=0: {qe_psi[0, 0]}")
        print(f"    - Band 2, G=0: {qe_psi[0, 1]}")

        # ❌ set_coefficients_miller 不能用于加载 QE 数据
        #    因为它从 (h,k,l) 计算 FFT 索引，与 QE 的 nl_d 映射不一致
        # ✅ 正确方法：使用 nl_d 和 nlm_d 映射直接填充 FFT grid

        print(f"\n  使用 nl_d/nlm_d 映射加载 QE 波函数...")

        # 获取 nl_d 映射 (QE 的 G → FFT 索引)
        nl_d = grid.get_nl_d()
        nlm_d = grid.get_nlm_d()

        # 准备主机内存 (nnr × nbands)
        # DFTcu 使用 band-major 布局: data[ib * nnr + ifft]
        psi_host = np.zeros(grid.nnr() * NBANDS, dtype=np.complex128)

        # 填充数据：对每个 G-vector，使用 nl_d 查找 FFT 索引
        for ig in range(npw):
            ifft = nl_d[ig]  # +G 的 FFT 索引
            ifft_m = nlm_d[ig]  # -G 的 FFT 索引

            for ib in range(NBANDS):
                val = qe_psi[ig, ib]

                # 填充 +G 位置 (band-major: idx = ib * nnr + ifft)
                psi_host[ib * grid.nnr() + ifft] = val

                # 填充 -G 位置 (复共轭，Hermitian 对称性)
                if ig > 0:  # G≠0 才需要填充 -G
                    psi_host[ib * grid.nnr() + ifft_m] = np.conj(val)

        # 传输到 GPU
        psi.copy_from_host(psi_host)

        print(f"  ✓ 使用 nl_d/nlm_d 映射完成加载")

        print(f"  ✓ 波函数已从 QE 加载 ({NBANDS} bands)")
        print(f"    - 使用 QE iter 0 的初始波函数")
        print(f"    - 确保 DFTcu 和 QE 从相同起点开始 Davidson 迭代")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 7.5: 验证加载的波函数（诊断）
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 7.5: 验证加载的波函数")

    print("\n从 DFTcu 读取加载后的波函数系数...")

    # 分配主机内存
    psi_dftcu_full = np.zeros(grid.nnr() * NBANDS, dtype=np.complex128)

    # 从 GPU 复制到主机
    psi.copy_to_host(psi_dftcu_full)

    # 重新排列为 (nbands, nnr) - band-major 布局
    psi_dftcu_full = psi_dftcu_full.reshape((NBANDS, grid.nnr()))

    print(f"  ✓ 已读取 DFTcu 波函数: shape = {psi_dftcu_full.shape}")

    # 获取 nl_d 映射（从 FFT grid 提取 Smooth G-vectors）
    nl_d = grid.get_nl_d()
    npw = grid.ngw()

    # 提取 Smooth grid 上的系数
    psi_dftcu_smooth = np.zeros((npw, NBANDS), dtype=np.complex128)
    for ig in range(npw):
        ifft = nl_d[ig]
        for ib in range(NBANDS):
            psi_dftcu_smooth[ig, ib] = psi_dftcu_full[ib, ifft]

    print(f"  ✓ 提取 Smooth grid 系数: shape = {psi_dftcu_smooth.shape}")

    # 与 QE 数据逐系数对比
    print("\n  逐 Band 对比系数（前 5 个 G-vectors）:")
    print(
        f"  {'Band':<6} {'G-vec':<6} {'DFTcu Re':>15} {'QE Re':>15} {'Diff Re':>12} {'DFTcu Im':>15} {'QE Im':>15} {'Diff Im':>12}"
    )
    print("  " + "-" * 100)

    for ib in range(min(NBANDS, 2)):  # 只检查前 2 个 band
        print(f"\n  Band {ib+1}:")
        for ig in range(min(5, npw)):
            dftcu_val = psi_dftcu_smooth[ig, ib]
            qe_val = qe_psi[ig, ib]
            diff_re = abs(dftcu_val.real - qe_val.real)
            diff_im = abs(dftcu_val.imag - qe_val.imag)

            print(
                f"  {ib+1:<6} {ig+1:<6} {dftcu_val.real:15.10f} {qe_val.real:15.10f} {diff_re:12.2e} "
                f"{dftcu_val.imag:15.10f} {qe_val.imag:15.10f} {diff_im:12.2e}"
            )

    # 计算整体误差
    print("\n  整体误差统计:")
    for ib in range(NBANDS):
        diff_band = psi_dftcu_smooth[:, ib] - qe_psi[:, ib]
        max_diff = np.max(np.abs(diff_band))
        rms_diff = np.sqrt(np.mean(np.abs(diff_band) ** 2))
        print(f"    Band {ib+1}: max_diff = {max_diff:.6e}, rms_diff = {rms_diff:.6e}")

    print("\n✓ 波函数加载验证完成")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 7.6: 计算并导出 H|ψ> 分项诊断数据
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 7.6: 计算并导出 H|ψ> 分项诊断数据")

    print("\n计算 Hamiltonian 各项作用在初始波函数上...")
    print("  H|ψ> = T|ψ> + V_loc|ψ> + V_NL|ψ>")

    # 创建输出目录
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 计算 T|ψ> (动能项)
    print("\n[1/4] 计算 T|ψ> (动能项)...")
    h_psi_temp = dftcu.Wavefunction(grid, NBANDS, ECUTWFC_RY / 2)
    ham.apply_kinetic(psi, h_psi_temp)

    # 导出 T|ψ>
    tpsi_data = np.zeros(grid.nnr() * NBANDS, dtype=np.complex128)
    h_psi_temp.copy_to_host(tpsi_data)
    tpsi_data = tpsi_data.reshape((NBANDS, grid.nnr()))

    tpsi_smooth = np.zeros((npw, NBANDS), dtype=np.complex128)
    for ig in range(npw):
        ifft = nl_d[ig]
        for ib in range(NBANDS):
            tpsi_smooth[ig, ib] = tpsi_data[ib, ifft]

    np.savetxt(
        f"{OUTPUT_DIR}/dftcu_tpsi_iter0.txt",
        np.column_stack([np.arange(1, npw + 1), tpsi_smooth[:, 0].real, tpsi_smooth[:, 0].imag]),
        fmt="%5d %25.16e %25.16e",
        header=f"DFTcu T|psi> iter=0 band=1 (npw={npw}) [Ha]",
    )
    print(f"  ✓ 已导出 T|ψ> → {OUTPUT_DIR}/dftcu_tpsi_iter0.txt")
    print(f"    |mean| = {np.abs(tpsi_smooth[:, 0]).mean():.6e} Ha")

    # 2. 计算 (T + V_loc)|ψ>
    print("\n[2/4] 计算 (T + V_loc)|ψ> (动能 + 局域势)...")
    h_psi_temp = dftcu.Wavefunction(grid, NBANDS, ECUTWFC_RY / 2)
    ham.apply_kinetic(psi, h_psi_temp)
    ham.apply_local(psi, h_psi_temp)

    tvlocpsi_data = np.zeros(grid.nnr() * NBANDS, dtype=np.complex128)
    h_psi_temp.copy_to_host(tvlocpsi_data)
    tvlocpsi_data = tvlocpsi_data.reshape((NBANDS, grid.nnr()))

    tvlocpsi_smooth = np.zeros((npw, NBANDS), dtype=np.complex128)
    for ig in range(npw):
        ifft = nl_d[ig]
        for ib in range(NBANDS):
            tvlocpsi_smooth[ig, ib] = tvlocpsi_data[ib, ifft]

    np.savetxt(
        f"{OUTPUT_DIR}/dftcu_tvlocpsi_iter0.txt",
        np.column_stack(
            [np.arange(1, npw + 1), tvlocpsi_smooth[:, 0].real, tvlocpsi_smooth[:, 0].imag]
        ),
        fmt="%5d %25.16e %25.16e",
        header=f"DFTcu (T+Vloc)|psi> iter=0 band=1 (npw={npw}) [Ha]",
    )
    print(f"  ✓ 已导出 (T+V_loc)|ψ> → {OUTPUT_DIR}/dftcu_tvlocpsi_iter0.txt")

    # 计算 V_loc|ψ>
    vlocpsi_smooth = tvlocpsi_smooth - tpsi_smooth
    print(f"    V_loc|ψ> |mean| = {np.abs(vlocpsi_smooth[:, 0]).mean():.6e} Ha")

    # 3. 计算完整 H|ψ>
    print("\n[3/4] 计算完整 H|ψ> (包含非局域势)...")
    h_psi_full = dftcu.Wavefunction(grid, NBANDS, ECUTWFC_RY / 2)
    ham.apply(psi, h_psi_full)

    hpsi_data = np.zeros(grid.nnr() * NBANDS, dtype=np.complex128)
    h_psi_full.copy_to_host(hpsi_data)
    hpsi_data = hpsi_data.reshape((NBANDS, grid.nnr()))

    hpsi_smooth = np.zeros((npw, NBANDS), dtype=np.complex128)
    for ig in range(npw):
        ifft = nl_d[ig]
        for ib in range(NBANDS):
            hpsi_smooth[ig, ib] = hpsi_data[ib, ifft]

    np.savetxt(
        f"{OUTPUT_DIR}/dftcu_hpsi_iter0.txt",
        np.column_stack([np.arange(1, npw + 1), hpsi_smooth[:, 0].real, hpsi_smooth[:, 0].imag]),
        fmt="%5d %25.16e %25.16e",
        header=f"DFTcu H|psi> iter=0 band=1 (npw={npw}) [Ha]",
    )
    print(f"  ✓ 已导出 H|ψ> → {OUTPUT_DIR}/dftcu_hpsi_iter0.txt")
    print(f"    |mean| = {np.abs(hpsi_smooth[:, 0]).mean():.6e} Ha")

    # 4. 计算 V_NL|ψ>
    print("\n[4/4] 计算 V_NL|ψ> (非局域势项)...")
    vnlpsi_smooth = hpsi_smooth - tvlocpsi_smooth

    np.savetxt(
        f"{OUTPUT_DIR}/dftcu_vnlpsi_iter0.txt",
        np.column_stack(
            [np.arange(1, npw + 1), vnlpsi_smooth[:, 0].real, vnlpsi_smooth[:, 0].imag]
        ),
        fmt="%5d %25.16e %25.16e",
        header=f"DFTcu VNL|psi> iter=0 band=1 (npw={npw}) [Ha]",
    )
    print(f"  ✓ 已导出 V_NL|ψ> → {OUTPUT_DIR}/dftcu_vnlpsi_iter0.txt")
    print(f"    |mean| = {np.abs(vnlpsi_smooth[:, 0]).mean():.6e} Ha")

    print("\n✓ Hamiltonian 分项诊断数据导出完成")
    print(f"  所有数据已保存到: {OUTPUT_DIR}/")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 8: 运行 NonSCFSolver
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 8: 运行 NonSCFSolver")

    print("\n创建 NonSCFSolver...")
    nscf = dftcu.NonSCFSolver(grid)

    # 启用诊断模式
    print("\n启用诊断模式...")
    diag = dftcu.NonSCFDiagnostics()
    diag.enable_all()  # 启用所有诊断输出
    diag.output_dir = OUTPUT_DIR
    nscf.enable_diagnostics(diag)
    print(f"✓ 诊断模式已启用，输出目录: {OUTPUT_DIR}")

    # ────────────────────────────────────────────────────────────────────────────
    # 在 solver.solve() 前导出初始波函数用于诊断
    # ────────────────────────────────────────────────────────────────────────────
    print("\n[诊断] 导出初始波函数...")

    # 创建输出目录
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 导出初始波函数（不导出，solver 会自动导出）
    print(f"  ✓ 将由 NonSCFSolver 自动导出诊断数据")
    print("✓ 诊断配置完成")

    print("\n调用 nscf.solve() ...")
    print("  - 使用固定的 vrs (potinit 阶段已计算)")
    print("  - Davidson 迭代求解本征值")
    print("  - 计算占据数和能量分解")

    result = nscf.solve(
        ham=ham,
        psi=psi,
        nelec=NELEC,
        atoms=atoms,
        ecutrho=ECUTRHO_RY,  # Ry
        rho_scf=None,  # 使用混合泛函模式（从波函数重算密度）
    )

    print("\n✓ NonSCFSolver 完成")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 9: 输出结果
    # ────────────────────────────────────────────────────────────────────────────
    print_section("Step 9: 结果总结")

    print("\n[9.1] 本征值 (Ha):")
    for i, e in enumerate(result.eigenvalues):
        e_ev = e * dftcu.constants.HA_TO_EV
        print(f"  Band {i+1}: {e:16.10f} Ha  ({e_ev:12.6f} eV)")

    print("\n[9.2] 能量分解:")
    print(f"  E_band  = {result.eband:16.10f} Ha  (能带能量)")
    print(f"  deband  = {result.deband:16.10f} Ha  (双重计数修正)")
    print(f"  E_H     = {result.ehart:16.10f} Ha  (Hartree 能量)")
    print(f"  E_XC    = {result.etxc:16.10f} Ha  (XC 能量)")
    print(f"  E_Ewald = {result.eewld:16.10f} Ha  (Ewald 能量)")
    print(f"  alpha   = {result.alpha:16.10f} Ha  (G=0 修正)")
    print(f"  ───────────────────────────────────")
    print(f"  E_tot   = {result.etot:16.10f} Ha  (总能量)")
    print(f"          = {result.etot * dftcu.constants.HA_TO_EV:16.10f} eV")

    print("\n[9.3] 诊断文件:")
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        for fname in [
            "dftcu_psi_initial.txt",
            "dftcu_eigenvalues.txt",
            "dftcu_occupations.txt",
            "dftcu_energy_breakdown.txt",
        ]:
            fpath = output_path / fname
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"  ✓ {fname:30} ({size:,} bytes)")
            else:
                print(f"  ✗ {fname:30} (not created)")

    # ────────────────────────────────────────────────────────────────────────────
    # 完成
    # ────────────────────────────────────────────────────────────────────────────
    print_section("完成")
    print(f"\n  ✅ NSCF 计算完成")
    print(f"  📁 诊断文件已导出到: {OUTPUT_DIR}/")
    print(f"\n  下一步:")
    print(f"  1. 运行 python compare_qe.py 对比 QE 结果")
    print(f"  2. 检查诊断文件格式是否正确")
    print(f"  3. 验证本征值和能量的精度")

    return 0


if __name__ == "__main__":
    sys.exit(main())
