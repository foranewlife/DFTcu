#!/usr/bin/env python3
"""
处理 QE vloc_psi 调试输出，转换为标准测试数据格式
新版本：支持物理索引 (h,k,l) 和 (ix,iy,iz)
"""

import os

# 路径配置
QE_REF_DIR = "tests/data/qe_reference/sic_minimal"
OUTPUT_DIR = "tests/data/qe_reference/sic_minimal"


def process_psi_input():
    """处理输入波函数 psi(G) - 带米勒指数"""
    input_file = f"{QE_REF_DIR}/dftcu_debug_psi_input_to_vloc.txt"
    output_file = f"{OUTPUT_DIR}/vloc_psi_input.dat"

    data = []
    with open(input_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 6:
                band, h, k, l, re, im = parts
                # 转为 0-based band 索引
                data.append([int(band) - 1, int(h), int(k), int(l), float(re), float(im)])

    # 按 (band, h, k, l) 排序
    data = sorted(data, key=lambda x: (x[0], x[1], x[2], x[3]))

    with open(output_file, "w") as f:
        f.write("# DFTcu: Input wavefunction psi(G) to vloc_psi\n")
        f.write("# Format: band h k l Re(psi) Im(psi)\n")
        f.write("# Unit: dimensionless (Gamma-only, sqrt(2) normalized for G≠0)\n")
        f.write("# Source: QE vloc_psi_gamma.f90 (first call)\n")
        for band, h, k, l, re, im in data:
            f.write(f"{band:6d} {h:6d} {k:6d} {l:6d} {re:25.16e} {im:25.16e}\n")

    print(f"✓ 处理 vloc_psi_input.dat ({len(data)} 个数据点)")
    return len(data)


def process_psi_r():
    """处理实空间波函数 psi(r) - 带 (ix,iy,iz)"""
    input_file = f"{QE_REF_DIR}/dftcu_debug_psi_r_gamma.txt"
    output_file = f"{OUTPUT_DIR}/vloc_psi_r.dat"

    count = 0
    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            fout.write("# DFTcu: Wavefunction psi(r) after IFFT (Gamma-only packed)\n")
            fout.write("# Format: ir ix iy iz Re(psi) Im(psi)\n")
            fout.write("# Note: Packed two bands (psi1 + i*psi2)\n")
            fout.write("# Source: QE vloc_psi_gamma.f90 after wave_g2r\n")
            for line in fin:
                if not line.startswith("#"):
                    fout.write(line)
                    count += 1

    print(f"✓ 处理 vloc_psi_r.dat ({count} 个实空间点)")
    return count


def process_vpsi_r():
    """处理 V(r)·psi(r) - 带 (ix,iy,iz)"""
    input_file = f"{QE_REF_DIR}/dftcu_debug_vpsi_r.txt"
    output_file = f"{OUTPUT_DIR}/vloc_vpsi_r.dat"

    count = 0
    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            fout.write("# DFTcu: V(r)·psi(r) after multiplication\n")
            fout.write("# Format: ir ix iy iz Re(V·psi) Im(V·psi)\n")
            fout.write("# Source: QE vloc_psi_gamma.f90 after V(r) multiplication\n")
            for line in fin:
                if not line.startswith("#"):
                    fout.write(line)
                    count += 1

    print(f"✓ 处理 vloc_vpsi_r.dat ({count} 个实空间点)")
    return count


def process_vpsi_g_output():
    """处理输出 V_loc|psi>(G) - 带米勒指数"""
    input_file = f"{QE_REF_DIR}/dftcu_debug_vpsi_g_scaled.txt"
    output_file = f"{OUTPUT_DIR}/vloc_psi_output.dat"

    data = []
    with open(input_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 6:
                band, h, k, l, re, im = parts
                # 转为 0-based band 索引
                data.append([int(band) - 1, int(h), int(k), int(l), float(re), float(im)])

    # 按 (band, h, k, l) 排序
    data = sorted(data, key=lambda x: (x[0], x[1], x[2], x[3]))

    with open(output_file, "w") as f:
        f.write("# DFTcu: V_loc|psi> output in G-space (after fac scaling)\n")
        f.write("# Format: band h k l Re(Vloc*psi) Im(Vloc*psi)\n")
        f.write("# Unit: Ry (QE convention)\n")
        f.write("# Source: QE vloc_psi_gamma.f90 after wave_r2g and fac scaling\n")
        for band, h, k, l, re, im in data:
            f.write(f"{band:6d} {h:6d} {k:6d} {l:6d} {re:25.16e} {im:25.16e}\n")

    print(f"✓ 处理 vloc_psi_output.dat ({len(data)} 个数据点)")
    return data


def compute_energy(vpsi_data):
    """计算能量期望值 <psi|V_loc|psi>"""
    # 读取输入 psi
    psi_data = {}
    with open(f"{OUTPUT_DIR}/vloc_psi_input.dat", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            band, h, k, l, re, im = line.split()
            key = (int(band), int(h), int(k), int(l))
            psi_data[key] = complex(float(re), float(im))

    # 计算 <psi|V_loc|psi> = Σ_band Σ_G psi*(G) * (V_loc*psi)(G)
    # Gamma-only: G≠0 项乘以 2
    energy = 0.0
    for band, h, k, l, re, im in vpsi_data:
        key = (band, h, k, l)
        if key in psi_data:
            vpsi_val = complex(re, im)
            contrib = (psi_data[key].conjugate() * vpsi_val).real

            # Gamma-only: G≠0 项乘以 2
            if not (h == 0 and k == 0 and l == 0):
                contrib *= 2.0

            energy += contrib

    output_file = f"{OUTPUT_DIR}/vloc_psi_energy.dat"
    with open(output_file, "w") as f:
        f.write("# DFTcu: <psi|V_loc|psi> expectation value\n")
        f.write("# Unit: Ry\n")
        f.write("# Computed from psi_input and vloc_psi_output\n")
        f.write(f"{energy:.16e}\n")

    print(f"✓ 计算能量期望值: {energy:.10f} Ry ({energy*0.5:.10f} Ha)")
    return energy


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("处理 V_loc_psi QE 参考数据（带物理索引）")
    print("=" * 60)
    print()

    # 处理各个文件
    process_psi_input()
    process_psi_r()
    process_vpsi_r()
    vpsi_data = process_vpsi_g_output()
    compute_energy(vpsi_data)

    print()
    print("=" * 60)
    print("✓ 数据处理完成")
    print("=" * 60)
    print()
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    print("生成的文件:")
    for fname in [
        "vloc_psi_input.dat",
        "vloc_psi_r.dat",
        "vloci_r.dat",
        "vloc_psi_output.dat",
        "vloc_psi_energy.dat",
    ]:
        fpath = f"{OUTPUT_DIR}/{fname}"
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  - {fname:30s} ({size:>8d} bytes)")

    print()
    print("✅ 所有数据都包含物理索引:")
    print("  ✓ G 空间: Miller 指数 (h, k, l)")
    print("  ✓ 实空间: 网格坐标 (ix, iy, iz)")
    print("  ✓ 符合 tests/README.md 的 Index-Locked 规范")


if __name__ == "__main__":
    main()
