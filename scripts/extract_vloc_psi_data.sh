#!/bin/bash
# ============================================================================
# 从 QE 运行中提取 V_loc_psi 参考数据
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QE_DIR="$PROJECT_ROOT/external/qe"
TEST_DIR="$PROJECT_ROOT/tests/nscf_step_by_step"
OUTPUT_DIR="$PROJECT_ROOT/tests/data/qe_reference/sic_minimal"

echo "=========================================="
echo "提取 V_loc_psi QE 参考数据"
echo "=========================================="

# 检查 QE 是否已编译
if [ ! -f "$QE_DIR/build/bin/pw.x" ]; then
    echo "错误: QE 未编译，请先运行: cd external/qe && cmake --build build"
    exit 1
fi

# 检查测试输入文件
if [ ! -f "$TEST_DIR/qe_run/sic_minimal.in" ]; then
    echo "错误: 找不到 sic_minimal.in"
    exit 1
fi

# 创建临时运行目录
RUN_DIR="$TEST_DIR/qe_run/vloc_psi_extract"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# 复制输入文件
cp "$TEST_DIR/qe_run/sic_minimal.in" ./pw.in

# 复制赝势文件
mkdir -p pseudo
cp "$TEST_DIR"/*.UPF pseudo/ 2>/dev/null || true

echo ""
echo "运行 QE pw.x (NSCF 计算)..."
echo "工作目录: $RUN_DIR"
echo ""

# 运行 QE (只运行一次，vloc_psi 会在第一次调用时输出数据)
"$QE_DIR/build/bin/pw.x" < pw.in > pw.out 2>&1

echo ""
echo "QE 运行完成，检查输出文件..."
echo ""

# 检查是否生成了调试文件
EXPECTED_FILES=(
    "dftcu_debug_psi_input_to_vloc.txt"
    "dftcu_debug_psi_r_gamma.txt"
    "dftcu_debug_vpsi_r.txt"
    "dftcu_debug_vpsi_g_raw.txt"
    "dftcu_debug_vpsi_g_scaled.txt"
)

MISSING_FILES=()
for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "警告: 以下文件未生成:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "请检查 QE 输出:"
    tail -50 pw.out
    exit 1
fi

echo "✓ 所有调试文件已生成"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 转换数据格式为标准索引格式
echo "转换数据格式..."

# 1. psi_input: (ig, ibnd, Re, Im) -> (band, ig, Re, Im)
python3 << 'EOF'
import sys
import numpy as np

# 读取 QE 输出 (ig, ibnd, Re, Im)
data = []
with open('dftcu_debug_psi_input_to_vloc.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 4:
            ig, ibnd, re, im = parts
            data.append([int(ibnd)-1, int(ig)-1, float(re), float(im)])  # 转为 0-based

# 按 (band, ig) 排序
data = sorted(data, key=lambda x: (x[0], x[1]))

# 输出标准格式
with open('psi_g_input.dat', 'w') as f:
    f.write('# DFTcu: Input wavefunction psi(G) to vloc_psi\n')
    f.write('# Format: band ig Re(psi) Im(psi)\n')
    f.write('# Unit: dimensionless (Gamma-only, sqrt(2) normalized for G≠0)\n')
    for band, ig, re, im in data:
        f.write(f'{band:6d} {ig:6d} {re:25.16e} {im:25.16e}\n')

print('✓ 转换 psi_g_input.dat')
EOF

# 2. psi_r: (ir, Re, Im)
python3 << 'EOF'
with open('dftcu_debug_psi_r_gamma.txt', 'r') as fin:
    with open('psi_r.dat', 'w') as fout:
        fout.write('# DFTcu: Wavefunction psi(r) after IFFT (Gamma-only packed)\n')
        fout.write('# Format: ir Re(psi) Im(psi)\n')
        fout.write('# Note: Packed two bands (psi1 + i*psi2)\n')
        for line in fin:
            if not line.startswith('#'):
                fout.write(line)

print('✓ 转换 psi_r.dat')
EOF

# 3. vpsi_r: (ir, Re, Im)
python3 << 'EOF'
with open('dftcu_debug_vpsi_r.txt', 'r') as fin:
    with open('vpsi_r.dat', 'w') as fout:
        fout.write('# DFTcu: V(r)·psi(r) after multiplication\n')
        fout.write('# Format: ir Re(V·psi) Im(V·psi)\n')
        for line in fin:
            if not line.startswith('#'):
                fout.write(line)

print('✓ 转换 vpsi_r.dat')
EOF

# 4. vpsi_g_output: (ig, band, Re, Im) -> (band, ig, Re, Im)
python3 << 'EOF'
# 使用 scaled 版本（已应用 fac 因子）
data = []
with open('dftcu_debug_vpsi_g_scaled.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 4:
            ig, band, re, im = parts
            data.append([int(band)-1, int(ig)-1, float(re), float(im)])  # 0-based

# 按 (band, ig) 排序
data = sorted(data, key=lambda x: (x[0], x[1]))

with open('vloc_psi_g_output.dat', 'w') as f:
    f.write('# DFTcu: V_loc|psi> output in G-space (after fac scaling)\n')
    f.write('# Format: band ig Re(Vloc*psi) Im(Vloc*psi)\n')
    f.write('# Unit: Ry (QE convention)\n')
    for band, ig, re, im in data:
        f.write(f'{band:6d} {ig:6d} {re:25.16e} {im:25.16e}\n')

print('✓ 转换 vloc_psi_g_output.dat')
EOF

# 5. 计算能量期望值 <psi|V_loc|psi>
python3 << 'EOF'
import numpy as np

# 读取 psi_input 和 vloc_psi_output
psi_data = {}
vpsi_data = {}

with open('psi_g_input.dat', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        band, ig, re, im = line.split()
        key = (int(band), int(ig))
        psi_data[key] = complex(float(re), float(im))

with open('vloc_psi_g_output.dat', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        band, ig, re, im = line.split()
        key = (int(band), int(ig))
        vpsi_data[key] = complex(float(re), float(im))

# 计算 <psi|V_loc|psi> = Σ_band Σ_ig psi*(ig) * (V_loc*psi)(ig)
# 注意: Gamma-only 需要考虑 G≠0 的因子 2
energy = 0.0
for key in psi_data:
    band, ig = key
    if key in vpsi_data:
        # psi* · (V_loc*psi)
        contrib = (psi_data[key].conjugate() * vpsi_data[key]).real

        # Gamma-only: G≠0 项乘以 2 (因为 -G 对称)
        if ig > 0:
            contrib *= 2.0

        energy += contrib

with open('vloc_psi_energy.dat', 'w') as f:
    f.write('# DFTcu: <psi|V_loc|psi> expectation value\n')
    f.write('# Unit: Ry\n')
    f.write(f'{energy:.16e}\n')

print(f'✓ 计算能量期望值: {energy:.6f} Ry')
EOF

# 移动文件到输出目录
mv psi_g_input.dat "$OUTPUT_DIR/"
mv psi_r.dat "$OUTPUT_DIR/"
mv vpsi_r.dat "$OUTPUT_DIR/"
mv vloc_psi_g_output.dat "$OUTPUT_DIR/"
mv vloc_psi_energy.dat "$OUTPUT_DIR/"

echo ""
echo "=========================================="
echo "✓ 数据提取完成"
echo "=========================================="
echo ""
echo "输出文件位置: $OUTPUT_DIR"
echo ""
find "$OUTPUT_DIR" -name "*.dat" -type f -exec ls -lh {} \; | tail -5
echo ""
echo "下一步: 编写测试代码验证 DFTcu 的 V_loc_psi 实现"
