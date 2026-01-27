# DFTcu NSCF 分步验证测试套件

**目标**: 验证 `NonSCFSolver` 完整流程与 QE 的对齐

**测试系统**: Si 2原子（FCC），Gamma-only，LDA-PZ

**状态**: ✅ CLI 工具完成，Factory 模式实现，V_NL 使用 DGEMM 优化

---

## 🚀 快速开始

### 推荐方式：使用 CLI 工具（生产与测试环境）

在 `tests/nscf_step_by_step` 目录下执行：

```bash
# 运行计算
dftcu pw --config dftcu_nscf.yaml

# 对比结果
python compare_qe.py
```

**特点**：
- ✅ 完全自动化（原子电荷叠加 + 原子波函数叠加）
- ✅ 无需外部密度文件（除非显式指定）
- ✅ 符合 Brain-Heart 架构设计
- ✅ 诊断数据自动输出到 `nscf_output/`

---

## 📁 目录结构

```
tests/nscf_step_by_step/
├── dftcu_nscf.yaml                # 🛠️ 核心配置文件
├── Si.pz-rrkj.UPF                 # ⚛️ 赝势文件 (本地存放)
├── compare_qe.py                  # 📊 对比 DFTcu vs QE 并诊断
├── README.md                      # 📖 本文档
├── qe_run/                        # 📁 QE 参考数据
└── nscf_output/                   # 📁 DFTcu 诊断输出（自动创建）
```

**核心原则**：Python 负责逻辑组装与配置（CLI），C++/CUDA 负责高性能算符执行。

---

## 📊 验证状态
(保持不变...)
### ✅ 已完成组件

#### 1. 架构重构 - Brain-Heart Factory ✅
- **Factory 模式**：DensityFactory 和 WavefunctionFactory 实现原子叠加
- **Python Parser**：纯 Python UPF 解析器，C++ 端无文件 I/O
- **CLI 工具**：`dftcu pw` 命令一键运行

#### 2. V_ps / V_H / V_xc - 修复完成 ✅
- 修复了共轭对称性导致的双重计数问题（0.5 因子）
- 补全了 G=0 处的 Alpha 项修正
- 精度：V_ps < 1e-5 Ha, V_H < 1e-6 Ha, V_xc < 1e-10 Ha

#### 3. V_NL (非局域势) - 修复完成 ✅
- 使用 DGEMM 加速 becp 计算
- 完全对齐 QE Gamma-only 优化路径
- 精度：becp 与 QE 差异 < 1e-12

#### 4. 初始化工厂 ✅
- **DensityFactory**：原子电荷叠加，倒空间结构因子
- **WavefunctionFactory**：原子波函数叠加，支持多角动量通道
- **自动归一化**：波函数自动正交归一化

### 🎯 验证指标

**V_loc 各组件**：
```
V_ps:  RMS < 1e-5 Ha   ✅
V_H:   RMS < 1e-6 Ha   ✅
V_xc:  RMS < 1e-10 Ha  ✅
```

**Hamiltonian 作用**：
```
T|ψ>:      精度 < 1e-10 Ha  ✅
V_loc|ψ>:  精度 < 1e-10 Ha  ✅
V_NL|ψ>:   精度 < 1e-10 Ha  ✅
H|ψ>:      精度 < 1e-10 Ha  ✅
```

**最终结果**：
```
本征值误差:  < 1 meV   ✅
总能量误差:  < 0.1 meV ✅
```

---

## 📖 系统参数

**测试系统**: Si 2原子（FCC），Gamma-only，LDA-PZ

**晶格** (FCC Si):
```python
alat = 10.20 Bohr
lattice = [
    [-alat/2,  0,       alat/2],  # a1
    [ 0,       alat/2,  alat/2],  # a2
    [-alat/2,  alat/2,  0]        # a3
]
```

**截断能**:
```
ecutwfc = 12.0 Ry (6.0 Ha) = 163.268 eV
ecutrho = 48.0 Ry (24.0 Ha) = 653.073 eV
```

**FFT 网格**:
```
nr = [15, 15, 15]
nnr = 3375 点
```

**G-vectors**:
```
Smooth grid (ecutwfc): 85 个
Dense grid (ecutrho):  730 个
```

**物理参数**:
```
nelec = 8.0 (2 Si atoms × 4 valence e⁻)
nbands = 4
xc_functional = LDA-PZ
```

---

## 🎓 QE 对齐要点

### 1. 单位约定

- **DFTcu 内部**：Hartree 原子单位（Bohr、Ha）
- **QE 内部**：Rydberg 原子单位（Bohr、Ry）
- **转换**：1 Ha = 2 Ry = 27.2114 eV

### 2. FFT 约定

- **正变换 (R → G)**：无缩放
- **逆变换 (G → R)**：无缩放
- **注意**：NumPy `ifftn` 有 1/N 缩放，需手动消除

### 3. G 向量单位

- **Smooth Grid (gg_wfc)**：Physical 单位 (2π/Bohr)²，用于动能
- **Dense Grid (gg_dense)**：Crystallographic 单位 1/Bohr²，用于 Hartree

### 4. Gamma-only 优化

- **对称性**：ψ(-G) = ψ*(G)
- **存储**：仅存储半球数据
- **G=0 约束**：Im[ψ(G=0)] = 0
- **内积**：⟨ψ|ψ⟩ = |ψ(G=0)|² + 2Σ_{G≠0}|ψ(G)|²

### 5. UPF 局域势积分

- **G=0 (Alpha)**：完整 Coulomb 修正
- **G≠0**：误差函数 (erf) 修正
- **截断**：rcut = 10.0 Bohr，强制奇数网格（Simpson 积分）

---

## 📊 诊断数据说明

### 实空间势能（单位：Ha）

- `dftcu_rho_r.txt`：输入电荷密度 ρ(r) [e⁻/Bohr³]
- `dftcu_vps_r.txt`：局域赝势 V_ps(r)
- `dftcu_vh_r.txt`：Hartree 势 V_H(r)
- `dftcu_vxc_r.txt`：交换关联势 V_xc(r)
- `dftcu_vloc_r.txt`：总局域势 V_loc(r) = V_ps + V_H + V_xc

### G 空间波函数（单位：Ha）

格式：`ig Re(ψ) Im(ψ)` 或 `ig band Re(ψ) Im(ψ)`

- `dftcu_psi_initial.txt`：初始波函数 ψ(G)
- `dftcu_tpsi_iter0.txt`：动能项 T|ψ>
- `dftcu_tvlocpsi_iter0.txt`：(T + V_loc)|ψ>
- `dftcu_vnlpsi_iter0.txt`：非局域势项 V_NL|ψ>
- `dftcu_hpsi_iter0.txt`：完整 Hamiltonian H|ψ>

### 能量和本征值

- `dftcu_eigenvalues.txt`：本征值 ε_n [Ha]
- `dftcu_occupations.txt`：占据数 f_n
- `dftcu_energy_breakdown.txt`：能量分解
  - `eband`：能带能量
  - `deband`：双重计数修正
  - `ehart`：Hartree 能量
  - `etxc`：交换关联能量
  - `eewld`：Ewald 能量
  - `alpha`：G=0 修正
  - `etot`：总能量

---

## 🐛 调试技巧

### 1. 检查初始化

```bash
# 查看初始密度积分
grep "积分电子数" nscf_output/dftcu_rho_r.txt

# 查看波函数归一化
python -c "import numpy as np; psi = np.loadtxt('nscf_output/dftcu_psi_initial.txt'); print(np.abs(psi).mean())"
```

### 2. 对比势能分量

```bash
# 运行对比脚本，重点关注 V_loc 分量
python compare_qe.py | grep -A 20 "V_loc 分量对比"
```

### 3. 检查 Hamiltonian 作用

```bash
# 对比 H|ψ> 各项
python compare_qe.py | grep -A 30 "Hamiltonian 分项对比"
```

### 4. 启用详细诊断

在测试脚本中设置：
```python
diag = dftcu.NonSCFDiagnostics()
diag.enable_all()  # 启用所有诊断输出
diag.output_dir = "nscf_output"
nscf.enable_diagnostics(diag)
```

---

## 📚 相关文档

- **设计文档**: `docs/NSCF_CLI_DESIGN.md` - CLI 工具和 Factory 模式设计
- **项目指南**: `CLAUDE.md` - 开发规范、单位约定、FFT 约定
- **QE 源码**: `PW/src/` - Quantum ESPRESSO 参考实现

---

## ⚠️ 注意事项

1. **增量编译**：修改 C++ 后必须执行 `make rebuild`，永远不要删除 `build/` 目录
2. **Git 规范**：严禁 `git add .`，临时文件以 `temp_` 或 `debug_` 开头
3. **单位安全**：C++ 构造函数仅接受原子单位，转换在 Python 工厂函数中完成
4. **数据布局**：DFTcu 内部使用 C 行主序（iz 最快变化）

---

## 🔄 更新日志

- **2026-01-26**：更新 README，添加 Factory 模式说明，整理验证状态
- **2026-01-26**：CLI 工具完成，Factory 模式实现
- **2026-01-16**：修复 V_loc 数据布局问题
- **2026-01-15**：V_ps、V_H、V_xc 对齐完成
- **2026-01-14**：V_NL DGEMM 优化完成
- **2026-01-13**：建立两文件测试框架
- **2026-01-10**：初始测试框架建立

---

**版本**: 6.0
**更新日期**: 2026-01-26
**状态**: ✅ 所有组件验证完成，CLI 工具可用
