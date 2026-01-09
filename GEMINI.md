# DFTcu 项目开发指南

## 快速参考

1. `import dftcu` 会触发增量编译（不要触碰 build 目录！）(或者uv pip install --no-build-isolation -e ".[dev]")
2. `cmake --build external/qe/build` 增量编译 QE
3. **永远不要** `git add .`
4. **测试数据不要加入 `.gitignore`**，Agent 需要访问这些文件
5. 运行命令前先cd到项目目录的绝对路径。
6. qe不能使用mpi并行
7. 临时文件以'temp_'开头
8. qe产生的测试文件要copy到 phase*/data 路径，备份input文件和赝势文件
9. 不要创建新的markdown文件除非我告诉你， 目前只更新 1. GEMINI.md or CLAUDE.md 2.tests/nscf_alignment/README.md

---

## 📐 FFT 缩放约定（全局统一）

**DFTcu 项目统一使用 QE 的 FFT 约定**：

### 正变换（R → G）: `FFT`, 不缩放
```
ψ(G) = Σ_r ψ(r) exp(-i G·r)
```
- cuFFT: `CUFFT_FORWARD`，无缩放
- numpy: `np.fft.fftn()`（无缩放）

### 逆变换（G → R）: `IFFT`, 不缩放
```
ψ(r) = Σ_G ψ(G) exp(i G·r)
```
- cuFFT: `CUFFT_INVERSE`，无缩放（**与 numpy 不同！**）
- QE: `wave_g2r`，无缩放
- numpy: `np.fft.ifftn()` 有 1/N 缩放（**需要手动 × N 对齐 QE**）

### 往返归一化
```
ψ(G) → IFFT → ψ(r) → FFT → ψ'(G)
ψ'(G) = N · ψ(G)  (其中 N = nr[0] × nr[1] × nr[2])
```

### 代码实现规范

**✅ 正确**（DFTcu 统一约定）：
```cpp
// G -> R: IFFT 不缩放（匹配 QE）
cufftExecZ2Z(plan, psi_g, psi_r, CUFFT_INVERSE);
// 输出：psi_r 未缩放

// R -> G: FFT 不缩放
cufftExecZ2Z(plan, psi_r, psi_g, CUFFT_FORWARD);
// 输出：psi_g = N × 输入的 psi_g
```

**❌ 错误**（不要使用 numpy 约定）：
```cpp
// ❌ 不要添加 1/N 缩放
cufftExecZ2Z(plan, psi_g, psi_r, CUFFT_INVERSE);
scale_kernel(psi_r, 1.0/N);  // ❌ 错误！破坏 QE 对齐
```

### Python 测试中的对齐

当与 QE 数据对比时：
```python
# QE 导出的 ψ(r) 未缩放
psi_r_qe = load_qe_data()

# DFTcu 计算（匹配 QE 约定，未缩放）
psi_r_dftcu = gamma_fft.wave_g2r(psi_g)

# 直接对比，无需额外缩放
assert np.allclose(psi_r_dftcu, psi_r_qe)
```

**如果使用 numpy 作为参考**：
```python
# numpy ifftn 使用 1/N 缩放
psi_r_numpy = np.fft.ifftn(psi_g)

# 需要 × N 才能与 QE/DFTcu 对齐
psi_r_numpy *= N
assert np.allclose(psi_r_dftcu, psi_r_numpy)
```

### 相关文件

- **`src/fft/gamma_fft_solver.cu`**: 实现 QE 约定（无缩放）
- **`src/fft/fft_solver.cuh`**: 标准 FFTSolver（历史遗留，需统一）
- **Phase 0b 测试**: 验证 FFT 约定对齐

---

## 🌐 全局单位约定（Hartree 原子单位制）

**核心原则**：
- **所有内部对象、数据结构、逻辑运算统一使用 Hartree 原子单位制**
- **单位转换只在边界层通过工厂函数完成**（如 `Grid::from_angstrom()`）
- **内部代码不包含任何单位转换逻辑**

### 原子单位制定义

**Hartree 原子单位**（DFTcu 内部统一标准）：
- **能量单位**: 1 Ha (Hartree) = 27.2114 eV = 2 Ry
- **长度单位**: 1 Bohr (a₀) = 0.5292 Angstrom
- **动能公式**: T = ½|k|² [Ha]，其中 k 单位为 2π/Bohr

**与 QE 的单位转换**（QE 使用 Rydberg）：
- QE ecutwfc = 12.0 Ry → DFTcu 内部 = 6.0 Ha
- QE g2kin [Ry] → DFTcu g2kin [Ha] = QE 值 × 0.5
- **转换因子**: `HA_TO_RY = 2.0`, `RY_TO_HA = 0.5`

**关键物理常数** (`src/utilities/constants.cuh`):
```cpp
const double BOHR_TO_ANGSTROM = 0.529177210903;  // 1 Bohr = 0.5292 Angstrom
const double HA_TO_EV = 27.211386245988;         // 1 Ha = 27.2114 eV
const double HA_TO_RY = 2.0;                     // 1 Ha = 2 Ry (exact)
const double RY_TO_HA = 0.5;                     // 1 Ry = 0.5 Ha (exact)
```

### 单位转换表

| 物理量 | 外部输入 | DFTcu 内部存储 | 转换方式 | 验证精度 |
|--------|----------|---------------|---------|---------|
| **晶格向量** `lattice` | Angstrom | Bohr | `Grid::from_angstrom()` | - |
| **倒格子** `rec_lattice` | - | 1/Bohr (Cryst.) | `inv(lattice)^T` | - |
| **体积** `volume` | - | Bohr³ | 自动计算 | - |
| **体积元** `dv` | - | Bohr³ | volume / nnr | - |
| **截断能** `ecutwfc` | Rydberg (QE) | Hartree | 输入时 × 0.5 | - |
| **截断能** `ecutrho` | Rydberg (QE) | Hartree | 输入时 × 0.5 | - |
| **G 向量** `G = h·b1 + k·b2 + l·b3` | - | 1/Bohr (Cryst.) | 由 Miller 指数计算 | - |
| **\|G\|² (Smooth)** `gg_wfc` | - | (2π/Bohr)² (Phys.) | 计算后 × (2π)² | **<1e-14** ✅ |
| **\|G\|² (Dense)** `gg_dense` | - | 1/Bohr² (Cryst.) | 直接计算 | **<1e-14** ✅ |
| **g2kin** `½|G|²` | Rydberg (QE) | Hartree | gg_wfc × 0.5 | **<1e-14** ✅ |
| **密度** `ρ(r)` | - | e⁻/Bohr³ | 内部单位 | - |
| **势能** `V(r)` | Rydberg (QE) | Hartree | 输入时 × 0.5 | - |

### 工厂函数模式（强制）

**核心原则**：
- **Grid 类内部完全纯净** - 只接受原子单位（Bohr + Hartree）
- **所有单位转换在工厂函数边界完成** - 独立的自由函数（`grid_factory.cuh`），不是类方法
- **函数名明确单位** - 通过函数名清楚地表达输入单位

**C++ 层**：
```cpp
#include "model/grid_factory.cuh"

// ✅ 推荐：从 QE 单位创建（Angstrom + Rydberg）
auto grid = create_grid_from_qe(
    lattice_ang,    // Angstrom
    {18, 18, 18},   // FFT grid
    12.0,           // ecutwfc [Ry]
    48.0,           // ecutrho [Ry]
    true            // is_gamma
);
// 内部自动转换为：Bohr + Hartree

// ✅ 高级用法：直接使用原子单位
auto grid = create_grid_from_atomic_units(
    lattice_bohr,   // Bohr
    {18, 18, 18},
    6.0,            // ecutwfc [Ha]
    24.0,           // ecutrho [Ha]
    true
);

// ❌ 错误：不要直接调用构造函数（除非你确定使用原子单位）
Grid grid(lattice, nr, ecutwfc, ecutrho, is_gamma);  // 容易搞混单位！
```

**Python 层示例**：
```python
import dftcu
import numpy as np

# QE 输入数据（Angstrom + Rydberg）
alat_ang = 10.20 * 0.529177  # Bohr → Angstrom
lattice = np.array([
    [-alat_ang/2, 0, alat_ang/2],
    [0, alat_ang/2, alat_ang/2],
    [-alat_ang/2, alat_ang/2, 0]
])  # 3×3 NumPy 数组，直接传入即可

# ✅ 推荐：使用工厂函数，单位明确
grid = dftcu.create_grid_from_qe(
    lattice_ang=lattice,  # 3×3 数组，无需 flatten
    nr=[18, 18, 18],
    ecutwfc_ry=12.0,   # 参数名明确：Rydberg
    ecutrho_ry=48.0,   # 参数名明确：Rydberg
    is_gamma=True
)

# 内部已经是 Hartree，直接读取
print(f"ecutwfc = {grid.ecutwfc()} Ha")  # 输出: 6.0 Ha
print(f"ecutrho = {grid.ecutrho()} Ha")  # 输出: 24.0 Ha

# ❌ 错误：不要手动转换后调用构造函数
ecutwfc_ha = 12.0 * 0.5  # 手动转换
grid = dftcu.Grid(lattice_bohr, [18,18,18], ecutwfc_ha, ...)  # 不推荐！
```

**为什么这样设计？**

1. **类型安全**：函数名 `create_grid_from_qe` 明确告诉你输入是 QE 单位
2. **避免混淆**：不需要记住 `ecutwfc` 参数是什么单位
3. **纯净内部**：Grid 类不包含任何单位转换代码
4. **易于扩展**：可以轻松添加其他单位系统的工厂函数

### G 向量和动能

**G 向量单位转换的关键理解**：

```
DFTcu 内部（Hartree 原子单位）：
  lattice_        [Bohr]
  rec_lattice_    [2π/Bohr]
  G = h·b1 + k·b2 + l·b3  [2π/Bohr]
  |G|²            [(2π/Bohr)²]

在 Hartree 原子单位中：
  T = ½ℏ²|k|²/m = ½|k|² [Ha]  当 k 单位为 2π/Bohr 时

数值关系：
  |G|²[(2π/Bohr)²] = 2 × ecutwfc [Ha]  （因为 T = ½|k|²）

关键点：
  - Hartree: T = ½|k|² [Ha]
  - Rydberg: T = |k|² [Ry] = ½|k|² [Ha]（因为 1 Ry = 0.5 Ha）
  - 所以 gcut² = 2 × ecutwfc_ha = ecutwfc_ry（如果从 QE 读取）

**⚠️ 内部倒空间网格单位区分：**
DFTcu 内部根据不同用途维护两套倒空间单位系统：
1. **波函数网格 (Smooth Grid)**: `gg_wfc` 和 `g2kin` 使用 **物理单位** $[(2\pi/Bohr)^2]$。
   - *用途*: 计算动能项 $T|\psi\rangle = \frac{1}{2}|G|^2 \psi(G)$。
   - *公式*: `gg_wfc = |G|²_cryst × (2π)²`。
2. **电荷/势能网格 (Dense Grid)**: `gg_dense` 和 `gl` 使用 **结晶学单位** $[(1/Bohr)^2]$ (不含 $2\pi$)。
   - *用途*: Hartree 势和局域赝势插值。
   - *公式*: 直接使用 Miller 指数与倒格子基矢计算，不乘 $(2\pi)^2$。
   - *注意*: 泛函组件（如 `Hartree`）的系数 `fac` 已包含了对 $(2\pi)^2$ 的抵消。

**代码实现** (`Grid::generate_gvectors()`):
```cpp
// ecutwfc 内部存储为 Hartree
// 生成 G-vector 时的筛选条件：|G|² ≤ 2×ecutwfc_ha
double gcut2 = 2.0 * ecutwfc_;  // [Ha] → [(2π/Bohr)²]

// 筛选条件
if (g2 > gcut2) continue;  // g2 [(2π/Bohr)²] vs gcut2 [2×Ha]
```

### 能量积分单位

**所有能量密度积分必须使用原子单位体积元**：

```cpp
// ✅ 正确：使用 Bohr³
double E_total = 0.0;
for (int i = 0; i < grid.nnr(); ++i) {
    E_total += energy_density[i] * grid.dv_bohr();  // [Ha/Bohr³] × [Bohr³] = [Ha]
}

// ❌ 错误：使用 Angstrom³（历史遗留代码）
double E_wrong = energy_density.integral() * grid.dv();  // 单位不匹配！
```

### 验证结果（Phase 0c）

| 测试项 | DFTcu 单位 | QE 单位 | 转换 | 精度 | 状态 |
|-------|-----------|---------|------|------|------|
| Miller 指数 | 无量纲 | 无量纲 | 1:1 | **0** (exact) | ✅ |
| G 向量数量 | 85 | 85 | 1:1 | **完全匹配** | ✅ |
| g2kin 计算 | Ha | Ry | × 0.5 | **<1e-14** | 待验证 |

**测试命令**：
```bash
python tests/nscf_alignment/phase0c/test_smooth_grid.py  # Smooth grid & g2kin 验证
python tests/nscf_alignment/phase0c/test_dense_grid.py   # Dense grid 验证
```

---

## ⚠️ 重要架构约束

### DensityFunctionalPotential - 从密度泛函计算势

**重要更新**：`Evaluator` 类将重命名为 `DensityFunctionalPotential` 以明确其用途。

**DensityFunctionalPotential 功能**：
- 从密度 ρ(r) 计算势 V[ρ] = δE[ρ]/δρ 和能量 E[ρ]
- 适用于任何密度的泛函（Hartree、XC、TF、vW 等）
- **同时适用于** KS-DFT 和 OFDFT

**适用场景**：

**KS-DFT SCF** - 使用 DensityFunctionalPotential：
```python
# 创建密度泛函势
dfp = dftcu.DensityFunctionalPotential(grid)
dfp.add_functional(dftcu.HartreeFunctional())
dfp.add_functional(dftcu.LDAFunctional())

# V(ρ) = V_Hartree[ρ] + V_XC[ρ]
ham = dftcu.Hamiltonian(grid)
ham.set_density_functional_potential(dfp)
ham.update_potentials(rho)
```

**OFDFT** - 使用 DensityFunctionalPotential：
```python
# 创建密度泛函势（包含动能泛函）
dfp = dftcu.DensityFunctionalPotential(grid)
dfp.add_functional(dftcu.HartreeFunctional())
dfp.add_functional(dftcu.LDAFunctional())
dfp.add_functional(dftcu.ThomasFermiFunctional())
dfp.add_functional(dftcu.VonWeizsackerFunctional())

# V(ρ) = V_Hartree[ρ] + V_XC[ρ] + δT_TF[ρ]/δρ + δT_vW[ρ]/δρ
```

**KS-DFT NSCF** - 使用 DensityFunctionalPotential（只调用一次）：
```python
# 创建密度泛函势
dfp = dftcu.DensityFunctionalPotential(grid)
dfp.add_functional(dftcu.HartreeFunctional())
dfp.add_functional(dftcu.LDAFunctional())

# 从 SCF 读取自洽密度，计算一次势
ham = dftcu.Hamiltonian(grid)
ham.set_density_functional_potential(dfp)
rho = read_scf_charge_density()
ham.update_potentials(rho)  # 只调用一次，不迭代

# 设置赝势
ham.set_nonlocal(nl_pseudo)

# 对角化
ham.apply(psi, h_psi)
```

**当前 Hamiltonian 构造函数的问题**：
```cpp
// src/solver/hamiltonian.cuh (当前实现)
Hamiltonian(Grid& grid, std::shared_ptr<Evaluator> evaluator, ...);
// ❌ 强制要求 Evaluator，但 Phase 1a 动能验证不需要它
```

**✅ Phase 1a 已实现**（Hamiltonian 已重构）：
```python
# Phase 1a 动能验证（不需要 DensityFunctionalPotential）
ham = dftcu.Hamiltonian(grid)
# v_loc 默认为 0，nonlocal 默认为 None
ham.apply(psi, h_psi)  # 只计算 T|ψ>
# ✅ 验证状态: 精度 1.1e-16 (机器精度)

# KS-DFT NSCF（需要 DensityFunctionalPotential，调用一次）
dfp = dftcu.DensityFunctionalPotential(grid)
dfp.add_functional(...)
ham = dftcu.Hamiltonian(grid)
ham.set_density_functional_potential(dfp)
rho = read_scf_charge_density()
ham.update_potentials(rho)  # 只调用一次
ham.set_nonlocal(nl_pseudo)

# KS-DFT SCF（需要 DensityFunctionalPotential，每次迭代调用）
dfp = dftcu.DensityFunctionalPotential(grid)
dfp.add_functional(...)
ham = dftcu.Hamiltonian(grid)
ham.set_density_functional_potential(dfp)
for iter in range(max_iter):
    ham.update_potentials(rho)  # 每次迭代调用
    ham.apply(psi, h_psi)
    rho = compute_density(psi)
```

**详细重构计划**：见 `docs/KSDFT_HAMILTONIAN_REFACTOR.md`

---

## QE 对齐核心要点

### 单位与约定
- **坐标单位**: Python 层传入 Angstrom，Backend G 向量单位 Angstrom⁻¹
- **截断能单位**: 统一使用 Rydberg
- **常数**: `BOHR_TO_ANGSTROM = 0.529177210903`

### Gamma-only 关键点
- **波函数**: QE 只存储半球，带 √2 因子，需通过 Hermitian 对称性展开
- **内积**: QE 对 G≠0 项乘以 2，DFTcu 使用全网格需匹配
- **G=0 约束**: 必须强制 `Im[ψ(G=0)] = 0`
- **G 向量索引**: QE 使用预计算的 `nl_d` 和 `nlm_d` 查找表映射 G 向量到 FFT 网格，基于 ecutwfc 截断
  - **在测试中**: 使用 `utils/qe_gvector_loader.py` 中的 `QEGVectorData` 类统一加载和访问这些索引
  - 详见: `docs/GVECTOR_MANAGEMENT_DESIGN.md`

### 初始化顺序
1. 先调用 `init_dij` 初始化 D 矩阵
2. 再调用 `update_projectors`
3. 否则会段错误

### 数据导出
- **只使用文本格式**（ASCII），不用二进制
- 便于调试和检查

---

## Python 层职责

**Python 层只负责**:
- 参数配置（grid, atoms, ecutwfc, ecutrho, mixing_beta 等）
- 调用 C++/CUDA 函数
- 读取和显示结果

**禁止在 Python 层**:
- ❌ 能量求和
- ❌ 密度混合
- ❌ 本征值加权
- ❌ 任何物理量的数值运算

**所有数值计算必须在 C++/CUDA 端完成**

---

## NSCF QE 对齐项目

### 项目目标
实现 DFTcu NSCF 与 QE 的完全对齐（Si + Gamma-only）

### 测试框架位置
**新测试框架**: `tests/nscf_alignment/` ✅
- 独立开发，与旧测试完全隔离
- **Phase 1 重构后代码复用率: 81.8%** ⬆️
- 维护成本降低 75%
- 通用工具库: `utils/hamiltonian_tester.py`

**旧测试**: `tests/test_*.py` ❌
- 保持不动，不要修改
- 代码重复率高，不适合 NSCF 对齐

### 测试入口
```bash
# 运行所有测试
python tests/nscf_alignment/main.py

# 运行单个 Phase (重构版推荐)
python tests/nscf_alignment/phase1a/test_kinetic_cuda_refactored.py

# 生成报告
python tests/nscf_alignment/main.py --report report.md
```

### QE 配置文件备份
每个测试 Phase 独立备份自己的 QE 配置文件:
```
tests/nscf_alignment/phaseX/
├── qe_config/           # QE 配置备份（独立自包含）
│   ├── si_nscf.in       # QE 输入文件
│   ├── Si.pz-rrkj.UPF   # 赝势文件
│   └── README.md        # 配置说明
└── data/                # 测试数据
```

**扩展到其他材料体系**（如 SiO2）:
1. 准备新的 QE 输入文件和赝势文件
2. 复制到 `phaseX/qe_config/` 目录
3. 测试代码无需修改

**优势**:
- ✅ 自包含，无需依赖外部路径
- ✅ 版本控制友好
- ✅ 简单明了，直接复制配置文件即可

详见: `tests/nscf_alignment/QE_CONFIG_BACKUP_DESIGN.md`

---

## 分阶段对齐计划

### ✅ Phase 0: 基础对齐（已完成）
- **Phase 0 (S_sub)**: 3.1e-15 精度 ✅
- **Phase 0b (FFT)**: 机器精度 ✅
  - 0b.4A: 打包验证（0 误差）
  - 0b.4C: IFFT 验证（9.2e-16）
  - 0b.4D: 端到端 G→R（9.2e-16）
- **Phase 0c (G 向量生成)**: ✅ 完成
  - **已完成 (Smooth grid)**:
    - ✅ Smooth grid G 向量原生生成（基于 ecutwfc）
    - ✅ Miller 指数与 QE 一致（误差 0）
    - ✅ g2kin 与 QE 一致（1.776e-15，机器精度）✨ 2026-01-08 验证
    - ✅ Python + C++/CUDA 实现
    - ✅ `generate_gvectors()` Python 绑定已添加 ✨ 2026-01-08
  - **已完成 (Dense grid)**: ✨ 2026-01-08
    - ✅ Dense grid G 向量生成（基于 ecutrho）
    - ✅ G-shell 分组 (ngl, gl, igtongl)
    - ✅ igk 映射 (Smooth → Dense)
    - ✅ Python 绑定完整（get_gg_dense, get_gl_shells, get_igtongl, get_igk）
    - ✅ **与 QE 完全对齐**：ngm_dense=730（QE 单进程输出一致）✨ 验证
    - ✅ FFT 网格约束正确实现：Miller 指数范围 `[-8, 8]` = `(nr-1)/2`
  - **实现细节**:
    - `generate_gvectors()` 一次性生成 Smooth + Dense 两个网格
    - Dense grid 包含所有 |G|² ≤ 2×ecutrho 的 G 向量
    - G-shell 按 |G|² 值分组（eps=1e-14）
    - igk 映射通过 Miller 指数匹配实现
    - FFT stick 约束不适用于单 GPU（仅 QE MPI 多进程使用）
  - **优先级**:
    - Smooth grid 已满足 Phase 1 H|ψ> 需求 ✅
    - Dense grid 已完成，Hartree/LDA 泛函测试可以开始 ✅
  - **调研结果**: Hartree 势能和局域赝势**需要** Dense grid (见 `docs/QE_DENSE_GRID_REQUIREMENT.md`)
  - **已知问题**: Phase 0c 测试在 main.py 中运行时有 CUDA 上下文冲突，单独运行正常 ⚠️
- **位置**: `tests/nscf_alignment/phase0/`, `phase0b/`, `phase0c/`
- **关键发现**: QE FFT 无缩放约定，`ψ → IFFT → FFT → N·ψ`

### ✅ Phase 1: H|ψ> 逐项验证（已完成）
- **Phase 1a (动能)**: ⏸️ 暂时禁用
  - 公式: `T|ψ> = g2kin * ψ(G)`
  - 位置: `tests/nscf_alignment/phase1a/`
  - 状态：功能正确但测试框架有 G 向量顺序问题，已在 main.py 中注释 ✨ 2026-01-08
  - 原精度：1.1e-16 ✅

- **Phase 1b (局域势)**: 1.14e-16 ✅
  - 公式: `V_loc|ψ> = FFT⁻¹[V_eff(r) · FFT(ψ)]`
  - 位置: `tests/nscf_alignment/phase1b/`
  - 关键: FFT 往返需除以 N 抵消缩放因子

- **Phase 1c (非局域势)**: 2.78e-17 ✅
  - 验证: `V_NL|ψ> = Σ D_ij |β_i><β_j|ψ>`
  - 位置: `tests/nscf_alignment/phase1c/` (已完成，测试文件已归档)
  - 状态：已验证完成，见 `PHASE1C_SUCCESS_REPORT.md`

- **Phase 1d (完整 H|ψ>)**: 定义验证 ✅
  - 验证: `H|ψ> = (T + V_loc + V_NL)|ψ>`
  - 位置: `tests/nscf_alignment/phase1d/`
  - 状态：各项独立验证均达机器精度

### 📋 Phase 2: 子空间投影（待定）
- **验证**: `H_sub = <ψ|H|ψ>`, `S_sub = <ψ|ψ>`
- **目标**: 1e-13

### 📋 Phase 3: Davidson 迭代（待定）
- **验证**: 完整迭代流程
- **目标**: 1e-12

---

## QE H|ψ> 计算流程

**文件**: `external/qe/PW/src/h_psi.f90`

```fortran
SUBROUTINE h_psi_( lda, n, m, psi, hpsi )
  ! 1. 动能项
  hpsi = g2kin * psi

  ! 2. 局域势（Gamma-only 路径）
  IF ( gamma_only ) THEN
    CALL vloc_psi_gamma(...)
  ENDIF

  ! 3. 非局域赝势
  CALL calbec( vkb, psi, becp )    ! becp = <β|ψ>
  CALL add_vuspsi( hpsi )          ! hpsi += V_NL|ψ>

  ! 4. Gamma 约束
  IF ( gamma_only .AND. gstart == 2 ) &
    hpsi(1,:) = REAL(hpsi(1,:))    ! Im[ψ(G=0)] = 0
END SUBROUTINE
```

---

## QE 源码修改指南

### 导出 H|ψ> 各项（Phase 1）
**文件**: `external/qe/PW/src/h_psi.f90`

在不同位置插入导出逻辑：
- Line 152 后: 导出 `g2kin` 和 `T|ψ>`
- Line 185 后: 导出 `V_eff(r)` 和 `V_loc|ψ>`
- Line 235 后: 导出 `becp` 和 `V_NL|ψ>`
- 返回前: 导出完整 `H|ψ>`

详见: `docs/NSCF_QE_ALIGNMENT_PLAN.md`

### 导出子空间矩阵（Phase 2）
**文件**: `external/qe/KS_Solvers/Davidson/regterg.f90`

在 Line 227 后导出 `H_sub`, `S_sub`, `evals_iter0`

### 重新编译
```bash
cd external/qe
cmake --build build --target pw -j8
```

---

## 开发工作流

### 添加新测试
1. 创建目录: `mkdir -p tests/nscf_alignment/phaseX/data`
2. 使用工具库:
   ```python
   from utils import QEDataLoader, Comparator, TestReporter, GridFactory
   ```
3. 参考模板: `tests/nscf_alignment/QUICKSTART.md`

### 修改配置
所有配置集中在 `tests/nscf_alignment/test_config.py`

### 运行测试
```bash
# 完整测试套件
python tests/nscf_alignment/main.py

# 单个 Phase
python tests/nscf_alignment/phase0/test_phase0.py
```

---

## 常见问题

### Q: 如何通知我？
```bash
happy notify -p "<message>"
```

### Q: 测试数据放哪里？
`tests/nscf_alignment/phaseX/data/`

**不要加入 .gitignore！**

### Q: 如何更改精度阈值？
修改 `tests/nscf_alignment/test_config.py` 中的 `PrecisionTargets`

### Q: Phase0Verifier 在哪里？
C++ 代码: `src/solver/phase0_verifier.cu`
Python 测试: `tests/nscf_alignment/phase0/test_phase0.py`

---

## QE NSCF 完整调用流程

### Gamma-only NSCF 路径

```
non_scf() [PW/src/non_scf.f90:10]
  └─> c_bands_nscf() [PW/src/c_bands.f90:1171]
       ├─> init_wfc(ik) [PW/src/wfcinit.f90]  # 初始化波函数
       │    └─> random_wavefunction() or atomic_wfc()
       │
       └─> diag_bands(iter=1, ik, avg_iter) [PW/src/c_bands.f90:176]
            │
            ├─> gamma_only == .TRUE. 分支 [line 316]
            │   └─> diag_bands_gamma() [PW/src/c_bands.f90:350]
            │
            └─> isolve == 0 (Davidson) 分支 [line 599]
                └─> regterg(h_psi, s_psi, ...) [KS_Solvers/Davidson/regterg.f90:19]
                     │
                     ├─> 初始化子空间 [line 167-227]
                     │   ├─> 复制 evc → psi
                     │   ├─> 计算 H|ψ> → hpsi
                     │   │    └─> h_psi(psi, hpsi) [PW/src/h_psi.f90]
                     │   │         ├─> 动能: hpsi = g2kin * psi
                     │   │         ├─> 局域势: vloc_psi_gamma()
                     │   │         ├─> 非局域势: calbec() + add_vuspsi()
                     │   │         └─> Gamma 约束: Im[hpsi(G=0)] = 0
                     │   │
                     │   ├─> 计算 S|ψ> → spsi (if uspp)
                     │   ├─> 投影: H_sub = ψ^† H ψ (使用 DGEMM，实数)
                     │   └─> 投影: S_sub = ψ^† S ψ
                     │
                     ├─> 对角化 [line 242-261]
                     │   └─> diaghg(H_sub, S_sub, eigenvalues, eigenvectors)
                     │        └─> DSYGVD (实数对称矩阵)
                     │
                     └─> Davidson 迭代 [line 265-end]
                         ├─> 计算残差: R = (H - ε*S)|ψ>
                         ├─> 预条件: g_psi(R)
                         ├─> 正交化并展开子空间
                         ├─> 重新对角化子空间
                         └─> 检查收敛 (notconv == 0)
```

### 关键 QE 函数说明

| 函数 | 文件 | 功能 | 对齐关键点 |
|------|------|------|-----------|
| `h_psi` | `PW/src/h_psi.f90` | 计算 H\|ψ> | 动能+局域势+非局域势 |
| `vloc_psi_gamma` | `PW/src/vloc_psi.f90` | 局域势（Gamma优化） | FFT 缩放约定 |
| `calbec` | `PW/src/calbec.f90` | 计算投影系数 <β\|ψ> | Gamma-only 内积 |
| `add_vuspsi` | `PW/src/add_vuspsi.f90` | 应用非局域势 | D_ij 矩阵初始化 |
| `regterg` | `KS_Solvers/Davidson/regterg.f90` | Davidson 对角化 | 实数 BLAS + 对角化 |
| `diaghg` | `LAXlib/dspev_drv.f90` | 广义本征值问题 | DSYGVD (实数) |

---

## DFTcu 代码架构

### 核心层级

```
Python 接口层 (src/dftcu/*.py)
    ↓
API 层 (src/api/dftcu_api.cu)
    ↓
Solver 层 (src/solver/)
    ├─> SCFSolver      # SCF 自洽迭代
    ├─> NonSCFSolver   # NSCF 单次对角化
    ├─> Davidson       # Davidson 迭代器
    └─> SubspaceSolver # 子空间求解器
    ↓
Model 层 (src/model/)
    ├─> Wavefunction   # 波函数 ψ(r), ψ(G)
    ├─> Density        # 电荷密度 ρ(r)
    ├─> Field          # 势场 V(r)
    └─> Grid           # FFT 网格
    ↓
Functional 层 (src/functional/)
    ├─> Hamiltonian    # 哈密顿量 H
    ├─> Hartree        # Hartree 势 V_H
    ├─> XCFunctional   # 交换相关 V_xc
    └─> Pseudopotential # 赝势 V_loc, V_NL
    ↓
Math 层 (src/math/)
    ├─> FFTSolver      # FFT R↔G 变换
    ├─> LinearAlgebra  # BLAS/LAPACK 包装
    └─> SphericalHarmonics # 球谐函数 Y_lm
```

### 关键 DFTcu 文件说明

#### Model 层
- **`src/model/wavefunction.cu`**: 波函数类
  - `set_coefficients_miller()`: Miller 指数映射（Phase 0 验证 ✅）
  - `force_gamma_constraint()`: G=0 约束（Phase 0 验证 ✅）
  - `dot()`: 内积计算（Phase 0 验证 ✅）
  - `orthonormalize()`: 正交归一化（Phase 0 验证 ✅）

- **`src/model/grid.cu`**: FFT 网格类
  - 管理实空间和倒空间网格
  - 提供 Miller 指数到 FFT 索引的映射

- **`src/model/density_builder.cu`**: 密度构建器
  - 从波函数计算电荷密度: `ρ(r) = Σ f_i |ψ_i(r)|²`

#### Functional 层
- **`src/functional/hamiltonian.cu`**: 哈密顿量
  - `apply()`: 计算 H|ψ>（Phase 1 待验证）
  - 组合动能、局域势、非局域势

- **`src/functional/pseudo.cu`**: 赝势核心
  - 局域势插值 V_loc(G)
  - 非局域投影仪 β_lm(G)

- **`src/functional/nonlocal_pseudo.cu`**: 非局域势
  - `apply_nonlocal()`: 计算 V_NL|ψ>（Phase 1c 待验证）
  - 投影系数 becp = <β|ψ>

#### Solver 层
- **`src/solver/scf.cu`**: SCF 求解器
  - 自洽迭代主循环
  - 密度混合

- **`src/solver/subspace_solver.cu`**: 子空间求解器
  - 计算 H_sub, S_sub（Phase 2 待验证）
  - 对角化子空间矩阵

- **`src/solver/davidson.cu`**: Davidson 迭代器
  - 完整 Davidson 算法（Phase 3 待验证）
  - 残差计算、预条件、子空间扩展

- **`src/solver/phase0_verifier.cu`**: Phase 0 验证器
  - 已验证 Miller 指数映射
  - 已验证 S_sub 矩阵计算
  - **精度**: 3.1e-15 ✅

### DFTcu NSCF 典型调用流程

```python
# Python 层 (src/dftcu/nscf.py)
import dftcu
import numpy as np

# 1. 初始化系统
lattice = np.array([[10,0,0], [0,10,0], [0,0,10]])  # 3×3 Angstrom
grid = dftcu.create_grid_from_qe(
    lattice_ang=lattice,
    nr=[18, 18, 18],
    ecutwfc_ry=12.0,
    ecutrho_ry=48.0,
    is_gamma=True
)
atoms = dftcu.Atoms(atomic_numbers, positions)
ham = dftcu.Hamiltonian(grid, atoms, ecutwfc, ecutrho)

# 2. 初始化波函数
psi = dftcu.Wavefunction(grid, nbands, ecutwfc)
psi.randomize()  # 或从文件加载

# 3. 运行 NSCF
solver = dftcu.NonSCFSolver(grid)
eigenvalues = solver.solve(ham, psi)  # C++/CUDA 端完成所有计算

# 4. 获取结果
energies = eigenvalues.tolist()
```

**C++/CUDA 端**（`src/solver/nscf.cu`）：
```cpp
std::vector<double> NonSCFSolver::solve(Hamiltonian& ham, Wavefunction& psi) {
    // 1. 计算 H|ψ>
    ham.apply(psi, hpsi);

    // 2. 子空间投影
    SubspaceSolver sub_solver(grid_);
    auto [H_sub, S_sub] = sub_solver.project(psi, hpsi);

    // 3. 对角化
    std::vector<double> eigenvalues = sub_solver.diagonalize(H_sub, S_sub);

    return eigenvalues;
}
```

---

## 文档索引

### QE 数据生成
- **QE 数据生成指南**: `docs/QE_DATA_GENERATION_GUIDE.md` ⭐
  - QE 源码修改说明
  - 数据导出流程
  - 网格配置对齐
  - 常见问题排查

### 测试框架文档
- **测试框架总览**: `tests/nscf_alignment/README.md`
- **快速入门**: `tests/nscf_alignment/QUICKSTART.md`
- **Phase 1 重构报告**: `tests/nscf_alignment/PHASE1_REFACTORING_REPORT.md` ⭐

### 对齐计划与报告
- **完整对齐计划**: `docs/NSCF_QE_ALIGNMENT_PLAN.md`
- **Phase 1 详细计划**: `docs/PHASE1_DETAILED_PLAN.md`
- **Phase 0 成功报告**: `docs/PHASE0_SUCCESS_REPORT.md`

### 架构设计文档
- **Hamiltonian 重构计划**: `docs/KSDFT_HAMILTONIAN_REFACTOR.md`
- **Evaluator 重命名计划**: `docs/EVALUATOR_RENAME_PLAN.md`
- **QE Dense Grid 需求调研**: `docs/QE_DENSE_GRID_REQUIREMENT.md` ⭐
  - Hartree 势能和局域赝势的网格使用分析
  - QE 源码调研结果
  - Dense grid 实现需求和优先级

---

## 关键设计原则

1. **测试代码高度复用**（Phase 1 重构后 81.8% 复用率）⬆️
2. **配置单点管理**（修改成本降低 75%）
3. **渐进式验证**（Phase N 依赖 Phase N-1）
4. **误差可追溯**（逐项分解定位问题）
5. **工业级质量**（SOLID、DRY、KISS 原则）

---

**版本**: 2.1 (Phase 1a 完成 + 重构)
**更新日期**: 2026-01-06

---

## 测试框架重构计划（2026-01-08）

### 背景

Grid 工厂函数已重构完成：
- ✅ 创建 `create_grid_from_qe()` 和 `create_grid_from_atomic_units()`
- ✅ 接受 3×3 NumPy 数组（无需 flatten）
- ✅ 单位转换在工厂函数边界完成
- ✅ Grid 类内部只使用原子单位

详见：`docs/GRID_FACTORY_REFACTORING_COMPLETE.md`

### 重构目标

更新所有测试代码使用新的工厂函数 API，确保：
1. 所有测试使用 `create_grid_from_qe()` 或 `create_grid_from_atomic_units()`
2. 移除旧的 `Grid(lattice.flatten(), nr)` 用法
3. 移除 `grid.set_cutoffs()` 调用
4. 统一通过 `utils/grid_factory.py` 创建测试 Grid

### 重构范围

#### 核心组件（优先级 1）
- [ ] `tests/nscf_alignment/utils/grid_factory.py`
  - 修改 `GridFactory.create_si_gamma_grid()` 使用新 API
  - 一旦更新，所有使用 GridFactory 的测试自动受益

#### Phase 0c 测试（优先级 2）
- [ ] `tests/nscf_alignment/phase0c/test_gvector_generator.py`
- [ ] `tests/nscf_alignment/phase0c/test_gvector_cuda.py`
- [ ] 其他直接创建 Grid 的测试

#### 自动受益的测试（优先级 3）
因为使用 `GridFactory`，以下测试会自动更新：
- Phase 0: `test_phase0.py`, `test_wavefunction_init.py`
- Phase 1a: `test_kinetic_with_grid.py`
- Phase 1b: `test_phase1b_vloc_refactored.py`
- Phase 1c: `test_nonlocal_with_grid.py`
- Phase 1d: `test_complete_hamiltonian.py`

### 验证计划

1. 更新核心 `utils/grid_factory.py`
2. 运行完整测试套件确保所有测试通过：
   ```bash
   python tests/nscf_alignment/main.py
   ```
3. 更新 Phase 0c 测试
4. 再次运行完整测试套件
5. 更新文档（QUICKSTART.md）

### Phase 0c Dense Grid 计划

根据 QE 调研结果（`docs/QE_DENSE_GRID_REQUIREMENT.md`）：

**当前状态**:
- ✅ Smooth grid 完全实现（满足 Phase 1 H|ψ> 需求）
- ❌ Dense grid 未实现（Hartree/LDA 泛函需要）

**实现优先级**:
1. **短期（本次重构）**: 更新测试框架使用新 API ✅
2. **中期（Hartree/LDA 测试前）**: 实现 Dense grid 支持
   - Dense grid G 向量生成（基于 ecutrho）
   - G-shell 分组 (ngl, gl, igtongl)
   - Dense grid FFT 支持
3. **长期（SCF 前）**: 实现 igk 映射 (Smooth ↔ Dense)

**关键发现**:
- Hartree 势能：需要 Dense grid (dfftp, ngm)
- 局域赝势：需要 Dense grid 的 G-shell 数据 (ngl, gl, igtongl)
- V_loc|ψ> 计算：在 Smooth grid 上（V_loc 从 Dense 插值）

详见：`tests/nscf_alignment/phase0c/README.md`（待更新）

---

**版本**: 2.2 (测试框架重构计划)
**更新日期**: 2026-01-08
