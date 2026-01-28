# DFTcu 项目开发指南

## ⚡ 快速参考

| 操作 | 命令 |
|------|------|
| 增量编译 | `make rebuild`（**禁止**删除 `build/`） |
| Debug 编译 | `make debug`（启用调试符号） |
| 运行全部测试 | `make test`（C++ ctest + Python pytest） |
| 运行 CLI | `dftcu pw --config examples/nscf_si.yaml` |
| QE 编译 | `cd external/qe; cmake --build build` |
**开发红线**：
- ❌ `git add .`（严禁）
- ❌ C++ 层文件 I/O（调试用单元测试）
- ✅ 临时文件以 `temp_` 或 `debug_` 开头

---

## 🎯 项目愿景

DFTcu 是基于 CUDA 加速的密度泛函理论 (DFT) 计算框架，深度对齐 Quantum ESPRESSO (QE)。

**核心设计哲学**：Python 逻辑组装，C++/CUDA 高性能执行

---

## 🏗️ 核心架构

### Brain-Heart 架构

**🧠 Python 层 (Brain)**：
- 文件解析：YAML、UPF、POSCAR（C++ 无 I/O）
- 模型工厂：调用 C++ 工厂函数生产对象
- 物理预处理：原子电荷叠加、波函数初始化

**🫀 C++/CUDA 层 (Heart)**：
- 纯粹计算内核：$H|\psi\rangle$、FFT、Davidson
- 无文件 I/O，无配置解析

### 四层架构

```
Workflow 层 → Solver 层 → Functional 层 → Model 层
(流程编排)    (算法实现)   (泛函/算符)     (数据模型)
```

#### Model 层：数据模型（无副作用）
- **职责**：数据容器 + 轻量工厂
- **允许**：纯数据类、`create_*` 工厂函数、Builder 模式
- **禁止**：执行物理计算、调用 Solver 层、管理迭代

#### Functional 层：泛函与算符（最小副作用）
- **职责**：数学定义 + 算符实现
- **允许**：泛函（Hartree, XC）、算符（LocalPseudoOperator）、`build_*` 构建函数
- **禁止**：管理迭代、知道 Solver 的存在

#### Solver 层：算法实现（有副作用，明确标注）
- **职责**：迭代算法 + kernel 调用
- **允许**：迭代算法（Davidson, SCF）、`apply_*` 算符、`[SIDE_EFFECT]` 标注
- **禁止**：解析文件、创建数据对象

#### Workflow 层：流程编排（封装副作用）
- **职责**：组件组装 + 流程定义
- **允许**：组装组件、定义流程、配置管理
- **禁止**：实现算法细节、直接调用 kernel

### 依赖关系

```
Workflow → Solver → Functional → Model
   ↓         ↓          ↓          ↓
   └─────────┴──────────┴──────→ Math/Utilities
```

**禁止**：Model → Solver, Functional → Solver, Math → Model

---

## 🏷️ 命名约定

### 类命名

| 后缀 | 含义 | 示例 |
|------|------|------|
| `*Data` | 0D/1D 原始数据（无计算） | `PseudopotentialData` |
| `*Field` | 3D 场数据 | `RealField`, `ComplexField` |
| `*Operator` | 作用于波函数的算符 | `LocalPseudoOperator`, `NonLocalPseudoOperator` |
| `*Builder` | 重量级构建器（涉及 FFT/积分） | `WavefunctionBuilder`, `DensityBuilder` |
| 无后缀 | 约定俗成 | `Grid`, `Atoms`, `Wavefunction` |

### 函数命名

| 前缀 | 含义 | 复杂度 | 示例 |
|------|------|--------|------|
| `create_*` | 轻量工厂（单位转换） | O(N) | `create_grid_from_qe`, `create_atoms_from_structure` |
| `build_*` | 重量构建（FFT/积分） | O(N log N)+ | `build_local_pseudo_from_upf` |
| `compute_*` | 计算物理量 | - | `compute_energy`, `compute_potential` |
| `apply_*` | 算符作用 | - | `apply_hamiltonian` |
| `*_inplace` | 原地修改（有副作用） | - | `orthonormalize_inplace` |
| `initialize_*` | 初始化 | - | `initialize_potentials` |

### 函数标注

```cpp
// [PURE]         无副作用，相同输入 → 相同输出
// [CONST]        不修改对象状态
// [SIDE_EFFECT]  修改输入参数或内部状态
// [KERNEL]       GPU kernel
// [FACTORY]      工厂函数：创建新对象
// [BUILDER]      构建器：逐步构建对象
```

---

## 📐 单位制

**内部统一使用 Hartree 原子单位**：
- 能量：Hartree (1 Ha = 2 Ry = 27.2114 eV)
- 长度：Bohr (1 Bohr = 0.5292 Å)
- 动能：$T = \frac{1}{2}|G|^2$ [Ha]（$G$ 单位为 $2\pi/Bohr$）

**单位转换在工厂函数边界完成**：
```cpp
// ✅ 工厂函数接受外部单位
create_grid_from_qe(lattice_ang, nr, ecutwfc_ry);  // Å, Ry

// ✅ 内部类只接受原子单位
Grid(lattice_bohr, nr, ecutwfc_ha);  // Bohr, Ha
```

**FFT 缩放约定（QE 对齐）**：
- 正变换 (R → G)：无缩放
- 逆变换 (G → R)：无缩放
- *注意*：NumPy `ifftn` 有 $1/N$ 缩放，需手动消除

---

## ⚙️ QE 对齐核心细节

### G 向量单位与网格
- `gg_wfc` (Smooth Grid)：**Physical 单位** $(2\pi/Bohr)^2$，包含 $tpiba^2$，用于动能
- `gg_dense` (Dense Grid)：**Crystallographic 单位** $1/Bohr^2$，用于 Hartree 和局域势

### Gamma-only 优化
- **波函数**：利用 Hermitian 对称性 $\psi(-G) = \psi^*(G)$，仅存储半球
- **G=0 约束**：强制 $Im[\psi(G=0)] = 0$
- **计算**：实数 BLAS 优化，$G \neq 0$ 项内积乘 2

### UPF 局域势积分
遵循 QE `vloc_mod.f90` 约定：
- **G=0 (Alpha)**：Coulomb 修正 $r \cdot (r \cdot V_{loc}(r) + Z \cdot e^2)$
- **G≠0**：erf 修正 $(r \cdot V_{loc}(r) + Z \cdot e^2 \cdot erf(r)) \cdot sin(qr)/q$
- **截断**：`rcut = 10.0 Bohr`，强制奇数网格（Simpson 积分）

### NSCF 约束
有效势 $V_{eff} = V_{ps} + V_H + V_{xc}$ 在 `initialize_potentials()` 阶段计算并固定。

**验证指标**：本征值差异 < 1 meV，总能量差异 < 0.1 meV

---

## 🗺️ 目录结构

```
src/
├── model/       # 数据模型 + 轻量工厂
│   ├── grid.cuh, atoms.cuh, wavefunction.cuh, field.cuh
│   ├potential_data.cuh  # 1D 赝势数据
│   └── *_factory.cuh             # create_* 函数
│
├── functional/  # 泛函 + 算符 + 重量构建器
│   ├── hartree.cuh, xc/          # 泛函
│   ├── local_pseudo.cuh          # → LocalPseudoOperator
│   ├── nonlocal_pseudo.cuh       # → NonLocalPseudoOperator
│   └── *_builder.cuh             # build_* 函数
│
├── solver/      # 迭代算法
│   ├── hamiltonian.cuh           # H|ψ⟩ 组合算符
│   ├── davidson.cuh, scf.cuh     # 迭代求解器
│   └── nscf.cuh
│
├── workflow/    # 流程编排
│   └── nscf_workflow.cuh
│
├── fft/         # FFT 求解器
├── math/        # 数学工具（Bessel, 球谐函数）
└── utilities/   # GPU 工具、常量
```

---

## 📊 数据流

```
UPF 文件 ──parse──▶ PseudopotentialData (1D)
                           │
                           │ build_*_from_upf (重量)
                           ▼
Grid + Atoms ────────▶ LocalPseudoOperator (3D)
                           │
                           │ apply
                           ▼
WavefunctionBuilder ──▶ Wavefunction ──▶ Hamiltonian ──▶ H|ψ⟩
     (重量)                                  │
                                            ▼
                                      Davidson/SCF
```

---

## 🧪 测试与验证

### ⚠️ 测试开发流程（必读）

添加新测试时，**必须遵循 `tests/README.md` 的四步法**：

1. **代码分解分析**：识别 `[PURE]` vs `[SIDE_EFFECT]` 函数
2. **测试类型矩阵**：确定测试策略（解析解 / QE Golden Data）
3. **QE 数据截获**：从 QE 真实运行中截获带索引的中间数据
4. **数据标准化**：静态物理档案入库（manifest.json + 数值文件）

**禁止**：直接编写测试代码而跳过数据准备步骤。

### 测试架构（物理契约体系）

基于 **物理契约 (Physical Contract)** 的 GTest C++ 测试框架，详见 `tests/README.md`。

| 维度 | 验证重点 | 工具 |
|------|----------|------|
| 物理算子 (Functional) | 1D/3D 插值、径向积分、泛函公式 | GTest + Indexed Ref |
| 代数算法 (Solver) | $H|\psi\rangle$ 作用、子空间对角化 | GTest + Analytic Check |
| 端到端 (Workflow) | SiC 体系全流程对齐 | CLI + Regression Pack |

### 索引锁定规范
- **实空间**：`(ix, iy, iz)` 逻辑网格坐标
- **倒空间**：Miller 指数 `(h, k, l)`
- **原子/投影项**：`(atom_type, atom_index, proj_l, proj_m)`

### 运行测试
```bash
# 全部测试（推荐）
make test

# 仅 C++ 测试
cd build && ctest --output-on-failure

# 仅 Python 测试
make test-python
```

### ⚠️ 过时目录
- `tests/nscf_step_by_step/` — 已过时，请使用 `tests/unit/` 和 `tests/data/`

---

## ✅ 重构进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 清理 I/O 操作 | ✅ 已完成 |
| 2 | 重命名副作用函数 (`*_inplace`) | ✅ 已完成 |
| 3 | 统一命名约定 (`*Operator`, `*Builder`) | ✅ 已完成 |
| 4 | 移除格式依赖后缀 (`*_from_upf`) | ✅ 已完成 |
| 5 | 修复依赖关系（PseudopotentialData 移到 model 层，UPFParser 移到 Python 层） | ✅ 已完成 |
| 6 | 分离双重接口（构造函数私有化） | 🔵 可选优化 |

---

## 📚 详细文档

| 文档 | 内容 |
|------|------|
| `docs/architecture_naming_proposal.md` | 命名约定详解 |
| `docs/ARCHITECTURE_CONVENTIONS.md` | 架构约定与示例 |
| `docs/TESTABILITY_ANALYSIS.md` | 测试架构与指南 |
| `docs/architecture_violations_analysis.md` | 架构违规分析 |
