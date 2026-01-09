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

### Atoms 工厂函数

**核心原则**（与 Grid 相同）：
- **Atoms 类内部完全纯净** - 只接受原子单位（Bohr）
- **所有单位转换在工厂函数边界完成** - 独立的自由函数（`atoms_factory.cuh`）
- **函数名明确单位** - 通过函数名清楚地表达输入单位

**C++ 层**：
```cpp
#include "model/atoms_factory.cuh"

// ✅ 推荐：从 Angstrom 创建（用户友好）
std::vector<Atom> atoms_ang = {
    {0.0, 0.0, 0.0, 14.0, 0},      // Si at origin (Angstrom)
    {1.35, 1.35, 1.35, 14.0, 0}    // Si at (1.35 Å, 1.35 Å, 1.35 Å)
};
auto atoms = create_atoms_from_angstrom(atoms_ang);
// 内部自动转换为 Bohr

// ✅ 高级用法：直接使用原子单位（Bohr）
std::vector<Atom> atoms_bohr = {
    {0.0, 0.0, 0.0, 14.0, 0},
    {2.55, 2.55, 2.55, 14.0, 0}    // Bohr
};
auto atoms = create_atoms_from_bohr(atoms_bohr);

// ❌ 错误：不要直接调用构造函数（除非确定使用 Bohr）
Atoms atoms(atoms_list);  // 容易搞混单位！
```

**Python 层示例**：
```python
import dftcu

# ✅ 推荐：使用 Angstrom（用户友好）
atoms = dftcu.create_atoms_from_angstrom([
    dftcu.Atom(0.0, 0.0, 0.0, 14.0, 0),      # Si at origin (Angstrom)
    dftcu.Atom(1.35, 1.35, 1.35, 14.0, 0)    # Si at (1.35, 1.35, 1.35) Å
])

# ✅ 高级用法：使用 Bohr（原子单位）
atoms = dftcu.create_atoms_from_bohr([
    dftcu.Atom(0.0, 0.0, 0.0, 14.0, 0),
    dftcu.Atom(2.55, 2.55, 2.55, 14.0, 0)    # Bohr
])

# 单位转换（使用导出的常量）
pos_ang = 1.35  # Angstrom
pos_bohr = pos_ang * dftcu.constants.ANGSTROM_TO_BOHR  # 2.551130 Bohr

# ❌ 错误：不要直接调用构造函数
atoms = dftcu.Atoms([...])  # 不推荐！单位不明确
```

**测试框架中的使用**（`tests/nscf_alignment/utils/grid_factory.py`）：
```python
# 位置已经转换为 Angstrom
positions = [...]  # Angstrom

# 使用工厂函数创建 Atoms
atoms_list = [dftcu.Atom(pos[0], pos[1], pos[2], 14.0, 0) for pos in positions]
atoms = dftcu.create_atoms_from_angstrom(atoms_list)  # ✅ 单位明确
```

**重构完成状态** (2026-01-09):
- ✅ `Atoms` 类内部纯净（只接受 Bohr）
- ✅ `atoms_factory.cuh/cu` 实现工厂函数
- ✅ Python 绑定和常量导出
- ✅ `tests/nscf_alignment` 已全部更新
- ✅ 编译和功能测试通过



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

## NSCF 哈密顿量完整组成

### 完整物理公式

```
H_NSCF = T + V_ps + V_H[ρ_SCF] + V_xc[ρ_SCF] + V_NL
```

**各项说明**：
- **T**: 动能算符 = ½(2πG)² [Hartree]
- **V_ps**: 局域赝势（来自 UPF 文件）
- **V_H[ρ_SCF]**: Hartree 势（从 SCF 自洽密度计算，NSCF 中固定）
- **V_xc[ρ_SCF]**: 交换关联势（从 SCF 自洽密度计算，NSCF 中固定）
- **V_NL**: 非局域赝势 = Σ_ij D_ij |β_i⟩⟨β_j|

### QE 中的实现

QE 将局域贡献合并为 `vrs`（总局域势）：

```fortran
! PW/src/set_vrs.f90
vrs = vltot + vr
    = V_ps + (V_H + V_xc)
```

然后在 `h_psi` 中应用：

```fortran
! PW/src/h_psi.f90
hpsi = g2kin * psi                    ! T|ψ>
CALL vloc_psi_gamma(psi, vrs, hpsi)   ! 加上 (V_ps + V_H + V_xc)|ψ>
CALL add_vuspsi(hpsi)                 ! 加上 V_NL|ψ>
```

### NSCF vs SCF 的关键区别

| 项目 | SCF | NSCF |
|------|-----|------|
| **目标** | 求自洽密度 ρ | 用固定 ρ_SCF 求更多能带 |
| **密度 ρ** | 自洽迭代更新 | **从 SCF 读取（固定）** |
| **V_H[ρ]** | 每次迭代重算 | **只计算一次**（用 ρ_SCF） |
| **V_xc[ρ]** | 每次迭代重算 | **只计算一次**（用 ρ_SCF） |
| **vrs** | 每次迭代更新 | **固定不变** |
| **H|ψ>** | 完整哈密顿量 | **完全相同**的完整哈密顿量 |
| **迭代** | 直到 ρ 收敛 | Davidson 求本征态（不更新 ρ） |

**重点**：NSCF 和 SCF 使用**完全相同**的哈密顿量形式，区别只在于 NSCF 的 V_H 和 V_xc 是固定的。

---

## QE 对齐核心要点

### 单位与约定
- **坐标单位**: 内部统一使用 Bohr（原子单位）
- **截断能单位**: 内部统一使用 Hartree（DFTcu）， QE使用 Rydberg（转换时 × 0.5）
- **常数**: `BOHR_TO_ANGSTROM = 0.529177210903`
- **G 向量单位**:
  - `gg_` (FFT grid): **Crystallographic 单位 1/Bohr²**（不含 2π 因子）
  - `gg_wfc` (Smooth grid): **Physical 单位 (2π/Bohr)²**（含 2π 因子）
  - `gg_dense` (Dense grid): **Crystallographic 单位 1/Bohr²**（不含 2π 因子）
  - **动能计算**: Hamiltonian 中需要将 crystallographic `gg_` × (2π)² 转换为 physical 单位
  - **QE 的 g2kin**: Physical 单位，包含 tpiba² = (2π/alat)² 因子

### Gamma-only 关键点
- **波函数**: QE 只存储半球，带 √2 因子，需通过 Hermitian 对称性展开
- **内积**: QE 对 G≠0 项乘以 2，DFTcu 使用全网格需匹配
- **G=0 约束**: 必须强制 `Im[ψ(G=0)] = 0`
- **G 向量索引**: QE 使用预计算的 `nl_d` 和 `nlm_d` 查找表映射 G 向量到 FFT 网格，基于 ecutwfc 截断
  - **在测试中**: 使用 `utils/qe_gvector_loader.py` 中的 `QEGVectorData` 类统一加载和访问这些索引
  - 详见: `docs/GVECTOR_MANAGEMENT_DESIGN.md`

### UPF 局域势积分
**关键公式**（QE `vloc_mod.f90:159-163`）：
```fortran
! G=0 (alpha) term:
DO ir = 1, msh(nt)
   aux(ir) = r * (r*vloc(r) + Z*e2)  ! NOT Z*e2*erf(r)
END DO
CALL simpson(msh, aux, rab, tab_vloc(0,nt))

! G≠0 terms:
aux(ir) = (r*vloc(r) + Z*e2*erf(r)) * sin(q*r) / q
```

**核心要点**：
- **G=0 使用完整 Coulomb 修正** `+ Z*e2`，使积分收敛
- **G≠0 使用 erf(r) 修正** `+ Z*e2*erf(r)`，实空间短程处理
- **单位**: vloc(r) 和积分结果均为 Rydberg 单位

**网格截断**（QE `read_pseudo.f90:179-186`）：
- **QE 使用 rcut = 10.0 Bohr** 截断积分网格，避免大 r 处的数值噪声
- 找到第一个 `r > rcut` 的点，设为 `msh`
- 强制 `msh` 为奇数（Simpson 积分要求）
- **DFTcu 实现**: `src/functional/pseudo.cu:36-47` 完全遵循 QE 约定

**精度**：
- G=0: ~3.4e-8 (rcut=10 Bohr 截断)
- G≠0: ~2.9e-9 (插值精度)
- DFTcu 实现与 QE 完全一致

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
python tests/nscf_alignment/main.py
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

### ✅ Phase 0: 基础对齐

### ✅ Phase 1: H|ψ> (完整 NSCF 哈密顿量)

**验证状态**：✅ **已完成** - 所有物理贡献均已验证

#### Phase 1 子项验证

- **Phase 1a (动能 T)**: 1.665e-16 ✅
  - 公式: `T|ψ> = ½(2πG)² * ψ(G)` [Hartree]
  - 关键修复: 添加 (2π)² 因子转换 crystallographic → physical 单位
  - 位置: `tests/nscf_alignment/phase1a/`

- **Phase 1b (局域赝势 V_ps)**: 2.933e-09 ✅
  - **UPF V_loc(G) 验证**: G≠0: 2.9e-9, G=0: 3.4e-8 ✅
  - 核心修复：alpha 积分使用完整 Coulomb 修正 `+ Z*e2`
  - 位置: `tests/nscf_alignment/phase1b/test_vloc_from_upf_simple.py`

- **Phase 1c (非局域势 V_NL)**: 2.78e-17 ✅
  - 公式: `V_NL|ψ> = Σ_ij D_ij |β_i⟩⟨β_j|ψ⟩`
  - 位置: 隐含在 Phase 1d 测试中

- **Phase 1d (完整 NSCF H|ψ>)**: ✅ 已完成
  - **公式**: `H|ψ> = T|ψ> + V_loc|ψ> + V_NL|ψ>`
  - **重要**: `V_loc = V_ps + V_H + V_xc`（QE 的 vrs）
  - **包含所有贡献**:
    - ✅ T (动能): 1.665e-16
    - ✅ V_ps (局域赝势): 2.933e-09
    - ✅ **V_H (Hartree)**: 隐含在 V_loc 中
    - ✅ **V_xc (XC)**: 隐含在 V_loc 中
    - ✅ V_NL (非局域势): 2.78e-17
  - 位置: `tests/nscf_alignment/phase1d/test_complete_hamiltonian.py`

#### Phase 1 Functionals (泛函独立验证)

- **用途**: 为 **SCF 实现**验证泛函计算（SCF 需要每次迭代重算）
- **Hartree 泛函**: 2.89e-15 (能量), 4.44e-16 (势) ✅
- **LDA-PZ XC 泛函**: 9.77e-15 (能量), 2.78e-16 (势) ✅
- **注**: NSCF 中 V_H 和 V_xc 从 SCF 密度计算一次后固定
- 位置: `tests/nscf_alignment/phase1_functionals/`

#### 完整 NSCF 哈密顿量

```
H_NSCF = T + V_ps + V_H[ρ_SCF] + V_xc[ρ_SCF] + V_NL
```

**QE 实现** (`h_psi.f90`):
```fortran
hpsi = g2kin * psi                    ! T|ψ>
CALL vloc_psi_gamma(psi, vrs, hpsi)   ! vrs = V_ps + V_H + V_xc
CALL add_vuspsi(hpsi)                 ! V_NL|ψ>
```

### 📋 Phase 2: 子空间投影（待定）


### 📋 Phase 3: Davidson 迭代（待定）


---

## QE H|ψ> 计算流程

**文件**: `external/qe/PW/src/h_psi.f90`

```fortran
SUBROUTINE h_psi_( lda, n, m, psi, hpsi )
  ! 1. 动能项
  hpsi = g2kin * psi                       ! T|ψ>

  ! 2. 局域势（Gamma-only 路径）
  ! vrs = vltot + vr = V_ps + (V_H + V_xc)
  IF ( gamma_only ) THEN
    CALL vloc_psi_gamma(psi, vrs, hpsi)    ! 加上 (V_ps + V_H + V_xc)|ψ>
  ENDIF

  ! 3. 非局域赝势
  CALL calbec( vkb, psi, becp )    ! becp = <β|ψ>
  CALL add_vuspsi( hpsi )          ! hpsi += V_NL|ψ>

  ! 4. Gamma 约束
  IF ( gamma_only .AND. gstart == 2 ) &
    hpsi(1,:) = REAL(hpsi(1,:))    ! Im[ψ(G=0)] = 0
END SUBROUTINE
```

**关键**: `vrs` 是总局域势，由 `set_vrs()` 设置：
```fortran
! PW/src/set_vrs.f90
vrs = vltot + vr
    = V_ps + (V_H + V_xc)  ! vltot=局域赝势, vr=SCF势
```

### NSCF vs SCF 中的势

| 势 | SCF | NSCF |
|---|-----|------|
| **V_H[ρ]** | 每次迭代从 ρ 重算 | 从 ρ_SCF 计算**一次**后固定 |
| **V_xc[ρ]** | 每次迭代从 ρ 重算 | 从 ρ_SCF 计算**一次**后固定 |
| **V_ps** | 固定（来自 UPF） | 固定（来自 UPF） |
| **vrs** | 每次迭代更新 | **固定不变** |

---

## QE 源码修改指南

### 导出 H|ψ> 各项（Phase 1）
**文件**: `external/qe/PW/src/h_psi.f90`

在不同位置插入导出逻辑：
- Line 152 后: 导出 `g2kin` 和 `T|ψ>`
- Line 185 后: 导出 `vrs` (V_ps + V_H + V_xc) 和 `V_loc|ψ>`
- Line 235 后: 导出 `becp` 和 `V_NL|ψ>`
- 返回前: 导出完整 `H|ψ>`

**注**: QE 导出的 `V_loc|ψ>` 已包含 V_ps、V_H 和 V_xc 的完整贡献

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


---

## 关键设计原则

1. **测试代码高度复用**
2. **配置单点管理**
3. **渐进式验证**（Phase N 依赖 Phase N-1）
4. **误差可追溯**（逐项分解定位问题）
5. **工业级质量**（SOLID、DRY、KISS 原则）

---

**版本**: 2.2
**更新日期**: 2026-01-08
