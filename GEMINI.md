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

**⚠️ 重要变更 (2026-01-12)**：

**当前测试框架**: `tests/nscf_step_by_step/` ✅
- **简洁独立**: 零外部依赖，只使用 DFTcu 核心 API
- **逐步验证**: 5 个测试步骤，分步验证 NSCF 哈密顿量各组成部分
- **易于维护**: 每个文件 < 200 行，逻辑清晰
- **完整文档**: README.md 包含详细使用说明
- **状态**: ✅ **Step 1 已完成** (2026-01-12)
  - 动能项验证通过: 2.66e-07 Ha (相对误差 ~1 ppm)
  - QE 数据导出已修复 (机器精度 3.5e-18 Ry)
  - Step 2-5 待验证

**已废弃**: `tests/nscf_alignment/` ❌
- 过于复杂，工具链繁琐
- 代码重复率高，难以维护
- **已不再使用**，仅作参考保留

**旧测试**: `tests/test_*.py` ❌
- 保持不动，不要修改
- 历史遗留，不适合 NSCF 对齐

### 测试入口
```bash
# 进入测试目录
cd tests/nscf_step_by_step

# 运行完整测试套件
python run_scf.py && python compare_qe.py


# 查看文档
cat README.md
```

### QE 数据目录
测试数据位于:
```
tests/nscf_step_by_step/qe_run/
├── Si.pz-rrkj.UPF           # 赝势文件
├── si_nscf.in               # QE NSCF 输入文件
├── si_scf.out               # QE SCF 输出
├── si_nscf.out              # QE NSCF 输出
├── dftcu_debug_*.txt        # QE 导出的调试数据
└── run_qe.sh                # QE 运行脚本
```

**QE 导出数据文件**:
- `dftcu_debug_miller_smooth.txt` - Miller 指数
- `dftcu_debug_nl_mapping.txt` - G → FFT 映射
- `dftcu_debug_psi_iter0.txt` - 初始波函数
- `dftcu_debug_tpsi_iter0.txt` - T|ψ>
- `dftcu_debug_vps_r.txt` - 局域赝势
- `dftcu_debug_v_hartree.txt` - Hartree 势
- `dftcu_debug_v_xc.txt` - XC 势
- `dftcu_debug_tvlocpsi_iter0.txt` - (T + V_loc)|ψ>
- `dftcu_debug_fullhpsi_iter0.txt` - 完整 H|ψ>
- `dftcu_debug_hr_iter0.txt` - H_sub 矩阵
- `dftcu_debug_sr_iter0.txt` - S_sub 矩阵

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

## 常见问题

### Q: 如何通知我？
```bash
happy notify -p "<message>"
```

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


---

## 关键设计原则

1. **测试代码高度复用**
2. **配置单点管理**
3. **渐进式验证**（Phase N 依赖 Phase N-1）
4. **误差可追溯**（逐项分解定位问题）
5. **工业级质量**（SOLID、DRY、KISS 原则）
