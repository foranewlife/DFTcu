# DFTcu NSCF 完整测试

**目标**: 验证 `NonSCFSolver` 完整流程与 QE 的对齐

**测试系统**: Si 2原子（FCC），Gamma-only，LDA-PZ

**状态**: 🔧 调试中 - V_NL becp 计算验证

---

## 📁 目录结构

```
tests/nscf_step_by_step/
├── run_nscf.py                    # ✅ 文件1：运行 DFTcu NSCF 并导出数据
├── compare_qe.py                  # ✅ 文件2：对比 DFTcu vs QE 并诊断
├── README.md                      # 📖 本文档
├── qe_run/                        # 📁 QE 参考数据
│   ├── si_nscf.in                 # QE NSCF 输入
│   ├── si_nscf.out                # QE NSCF 输出（包含本征值）
│   ├── Si.pz-rrkj.UPF             # 赝势文件
│   ├── qe_rho_r.txt               # QE SCF 密度 ρ(r)
│   ├── dftcu_debug_psi_iter0.txt  # QE 导出：初始波函数 ψ(G)
│   ├── dftcu_debug_tpsi_iter0.txt # QE 导出：T|ψ>
│   ├── dftcu_debug_tvlocpsi_iter0.txt  # QE 导出：(T + V_loc)|ψ>
│   └── dftcu_debug_fullhpsi_iter0.txt  # QE 导出：H|ψ>
└── nscf_output/                   # 📁 DFTcu 诊断输出
    ├── dftcu_eigenvalues.txt      # DFTcu 本征值
    ├── dftcu_occupations.txt      # DFTcu 占据数
    └── dftcu_energy_breakdown.txt # DFTcu 能量分解
```

**核心原则**：只维护两个Python文件，一个运行DFTcu，一个分析对比。

---

## 🚀 快速开始

### 1. 运行 NSCF 计算

```bash
cd tests/nscf_step_by_step
python run_nscf.py
```

**功能**：
1. 从 `qe_run/qe_rho_r.txt` 加载 QE SCF 自洽密度
2. 构建完整的 NSCF Hamiltonian：**H = T + V_ps + V_H[ρ] + V_xc[ρ] + V_NL**
3. 运行 `NonSCFSolver.solve()` 求解本征值
4. 自动导出诊断数据到 `nscf_output/`

**关键步骤**：
- 坐标转换：QE Fortran 列主序 → DFTcu C 行主序
- 泛函计算：**V_loc = V_ps + V_H[ρ] + V_xc[ρ]**（QE 的 `vrs`）
- 非局域势：V_NL = Σ_ij D_ij |β_i⟩⟨β_j|
- Davidson 迭代：求解 H|ψ> = ε|ψ>

### 2. 对比 QE 结果

```bash
python compare_qe.py
```

**对比内容**：
- **本征值**：DFTcu vs QE（目标精度 < 1 meV）
- **Hamiltonian 各项**：T|ψ>, V_loc|ψ>, V_NL|ψ>, H|ψ>
- **能量分解**：E_band, E_Hartree, E_XC, E_Ewald, E_tot

---

## 📊 当前状态（2026-01-14）

### ✅ 已完成修复

#### 1. V_ps (局域赝势) - 修复完成 ✅

**问题1：Hermitian 双重计数**
- **根因**：`scatter_dense_to_fft_kernel` 存储 +G 和 -G（共轭），IFFT 对每个 G 计数两次
- **修复**：在 `src/functional/pseudo.cu:428-440` 添加 0.5 缩放
  ```cpp
  scale_complex_kernel<<<...>>>(grid_.nnr(), v_g_->data(), 0.5);
  ```
- **验证**：RMS = 4.32e-6 Ha (0.12 meV) ✅

**问题2：缺少 alpha 项 (v_of_0)**
- **根因**：G=0 项被提取但未加回 R 空间
- **修复**：在 `src/functional/pseudo.cu:462-470` 添加回 `v_of_0_ * 0.5`
  ```cpp
  add_scalar_kernel<<<...>>>(grid_.nnr(), v.data(), v_of_0_ * 0.5);
  ```
- **验证**：包含在 4.32 µHa RMS 误差中 ✅

#### 2. V_H (Hartree 势) - 修复完成 ✅

**问题：Hermitian 双重计数**
- **根因**：`map_dense_to_fft_gamma_kernel` 同样的双重计数问题
- **修复**：在 `src/functional/hartree.cu:287-300` 添加 0.5 缩放
  ```cpp
  scale_complex_kernel<<<...>>>(nnr, rho_g_->data(), 0.5);
  ```
- **验证**：RMS < 1e-6 Ha（完美匹配）✅

#### 3. V_xc (交换关联势) - 验证正确 ✅

- **状态**：RMS = 1.42e-16 Ha（机器精度）
- **结论**：无需修复 ✅

### 🔧 当前调试重点：V_NL (非局域势)

#### 问题定位

**Hamiltonian 各项统计**（来自 `compare_qe.py` 的 QE 数据）：
```
T|ψ>:      |mean| = 0.0195 Ha
V_loc|ψ>:  |mean| = 0.0333 Ha  ✅ 已修复
V_NL|ψ>:   |mean| = 0.0499 Ha  ⚠️ 最大贡献！
H|ψ>:      |mean| = 0.0819 Ha
```


#### 已实现的修复（待验证）

**位置**：`src/functional/nonlocal_pseudo.cu:365-580`

**完整 DGEMM 实现**：
1. ✅ **提取紧凑数组**（lines 370-391）：
   ```cpp
   extract_smooth_to_packed_kernel<<<...>>>(
       npw, grid_.nl_d(),
       d_projectors_.data() + iproj * nnr,
       beta_packed.data() + iproj * npw
   );
   ```

2. ✅ **DGEMM 计算 becp**（lines 410-416）：
   ```cpp
   cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
               num_projectors_, nb, 2*npw,
               2.0,  // Gamma-only 因子
               beta_real, 2*npw,
               psi_real, 2*npw,
               0.0, becp_real.data(), num_projectors_);
   ```

3. ✅ **减去 G=0 重复计数**（lines 434-456）：
   ```cpp
   if (gstart == 2) {
       cublasDger(h, num_projectors_, nb, -1.0,
                  beta_real, 2*npw,
                  psi_real, 2*npw,
                  becp_real.data(), num_projectors_);
   }
   ```

4. ✅ **D-matrix 耦合 + 最终组合**（lines 496-559）

**验证状态**：🔧 代码已重新编译，待运行测试验证

### 📈 诊断数据（修复前）

**V_loc 各组件**：
```
V_ps:  RMS = 4.32e-6 Ha  (0.12 meV)   ✅ 修复后
V_H:   RMS < 1e-6 Ha                  ✅ 修复后
V_xc:  RMS = 1.42e-16 Ha              ✅ 验证正确

V_loc 总计：所有组件已验证 ✅
```

**本征值误差**（修复前）：
```
DFTcu vs QE 差异: ~72 eV (~2.6 Ha)  ❌

归因：V_NL becp 计算错误（6000× 误差）
```

---

## 🛠️ 下一步验证

### 立即任务

1. **运行测试验证 V_NL 修复**
   ```bash
   cd tests/nscf_step_by_step
   python run_nscf.py 2>&1 | grep -E "(DEBUG|becp|V_NL)" | head -50
   ```

   **预期输出**：
   - `[DEBUG becp] After DGEMM (before G=0 correction)`: becp 值应接近 QE
   - `[DEBUG V_NL] beta_packed at G=0`: β(G=0) ≈ 0.5826
   - `[DEBUG V_NL] vnl_packed after DGEMM`: V_NL|ψ> 值

2. **验证 becp 精度**
   - 目标：becp 与 QE 差异 < 1e-10
   - QE 参考值：becp[1] = 1.117
   - 修复前：becp[1] = 0.000182（误差 6000×）
   - 修复后：待验证

3. **运行完整对比**
   ```bash
   python compare_qe.py | tail -100
   ```

   **验证指标**：
   - 本征值误差：从 ~72 eV 降至 < 1 meV ✅
   - V_NL|ψ> 误差：< 1e-10 Ha ✅
   - H|ψ> 总误差：< 1e-10 Ha ✅

---

## 📖 系统参数

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
ecutwfc = 12.0 Ry (6.0 Ha)
ecutrho = 48.0 Ry (24.0 Ha)
```

**FFT 网格**:
```
nr = [15, 15, 15]  # 匹配 QE si_nscf.in
nnr = 3375 点
```

**G-vectors**:
```
Smooth grid (ecutwfc): 85 个
Dense grid (ecutrho):  730 个
```

---

## 🎯 成功标准

### V_loc 组件验证 ✅
- ✅ **V_ps**: RMS < 1e-5 Ha（已达标：4.32e-6 Ha）
- ✅ **V_H**: RMS < 1e-6 Ha（已达标：< 1e-6 Ha）
- ✅ **V_xc**: RMS < 1e-10 Ha（已达标：1.42e-16 Ha）

### V_NL 组件验证 🔧
- 🔧 **becp 投影系数**: 与 QE 差异 < 1e-10（待验证）
- 🔧 **V_NL|ψ>**: RMS 误差 < 1e-10 Ha（待验证）

### 最终验证目标 🎯
- **本征值**: 与 QE 差异 < 1 meV（当前 ~72 eV ❌）
- **H|ψ>**: RMS 误差 < 1e-10 Ha（待验证）
- **总能量**: 差异 < 0.1 meV（待验证）

---

## 📚 相关文档

- **QE 完整流程**: `NSCF_WORKFLOW.md`
- **CLAUDE 指南**: `../../CLAUDE.md`（单位约定、FFT 约定）

---

## 📝 修复历史

### 2026-01-14: V_loc 组件修复完成
- ✅ 修复 V_ps Hermitian 双重计数（0.5 因子）
- ✅ 修复 V_ps 缺少 alpha 项
- ✅ 修复 V_H Hermitian 双重计数（0.5 因子）
- ✅ 验证 V_xc 正确（机器精度）
- 🔧 V_NL DGEMM 实现已完成，待验证

### 2026-01-13: 建立两文件测试框架
- ✅ 创建 `run_nscf.py`：运行 DFTcu 并导出数据
- ✅ 创建 `compare_qe.py`：分析对比并诊断 Hamiltonian 各项

---

**版本**: 5.0
**更新日期**: 2026-01-14
**状态**: 🔧 V_NL becp 计算验证中
