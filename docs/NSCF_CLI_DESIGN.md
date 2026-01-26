# 📝 DFTcu NSCF (noscf) 流程设计文档

## 1. QE 电荷密度与波函数初始化调研

在 Quantum ESPRESSO (`pw.x`) 中，初始场（电荷密度和波函数）的来源通常有以下几种：

### 电荷密度 (`starting_rho`)
1.  **从文件读取 (`file`)**: 从之前的 SCF 计算结果中加载。
2.  **原子电荷叠加 (`atomic`)**: 这是最常用的初始化方法。
    *   **原理**: $\rho(\mathbf{r}) = \sum_{I} \rho_{atom}^{I}(|\mathbf{r} - \mathbf{R}_I|)$。
    *   **计算过程**:
        1.  从 UPF 文件中读取径向原子密度 $\rho_{at}(r)$。
        2.  在倒空间利用结构因子 $S(\mathbf{G})$ 进行叠加。
        3.  通过 FFT 变换到实空间。

### 波函数 (`starting_wfc`)
1.  **原子轨道叠加 (`atomic`)**: 利用赝势中的伪原子轨道 ($\chi(r)$) 构建初始基组。
2.  **随机初始化 (`random`)**: 生成随机相位的平面波系数。

**结论**: 为了实现高度对齐 QE 的 NSCF，我们在 Python 端解析 UPF 轨道数据，并驱动 C++ 的工厂类执行叠加算法。

---

## 2. Python-C++ 交互架构 (Brain vs. Heart)

### A. Python 层职责 (大脑 - Brain)
1.  **文件解析**: 使用 Python 纯脚本解析 YAML、UPF (XML) 和 POSCAR。**C++ 端不包含任何文件 I/O 逻辑**。
2.  **数据注入**: 将解析出的数值数组（径向网格、势能、原子轨道）通过 Setter 注入 `PseudopotentialData`。
3.  **模型组装**: 充当总装配线，创建 `Grid`, `Atoms`, `Hamiltonian` 等对象。
4.  **初始化驱动**: 实例化 `DensityFactory` 和 `WavefunctionFactory`，驱动原子叠加算法。

### B. C++ 层职责 (心脏 - Heart)
1.  **数值工厂**: `DensityFactory` 和 `WavefunctionFactory` 提供高性能的 CUDA Kernel 执行原子叠加计算。
2.  **高性能内核**: 执行 FFT ($R \leftrightarrow G$)、Davidson 对角化及能量求和。
3.  **生命周期管理**: 工作流（Workflow）对象管理计算中间状态。

---

## 3. `pw.py` 最终执行流程

1.  **Step 1: 数据解析 (Python)**
    *   `DFTcuConfig` 加载 YAML。
    *   `PythonUPFParser` 提取 $r$, $vloc$, $beta$, $rho\_at$, $chi$。
2.  **Step 2: 生成模型对象**
    *   `grid`, `atoms` 生产。
    *   组装 `Hamiltonian` (Hartree + XC + Local/Nonlocal Pseudo)。
3.  **Step 3: 自动化初始化 (Factories)**
    *   `DensityFactory`: 执行原子电荷叠加。
    *   `WavefunctionFactory`: 执行原子波函数叠加。
4.  **Step 4: 运行工作流**
    *   调用 `dftcu.NSCFWorkflow(grid, atoms, ham, psi, rho_data, config)`。
    *   执行 `execute()` 并输出本征值。

---

## 4. 实施状态总结 (2026-01-26)

*   ✅ **Python UPF Parser**: 完整实现，支持解析所有物理字段。
*   ✅ **Factory 重构**: 统一了 `DensityFactory` 和 `WavefunctionFactory` 的命名与接口。
*   ✅ **CLI 集成**: `dftcu pw` 已实现一键运行，完全脱离外部密度文件依赖。
*   ✅ **物理对齐**: 修复了倒空间插值的单位问题，确保本征值结果符合物理预期。
