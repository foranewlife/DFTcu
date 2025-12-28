# 技术规划：在 DFTcu 中实现平面波 KS-DFT {#blog_ks_dft_plan}

**日期**: 2025-12-28
**状态**: 规划中
**目标**: 扩展 DFTcu 架构以支持基于轨道的 Kohn-Sham DFT 计算

---

## 1. 引言

目前 DFTcu 已实现了高效的无轨道 DFT (OF-DFT) 框架。为了提升计算精度并支持更广泛的化学体系，我们需要引入基于轨道的 Kohn-Sham DFT。KS-DFT 的核心挑战在于处理单粒子波函数（轨道）以及求解哈密顿量的本征值问题。

## 2. 核心组件开发计划

实现平面波 KS-DFT 需要在现有架构上新增以下核心模块：

### 2.1 波函数与占据数管理 (Wavefunctions)
*   **数据结构**: 需要支持 `ComplexField` 的集合，用于存储 $\psi_{n,k}$（n 为带索引，k 为 k 点）。
*   **正交化**: 实现并行化的 Gram-Schmidt 或 Lowdin 正交化算法。
*   **占据数**: 实现费米-狄拉克分布（Fermi-Dirac distribution）以处理金属体系。

### 2.2 哈密顿量算符 ($\\hat{H}\\psi$)
需要实现高效的算符作用逻辑：
*   **动能算符 ($\\hat{T}$)**: 在倒空间利用 $G^2$ 进行对角操作。
*   **局域势算符 ($\\hat{V}_{loc}$)**: 在实空间利用表达式模板进行高效相乘。
*   **非局域赝势 ($\\hat{V}_{nl}$)**: 实现 Kleinman-Bylander 形式，这是平面波 KS-DFT 的性能关键。

### 2.3 本征值求解器 (Eigensolver)
由于平面波基组规模巨大，不能直接对角化。需要迭代法求解：
*   **LOBPCG**: 现代且高度并行化的预条件共轭梯度法。
*   **Davidson**: 经典的子空间迭代法。
*   **预条件器**: 实现基于动能的 Teter-Payne-Allan 预条件器。

### 2.4 电荷密度混合 (Density Mixing)
*   复用并优化现有的 **Pulay Mixing (DIIS)** 算法，确保 SCF 循环的高效收敛。

---

## 3. 分阶段实施路线图

### 第一阶段：基础设施 (Infrastructure)
1.  定义 `Wavefunction` 类，支持多带（Multi-band）操作。
2.  扩展 `Evaluator` 以处理轨道相关的能量项。
3.  实现简单的实空间非局域算符接口。

### 第二阶段：哈密顿量与本征值 (Hamiltonian & Solver)
1.  实现 `Hamiltonian::apply(const Wavefunction& psi)` 方法。
2.  集成 cuBLAS/cuSOLVER 以加速子空间对角化。
3.  实现 LOBPCG 求解器的初始版本。

### 第三阶段：集成与验证 (Integration & Validation)
1.  完成完整的 SCF 循环逻辑。
2.  针对标准体系（如 Si, Al）进行精度对标（vs DFTpy/VASP）。
3.  性能调优：利用 Nsight Systems 分析 $\\hat{H}\\psi$ 的瓶颈。

---

## 4. 技术决策：混合优化

我们将继续贯彻 **cuBLAS + 表达式模板** 的混合策略：
*   **表达式模板**: 用于实空间的势场相加和电荷密度构建（$\rho = \sum |\psi|^2$）。
*   **cuBLAS**: 用于波函数之间的正交化（大矩阵乘法）和子空间投影。

---

## 5. 结论

通过引入 KS-DFT，DFTcu 将从一个专门的无轨道计算工具演变为一个通用的平面波 DFT 平台。这将充分利用我们已经建立的高效网格操作、FFT 求解器以及 cuBLAS 集成优势。
