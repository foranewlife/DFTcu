# 技术规划：对标 VASP 实现平面波 KS-DFT {#blog_ks_dft_plan}

**日期**: 2025-12-28
**状态**: 规划中 (对标 VASP 架构)
**目标**: 在 DFTcu 中实现高性能平面波 Kohn-Sham DFT，核心算法与数据结构对标 VASP 6.x 及其 GPU 加速版本。

---

## 1. 引言

VASP (Vienna Ab initio Simulation Package) 是平面波 DFT 计算的工业标准。其 GPU 版本利用了高效的线性代数库和自定义 Kernel。DFTcu 将在现有 C++/CUDA 架构基础上，参考 VASP 的核心算法逻辑，实现高性能的 KS-DFT 求解器。

## 2. 核心模块对标说明

我们将 DFTcu 的开发任务与 VASP 的核心源文件进行功能对标，以确保算法的严谨性与效率。

### 2.1 波函数与基组 (Basis & Wavefunctions)
*   **VASP 对标**: `wave.F`, `wave_struct.F`
*   **DFTcu 实现**:
    *   `Wavefunction` 类：存储系数 $C_{n,k}(G)$。
    *   **平面波截断**: 实现 `ENCUT` 控制的倒空间球形截断。
    *   **K 点并行**: 初始版本支持单 K 点（Gamma-only 优化），后续扩展至多 K 点。

### 2.2 哈密顿量作用 ($\\hat{H}\\psi$)
*   **VASP 对标**: `hamil.F`, `nonl.F` (倒空间), `nonlr.F` (实空间)
*   **DFTcu 实现**:
    *   **动能项**: 倒空间对角操作。
    *   **局域势**: 实空间相乘。利用 DFTcu 现有的表达式模板加速 $\\hat{V}_{loc} \\psi$。
    *   **非局域势 (Non-local)**: 实现 Kleinman-Bylander 形式。
        *   对标 VASP 的 `LREAL = .FALSE.` (倒空间投影)。
        *   关键计算：$\\sum_{lm} \\langle \\psi_n | \\beta_{lm} \\rangle \\langle \\beta_{lm} | \\psi_n \\rangle$。
        *   **优化**: 使用 cuBLAS 的 `cublasZgemm` 处理投影系数矩阵。

### 2.3 电子子问题求解器 (Electronic Minimization)
*   **VASP 对标**: `davidson.F` (ALGO=Normal), `rmm-diis.F` (ALGO=VeryFast)
*   **DFTcu 实现**:
    *   **阻尼迭代/Davidson**: 用于初始步骤的快速收敛。
    *   **RMM-DIIS (Residual Minimization Method)**: 生产计算中的核心算法。
        *   通过最小化残差 $R = (\\hat{H} - \\epsilon) \\psi$ 来更新波函数。
        *   **GPU 优化**: 实现 Blocked RMM-DIIS，最大化矩阵乘法比例。

### 2.4 电荷密度构建与混合 (Charge Density & Mixing)
*   **VASP 对标**: `charge.F`, `mix.F`
*   **DFTcu 实现**:
    *   **Density Construction**: $\\rho(r) = \\sum_n f_n |\\psi_n(r)|^2$。
    *   **Mixing**: 对标 VASP 的 `IMIX = 4` (Pulay 混合)。
    *   **优化**: 利用现有的表达式模板实现高效的密度空间运算。

---

## 3. 开发阶段规划 (对标 VASP 工作流)

### 第一阶段：单电子算子与基础结构
1.  **数据容器**: 完成 `Wavefunction` 类，支持 `CPU <-> GPU` 零拷贝同步。
2.  **赝势读取**: 扩展 `LocalPseudo` 模块以支持 VASP 风格的 `POTCAR`（读取 PAW 数据中的局域部分和非局域投影器参数）。
3.  **FFT 映射**: 对标 VASP 的 `PREC` 参数，建立精确的 FFT 网格映射逻辑。

### 第二阶段：本征值求解与哈密顿量
1.  **$\\hat{H}\\psi$ 核心实现**: 集成非局域投影器计算。
2.  **Davidson 求解器**: 实现基础的对角化功能，确保能得到正确的能级。
3.  **子空间对角化**: 集成 cuSOLVER 处理子空间哈密顿矩阵 $H_{nm} = \\langle \\psi_n | \\hat{H} | \\psi_m \\rangle$。

### 第三阶段：全自洽场 (SCF) 循环
1.  **费米能级计算**: 实现高效的四面体法或 Smearing 算法。
2.  **混合器集成**: 将 Pulay 混合器应用于电荷密度。
3.  **收敛判据**: 对标 VASP 的 `EDIFF` 参数。

---

## 4. 关键算法挑战：非局域算符优化

在 GPU 上，非局域势的投影是最大的计算瓶颈。
*   **VASP 经验**: 通过将多个波函数和投影器打包成大矩阵，将计算转化为 `GEMM`。
*   **DFTcu 策略**:
    1.  将投影器 $\\beta_i$ 预存储在 GPU。
    2.  利用 `cublasDgemm` 计算所有带的投影系数 $\\langle \\beta_i | \\psi_n \\rangle$。
    3.  利用自定义 Kernel 或表达式模板完成最后的重构。

---

## 5. 结论

通过深度对标 VASP 的架构与算法，DFTcu 将具备处理复杂体系的 KS-DFT 计算能力。我们将继承 VASP 在算法稳定性上的优势，并利用现代 C++ 和 CUDA 提供的更高效内存管理与计算融合技术进行超越。
