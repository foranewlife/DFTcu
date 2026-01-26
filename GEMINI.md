# 📝 DFTcu 项目开发指南 (GEMINI 版)

## ⚡ 快速开发参考
1.  **增量编译**: 修改 C++ 后使用 `make rebuild`。**永远不要**直接删除 `build/` 目录。
2.  **Git 规范**: **严禁**执行 `git add .`。临时调试文件请以 `temp_` 或 `debug_` 开头。
3.  **运行环境**: 在项目根目录下执行命令。确保已进入 `.venv` 并安装了开发依赖。
4.  **测试入口**: NSCF 对齐验证请使用 `tests/nscf_step_by_step/`。
5.  **单位检查**: 核心类构造函数仅接受原子单位 (Bohr/Ha)，转换逻辑封装在 Python 工厂函数中。

---

## 🎯 项目愿景
DFTcu 是一个基于 CUDA 加速的密度泛函理论 (DFT) 计算框架，深度对齐 Quantum ESPRESSO (QE)。其核心设计哲学是 **“Python 逻辑组装，C++/CUDA 高性能执行”**。

---

## 🏗️ 核心架构 (Brain-Heart Architecture)

### 🧠 Python 层：决策与组装 (The Brain)
*   **文件解析 (No-I/O in C++)**：所有外部文件（YAML 配置、UPF 赝势、POSCAR、电荷密度）均由 Python 解析。
*   **模型工厂**：利用 Python 灵活的逻辑，调用 C++ 暴露的工厂函数生产基础对象（Grid, Atoms, PseudopotentialData）。
*   **物理预处理**：实现启动算法，如“原子电荷叠加”生成初始密度，或加载 QE 导出的波函数进行验证。

### 🫀 C++/CUDA 层：执行与加速 (The Heart)
*   **纯粹计算内核**：专注于 $H|\psi\rangle$ 作用、FFT 变换、Davidson 对角化及能量分解。
*   **层次化结构**：
    *   **Solver 层**: SCF, NonSCFSolver, Davidson 迭代。
    *   **Functional 层**: Hamiltonian 组装, Hartree, XC, Pseudopotential。
    *   **Model 层**: Wavefunction, Density, Field, Grid。
    *   **Math 层**: FFT 求解器, 线性代数包装, 球谐函数。

---

## 🗺️ 项目地图 (Project Mapping)

| 目录/文件 | 职责说明 |
| :--- | :--- |
| `src/dftcu/` | Python 接口、配置模型 (Pydantic) 与工具类。 |
| `src/api/` | Pybind11 绑定，定义 Python 与 C++ 的交互边界。 |
| `src/workflow/` | 计算流程封装（如 `NSCFWorkflow`），一键启动复杂计算。 |
| `src/solver/` | 数值求解算法库（Hamiltonian, Davidson, Subspace）。 |
| `src/functional/` | 物理泛函实现（Hartree, XC, UPF 赝势模型）。 |
| `src/model/` | 基础物理对象及其 GPU 内存管理。 |
| `tests/nscf_step_by_step/` | **当前主测试集**：分步骤对齐 QE NSCF 的关键路径。 |

---

## 📐 物理约定与单位制

### 1. 全局单位：Hartree 原子单位
项目内部**统一且强制**使用 Hartree 原子单位。
*   **能量**: Hartree (Ha)。1 Ha = 2 Ry = 27.2114 eV。
*   **长度**: Bohr ($a_0$)。1 Bohr = 0.5292 Angstrom。
*   **动能**: $T = \frac{1}{2}|G|^2$ [Ha]（$G$ 单位为 $2\pi/Bohr$）。

### 2. FFT 缩放约定 (QE 对齐)
*   **正变换 (R → G)**: `FFT`，无缩放。
*   **逆变换 (G → R)**: `IFFT`，无缩放。
*   *注意*：NumPy 默认 `ifftn` 有 $1/N$ 缩放，与 QE/DFTcu 交互时需手动消除。

---

## ⚙️ QE 对齐核心细节

### 1. G 向量单位与网格
*   `gg_wfc` (Smooth Grid): **Physical 单位** $(2\pi/Bohr)^2$，包含 $tpiba^2$ 因子，用于动能计算。
*   `gg_dense` (Dense Grid): **Crystallographic 单位** $1/Bohr^2$，用于 Hartree 和局域势插值。

### 2. Gamma-only 优化
*   **波函数**: 利用 Hermitian 对称性 $\psi(-G) = \psi^*(G)$，仅存储半球数据。
*   **G=0 约束**: 必须强制 $Im[\psi(G=0)] = 0$。
*   **计算**: 使用实数 BLAS 优化，计算内积时 $G\neq 0$ 项需乘以 2。

### 3. UPF 局域势积分
完全遵循 QE `vloc_mod.f90` 约定：
*   **G=0 (Alpha)**: 使用完整 Coulomb 修正 $r \cdot (r \cdot V_{loc}(r) + Z \cdot e^2)$，确保积分收敛。
*   **G≠0**: 使用误差函数 (erf) 修正 $(r \cdot V_{loc}(r) + Z \cdot e^2 \cdot erf(r)) \cdot sin(qr)/q$。
*   **截断**: 使用 `rcut = 10.0 Bohr` 查找第一个点并强制为奇数网格（Simpson 积分要求）。

### 4. DensityFunctionalPotential (DFP)
*   **统一架构**: DFP 负责从密度 $\rho(r)$ 计算势 $V[\rho] = \delta E[\rho]/\delta\rho$ 和能量 $E[\rho]$。
*   **适用性**: 模块化设计使其同时适用于 KS-DFT（Hartree + XC）和 OFDFT（包含动能泛函）。
*   **NSCF 约束**: 有效势 $V_{eff} = V_{ps} + V_H + V_{xc}$ 在 `potinit` 阶段计算并固定，不再随波函数迭代更新。

---

## 🧪 验证与开发守则

### 1. 逐步验证框架 (Step-by-Step Validation)
测试重点位于 `tests/nscf_step_by_step/`，目标是实现 DFTcu NSCF 与 QE 的完全对齐。

**测试系统**: Si 2原子（FCC），Gamma-only，LDA-PZ。

**核心工具**:
*   `run_nscf.py`: 运行 DFTcu NSCF 计算并导出诊断数据到 `nscf_output/`。
*   `compare_qe.py`: 将 DFTcu 结果与 `qe_run/` 下的 QE 参考数据进行逐项对比。

**分项验证状态**:
1.  **V_ps (局域赝势)**: 修复了 Hermitian 双重计数（0.5 因子）及 alpha 项修正。
2.  **V_H (Hartree 势)**: 修复了 Hermitian 双重计数。
3.  **V_xc (交换关联势)**: 经验证已达到机器精度对齐。
4.  **V_NL (非局域势)**: 当前调试重点，使用 DGEMM 优化 `becp` ($\langle\beta|\psi\rangle$) 计算。
5.  **最终指标**: 本征值差异 < 1 meV，总能量差异 < 0.1 meV。

**运行测试与计算**:
*   **推荐方式 (CLI)**: 直接使用组装好的工具链。
    ```bash
    dftcu pw --config examples/nscf_si.yaml
    ```
*   **分步调试 (Developer)**: 使用底层脚本观察物理量对齐。
    ```bash
    cd tests/nscf_step_by_step
    python run_nscf.py && python compare_qe.py
    ```

### 2. 开发红线
1.  **增量编译**：修改 C++ 后必须执行 `make rebuild`。
2.  **C++ 无 I/O**：C++ 不得包含 `fopen` 或文件解析，所有调试导出需在 Python 侧发起或通过安全钩子。
3.  **Git 规范**：严禁执行 `git add .`。临时调试文件命名建议以 `temp_` 或 `debug_` 开头。
4.  **单位安全**：Setter 方法接收外部单位（如 Ry/Ang），构造函数内部仅允许原子单位。
