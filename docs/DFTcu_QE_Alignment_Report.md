# DFTcu 与 Quantum ESPRESSO 对齐报告

**目标**: 将 `DFTcu` 的单氧原子 SCF 计算结果与 Quantum ESPRESSO (QE) 对齐。

**基准**:
- **体系**: 单个氧原子，位于 10 Å x 10 Å x 10 Å 的立方晶格中。
- **赝势**: `O_ONCV_PBE-1.2.upf`
- **泛函**: LDA (Perdew-Zunger)，在 QE 中记为 `'sla+pz'`。
- **截断能**: `ecutwfc = 30 Ry` (合 15 Ha)。
- **K 点**: Gamma 点。
- **QE 总能量**: **-30.45101307 Ry**。

---

## 编译与开发流程

为了提高开发效率，我们建立了一套高效的半自动化编译流程。

### 1. `DFTcu` 核心库的自动增量编译

我们利用 Python 的 `import` 机制与 `pyproject.toml` 中的构建配置相结合，实现了在运行 Python 脚本时自动触发 `CMake` 编译。

**工作原理**:
- `pyproject.toml` 文件将 `cmake --build` 和 `cmake --install` 命令配置为包的构建过程。
- 当在 Python 脚本中首次 `import dftcu` 时，如果 C++ 源码发生了变动，`pip` 的可编辑安装模式会自动执行 `CMake` 增量编译，将最新的 `.so` 文件安装到虚拟环境中。

**操作流程**:
1. 修改任何 `.cu` 或 `.cuh` 文件。
2. 直接运行相关的 Python 测试脚本，如 `PYTHONPATH=build python3 tests/python/run_oxygen_scf.py`。
3. `dftcu` 模块会自动重新编译，无需手动执行 `make` 或 `cmake --build`。

### 2. Quantum ESPRESSO (QE) 子模块的增量编译

在调试与 QE 的对齐问题时，我们有时需要修改 QE 的 Fortran 源码以打印中间变量。

**操作流程**:
- QE 作为一个子模块位于 `external/qe`。
- 我们已经使用 `CMake` 配置好了它的构建系统。
- 若要增量编译 QE，只需执行：
  ```bash
  cmake --build external/qe/build --target pw.x -j$(nproc)
  ```
- 这将只重新编译发生变化的 QE 组件，速度远快于完全重新编译。

---

## 如何生成和使用 QE 基准数据

所有用于对齐的 Quantum ESPRESSO 基准数据都集中在 `run_qe_lda/` 目录下。

### 1. 生成 QE 输出

我们通过一个打了 `DFTCU` 补丁的 QE 版本来运行计算。这个补丁使得 QE 在计算过程中会额外输出 `DFTcu` 可直接使用的中间变量，如 G-space 的波函数、实空间的势场等。

**操作流程**:
1.  **准备输入文件**: `run_qe_lda/qe_in.in` 是标准的 QE 输入文件，其中 `verbosity = 'high'` 用于输出详细的调试信息。
2.  **运行 QE**: 在 `run_qe_lda/` 目录下执行 QE 计算：
    ```bash
    # 假设 pw.x 在环境变量中
    pw.x < qe_in.in > qe_out.out
    ```
3.  **收集输出**:
    - `qe_out.out`: QE 的主输出文件，包含了总能量、能量组件、能级、收敛信息等。
    - `pwscf.save/`: QE 的标准保存目录，其中 `data-file-schema.xml` 包含了完整的输入/输出参数和最终结果的结构化数据。
    - `charge-density.dat` 和 `wfc*.dat`: 分别是二进制格式的电荷密度和波函数文件。

### 2. 使用对比数据

`DFTcu` 的 Python 测试脚本 (`tests/python/run_oxygen_scf.py`) 通过读取 `qe_out.out` 和 `data-file-schema.xml` 来获取 QE 的最终能量和组件值，用于和 `DFTcu` 的计算结果进行实时对比。

**下一步的核心任务**就是实现对 `charge-density.dat` 和 `wfc*.dat` 的解析，以便我们可以：
-   **使用 QE 的波函数作为 `DFTcu` 的初始猜测**，而不是随机波函数。
-   **使用 QE 的电荷密度作为 `DFTcu` 的输入**，来单步验证我们的势场计算是否正确。

---

## 对齐进度与问题分析

### 已完成的工作

1.  **SCF 收敛框架**:
    - 成功实现了 **Broyden 混合器** (`BroydenMixer`)，解决了早期线性混合导致的密度震荡和 SCF 不收敛问题。
    - 目前 SCF 循环可以在约 30-40 次迭代内稳定收敛。

2.  **核心物理算符验证**:
    - **Ewald 能量**: 与 QE 的 `ewald contribution` **完美对齐**（误差 < 1e-7 Ry）。
    - **XC 能量 (LDA-PZ)**: 与 QE 的 `xc contribution` **基本对齐**（误差 ~0.01 Ry），在网格积分精度范围内。

### 当前主要问题：有效势场 $V_{eff}$ 错误

在我们将 `ecutwfc` 等关键参数与 QE 完全对齐后，总能量误差反而从 0.28 Ry 扩大到了 **3.1 Ry**。

通过详细的能量组件和本征值对比，我们发现问题的根源在于 **`DFTcu` 计算出的有效势场 $V_{eff}$ 与 QE 存在显著物理偏差**。

**关键证据**:

1.  **本征能级严重偏离**:
    - **QE**: 2s 轨道 $\approx -1.89$ Ry, 2p 轨道 $\approx -0.61$ Ry。
    - **`DFTcu`**: 2s 轨道 $\approx -2.78$ Ry, 2p 轨道 $\approx -0.54$ Ry。
    - **结论**: `DFTcu` 的能级不仅绝对值错误，**2s-2p 能级分裂间隔也与 QE 不符**，这直接证明了势场形状是错误的。

2.  **能量组件失衡**:
    - **One-electron Contribution**: `DFTcu` 比 QE **低了约 7 Ry**。
    - **Hartree Contribution**: `DFTcu` 比 QE **高了约 5 Ry**。
    - **结论**: `DFTcu` 的 Hartree 能量偏高，说明其计算出的电子密度比 QE 更向原子核收缩。这是由一个吸引过强的（过于负的）有效势场导致的。

---

## 下一步调试计划

当前的重点是**停止对总能量的宏观调试，转向对势场分量的微观对齐**。

**核心策略**: **使用 QE 已收敛的物理量作为输入，在 `DFTcu` 中进行单步计算，并直接对比中间结果。**

1.  **目标**: 精确对齐有效势 $V_{eff} = V_{loc} + V_H + V_{XC}$。

2.  **执行步骤**:
    - **加载 QE 数据**:
        - **任务**: 实现从 QE 的 `pwscf.save` 目录中加载已收敛的**电荷密度** (`charge-density.dat`) 和**波函数** (`wfc*.dat`) 的功能。
        - **挑战**: 这些文件是二进制格式。由于当前环境缺少 `h5py` 或 `scipy.io` 等库，需要编写一个简单的 C++ 或 Python 加载器，或者修改 QE 源码使其输出文本格式。

    - **单步验证**:
        - 使用 QE 的密度 $\rho_{QE}(r)$ 作为输入，在 `DFTcu` 中计算 $V_H(r)$ 和 $V_{XC}(r)$。
        - 直接将 `DFTcu` 生成的 $V_H(r)$ 和 $V_{XC}(r)$ 场与从 QE debug 输出中提取的相应势场进行逐点对比。
        - 使用 QE 的波函数 $\psi_{QE}(G)$ 和 `DFTcu` 的哈密顿量 $\hat{H}_{DFTcu}$，计算 $\hat{H}_{DFTcu} \psi_{QE}(G)$，并与 QE 的结果对比，以验证非局部算符。

通过这种方式，我们可以将误差来源精确定位到是 Hartree、XC 还是局部/非局部赝势的实现上。
