# DFTcu Truncated Newton (TN) 优化器对齐及验收报告

## 1. 任务背景
在电子结构计算中，优化算法的精度直接决定了结果的可靠性。此前 `DFTcu` 与参考软件 `DFTpy` 在 Truncated Newton (TN) 优化过程的第一步存在 $\sim 10^{-3}$ Ha 的能量分歧。通过本次任务，我们定位并修复了底层网格映射与赝势插值问题，并实现了与 `DFTpy` 逻辑完全对齐的 TN 优化器。

## 2. 编译方法
项目采用 CMake 结合 `uv` (Python 轮子管理) 的构建流程。
*   **完整编译**：
    ```bash
    make build
    ```
*   **增量开发编译**（更新 C++ 核心并重新安装 Python 接口）：
    ```bash
    make rebuild
    ```

## 3. 测试方法
验证通过对比测试脚本 `tests/python/test_tn_comparison.py` 完成。

### 3.1 验证逻辑
1.  **环境对齐**：在 `DFTpy` 和 `DFTcu` 中同时构建完全一致的铝 (Al) 原子 FCC 系统，使用相同的 LDA 泛函、Thomas-Fermi 动能泛函以及相同的实空间网格 ($32 \times 32 \times 32$)。
2.  **单步精确对齐**：执行一次 TN 优化迭代，对比 Step 0 的总能量和 Step 1 优化后的能量。
3.  **收敛轨迹验证**：对比 30 步迭代内的能量演化曲线（dE）。

### 3.2 关键对齐点（修复项）
*   **Grid Indexing**：修正了 $G$ 向量在内存中的映射顺序，确保 $n0$（最慢变化轴）到 $n2$（最快变化轴）的排列与 `cuFFT` 及 `DFTpy` 频率空间一致。这是能量和梯度对齐的基础。
*   **Radial Interpolation**：在 GPU 上实现了赝势的径向三次样条插值，消除了此前因 $G$ 点顺序敏感导致的势场误差。
*   **Cubic Line Search**：实现了基于能量和导数采样点的三次插值线性搜索，并引入 Armijo 条件回溯，确保每一步步长选择与 `DFTpy` 高度同步。

### 3.3 验收结论
*   **初始能量误差**：$< 10^{-10}$ Ha。
*   **第一步迭代能量**：`DFTcu` 与 `DFTpy` 结果均为 `-2.1177038618 Ha`，达到 **10 位有效数字对齐**。
*   **最终收敛能量差异**：$\sim 3 \times 10^{-5}$ Ha（高于预期）。

## 4. 参考代码位置
为了确保算法实现的 1:1 复现，开发过程中深度参考了以下 `DFTpy`（位于项目 `external/DFTpy/` 目录下）的源码：

*   **TN 核心算法**：`src/dftpy/optimization/optimization.py` 中的 `get_direction_TN` 函数，特别是关于 Hessian-vector product 的差分计算逻辑。
*   **线性搜索逻辑**：`src/dftpy/optimization/optimization.py` 中的 `ValueAndDerivative` 接口。
*   **倒空间网格定义**：`src/dftpy/grid.py` 中的频率生成与 Cartesian $G$ 向量映射。
*   **局部赝势处理**：`src/dftpy/functional/pseudo/__init__.py` 中的 `calc_vlines` 与径向格点插值方法。
