1. import dftcu 会触发增量编译 （不要触碰build目录!!!!!!）
2. cmake --build external/qe/build 会增量编译qe
3. qe测试在run_qe_lda
4. **.gitignore 注意事项**: 不要将 Agent 需要访问的文件或目录加入 `.gitignore`。Agent 无法看到被 git 忽略的文件，这会干扰调试和开发。
5. 永远不要git add .

## QE 对齐注意事项 (Alignment Notes)

1.  **单位转换**: Python 层传入 `Atoms` 的坐标必须是 **埃 (Angstrom)**。Backend 里的 G 向量单位是 $\text{Angstrom}^{-1}$，相位计算 $G \cdot R$ 需保持单位一致。
2.  **初始化**: 在调用 `update_projectors` 之前，必须先调用 `init_dij` 初始化 D 矩阵，否则后端会因内存未分配发生段错误。
3.  **相位约定**:
    *   结构因子相位为 `exp(-iG.R)`。
    *   原子波函数与投影项必须包含 $i^l$ 因子（$l=1$ 时即 $i$，$l=2$ 时为 $-1$）。
4.  **归一化**: 投影仪定义中包含 $1/\sqrt{\Omega}$ 归一化因子。
5.  **G 向量顺序**: QE 与 DFTcu 的 G 向量索引布局完全不同。进行点对点数值对比时，**必须**通过米勒指数 (Miller Indices) 进行重映射。
6.  **Gamma 点波函数**: QE 导出的波函数系数通常只包含半个网格，且带有隐含的 $\sqrt{2}$ 因子。直接导入时需注意通过 Hermitian 对称性 ($C_{-G} = C_G^*$) 进行全网格展开。
7.  **内存布局 (Layout)**: QE 导出的 3D 网格在文本中通常需要执行交换操作以匹配 DFTcu：`qe_aligned = qe_raw.reshape(nr).transpose(1, 0, 2).flatten()`。

## 初始哈密顿量 (V_eff) 对齐要点

1.  **截断能单位**: 项目中所有 `set_gcut` 接口接收的单位统一为 **Rydberg**。数值上，$|G|^2$ (以 $\text{Bohr}^{-1}$ 为单位) 直接等于其以 Rydberg 为单位的能量值。
2.  **Hartree 势 (V_H)**: QE 在计算 Hartree 势的逆变换时，通常**不应用**球截断（即使用整个 FFT 网格）。
3.  **交换相关势 (V_xc)**:
    *   对于包含噪声或负波动的电荷密度，QE 会使用 `abs(rho)` 保证稳定性。
    *   **关键阈值**: QE 内部硬编码了 `rho_threshold = 1e-10`。在 `LDA_PZ` 中必须同步设置此值才能达到 $10^{-12}$ 级别的对齐。
4.  **局部势 (V_loc)**:
    *   **G=0 Limit (Alpha Term)**: `tab_vloc[0]` 必须包含 Alpha 项 $\int (rV_{loc}(r) + 2Ze^2) dr$。注意 Rydberg 到 Hartree 的 0.5 转换。
    *   **插值索引**: 对于 $q > 0$ 的插值，必须从 `tab_vloc[1]` 开始（即使用索引 `i0` 而不是 `i0-1`），因为 `tab_vloc[0]` 是预留给 Alpha 项的。
    *   必须严格在 $G$ 空间应用 `ecutrho` 截断，否则在原子中心会产生巨大的吉布斯纹波。
5.  **随机波函数生成**: 必须采用动能衰减的平滑噪声逻辑 $\psi_{random}(G) \propto \frac{\text{randy()}}{G^2 + 1.0}$ 以提升收敛性。和QE对比时千万不要使用随机波函数。
6.  **标准流程**: 使用 `dftcu.initialize_hamiltonian` 函数可自动处理上述对齐细节，实现 $10^{-15}$ Ha 级别的精度。
