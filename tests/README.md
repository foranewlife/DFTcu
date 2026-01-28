# DFTcu 测试与数值验证框架

本目录包含 DFTcu 的全套 C++ 测试套件。我们的目标是建立一套基于**物理契约 (Physical Contract)** 的验证体系，确保计算精度严格对齐主流第一性原理软件（如 Quantum ESPRESSO）。

---

## 1. 核心设计哲学

### 1.1 物理契约胜过输出比对
我们不仅比对日志。测试的核心是验证代码是否履行了物理公式定义的“契约”。通过抽象出**带逻辑索引的标准数据集**，确保数值验证与底层实现（如内存排序、并行策略、库版本）彻底解耦。

### 1.2 函数式分解：[PURE] vs [SIDE_EFFECT]
- **[PURE] 纯计算逻辑**：将复杂的物理算子拆分为无副作用的纯数学函数（位于 `*_kernels.cuh`）。这些函数在单元测试中通过 Mock 输入直接验证，不依赖复杂的全局状态。
- **[SIDE_EFFECT] 状态管理逻辑**：负责 GPU 显存更新、FFT 调用、Stream 同步等。通过集成测试验证其对系统状态的正确修改。

---

## 2. 测试架构规范

### 2.1 测试矩阵与工具

| 维度 | 验证重点 | 推荐工具 |
| :--- | :--- | :--- |
| **物理算子 (Functional)** | 1D/3D 插值、径向积分、泛函公式 | GTest + Indexed Ref |
| **代数算法 (Solver)** | $H|\psi\rangle$ 作用、子空间对角化、厄米性 | GTest + Analytic Check |
| **物理不变性 (Invariant)** | 平移不变性、旋转不变性、单位一致性 | GTest + Symmetry Test |
| **端到端 (Workflow)** | SiC 体系全流程对齐（本征值/能量） | CLI + Regression Pack |

### 2.2 索引锁定 (Index-Locked)
所有空间分布数据强制携带逻辑键值，杜绝排序依赖：
- **实空间 (R-space)**：使用 `(ix, iy, iz)` 逻辑网格坐标。
- **倒空间 (G-space)**：使用 Miller 指数 `(h, k, l)`。
- **原子/投影项**：使用 `(atom_type, atom_index, proj_l, proj_m)`。

---

## 3. 参考数据 (Golden Data) 标准化要求

所有参考数据包必须作为**静态物理档案**存入仓库，严禁动态生成。

### 3.1 完整性契约 (The Golden Package)
一个合格的参考数据子目录（如 `tests/data/qe_reference/sic_zincblende/`）必须包含：
1. **输入档案**：产生该数据的 QE 输入文件 (`*.in`)。
2. **物理档案**：确切的赝势文件 (`*.UPF`)。
3. **环境档案**：注明 QE 版本及编译配置。
4. **数值档案**：带索引的截获数据（`vloc_tab`, `vloc_r`, `psi_g` 等）。

### 3.2 溯源与可靠性
- **QE 截获**：数据必须从 QE 真实运行中通过修改源码或导出工具截获。
- **静态固化**：一旦入库，非物理逻辑变更严禁随意修改基准数值。

---

## 4. 目录结构

```bash
tests/
│   ├── unit/                  # 单元测试：针对解耦后的 [PURE] 物理逻辑
│   │   ├── functional/        # 物理算符测试 (如 LocalPseudo, Hartree)
│   │   └── solver/            # 算法逻辑测试 (如 Hamiltonian, Davidson)
│   ├── fixtures/              # 测试脚手架：提供标准 Grid, Atoms (SiC) 预设
│   ├── data/                  # 静态测试数据
│   │   ├── pseudopotentials/  # 存档的 UPF 赝势
│   │   └── qe_reference/      # 固化的 Golden Package 数据 (.ref, .in)
│   ├── CMakeLists.txt         # 测试构建配置
│   └── README.md              # 本文档
```

---

## 5. 测试设计四步法

创建新测试时，遵循以下流程：

### 第一步：代码分解分析

分析目标代码能否进一步分解为可独立测试的单元：

```
目标函数 [SIDE_EFFECT]
    │
    ├── 纯数学函数 A [PURE] ← 可单独测试
    ├── 纯数学函数 B [PURE] ← 可单独测试
    └── 状态管理逻辑 [SIDE_EFFECT] ← 集成测试
```

**输出**：函数依赖图 + 可测试性评估

### 第二步：测试类型矩阵

确定每类函数的测试策略和参数覆盖：

| 函数类型 | 测试策略 | 参数覆盖 |
|----------|----------|----------|
| **数学公式** (interpolate, integrate) | 解析解对比 | 边界值、典型值、极端值 |
| **物理算子** (vloc_g, structure_factor) | QE Golden Data | 单原子、多原子、不同元素 |
| **不变性** | 对称性检验 | 平移、旋转、单位一致性 |
| **边界条件** | 特殊点验证 | G=0、r=0、截断边界 |

为每类函数定义「契约」，例如：
- `simpson_integrate(f, r)` 契约：对 $f(r)=r^2$ 积分误差 < 1e-12

### 第三步：QE 数据截获

准备 QE 输入，修改源码截获中间数据：

1. **设计最小化输入**：避免测试数据过大
   - 使用小晶胞（2-4 原子）
   - 使用低截断能（20-30 Ry）
   - 使用粗网格（足够验证精度即可）

2. **截获点规范化**：记录截获位置和单位
   ```yaml
   capture_points:
     - name: vloc_tab
       source: vloc_mod.f90:vloc_of_g()
       index: [element, ig]
       unit: Ry
   ```

### 第四步：数据标准化

整理数据为带物理量标记的版本化输入文件：

**目录结构**（路径索引，避免冗余）：
```
tests/data/
├── pseudopotentials/          # 共享赝势（复用）
│   ├── Si.pz-rrkj.UPF
│   └── C.pz-rrkj.UPF
├── inputs/                    # 共享 QE 输入
│   └── sic_scf.in
└── qe_reference/
    └── sic_zincblende/
        ├── manifest.json      # 元数据 + 相对路径索引
        ├── vloc_tab.bin       # 仅数值数据
        └── vloc_r.bin
```

**manifest.json 格式**：
```json
{
  "metadata": {
    "qe_version": "7.2",
    "capture_date": "2026-01-28",
    "input_file": "../inputs/sic_scf.in",
    "pseudopotentials": {
      "Si": "../pseudopotentials/Si.pz-rrkj.UPF",
      "C": "../pseudopotentials/C.pz-rrkj.UPF"
    }
  },
  "vloc_tab": {
    "index": ["element", "ig"],
    "unit": "Ry",
    "file": "vloc_tab.bin"
  }
}
```

---

## 6. 编写测试代码

完成四步设计后，编写 GTest 测试：

1. 继承 `SiCFixture` 获取一致的 Bohr/Ha/Grid 环境
2. 使用 `StandardDataLoader` 加载索引数据
3. 实现数值断言：`EXPECT_NEAR(actual, ref_value, 1e-12)`
4. **(进阶)**：增加显存审计断言，确保测试前后显存平衡

---

## 7. 工作日志 (2026-01-28)

### ✅ 今日进展
- **测试流程规范化**：定义「测试设计四步法」，从代码分解到数据标准化的完整流程
- **数据格式优化**：采用路径索引方式，避免赝势/输入文件冗余
- **LocalPseudo 分析**：完成代码结构分析，识别 [PURE] 和 [SIDE_EFFECT] 函数

### 🚀 下一步计划
- **SiC 基准构建**：设计最小化 QE 输入，截获 `vloc_tab` 和 `vloc_r` 数据
- **编写首个测试**：`test_local_pseudo_kernels.cu`，验证 `interpolate_vloc_phys()` 和 `compute_structure_factor_phase()`
