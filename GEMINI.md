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
2.  **Git 规范**：严禁执行 `git add .`。临时调试文件命名建议以 `temp_` 或 `debug_` 开头。
3.  **单位安全**：Setter 方法接收外部单位（如 Ry/Ang），构造函数内部仅允许原子单位。

---

## 🏛️ 架构约定与编码规范

### 1. 四层架构设计

DFTcu 采用清晰的四层架构，每层有明确的职责和边界：

```
Workflow 层 (流程编排)
    ↓
Solver 层 (算法实现)
    ↓
Functional 层 (数学定义)
    ↓
Model 层 (数据模型)
```

#### Model 层：数据模型（无副作用）
*   **职责**：数据容器 + 工厂函数
*   **允许**：
    *   纯数据类（Grid, Atoms, Wavefunction, Field）
    *   工厂函数（`create_grid_from_atomic_units`, `create_atoms_from_structure`）
    *   Builder 模式（`WavefunctionFactory`, `DensityFactory`）
    *   数据拷贝（仅修改自身）
*   **禁止**：
    *   执行物理计算
    *   调用 Solver 层
    *   管理迭代
    *   依赖全局状态

#### Functional 层：泛函定义（最小副作用）
*   **职责**：数学定义 + 接口抽象
*   **允许**：
    *   定义泛函接口（`Functional` 基类）
    *   实现具体泛函（`Hartree`, `LDA_PZ`, `LocalPseudo`）
    *   组合多个泛函（`DensityFunctionalPotential`）
    *   调用 kernel 计算能量和势
*   **禁止**：
    *   管理迭代
    *   依赖全局状态
    *   知道 Solver 的存在

#### Solver 层：算法实现（有副作用，明确标注）
*   **职责**：迭代算法 + kernel 调用
*   **允许**：
    *   实现迭代算法（`NonSCFSolver`, `Davidson`）
    *   应用算符（`Hamiltonian::apply`）
    *   修改输入参数（明确标注 `[SIDE_EFFECT]`）
    *   提供融合接口（`apply_fused`）
*   **禁止**：
    *   解析文件
    *   创建数据对象（应该用 Factory）
    *   知道 Workflow 的存在

#### Workflow 层：流程编排（封装副作用）
*   **职责**：组件组装 + 流程定义
*   **允许**：
    *   组装组件（在构造函数中）
    *   定义计算流程（`execute()`）
    *   配置管理（`NSCFWorkflowConfig`）
    *   错误处理
*   **禁止**：
    *   实现算法细节
    *   直接调用 kernel
    *   绕过 Solver 层

### 2. 函数分类标注

为了明确副作用和优化机会，使用以下标注：

```cpp
// [PURE]         纯函数：无副作用，相同输入总是产生相同输出
// [CONST]        常量函数：不修改对象状态
// [SIDE_EFFECT]  有副作用：修改输入参数或全局状态
// [FACTORY]      工厂函数：创建新对象
// [BUILDER]      构建器：逐步构建对象
// [KERNEL]       GPU kernel：在 GPU 上执行
// [FUSIBLE]      可融合的 kernel：可以与其他 kernel 融合
// [FUSED]        已融合的 kernel：多个 kernel 的融合版本
```

**标注示例**：

```cpp
// Model 层
// [FACTORY] [PURE]
Grid create_grid_from_atomic_units(...);

// [CONST]
int Wavefunction::num_bands() const;

// [SIDE_EFFECT] 仅修改自身
void Wavefunction::copy_from(const Wavefunction& other);

// Functional 层
// [CONST] [KERNEL]
double Hartree::compute_energy(const RealField& rho) const;

// [SIDE_EFFECT] [KERNEL] 修改 v
void Hartree::compute_potential(const RealField& rho, RealField& v) const;

// Solver 层
// [SIDE_EFFECT] [KERNEL] [FUSIBLE]
void Hamiltonian::apply_kinetic(const Wavefunction& psi_in, Wavefunction& psi_out) const;

// [SIDE_EFFECT] [KERNEL] [FUSED]
void Hamiltonian::apply_fused(const Wavefunction& psi_in, Wavefunction& psi_out) const;

// Workflow 层
// [SIDE_EFFECT] 执行完整计算流程
EnergyBreakdown NSCFWorkflow::execute();
```

### 3. 命名约定

*   **工厂函数**：`create_<type>_from_<source>`
    *   例：`create_grid_from_atomic_units`, `create_atoms_from_structure`
*   **查询函数**：`get_<property>` 或 `is_<property>`
    *   例：`get_num_bands()`, `is_converged()`
*   **计算函数**：`compute_<quantity>`
    *   例：`compute_energy()`, `compute_potential()`
*   **应用算符**：`apply_<operator>`
    *   例：`apply_hamiltonian()`, `apply_kinetic()`
*   **初始化函数**：`initialize_<component>`
    *   例：`initialize_density()`, `initialize_wavefunction()`
*   **融合 kernel**：`<name>_fused`
    *   例：`apply_hamiltonian_fused()`

### 4. 性能优化约定

#### Kernel 融合支持

为了支持未来的 kernel 融合优化，遵循以下约定：

```cpp
// 1. 在 Solver 层提供融合接口
class Hamiltonian {
public:
    // [SIDE_EFFECT] [KERNEL] [FUSIBLE] 标准接口
    void apply(const Wavefunction& psi_in, Wavefunction& psi_out) const;

    // [SIDE_EFFECT] [KERNEL] [FUSED] 融合接口
    void apply_fused(const Wavefunction& psi_in, Wavefunction& psi_out) const;

    // [CONST] 查询是否支持融合
    bool supports_fusion() const { return true; }
};

// 2. 在 Solver 层根据配置选择
if (enable_fusion && ham.supports_fusion()) {
    ham.apply_fused(psi, h_psi);  // 使用融合版本
} else {
    ham.apply(psi, h_psi);        // 使用标准版本
}

// 3. 在 Workflow 层配置
solver.set_fusion_enabled(config.enable_kernel_fusion);
```

#### 数据布局优化

为了支持缓存友好的数据访问，可以提供多种数据布局：

```cpp
class Atoms {
public:
    // [SOA] Structure of Arrays 布局（适合坐标级操作）
    const double* pos_x() const;
    const double* pos_y() const;
    const double* pos_z() const;

    // [AOS] Array of Structures 布局（适合原子级操作）
    const AtomData* atoms_data() const;

    // [SYNC] 同步函数
    void sync_soa_to_aos();
    void sync_aos_to_soa();
};
```

### 5. 依赖关系约定

**允许的依赖**（单向，无循环）：

```
Workflow → Solver → Functional → Model
   ↓         ↓          ↓          ↓
   └─────────┴──────────┴──────→ Math
   └─────────┴──────────┴──────→ Utilities
```

**禁止的依赖**：

*   ❌ Model → Solver（Model 不应该知道 Solver）
*   ❌ Model → Workflow（Model 不应该知道 Workflow）
*   ❌ Functional → Solver（Functional 不应该知道 Solver）
*   ❌ Math → Model（Math 应该是独立的）

### 6. 详细文档

完整的架构约定和编码规范请参考：
*   **`docs/ARCHITECTURE_CONVENTIONS.md`**：详细的架构约定、示例和检查清单
*   **`docs/kernel_fusion_architecture_analysis.md`**：Kernel 融合架构分析和优化方案
*   **`docs/atoms_optimization_proposal.md`**：数据布局优化方案（缓存友好）
*   **`docs/TESTABILITY_ANALYSIS.md`**：架构可测试性分析与单元测试指南

---

## 🧪 测试架构与可测试性

### 整体可测试性评分：⭐⭐⭐⭐（良好）

当前四层架构**非常适合单元测试**，每层的可测试性如下：

| 层次 | 可测试性 | 测试类型 | 主要优势 | 测试策略 |
|------|---------|---------|---------|---------|
| **Model 层** | ⭐⭐⭐⭐⭐ | 单元测试 | 纯函数，无副作用 | 大量单元测试 |
| **Functional 层** | ⭐⭐⭐⭐ | 单元测试 | 接口清晰 | 小网格 + Mock |
| **Solver 层** | ⭐⭐⭐ | 单元测试 | 职责明确 | Test Fixture |
| **Workflow 层** | ⭐⭐ | 集成测试 | 端到端验证 | 集成测试 |

### 测试示例

#### Model 层：极易测试
```cpp
// 纯函数，完美的单元测试
TEST(AtomsTest, CreateFromStructure) {
    std::vector<std::string> elements = {"Si", "Si"};
    std::vector<std::vector<double>> positions = {{0, 0, 0}, {1.35, 1.35, 1.35}};

    auto atoms = create_atoms_from_structure(
        elements, positions, lattice, true,
        {"Si"}, {{"Si", 4.0}}
    );

    EXPECT_EQ(atoms->nat(), 2);
    EXPECT_NEAR(atoms->h_pos_x()[1], 2.551, 1e-3);  // 1.35 Å → Bohr
}

TEST(WavefunctionFactoryTest, NumBands) {
    WavefunctionFactory factory(grid, atoms);
    factory.add_atomic_orbital(0, 0, r, chi_s, rab, msh);  // s 轨道
    factory.add_atomic_orbital(0, 1, r, chi_p, rab, msh);  // p 轨道

    EXPECT_EQ(factory.num_bands(), 8);  // 2 原子 × (1 s + 3 p) = 8
}
```

#### Functional 层：较易测试
```cpp
TEST(HartreeTest, ComputeEnergy) {
    Grid grid = create_test_grid(8, 8, 8);  // 小网格，快速测试
    RealField rho(grid, 1);
    fill_uniform_density(rho, 1.0);

    Hartree hartree;
    double energy = hartree.compute_energy(rho);

    EXPECT_NEAR(energy, expected, 1e-6);
}
```

#### Solver 层：使用 Test Fixture
```cpp
class HamiltonianTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid_ = create_test_grid(8, 8, 8);
        atoms_ = create_test_atoms();
        ham_ = std::make_unique<Hamiltonian>(*grid_);
    }

    std::unique_ptr<Grid> grid_;
    std::shared_ptr<Atoms> atoms_;
    std::unique_ptr<Hamiltonian> ham_;
};

TEST_F(HamiltonianTest, ApplyKinetic) {
    Wavefunction psi_in(*grid_, 1, 6.0);
    Wavefunction psi_out(*grid_, 1, 6.0);

    set_plane_wave(psi_in, 0, 0, 0);
    ham_->apply_kinetic(psi_in, psi_out);

    EXPECT_NEAR(compute_norm(psi_out), 0.0, 1e-10);
}
```

### 测试工具类

为简化测试，提供统一的测试工具：

```cpp
// tests/test_utils/test_utils.hpp
namespace dftcu::test {

// 创建测试用的小网格（快速测试）
Grid create_test_grid(int nr = 8, double ecutwfc = 6.0);

// 创建测试用的 Atoms
std::shared_ptr<Atoms> create_test_atoms(int nat = 2);

// 创建测试用的 Wavefunction
Wavefunction create_test_wavefunction(const Grid& grid, int nbands = 1);

// 填充均匀密度
void fill_uniform_density(RealField& rho, double value);

// 设置平面波
void set_plane_wave(Wavefunction& psi, int band, int gx, int gy, int gz);

// 计算范数
double compute_norm(const Wavefunction& psi, int band = 0);

// 计算重叠矩阵
std::vector<std::vector<double>> compute_overlap_matrix(const Wavefunction& psi);

}  // namespace dftcu::test
```

### 测试目录结构

```
tests/
├── unit/                    # 单元测试
│   ├── model/              # Model 层测试（纯函数，易测试）
│   │   ├── test_atoms.cpp
│   │   ├── test_grid.cpp
│   │   ├── test_wavefunction.cpp
│   │   └── test_factories.cpp
│   │
│   ├── functional/         # Functional 层测试（小网格）
│   │   ├── test_hartree.cpp
│   │   ├── test_lda_pz.cpp
│   │   └── test_local_pseudo.cpp
│   │
│   └── solver/             # Solver 层测试（Test Fixture）
│       ├── test_hamiltonian.cpp
│       ├── test_nscf_solver.cpp
│       └── test_davidson.cpp
│
├── integration/            # 集成测试
│   ├── test_nscf_workflow.cpp
│   └── test_scf_workflow.cpp
│
├── regression/             # 回归测试（与 QE 对比）
│   └── test_qe_alignment.py
│
└── test_utils/             # 测试工具
    ├── test_utils.hpp
    └── test_utils.cpp
```

### 提高可测试性的建议

#### 1. 依赖注入（推荐）⭐⭐⭐⭐⭐

使用接口抽象，方便 mock：

```cpp
// 定义接口
class IHamiltonian {
public:
    virtual void apply(const Wavefunction& psi_in, Wavefunction& psi_out) const = 0;
    virtual ~IHamiltonian() = default;
};

// 实现类
class Hamiltonian : public IHamiltonian {
public:
    void apply(const Wavefunction& psi_in, Wavefunction& psi_out) const override;
};

// 测试时使用 Mock
class MockHamiltonian : public IHamiltonian {
public:
    MOCK_METHOD(void, apply, (const Wavefunction&, Wavefunction&), (const, override));
};

TEST(NonSCFSolverTest, SolveMocked) {
    MockHamiltonian ham;
    EXPECT_CALL(ham, apply(_, _)).Times(10);

    NonSCFSolver solver(grid);
    solver.solve(ham, psi, ...);
}
```

#### 2. 小数据测试（推荐）⭐⭐⭐⭐

使用小网格减少测试时间：

```cpp
// ❌ 慢：真实网格（48×48×48 = 110592 点，测试需要 10 秒）
Grid grid = create_grid_from_atomic_units(lattice, {48, 48, 48}, 6.0, 24.0, true);

// ✅ 快：测试网格（8×8×8 = 512 点，测试需要 0.1 秒）
Grid grid = create_test_grid(8);
```

#### 3. 参数化测试（推荐）⭐⭐⭐

测试多种输入组合：

```cpp
class GridTest : public ::testing::TestWithParam<int> {};

TEST_P(GridTest, CreateGrid) {
    int nr = GetParam();
    auto grid = create_test_grid(nr);
    EXPECT_EQ(grid.nr()[0], nr);
}

INSTANTIATE_TEST_SUITE_P(
    DifferentSizes,
    GridTest,
    ::testing::Values(4, 8, 16, 32)
);
```

### 测试最佳实践

1.  **分层测试**：
    *   Model 层：大量单元测试（纯函数，易测试）
    *   Functional 层：单元测试 + 小网格
    *   Solver 层：单元测试 + Test Fixture
    *   Workflow 层：集成测试（端到端验证）

2.  **快速反馈**：
    *   使用小数据（8×8×8 网格）
    *   单元测试应在 0.1 秒内完成
    *   集成测试可以稍慢（1-10 秒）

3.  **隔离测试**：
    *   使用 Mock 对象隔离依赖
    *   使用 Test Fixture 管理测试环境
    *   避免测试之间的相互依赖

4.  **回归测试**：
    *   与 QE 参考值对比
    *   使用 `tests/nscf_step_by_step/` 进行逐步验证
    *   确保数值精度（本征值 < 1 meV，能量 < 0.1 meV）

详细的测试指南和示例请参考 **`docs/TESTABILITY_ANALYSIS.md`**。

---

## 🔍 代码现状分析与重构计划

### 当前代码问题总结

经过全面分析，发现以下**架构违规**和**代码质量问题**：

#### 🔴 严重问题（优先级 1 - 立即修复）

1. **Model 层包含 I/O 操作**（违反"C++ 无 I/O"原则）
   - `src/model/grid.cu`: 20+ 个 `printf` 调试语句
   - `src/model/density_factory.cu`: 3 个 `printf` 调试语句
   - **影响**: 违反架构设计，难以测试，性能开销

2. **Solver 层包含文件 I/O**（违反"C++ 无 I/O"原则）
   - `src/solver/hamiltonian.cu`: 15+ 个 `fprintf/printf` 语句
   - `src/solver/nscf.cu`: 6 个 `dump_*` 文件 I/O 方法
   - **影响**: 违反架构设计，难以控制输出

#### 🟡 中等问题（优先级 2 - 重构）

3. **命名不清晰的副作用函数**
   - `Wavefunction::apply_mask()` → 应改为 `apply_mask_inplace()`
   - `Wavefunction::orthonormalize()` → 应改为 `orthonormalize_inplace()`
   - `Hamiltonian::update_potentials()` → 应改为 `update_potentials_inplace()`
   - `NSCFWorkflow::potinit()` → 应改为 `initialize_potentials()`

4. **双重接口混淆**
   - `LocalPseudo::compute()` 有两个重载，语义不同
   - `Hartree::compute()` 有两个重载，语义不同
   - **建议**: 分离为不同的方法名

5. **依赖关系违规**
   - `src/model/wavefunction.cuh` 依赖 `fft/fft_solver.cuh`（Model → Math）
   - `src/model/density_factory.cu` 依赖 `fft/fft_solver.cuh`（Model → Math）
   - **建议**: 将 FFT 相关逻辑移到 Solver 层

6. **生命周期管理不清晰**
   - `LocalPseudo::grid_ptr_` 使用裸指针
   - `Hartree::grid_` 初始化为 `nullptr`
   - **建议**: 使用引用或共享指针

#### 🟢 轻微问题（优先级 3 - 优化）

7. **构造函数中的多步初始化**
   - `NSCFWorkflow` 构造函数执行 6 步初始化
   - **建议**: 将初始化逻辑移到 `initialize()` 方法

8. **缺少函数标注**
   - 大部分函数缺少 `[PURE]`, `[CONST]`, `[SIDE_EFFECT]` 标注
   - **建议**: 逐步添加标注

### 重构计划

#### 阶段 1：清理 I/O 操作（1 周）

**目标**: 移除所有 C++ 层的文件 I/O

**任务清单**:
- [ ] 移除 `src/model/grid.cu` 中的所有 `printf`
- [ ] 移除 `src/model/density_factory.cu` 中的所有 `printf`
- [ ] 移除 `src/solver/hamiltonian.cu` 中的所有 `fprintf/printf`
- [ ] 删除 `src/solver/nscf.cu` 中的 `dump_*` 方法
- [ ] 通过 Python 侧诊断模式替代（使用 `get_stats()` 和回调）

**预期收益**:
- ✅ 符合架构设计
- ✅ 提高可测试性
- ✅ 减少性能开销

#### 阶段 2：重命名副作用函数（3 天）

**目标**: 明确标注所有有副作用的函数

**任务清单**:
- [ ] `Wavefunction::apply_mask()` → `apply_mask_inplace()`
- [ ] `Wavefunction::orthonormalize()` → `orthonormalize_inplace()`
- [ ] `Wavefunction::force_gamma_constraint()` → `enforce_gamma_constraint_inplace()`
- [ ] `Hamiltonian::update_potentials()` → `update_potentials_inplace()`
- [ ] `NonLocalPseudo::update_projectors()` → `update_projectors_inplace()`
- [ ] `NSCFWorkflow::potinit()` → `initialize_potentials()`
- [ ] 更新所有调用点

**预期收益**:
- ✅ 代码意图更清晰
- ✅ 减少误用
- ✅ 提高可维护性

#### 阶段 3：修复依赖关系（1 周）

**目标**: 修复违反依赖关系的代码

**任务清单**:
- [ ] 将 `Wavefunction::compute_density()` 移到 Solver 层
- [ ] 移除 `src/model/wavefunction.cuh` 中的 `#include "fft/fft_solver.cuh"`
- [ ] 移除 `src/model/density_factory.cu` 中的 FFT 依赖
- [ ] 修复 `LocalPseudo::grid_ptr_` 为引用或共享指针
- [ ] 修复 `Hartree::grid_` 初始化问题

**预期收益**:
- ✅ 依赖关系清晰
- ✅ 易于单元测试
- ✅ 减少耦合

#### 阶段 4：分离双重接口（3 天）

**目标**: 消除重载方法的语义混淆

**任务清单**:
- [ ] `LocalPseudo::compute()` 分离为：
  - `compute_vloc_inplace(RealField& vloc_r)`
  - `compute_potential_from_density(const RealField& rho, RealField& v_out)`
- [ ] `Hartree::compute()` 分离为：
  - `compute_energy(const RealField& rho)`
  - `compute_potential(const RealField& rho, RealField& v_out)`
- [ ] 更新所有调用点

**预期收益**:
- ✅ 接口语义清晰
- ✅ 易于理解和使用
- ✅ 减少错误

#### 阶段 5：添加函数标注（持续进行）

**目标**: 为所有函数添加标注

**任务清单**:
- [ ] 为 Model 层函数添加 `[PURE]`, `[CONST]`, `[FACTORY]` 标注
- [ ] 为 Functional 层函数添加 `[CONST]`, `[KERNEL]` 标注
- [ ] 为 Solver 层函数添加 `[SIDE_EFFECT]`, `[KERNEL]`, `[FUSIBLE]` 标注
- [ ] 为 Workflow 层函数添加 `[SIDE_EFFECT]` 标注

**预期收益**:
- ✅ 代码意图清晰
- ✅ 易于识别优化机会
- ✅ 提高代码质量

### 重构时间表

| 阶段 | 任务 | 工作量 | 优先级 | 预期完成 |
|------|------|--------|--------|---------|
| 阶段 1 | 清理 I/O 操作 | 1 周 | 🔴 高 | Week 1 |
| 阶段 2 | 重命名副作用函数 | 3 天 | 🔴 高 | Week 2 |
| 阶段 3 | 修复依赖关系 | 1 周 | 🟡 中 | Week 3 |
| 阶段 4 | 分离双重接口 | 3 天 | 🟡 中 | Week 4 |
| 阶段 5 | 添加函数标注 | 持续 | 🟢 低 | 持续进行 |

### 重构原则

1. **向后兼容**: 保留旧接口，标记为 `@deprecated`
2. **渐进式迁移**: 逐步迁移现有代码到新接口
3. **测试驱动**: 每次重构前先写测试，确保行为不变
4. **文档同步**: 重构后立即更新文档

详细的代码分析和重构建议请参考 **`docs/CODE_REFACTORING_PLAN.md`**（待创建）。

---
