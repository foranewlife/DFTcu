# DFTcu: CUDA-Accelerated DFT

高性能 CUDA 加速的密度泛函理论（DFT）计算库，兼容 [DFTpy](https://gitlab.com/pavanello-research-group/dftpy)，参考 [GPUMD](https://github.com/brucefan1983/GPUMD) 架构设计。

## ✨ 特性

- **🚀 高性能 GPU 计算**
  - CUDA 原生网格操作 - GPU 上高效处理 3D 网格和场
  - cuFFT 加速 - 快速倒空间变换
  - ✅ **混合优化策略** - 结合 cuBLAS (BLAS L1/归约) 与 **表达式模板 (内核融合)**
  - ✅ **计算图优化** - Evaluator 自动融合势能累加操作，减少内存带宽占用
  - 增量编译支持 - 2-5s 快速重建（CMake）/ ~21s（Python 安装）

- **⚛️ DFT 泛函实现**
  - ✅ Hartree 势 - 基于 cuFFT 的快速求解器
  - ✅ Ewald 求和 - 精确离子-离子相互作用
  - ✅ 局域赝势 - 倒空间局域赝势计算
  - ✅ Thomas-Fermi KEDF - 机器精度验证
  - ✅ von Weizsäcker KEDF - 梯度动能修正
  - ✅ Wang-Teter KEDF - 非局域动能泛函
  - ✅ revHC KEDF - 非局域 GGA 动能泛函
  - ✅ LDA 交换关联 - Perdew-Zunger 泛函
  - ✅ GGA 交换关联 - PBE 泛函 (机器精度对标)

- **🔒 现代 C++ 设计**
  - 智能指针 - 完全使用 `std::shared_ptr`/`unique_ptr`
  - RAII 模式 - 自动 GPU 内存管理，修复了跨测试的状态污染
  - 类型擦除 - Functional 包装器实现多态组合
  - 表达式模板 - 实现逐元素运算的内核融合（Kernel Fusion）

- **🐍 Python 集成**
  - pybind11 绑定 - 零拷贝数据传输
  - Editable 安装 - 快速开发迭代
  - NumPy 兼容 - 无缝集成科学计算栈

- **🛠️ 开发者友好**
  - 增量编译 - CMake + uv 构建产物共享
  - 性能剖析 - 集成 **Nsight Systems (nsys)** 工作流
  - 技术博客 - Doxygen 集成了详细的设计笔记与算法分析报告
  - 完整测试 - 31/31 测试通过，覆盖核心算法与优化器

## 🚀 快速开始

### 前置要求

- NVIDIA GPU (建议 Ampere 架构及以上, sm_70+)
- CUDA Toolkit 11.0+ (建议 12.0+)
- CMake 3.18+
- Python 3.9+
- C++ 编译器（支持 C++14）

### 一键安装（推荐开发者）

```bash
# 1. 克隆仓库（包含 submodules）
git clone --recursive https://github.com/your-org/DFTcu.git
cd DFTcu

# 2. 完整环境设置（自动安装依赖）
make setup

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装开发模式（支持增量编译）
make install-dev

# 5. 运行测试
make test-python
```

就这么简单！🎉

**开发工作流**：修改 `.cu` 文件后只需 `make rebuild` (~21s) 即可重新编译并重新加载 C++ 扩展！

### 手动安装

如果你喜欢手动控制每一步：

```bash
# 1. 安装 Python 依赖（使用 uv）
uv sync --all-extras

# 2. 配置构建（选择适合你 GPU 的架构）
# -DBUILD_WITH_CUDA=OFF 可用于无 CUDA 环境的文档构建
cmake -B build -GNinja -DCMAKE_CUDA_ARCHITECTURES=89  # 手动指定

# 3. 构建
cmake --build build -j$(nproc)

# 4. 测试
cd build && ctest
```

## 📖 使用示例

### Python 示例

```python
import dftcu
import numpy as np

# 创建网格 (10 Bohr 立方晶胞)
lattice = np.eye(3) * 10.0
grid = dftcu.Grid(lattice.flatten().tolist(), [32, 32, 32])

# 创建密度场并初始化
rho = dftcu.RealField(grid, 1)
rho.fill(0.01)

# 使用组合式 Evaluator 计算多个泛函
evaluator = dftcu.Evaluator(grid)
evaluator.add_functional(dftcu.ThomasFermi())
evaluator.add_functional(dftcu.LDA_PZ())

# 高效计算总能量和势 (内部利用表达式模板进行 Kernel 融合)
v_tot = dftcu.RealField(grid, 1)
energy = evaluator.compute(rho, v_tot)
```

## 🛠️ 常用命令

```bash
# 📦 安装和开发
make setup           # 完整设置（首次运行）
make install-dev     # 开发模式安装（editable，支持增量编译）⭐
make rebuild         # 增量重建（仅 editable 模式，~21s）⭐

# 🔨 构建
make build           # 构建 C++ 库（2-5s 增量编译）⭐
make configure       # 配置 CMake (需在更改选项后运行)
make clean           # 清理构建产物

# 🧪 测试
make test            # 运行所有测试（C++ + Python）
pytest tests/python/test_benchmark_optimizers.py  # 运行基准测试

# 🎨 代码质量与文档
make format          # 格式化所有代码 (clang-format + black + isort)
make doc             # 生成 Doxygen 文档 (含技术博客) ⭐
```

**详细指南**：查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解 **Nsight Systems 性能剖析** 和 **增量编译机制**。

## 📁 项目结构

```
DFTcu/
├── src/                   # C++/CUDA 源代码
│   ├── model/            # Grid, Field, Atoms 类
│   ├── fft/              # FFT solver (cuFFT 封装)
│   ├── math/             # LineSearch, DCSRCH, 表达式模板
│   ├── functional/       # DFT 泛函 (TF, vW, revHC, LDA, PBE)
│   ├── solver/           # Evaluator, Optimizers (TN, CG, SD)
│   ├── utilities/        # CublasManager, NVRTCManager, Kernels
│   └── api/              # Python 绑定 (pybind11)
├── docs/                  # 文档与博客
│   ├── blog/             # 技术笔记与算法分析报告
│   └── Doxyfile.in       # Doxygen 配置
├── tests/                 # 完整测试套件
└── ...
```

## 📊 性能

### 精度验证（vs DFTpy）

所有泛函已通过高精度验证（误差远低于物理显着性）：

| 泛函 | 绝对误差 (Ha) | 状态 |
|------|----------|------|
| Thomas-Fermi / vW | < 10⁻¹⁵ | ✅ |
| Wang-Teter / revHC | < 10⁻¹³ | ✅ |
| LDA (PZ) / GGA (PBE) | < 10⁻¹⁴ | ✅ |
| Hartree / Ewald | < 10⁻¹³ | ✅ |

### 速度提升（vs DFTpy）

| 测试 | DFTpy (CPU) | DFTcu (GPU) | 加速比 |
|------|-------|-------|--------|
| 初始能量计算 | 30.8 ms | 4.6 ms | **6.7x** |
| 优化器迭代 (TN) | - | - | 显著提升 |

**测试配置**：NVIDIA RTX 4090 (sm_89), 32³ 网格

## 🗺️ 路线图

### ✅ 已完成 (v0.1.0)

- [x] 核心网格和场系统（表达式模板支持）
- [x] cuFFT & cuBLAS 深度集成
- [x] 完整泛函支持：Thomas-Fermi, vW, revHC, Hartree, Ewald, LDA, PBE
- [x] **直接密度优化器**：截断牛顿法 (TN), 共轭梯度 (CG), 最速下降 (SD)
- [x] 兼容 SciPy 的线搜索 (Line Search) 模块
- [x] **性能基准测试脚本**
- [x] 高级 Doxygen 文档系统 (集成技术博客)
- [x] 增量编译与 CI/CD 流程优化

### 🚧 进行中

- [ ] 更多 XC 泛函（如 SCAN）
- [ ] 非局域赝势 (Non-local pseudopotentials)
- [ ] GPU 多卡并行支持

### 📋 计划中

- [ ] 自适应网格支持
- [ ] 与分子动力学 (MD) 引擎集成
- [ ] 时间依赖 DFT (TD-DFT)
