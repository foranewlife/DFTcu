# DFTcu: CUDA-Accelerated DFT

高性能 CUDA 加速的密度泛函理论（DFT）计算库，兼容 [DFTpy](https://gitlab.com/pavanello-research-group/dftpy)，参考 [GPUMD](https://github.com/brucefan1983/GPUMD) 架构设计。

## ✨ 特性

- **🚀 高性能 GPU 计算**
  - CUDA 原生网格操作 - GPU 上高效处理 3D 网格和场
  - cuFFT 加速 - 快速倒空间变换
  - 增量编译支持 - 2-5s 快速重建（CMake）/ ~21s（Python 安装）

- **⚛️ DFT 泛函实现**
  - ✅ Hartree 势 - 基于 cuFFT 的快速求解器
  - ✅ Ewald 求和 - 精确离子-离子相互作用
  - ✅ 局域赝势 - 倒空间局域赝势计算
  - ✅ Thomas-Fermi KEDF - 机器精度验证
  - ✅ von Weizsäcker KEDF - 梯度动能修正
  - ✅ Wang-Teter KEDF - 非局域动能泛函
  - ✅ LDA 交换关联 - Perdew-Zunger 泛函

- **🔒 现代 C++ 设计**
  - 智能指针 - 完全使用 `std::shared_ptr`/`unique_ptr`
  - RAII 模式 - 自动 GPU 内存管理，零泄漏
  - 类型擦除 - Functional 包装器实现多态组合
  - 移动语义 - 防止意外拷贝

- **🐍 Python 集成**
  - pybind11 绑定 - 零拷贝数据传输
  - Editable 安装 - 快速开发迭代
  - NumPy 兼容 - 无缝集成科学计算栈

- **🛠️ 开发者友好**
  - 增量编译 - CMake + uv 构建产物共享
  - 完整测试 - 17/17 测试通过，覆盖所有核心功能
  - 详细文档 - API 文档 + 开发指南
  - 现代工具链 - CMake presets + uv + pre-commit hooks

## 🚀 快速开始

### 前置要求

- NVIDIA GPU (建议 sm_70+)
- CUDA Toolkit 11.0+
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

**开发工作流**：修改 `.cu` 文件后只需 `make rebuild` (~21s) 即可重新编译和安装！

### 手动安装

如果你喜欢手动控制每一步：

```bash
# 1. 安装 Python 依赖（使用 uv）
uv sync --all-extras

# 2. 配置构建（选择适合你 GPU 的架构）
cmake --preset=rtx4090    # RTX 4090
# 或
cmake --preset=a100       # A100
# 或
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86  # 手动指定

# 3. 构建
cmake --build build -j8

# 4. 测试
cd build && ctest
```

## 📖 使用示例

### Python 示例

```python
import dftcu
import numpy as np

# 创建网格
lattice = np.eye(3) * 10.0  # 10 Bohr 立方晶胞
grid = dftcu.Grid(lattice.flatten().tolist(), [32, 32, 32])

# 创建密度场
rho = dftcu.RealField(grid, 1)
rho_data = np.ones(32**3) * 0.01  # 均匀密度
rho.copy_from_host(rho_data)

# 使用组合式 Evaluator 计算多个泛函
evaluator = dftcu.Evaluator(grid)
evaluator.add_functional(dftcu.ThomasFermi(coeff=1.0))
evaluator.add_functional(dftcu.vonWeizsacker(coeff=1.0))
evaluator.add_functional(dftcu.WangTeter(coeff=1.0))

# 一次计算所有能量和势
v_tot = dftcu.RealField(grid, 1)
total_energy = evaluator.compute(rho, v_tot)

print(f"Total Energy: {total_energy} Ha")
```

### C++/CUDA 示例

```cpp
#include "model/grid.cuh"
#include "model/field.cuh"
#include "solver/evaluator.cuh"
#include "functional/kedf/tf.cuh"
#include "functional/kedf/vw.cuh"
#include <memory>

using namespace dftcu;

int main() {
    // 创建网格（使用 shared_ptr）
    std::vector<double> lattice = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    std::vector<int> nr = {32, 32, 32};
    auto grid = std::make_shared<Grid>(lattice, nr);

    // 创建场
    RealField rho(grid, 1);
    RealField v_tot(grid, 1);

    // 使用 Evaluator 组合多个泛函
    Evaluator evaluator(grid);
    evaluator.add_functional(Functional(std::make_shared<ThomasFermi>(1.0)));
    evaluator.add_functional(Functional(std::make_shared<vonWeizsacker>(1.0)));

    // 计算总能量
    double total_energy = evaluator.compute(rho, v_tot);

    return 0;
}
```

## 🛠️ 常用命令

```bash
# 📦 安装和开发
make setup           # 完整设置（首次运行）
make install-dev     # 开发模式安装（editable，支持增量编译）⭐
make rebuild         # 增量重建（仅 editable 模式，~21s）⭐
make install         # 标准安装（全量编译，~26s）

# 🔨 构建
make build           # 构建 C++ 库（2-5s 增量编译）⭐
make build-install   # 构建 C++ + 自动安装 Python
make configure       # 配置 CMake
make clean           # 清理构建产物

# 🧪 测试
make test            # 运行所有测试（C++ + Python）
make test-python     # 仅 Python 测试（推荐）⭐
make test-cpp        # 仅 C++ 测试

# 🎨 代码质量
make format          # 格式化所有代码
make lint            # 运行 linters

# 📚 文档
make doc             # 生成 Doxygen 文档

# 🐍 Python 依赖
make sync            # 同步依赖（uv sync）

# ℹ️ 其他
make help            # 显示所有命令
make info            # 项目信息
```

**⭐ 开发推荐流程**：
```bash
make install-dev     # 首次安装
# ... 编辑 .cu 文件 ...
make rebuild         # 快速增量编译
pytest tests/python/ # 运行测试
```

**详细指南**：查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解增量编译和构建产物共享机制。

## 📁 项目结构

```
DFTcu/
├── src/                   # C++/CUDA 源代码
│   ├── model/            # Grid, Field, Atoms 类
│   ├── fft/              # FFT solver (cuFFT 封装)
│   ├── functional/       # DFT 泛函
│   │   ├── kedf/        # 动能密度泛函
│   │   └── xc/          # 交换关联（未来）
│   ├── utilities/        # 工具函数和 kernels
│   └── api/              # Python 绑定 (pybind11)
├── tests/                 # 测试
│   ├── python/           # Python 测试 (pytest)
│   └── test_*.cu         # C++ 测试 (Google Test)
├── docs/                  # 文档配置 (Doxygen)
├── scripts/               # 辅助脚本
├── external/              # Git submodules
│   ├── DFTpy/            # Python DFT 参考
│   └── GPUMD/            # GPU MD 架构参考
├── CMakeLists.txt         # CMake 构建配置
├── CMakePresets.json      # CMake 预设
├── Makefile               # 便捷命令封装
├── pyproject.toml         # Python 项目配置
└── uv.lock                # 依赖锁定文件
```

## 🔧 CMake Presets

预配置的构建配置，适用于不同 GPU 和场景：

| Preset | 描述 | CUDA Arch |
|--------|------|-----------|
| `default` | 默认 Release 构建 | sm_86 |
| `debug` | Debug 构建，含符号 | sm_86 |
| `release` | 优化的 Release 构建 | sm_86 |
| `rtx4090` | RTX 4090 优化 | sm_89 |
| `rtx3090` | RTX 3090 优化 | sm_86 |
| `a100` | A100 优化 | sm_80 |
| `v100` | V100 优化 | sm_70 |
| `multi-gpu` | 多 GPU 架构支持 | 70;80;86;89 |
| `profile` | 性能分析构建 | sm_86 |

使用方法：
```bash
cmake --preset=rtx4090
cmake --build --preset=rtx4090
```

## 📦 依赖管理

DFTcu 使用 **uv** - 比 pip 快 10-100 倍的 Python 包管理器。

所有依赖在 `pyproject.toml` 中管理：
- **核心依赖**: numpy, scipy, pybind11, ase
- **开发工具**: pytest, black, flake8, isort, mypy, pre-commit
- **文档**: sphinx, sphinx-rtd-theme
- **基准测试**: matplotlib, pandas, jupyter

```bash
# 添加新包
uv add requests

# 删除包
uv remove requests

# 同步依赖
uv sync --all-extras

# 更新所有包
uv lock --upgrade
```

## 🤝 贡献

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：

- 开发环境设置
- 代码规范和风格指南
- 测试要求
- 提交流程
- 调试技巧

### 快速贡献流程

```bash
# 1. Fork 并克隆
git clone https://github.com/your-username/DFTcu.git
cd DFTcu

# 2. 创建分支
git checkout -b feature/your-feature

# 3. 开发
make setup
source .venv/bin/activate
# ... 编写代码 ...

# 4. 测试和格式化
make format
make test

# 5. 提交
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

## 📊 性能

### 精度验证（vs DFTpy）

所有泛函已通过高精度验证：

| 泛函 | 绝对误差 | 状态 |
|------|----------|------|
| Thomas-Fermi | < 10⁻¹⁵ Ha | ✅ |
| von Weizsäcker | < 10⁻¹⁵ Ha | ✅ |
| Wang-Teter (NL) | < 10⁻¹⁵ Ha | ✅ |
| LDA XC (PZ) | < 10⁻¹⁴ Ha | ✅ |
| Hartree | < 10⁻¹⁵ Ha | ✅ |
| Local Pseudo | < 10⁻¹⁴ Ha | ✅ |
| Ewald | < 10⁻¹³ Ha | ✅ |

**测试系统**：FCC Al (4原子，32³网格)，总能量误差 < 10⁻¹³ Ha

### 速度提升（vs DFTpy）

| 测试 | DFTpy | DFTcu | 加速比 |
|------|-------|-------|--------|
| 初始能量计算 | 30.8 ms | 4.6 ms | **6.7x** |
| TF KEDF | - | - | ~10x |

**测试配置**：NVIDIA GPU (sm_89), 32³网格

### 编译性能

| 构建方式 | 首次编译 | 增量编译 | 适用场景 |
|----------|----------|----------|----------|
| `make build` | ~25s | **2-5s** | C++ 开发 |
| `make rebuild` | ~26s | **~21s** | Python 开发 |
| `make install` | ~26s | ~26s | 发布构建 |

## 🗺️ 路线图

### ✅ 已完成 (v0.1.0)

- [x] 核心网格和场系统（智能指针 + RAII）
- [x] cuFFT 集成（正确归一化）
- [x] Hartree 势求解器
- [x] Ewald 求和（精确 + 快速算法）
- [x] 局域赝势
- [x] Thomas-Fermi KEDF
- [x] von Weizsäcker KEDF
- [x] Wang-Teter 非局域 KEDF
- [x] LDA 交换关联泛函（Perdew-Zunger）
- [x] Evaluator 组合式设计
- [x] SCF 优化器（DIIS + Anderson）
- [x] 增量编译支持
- [x] 完整测试覆盖（17/17 通过）

### 🚧 进行中

- [ ] 性能基准测试套件
- [ ] 更多 XC 泛函（PBE, SCAN）
- [ ] 非局域赝势
- [ ] GPU 多卡支持

### 📋 计划中

- [ ] 自适应网格
- [ ] 分子动力学集成
- [ ] 响应函数计算
- [ ] 时间依赖 DFT

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [DFTpy](https://gitlab.com/pavanello-research-group/dftpy) - Python DFT 框架
- [GPUMD](https://github.com/brucefan1983/GPUMD) - GPU 分子动力学，架构参考
- [pybind11](https://github.com/pybind/pybind11) - C++/Python 绑定
- [uv](https://github.com/astral-sh/uv) - 快速 Python 包管理器

## 📮 联系方式

- 问题反馈: [GitHub Issues](https://github.com/your-org/DFTcu/issues)
- 功能讨论: [GitHub Discussions](https://github.com/your-org/DFTcu/discussions)

---

**快速链接**: [开发指南](DEVELOPMENT.md) | [贡献指南](CONTRIBUTING.md) | [API 文档](docs/)

**版本**: v0.1.0 | **测试**: 17/17 通过 ✅ | **构建**: Ninja + uv
