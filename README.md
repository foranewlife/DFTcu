# DFTcu: CUDA-Accelerated DFT

高性能 CUDA 加速的密度泛函理论（DFT）计算库，兼容 [DFTpy](https://gitlab.com/pavanello-research-group/dftpy)，参考 [GPUMD](https://github.com/brucefan1983/GPUMD) 架构设计。

## ✨ 特性

- **CUDA 原生网格操作** - GPU 上高效处理 3D 网格和场
- **FFT 加速** - 集成 cuFFT 实现快速倒空间变换
- **DFT 泛函**
  - Hartree 势：基于 cuFFT 的快速求解器
  - 局域赝势：倒空间局域赝势计算
  - Thomas-Fermi 动能泛函：已验证与 DFTpy 机器精度一致
- **Python 集成** - 通过 pybind11 提供 Pythonic API
- **内存管理** - 自动 GPU 内存管理（GPUMD 风格）
- **现代化构建** - CMake + presets，uv 包管理，完整 CI/CD

## 🚀 快速开始

### 前置要求

- NVIDIA GPU (建议 sm_70+)
- CUDA Toolkit 11.0+
- CMake 3.18+
- Python 3.9+
- C++ 编译器（支持 C++14）

### 一键安装

```bash
# 1. 克隆仓库（包含 submodules）
git clone --recursive https://github.com/your-org/DFTcu.git
cd DFTcu

# 2. 完整环境设置（自动安装依赖）
make setup

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 构建项目
make build

# 5. 运行测试
make test
```

就这么简单！🎉

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
rho = dftcu.RealField(grid, rank=1)
rho_data = np.ones(32**3) * 0.01  # 均匀密度
rho.copy_from_host(rho_data)

# 计算 Thomas-Fermi 动能
tf = dftcu.ThomasFermi(coeff=1.0)
v_kedf = dftcu.RealField(grid, rank=1)
energy = tf.compute(rho, v_kedf)

print(f"TF Energy: {energy} Ha")
```

### C++/CUDA 示例

```cpp
#include "model/grid.cuh"
#include "model/field.cuh"
#include "functional/kedf/tf.cuh"

using namespace dftcu;

int main() {
    // 创建网格
    std::vector<double> lattice = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    std::vector<int> nr = {32, 32, 32};
    Grid grid(lattice, nr);

    // 创建场并计算
    RealField rho(grid, 1);
    RealField v_kedf(grid, 1);

    ThomasFermi tf(1.0);
    double energy = tf.compute(rho, v_kedf);

    return 0;
}
```

## 🛠️ 常用命令

```bash
# 开发环境
make setup          # 完整设置（首次运行）
make sync           # 同步 Python 依赖
make add PKG=X      # 添加 Python 包
make remove PKG=X   # 删除 Python 包

# 构建和测试
make build          # 构建项目
make rebuild        # 清理并重新构建
make test           # 运行所有测试
make test-cpp       # 仅 C++ 测试
make test-python    # 仅 Python 测试

# 代码质量
make format         # 格式化所有代码
make lint           # 运行 linters
make check          # format + lint

# CMake Presets
make preset-debug       # Debug 构建
make preset-release     # Release 构建
make list-presets       # 列出所有 presets

# 其他
make clean          # 清理构建
make clean-all      # 清理所有（包括 .venv）
make help           # 显示所有命令
make info           # 项目信息
```

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

Thomas-Fermi KEDF 实现已通过验证，与 DFTpy 相比：

- ✅ **精度**: 机器精度一致 (相对误差 < 10⁻¹⁵)
- ⚡ **速度**: GPU 加速（具体加速比取决于系统大小）

## 🗺️ 路线图

- [x] 核心网格和场系统
- [x] cuFFT 集成
- [x] Hartree 势求解器
- [x] 局域赝势
- [x] Thomas-Fermi KEDF
- [ ] von Weizsäcker KEDF
- [ ] Wang-Teter 非局域 KEDF
- [ ] LDA 交换关联泛函
- [ ] GGA 交换关联泛函
- [ ] 非局域赝势
- [ ] 自洽场（SCF）求解器

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

**快速链接**: [文档](docs/) | [贡献指南](CONTRIBUTING.md) | [更新日志](CHANGELOG.md)
