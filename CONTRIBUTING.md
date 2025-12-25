# Contributing to DFTcu

DFTcu 欢迎各种形式的贡献！本文档提供开发指南和最佳实践。

## 📋 目录

- [快速开始](#快速开始)
- [开发工具](#开发工具)
- [依赖管理](#依赖管理)
- [构建和测试](#构建和测试)
- [代码规范](#代码规范)
- [提交流程](#提交流程)

## 🚀 快速开始

### 环境设置

**一键设置（推荐）**:
```bash
make setup
source .venv/bin/activate
```

这会自动完成：
- ✅ 检查 CUDA、CMake、Python 环境
- ✅ 安装 uv（快速 Python 包管理器）
- ✅ 创建虚拟环境并安装所有依赖
- ✅ 设置 pre-commit 钩子

### 构建项目

```bash
# 配置并构建
make build

# 或使用 CMake presets
cmake --preset=debug
cmake --build --preset=debug
```

### 运行测试

```bash
# 运行所有测试
make test

# 仅 C++ 测试
make test-cpp

# 仅 Python 测试
make test-python
```

## 🛠️ 开发工具

### uv - 快速 Python 包管理

DFTcu 使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖，比 pip 快 10-100 倍。

**添加依赖**:
```bash
# 添加核心依赖
uv add matplotlib

# 添加开发依赖
uv add --dev pytest-asyncio
```

**删除依赖**:
```bash
uv remove matplotlib
```

**同步依赖**:
```bash
# 当 pyproject.toml 或 uv.lock 更新后
uv sync --all-extras
```

**依赖配置**: 所有依赖在 \`pyproject.toml\` 中管理。

### CMake Presets

使用 CMake presets 快速配置不同构建：

```bash
# 列出所有 presets
cmake --list-presets

# 使用 preset 配置
cmake --preset=rtx4090    # RTX 4090 (sm_89)
cmake --preset=debug      # Debug 构建
cmake --preset=release    # Release 构建
```

可用 presets: `default`, `debug`, `release`, `rtx4090`, `rtx3090`, `a100`, `v100`, `multi-gpu`

### Makefile 快捷命令

```bash
make setup        # 完整环境设置
make build        # 构建项目
make test         # 运行所有测试
make format       # 格式化代码
make clean        # 清理构建
make help         # 显示所有命令
```

## 📝 代码规范

### C++/CUDA 代码

使用 \`.clang-format\` 自动格式化：
```bash
make format-cpp
```

**规范**: 缩进 4 空格，行宽 100 字符，命名 snake_case

### Python 代码

使用 black + isort + flake8：
```bash
make format-python
```

**规范**: 遵循 PEP 8，行宽 100 字符

### Pre-commit 钩子

```bash
pre-commit install      # 安装钩子
pre-commit run --all-files  # 手动运行检查
```

## 🔄 提交流程

1. **创建分支**: `git checkout -b feature/your-feature`
2. **开发和测试**: 编写代码，运行 `make test`
3. **格式化代码**: `make format`
4. **提交更改**: `git commit -m "feat: add feature"`
5. **推送并创建 PR**

### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <subject>

feat: 新功能
fix: Bug 修复
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试相关
chore: 构建/工具
```

## 🐛 调试技巧

```bash
# CUDA 调试
cuda-gdb --args ./test_program
cuda-memcheck ./test_program

# Python 调试
python -m pdb tests/python/test_tf_kedf.py
```

## 💡 常见问题

**Q: 虚拟环境损坏？**
```bash
rm -rf .venv && make setup
```

**Q: 更新依赖？**
```bash
uv lock --upgrade && uv sync --all-extras
```

**Q: 构建失败？**
```bash
make clean && make rebuild
```

## 📚 资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [uv Documentation](https://docs.astral.sh/uv/)

感谢您的贡献！🎉
