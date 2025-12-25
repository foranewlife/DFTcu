# UV 依赖管理工作流程

DFTcu 项目使用 **uv** 进行现代化 Python 依赖管理。

## 快速开始

### 一键设置（推荐）

```bash
make setup
```

这个命令会自动完成：
- ✅ 检查 CUDA、CMake、Python 环境
- ✅ 安装 uv（如果未安装）
- ✅ 创建虚拟环境（.venv）
- ✅ 安装所有依赖（包括开发依赖）
- ✅ 生成 uv.lock 锁定文件
- ✅ 安装 pre-commit 钩子（如果在 git 仓库中）

### 激活虚拟环境

```bash
source .venv/bin/activate
```

## 依赖管理

### 添加新依赖

**核心依赖**（所有用户需要）：
```bash
make add PKG=matplotlib
# 或者
uv add matplotlib
```

**开发依赖**（仅开发者需要）：
```bash
uv add --dev pytest-asyncio
```

**可选依赖组**：
```bash
# 添加到 benchmark 组
uv add --optional benchmark plotly
```

### 删除依赖

```bash
make remove PKG=matplotlib
# 或者
uv remove matplotlib
```

### 同步依赖

当 `pyproject.toml` 或 `uv.lock` 更新后：
```bash
make sync
# 或者
uv sync --all-extras
```

### 更新依赖

```bash
# 更新所有依赖到最新兼容版本
uv lock --upgrade

# 更新特定包
uv lock --upgrade-package numpy
```

## 依赖配置文件

### pyproject.toml

所有依赖在 `pyproject.toml` 中管理：

```toml
[project]
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "pybind11>=2.10.0",
    "ase>=3.22.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    # ...
]
docs = [
    "sphinx>=6.0.0",
    # ...
]
benchmark = [
    "matplotlib>=3.5.0",
    # ...
]
```

### uv.lock

- **自动生成**：由 `uv sync` 或 `uv add` 自动创建/更新
- **版本锁定**：确保所有开发者使用完全相同的依赖版本
- **应该提交**：将 `uv.lock` 提交到版本控制
- **跨平台**：包含所有平台的锁定信息

## 为什么使用 uv？

### 速度对比

| 操作 | pip | uv | 提升 |
|------|-----|----|----|
| 安装依赖 | 45s | 0.5s | **90x** |
| 创建虚拟环境 | 2s | 0.1s | **20x** |
| 解析依赖 | 10s | 0.2s | **50x** |

### 主要优势

1. **极快的速度** - 用 Rust 编写，比 pip 快 10-100 倍
2. **可靠的锁定** - 类似 npm/yarn 的 lockfile 机制
3. **一致性保证** - 跨平台完全相同的依赖树
4. **现代化工具** - 集成虚拟环境管理
5. **向后兼容** - 完全兼容 pip/pyproject.toml 标准

## 常见任务

### 完整的开发设置
```bash
# 一键完成所有设置
make setup

# 激活环境
source .venv/bin/activate

# 构建项目
make build

# 运行测试
make test
```

### 更新项目依赖
```bash
# 拉取最新代码
git pull

# 同步依赖（使用 uv.lock 中的版本）
make sync

# 重新构建
make rebuild
```

### 添加新功能的依赖
```bash
# 添加依赖
make add PKG=requests

# 依赖会自动添加到 pyproject.toml
# uv.lock 会自动更新

# 提交更改
git add pyproject.toml uv.lock
git commit -m "Add requests dependency"
```

### 清理环境
```bash
# 清理构建产物
make clean

# 清理虚拟环境（重新开始）
make clean-all

# 重新设置
make setup
```

## 与旧方式的对比

| 方面 | 旧方式 (requirements.txt) | 新方式 (uv + pyproject.toml) |
|------|-------------------------|----------------------------|
| 配置文件 | requirements.txt, requirements-dev.txt | pyproject.toml (标准) |
| 锁定机制 | pip freeze (不可靠) | uv.lock (可靠) |
| 安装速度 | 慢 (45s) | 快 (0.5s) |
| 依赖管理 | 手动编辑 | `uv add/remove` |
| 虚拟环境 | 手动 venv/virtualenv | uv 自动管理 |
| 标准支持 | 非标准 | PEP 621 标准 |

## 故障排除

### 虚拟环境损坏
```bash
rm -rf .venv
make setup
```

### 依赖冲突
```bash
# 查看冲突详情
uv sync --all-extras

# 如果需要，强制重新解析
rm uv.lock
uv sync --all-extras
```

### Python 版本不匹配
```bash
# 检查 Python 版本
python3 --version

# pyproject.toml 要求 Python 3.9+
# 如果版本不符，需要安装更新的 Python
```

## 参考资源

- [uv 官方文档](https://docs.astral.sh/uv/)
- [PEP 621 - Python项目元数据](https://peps.python.org/pep-0621/)
- [pyproject.toml 规范](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

## Make 命令速查

```bash
make setup      # 完整设置（包含所有步骤）
make sync       # 同步依赖
make add        # 添加依赖 (需要 PKG=<name>)
make remove     # 删除依赖 (需要 PKG=<name>)
make lock       # 更新 uv.lock
make build      # 构建项目
make test       # 运行测试
make clean      # 清理构建
make clean-all  # 清理一切（包括 .venv）
```
