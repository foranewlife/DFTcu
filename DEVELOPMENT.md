/**
 * \page dev_guide 开发者指南
 */
# DFTcu 开发指南

## 构建方式对比

DFTcu 提供三种构建方式，各有优劣：

| 方式 | 命令 | 首次编译 | 增量编译 | 适用场景 |
|------|------|----------|----------|----------|
| **标准安装** | `make install` | ~26s | ~26s | 发布、最终测试 |
| **开发模式** | `make install-dev` | ~26s | ~21s | 日常开发 (推荐) |
| **CMake直接构建** | `make build` | ~25s | <5s | C++开发、调试 |

## 📦 构建产物共享机制

### 关键配置

1. **统一构建目录** (`pyproject.toml`):
   ```toml
   [tool.scikit-build]
   build-dir = "build"  # 不再使用 build/{wheel_tag}
   ```

2. **统一构建系统** - 使用 Ninja:
   - `make build` 使用 `-GNinja`
   - `uv pip install` 默认使用 Ninja
   - 两者共享 `build/` 目录和 `build.ninja` 文件

### 工作原理

```
┌─────────────────────────────────────────┐
│         DFTcu Build Workflow            │
├─────────────────────────────────────────┤
│                                         │
│  1️⃣  make build (CMake + Ninja)         │
│     ├─ 配置: build/CMakeCache.txt      │
│     ├─ 构建规则: build/build.ninja     │
│     └─ 产物: build/*.so, build/*.a     │
│                    ↓                    │
│  2️⃣  make rebuild OR uv pip install    │
│     ├─ 复用 build/ 目录                │
│     ├─ Ninja 检测变更文件              │
│     └─ 只编译修改的 .cu 文件 (增量)   │
│                                         │
└─────────────────────────────────────────┘
```

## 🚀 推荐的开发流程

### 方案 A：纯 CMake（C++ 开发者）

```bash
# 首次构建
make build

# 修改 .cu 文件后
make rebuild    # 超快，只编译变更的文件

# 需要测试 Python 接口时
uv pip install -e . --no-deps  # 使用已有构建产物，21s
pytest tests/python/

# 或者一步到位
make build-install  # 构建 C++ + 自动安装 Python包
```

### 方案 B：Python 优先（推荐日常开发）

```bash
# 首次安装（开发模式）
make install-dev

# 修改 .cu 文件后
make rebuild        # 增量编译 + 安装，~21s
pytest tests/python/

# 或直接
CUDACXX=/usr/local/cuda/bin/nvcc uv pip install -e . --no-deps
```

## 🔧 详细命令说明

### 安装命令

```bash
# 标准安装（每次全新编译）
make install
# 等价于：CUDACXX=/usr/local/cuda/bin/nvcc uv pip install .

# 开发模式安装（editable，支持增量）
make install-dev
# 等价于：CUDACXX=/usr/local/cuda/bin/nvcc uv pip install -e .

# 增量重建（仅开发模式有效）
make rebuild
# 等价于：CUDACXX=/usr/local/cuda/bin/nvcc uv pip install -e . --no-deps
```

### CMake 命令

```bash
# 配置项目
make configure

# 仅构建 C++
make build

# 构建 C++ + 自动安装 Python
make build-install  # 内部调用 uv pip install -e .

# 清理
make clean

# 运行测试
make test            # C++ + Python 测试
make test-python     # 仅 Python 测试
```

## ⚡ 增量编译加速技巧

### 1. 使用 `build/` 目录缓存

```bash
# ❌ 每次删除 build/ 会导致全量重编译
rm -rf build && make install-dev

# ✅ 保留 build/ 目录，只清理特定文件
cmake --build build --target clean
make rebuild
```

### 2. 修改单个文件后的最快流程

```bash
# 编辑 src/functional/kedf/vw.cu
vim src/functional/kedf/vw.cu

# 方案1：纯 CMake（最快，<5s）
make build
# 手动测试 C++ 部分

# 方案2：CMake + Python 安装（~21s）
make rebuild
pytest tests/python/test_vw_kedf.py
```

### 3. 并行编译

```bash
# Ninja 自动并行，无需额外配置
cmake --build build    # 自动使用所有 CPU 核心

# 或显式指定
cmake --build build -j$(nproc)
```

## 📊 性能对比

以下是修改单个 `.cu` 文件后的重新编译时间：

| 场景 | 时间 | 说明 |
|------|------|------|
| `make build` | 2-5s | Ninja 增量编译，只编译变更文件 |
| `make rebuild` | ~21s | CMake + Python 安装 |
| `make install` | ~26s | 全量重编译（不推荐开发时使用）|

## 🛠️ 高级功能

### 自动 Python 安装（CMake 触发）

```bash
# CMake 构建完成后自动运行 uv pip install
cmake -B build -DAUTO_PIP_INSTALL=ON -GNinja \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DBUILD_TESTING=OFF
cmake --build build

# 或使用 Makefile
make build-install
```

### Editable Install 说明

开发模式（`-e`）的工作原理：

```python
# 安装后，Python import 直接指向源码目录
import dftcu  # → 加载 build/dftcu.cpython-311-x86_64-linux-gnu.so

# 修改 .cu 文件后
# 1. 不需要重新 install（Python 路径不变）
# 2. 只需 make rebuild 重新编译 .so
# 3. 重新 import 自动加载新的 .so
```

配置参数（`pyproject.toml`）：

```toml
[tool.scikit-build]
editable.mode = "redirect"      # 使用符号链接模式
editable.rebuild = false        # 禁用自动重建（手动控制更好）
editable.verbose = true         # 显示详细构建信息
```

## 🐛 常见问题

### Q1: `ninja: error: Makefile:5: expected '=', got ':'`

**原因**：CMake 使用了 Unix Makefiles，但 `uv pip install` 使用 Ninja

**解决**：
```bash
rm -rf build
cmake -B build -GNinja ...  # 必须使用 -GNinja
```

### Q2: 修改代码后测试未更新

**原因**：未重新编译 `.so` 文件

**解决**：
```bash
make rebuild  # 重新编译并安装
```

### Q3: 构建产物未共享

**检查**：
```bash
# 应该看到 build.ninja，不是 Makefile
ls build/build.ninja  # ✅
ls build/Makefile     # ❌ 如果存在，删除 build/ 重新配置
```

## 📝 最佳实践

1. **首次设置**
   ```bash
   make install-dev
   ```

2. **日常开发循环**
   ```bash
   # 编辑 .cu 文件
   vim src/functional/kedf/wt.cu

   # 增量重建
   make rebuild

   # 运行测试
   pytest tests/python/test_wt_kedf.py -v
   ```

3. **提交前完整测试**
   ```bash
   make clean
   make install-dev
   make test-python
   ```

4. **性能调优开发**
   ```bash
   # 仅编译 C++，快速迭代
   make build

   # 完成后安装 Python
   uv pip install -e . --no-deps
   ```

## ⚡ 性能剖析指南

为了深入分析 GPU 代码的性能瓶颈，您可以使用 NVIDIA Nsight Systems (nsys) 工具。

### 1. 配置和构建项目

首先，您需要以启用性能剖析的模式重新配置和构建项目。这将确保 `nvcc` 使用 `-lineinfo` 标志编译 CUDA 代码，使剖析报告能够精确映射到源代码行。

```bash
# 清理旧的构建产物
make clean

# 配置 CMake，并启用 ENABLE_PROFILING
cmake -B build -GNinja -DENABLE_PROFILING=ON

# 构建项目
make build

# 安装到开发环境
make install-dev
```

### 2. 运行性能剖析器 (nsys)

安装完成后，您可以使用 `nsys profile` 命令运行您感兴趣的测试或代码。以剖析 Python 基准测试为例：

```bash
# 运行 nsys 剖析 Python 基准测试
# --force-overwrite true: 强制覆盖旧报告
# -w true: 等待被剖析的进程结束
# -o profiling_report: 指定报告文件名 (将生成 profiling_report.nsys-rep)
nsys profile --force-overwrite true -w true -o profiling_report .venv/bin/pytest tests/python/test_benchmark_optimizers.py
```

### 3. 查看剖析报告

`nsys profile` 命令执行完成后，会在当前目录下生成一个 `.nsys-rep` 文件（例如 `profiling_report.nsys-rep`）。

您可以使用 NVIDIA Nsight Systems GUI 工具打开这个报告文件，进行详细的性能分析，包括：

*   GPU Kernel 启动时间
*   CPU-GPU 同步开销
*   内存传输模式
*   CUDA API 调用追踪
*   Kernel 的源代码级性能分析

通过分析这些数据，您可以识别出代码中的热点区域和性能瓶颈，从而进行有针对性的优化。
