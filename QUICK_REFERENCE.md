# DFTcu 快速参考卡片

## ⚡ 常用命令

### 构建项目

```bash
# CMake 方式（推荐）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

# 传统 Makefile
make -j4
```

### 运行测试

```bash
# C++ 测试
cd build && ctest --output-on-failure

# Python 测试
export PYTHONPATH=$PWD/build:$PYTHONPATH
pytest tests/ -v

# 单个测试
./build/tests/test_kedf_tf
```

### 代码格式化

```bash
# 自动（推荐）
pre-commit install
git commit -m "message"  # 自动格式化

# 手动
./scripts/format_code.sh

# 分别执行
clang-format -i src/**/*.{cu,cuh}
black .
isort --profile black .
```

### 文档生成

```bash
cd build
make doc
firefox docs/html/index.html
```

---

## 📁 项目结构

```
DFTcu/
├── src/
│   ├── model/          # Grid, Field, Atoms
│   ├── functional/     # DFT functionals
│   │   ├── kedf/       # Kinetic energy functionals
│   │   └── xc/         # Exchange-correlation
│   ├── fft/            # FFT solver
│   ├── utilities/      # Helper functions
│   └── api/            # Python bindings
├── tests/              # Unit tests
├── docs/               # Documentation
├── scripts/            # Helper scripts
└── external/           # Dependencies (DFTpy, GPUMD)
```

---

## 🛠️ 配置文件

| 文件 | 用途 |
|------|------|
| `CMakeLists.txt` | CMake 构建配置 |
| `.clang-format` | C++/CUDA 代码格式 |
| `pyproject.toml` | Python 项目配置 |
| `.pre-commit-config.yaml` | Git hooks |
| `.github/workflows/ci.yml` | CI/CD 流程 |

---

## 🔧 CMake 选项

```bash
# Debug 模式
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# 禁用测试
cmake -B build -DBUILD_TESTING=OFF

# 启用性能分析
cmake -B build -DENABLE_PROFILING=ON

# 指定 CUDA 架构
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86

# 多个架构（通用二进制）
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;86"
```

**GPU 架构对照表**:
| GPU | Compute Cap | CMake 值 |
|-----|-------------|----------|
| RTX 4090 | 8.9 | 89 |
| RTX 3090 | 8.6 | 86 |
| A100 | 8.0 | 80 |
| V100 | 7.0 | 70 |

---

## 🧪 测试命令

```bash
# 运行所有测试
ctest

# 详细输出
ctest --output-on-failure

# 只运行特定测试
ctest -R kedf

# 并行测试
ctest -j4

# 重新运行失败的测试
ctest --rerun-failed
```

---

## 📝 提交规范

```bash
# 格式
<type>(<scope>): <subject>

# 类型
feat:     新功能
fix:      Bug 修复
docs:     文档更新
style:    代码格式
refactor: 重构
test:     测试相关
perf:     性能优化
ci:       CI/CD 配置

# 示例
git commit -m "feat(kedf): add von Weizsäcker functional"
git commit -m "fix(hartree): correct energy normalization"
git commit -m "docs: update installation guide"
```

---

## 🐛 故障排除

### CMake 找不到 CUDA
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cmake -B build -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

### Python 模块导入失败
```bash
export PYTHONPATH=$PWD/build:$PYTHONPATH
python3 -c "import dftcu; print(dftcu.__file__)"
```

### 编译错误: unsupported architecture
```bash
# 检查 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 使用兼容的架构
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=70  # V100+
```

### Pre-commit 失败
```bash
# 跳过 hooks（紧急情况）
git commit --no-verify

# 手动运行修复
pre-commit run --all-files
```

---

## 🚀 开发工作流

```bash
# 1. 创建分支
git checkout -b feature/my-feature

# 2. 开发...
# 编辑 src/...

# 3. 构建
cmake --build build -j

# 4. 测试
cd build && ctest

# 5. 提交
git add .
git commit -m "feat: add my feature"

# 6. 推送
git push origin feature/my-feature

# 7. 创建 Pull Request
```

---

## 📚 常用文档链接

- **完整指南**: `MODERNIZATION_GUIDE.md`
- **贡献指南**: `CONTRIBUTING.md`
- **架构分析**: `ARCHITECTURE_ANALYSIS.md`
- **进展总结**: `PROGRESS_SUMMARY.md`

---

## 💡 提示

1. **首次设置**: 运行 `./scripts/setup_dev.sh`
2. **激活环境**: `source .venv/bin/activate`
3. **检查 GPU**: `nvidia-smi`
4. **查看构建状态**: GitHub Actions 自动运行
5. **需要帮助**: GitHub Issues 或 Discussions

---

**快速开始**: `./scripts/setup_dev.sh && cmake -B build && cmake --build build -j`
