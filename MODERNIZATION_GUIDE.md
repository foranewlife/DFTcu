# DFTcu 现代化项目管理指南

## 概述

本文档介绍了 DFTcu 项目采用的现代化开发工具和最佳实践。

---

## 🎯 已实现的现代化工具

### 1. 版本管理 ✓

#### Git 配置
- **`.gitignore`**: 完善的忽略规则
  - Python 编译文件 (`*.pyc`, `__pycache__`)
  - CUDA 编译产物 (`*.o`, `*.ptx`, `*.so`)
  - 构建目录 (`build/`, `cmake-build-*`)
  - IDE 配置 (`.vscode/`, `.idea/`)
  - 文档生成 (`docs/_build/`, `docs/html/`)

#### 使用方法
```bash
# 初始化仓库（如果还没有）
git init
git add .
git commit -m "chore: initial commit with modern tooling"

# 推荐的分支策略
git checkout -b develop  # 开发分支
git checkout -b feature/new-kedf  # 功能分支
```

---

### 2. 代码格式化 ✓

#### clang-format (C++/CUDA)
- **配置文件**: `.clang-format`
- **风格**: 基于 Google Style，4 空格缩进，100 字符行宽
- **自动格式化**:
  ```bash
  # 格式化所有源文件
  find src -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

  # 或使用脚本
  ./scripts/format_code.sh
  ```

#### black & isort (Python)
- **black**: 零配置的 Python 格式化工具
- **isort**: 自动排序导入
- **使用**:
  ```bash
  black .
  isort --profile black .
  ```

#### EditorConfig
- **`.editorconfig`**: 跨编辑器的代码风格配置
- 支持 VSCode, Vim, Emacs, IntelliJ 等

---

### 3. 构建系统 - CMake ✓

#### 为什么选择 CMake？
相比传统 Makefile：
- ✅ 跨平台支持 (Linux, Windows, macOS)
- ✅ 自动依赖管理
- ✅ 更好的 IDE 集成
- ✅ 现代化的包管理 (FetchContent)
- ✅ 内置测试框架支持

#### 使用方法

**基本构建**:
```bash
# 配置（首次或修改 CMakeLists.txt 后）
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89

# 编译
cmake --build build -j$(nproc)

# 安装
cmake --install build --prefix ~/.local
```

**高级选项**:
```bash
# Debug 构建（带符号）
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# 启用性能分析
cmake -B build -DENABLE_PROFILING=ON

# 禁用测试构建（加快编译）
cmake -B build -DBUILD_TESTING=OFF

# 指定 CUDA 架构
# RTX 4090: 89, RTX 3090: 86, A100: 80
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

**与传统 Makefile 对比**:
```bash
# 旧方式
make clean
make -j4

# 新方式
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

### 4. 测试框架 ✓

#### Google Test (C++ 单元测试)
- **自动集成**: CMake FetchContent 自动下载
- **运行测试**:
  ```bash
  cd build
  ctest --output-on-failure

  # 或运行单个测试
  ./tests/test_kedf_tf -v
  ```

#### pytest (Python 测试)
- **配置**: `pyproject.toml` 中的 `[tool.pytest]`
- **运行**:
  ```bash
  export PYTHONPATH=$PWD/build:$PYTHONPATH
  pytest tests/ -v
  ```

#### 测试覆盖率
```bash
# C++ 覆盖率（需要 gcov）
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
cmake --build build
cd build && ctest
gcovr -r ..

# Python 覆盖率
pytest --cov=. --cov-report=html
```

---

### 5. 文档生成 - Doxygen ✓

#### 配置
- **`docs/Doxyfile.in`**: Doxygen 配置模板
- **自动识别**: CUDA 文件 (`*.cu`, `*.cuh`)
- **输出**: HTML + 调用图

#### 生成文档
```bash
cd build
make doc

# 查看文档
firefox docs/html/index.html
```

#### Doxygen 注释示例
```cpp
/**
 * @brief Compute Thomas-Fermi kinetic energy
 *
 * Implements the Thomas-Fermi functional:
 * \f$ E_{TF}[\rho] = C_{TF} \int \rho^{5/3}(\mathbf{r}) d\mathbf{r} \f$
 *
 * @param rho Input electron density field
 * @param v_kedf Output potential \f$ \delta E / \delta \rho \f$
 * @return Total kinetic energy in Hartree atomic units
 *
 * @note Requires density > 0 to avoid numerical issues
 * @see KEDF_Base for interface documentation
 */
double compute(const RealField& rho, RealField& v_kedf);
```

---

### 6. Pre-commit Hooks ✓

#### 安装
```bash
pip install pre-commit
pre-commit install
```

#### 功能
每次 `git commit` 前自动执行：
- ✅ 代码格式化 (clang-format, black)
- ✅ 导入排序 (isort)
- ✅ 语法检查 (flake8)
- ✅ 文件检查（尾随空格、大文件等）
- ✅ YAML 验证

#### 跳过 Hooks（紧急情况）
```bash
git commit --no-verify -m "urgent fix"
```

#### 手动运行所有 Hooks
```bash
pre-commit run --all-files
```

---

### 7. 持续集成 - GitHub Actions ✓

#### CI 工作流 (`.github/workflows/ci.yml`)

**触发条件**:
- Push 到 `main` 或 `develop` 分支
- 所有 Pull Requests

**执行内容**:
1. **代码格式检查**: clang-format, black, flake8
2. **编译**: CUDA 容器环境，CMake 构建
3. **测试**: C++ 单元测试 + Python 测试
4. **文档**: Doxygen 构建验证

#### 性能基准工作流 (`.github/workflows/benchmark.yml`)

**触发条件**:
- Push 到 `main`（发布）
- 每周一定时运行
- 手动触发

**需要**: 自建 GPU Runner

#### 查看 CI 结果
- 在 GitHub PR 页面自动显示
- 绿色勾号 ✓ = 通过
- 红色叉 ✗ = 失败，点击查看详情

---

### 8. Python 项目配置 - pyproject.toml ✓

#### 现代化 Python 配置
单文件管理所有工具配置：
- **项目元数据**: 名称、版本、依赖
- **black**: 格式化规则
- **isort**: 导入排序
- **pytest**: 测试配置
- **coverage**: 覆盖率设置

#### 依赖管理
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装文档依赖
pip install -e ".[docs]"

# 安装所有
pip install -e ".[dev,docs,benchmark]"
```

---

## 🚀 快速开始

### 首次设置
```bash
# 1. 运行自动化设置脚本
./scripts/setup_dev.sh

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 配置 CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 4. 构建
cmake --build build -j$(nproc)

# 5. 运行测试
cd build && ctest
```

### 日常开发流程

```bash
# 1. 创建功能分支
git checkout -b feature/my-new-feature

# 2. 编码...

# 3. 格式化（可选，pre-commit 会自动做）
./scripts/format_code.sh

# 4. 运行测试
cd build && ctest

# 5. 提交（pre-commit 自动检查）
git add .
git commit -m "feat(kedf): add new functional"

# 6. 推送并创建 PR
git push origin feature/my-new-feature
```

---

## 📊 工具对比

### 构建系统

| 特性 | Makefile | CMake |
|------|----------|-------|
| 跨平台 | ❌ | ✅ |
| 自动依赖 | ❌ | ✅ |
| IDE 集成 | ⚠️ | ✅ |
| 并行构建 | ✅ | ✅ |
| 包管理 | ❌ | ✅ (FetchContent) |
| 学习曲线 | 低 | 中 |
| **推荐** | 简单项目 | **现代项目** ✓ |

### 测试框架

| 框架 | 语言 | 特点 |
|------|------|------|
| Google Test | C++ | 业界标准，断言丰富 |
| pytest | Python | 简洁，插件丰富 |
| CTest | CMake | 跨语言测试运行器 |

### CI/CD 平台

| 平台 | 优点 | 缺点 |
|------|------|------|
| **GitHub Actions** ✓ | 免费，集成好，Docker 支持 | GPU 需自建 runner |
| GitLab CI | 自托管，功能强 | 配置复杂 |
| Travis CI | 简单 | 免费额度少 |

---

## 🛠️ 故障排除

### CMake 找不到 CUDA
```bash
# 设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cmake -B build -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

### clang-format 未安装
```bash
# Ubuntu/Debian
sudo apt install clang-format

# macOS
brew install clang-format

# 或使用 pre-commit 自动下载
pre-commit install
```

### 测试失败
```bash
# 详细输出
cd build
ctest --output-on-failure --verbose

# 单独运行失败的测试
./tests/test_kedf_tf --gtest_filter=ThomasFermiTest.UniformDensity
```

### Python 模块导入失败
```bash
# 确保构建目录在 PYTHONPATH
export PYTHONPATH=$PWD/build:$PYTHONPATH

# 或直接在 build 目录运行
cd build
python3 -c "import dftcu; print(dftcu.__file__)"
```

---

## 📚 推荐资源

### 学习资源
- **CMake**: [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- **Google Test**: [Primer](https://google.github.io/googletest/primer.html)
- **Pre-commit**: [Documentation](https://pre-commit.com/)
- **Doxygen**: [Manual](https://www.doxygen.nl/manual/)

### 最佳实践
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## 🎉 总结

DFTcu 现在采用现代化工具链：

✅ **版本管理**: Git + .gitignore
✅ **代码格式**: clang-format + black + EditorConfig
✅ **构建系统**: CMake（替代 Makefile）
✅ **测试框架**: Google Test + pytest
✅ **文档生成**: Doxygen
✅ **自动化检查**: Pre-commit hooks
✅ **持续集成**: GitHub Actions
✅ **依赖管理**: pyproject.toml
✅ **开发脚本**: setup_dev.sh, format_code.sh

这些工具使得：
- 代码质量更高（自动格式化）
- 协作更顺畅（统一风格）
- 测试更可靠（自动化 CI）
- 文档更完善（Doxygen）
- 开发更高效（CMake 并行构建）

---

**开始使用**: `./scripts/setup_dev.sh`

**贡献指南**: 参见 `CONTRIBUTING.md`

**问题反馈**: GitHub Issues
