# DFTcu Python Tests

完整的 Python 测试套件，验证所有 DFT 泛函的精度和性能。

## 测试概览

**测试统计**: 17/17 通过 ✅
**精度**: 所有泛函与 DFTpy 误差 < 10⁻¹³ Ha
**性能**: 平均加速 6.7x（初始能量计算）

## 运行测试

**快速运行所有测试**:
```bash
make test-python
# 或
pytest tests/python/ -v
```

**运行特定测试**:
```bash
pytest tests/python/test_tf_kedf.py -v
pytest tests/python/test_initial_energies.py -v -s  # 显示详细输出
```

## 测试分类

### 🔋 动能密度泛函 (KEDF)

#### test_tf_kedf.py
Thomas-Fermi KEDF 基础测试，验证机器精度一致性。

**运行**:
```bash
pytest tests/python/test_tf_kedf.py -v
```

#### test_vw_kedf.py
von Weizsäcker KEDF 测试，包含梯度动能修正。

**运行**:
```bash
pytest tests/python/test_vw_kedf.py -v
```

#### test_wt_kedf.py
Wang-Teter 非局域 KEDF 测试，验证倒空间卷积计算。

**运行**:
```bash
pytest tests/python/test_wt_kedf.py -v
```

### ⚡ 静电和赝势

#### test_hartree_pseudo.py
Hartree 势和局域赝势联合测试，验证泊松方程求解器。

**运行**:
```bash
pytest tests/python/test_hartree_pseudo.py -v
```

#### test_ewald.py
Ewald 求和单系统测试，验证离子-离子相互作用计算。

**运行**:
```bash
pytest tests/python/test_ewald.py -v
```

#### test_ewald_multi.py
Ewald 求和多系统测试，验证不同晶格结构。

**运行**:
```bash
pytest tests/python/test_ewald_multi.py -v
```

### 🔄 交换关联泛函 (XC)

#### test_lda_xc.py
LDA 交换关联泛函测试（Perdew-Zunger），验证局域密度近似。

**运行**:
```bash
pytest tests/python/test_lda_xc.py -v
```

### 🧮 综合测试

#### test_initial_energies.py ⭐
**最重要的综合测试**，验证所有泛函的完整能量计算：
- Thomas-Fermi KEDF
- von Weizsäcker KEDF
- Wang-Teter KEDF
- LDA XC (PZ)
- Hartree 势
- 局域赝势
- Ewald 求和
- **Evaluator 组合式计算**

输出格式化的能量对比表格，包含精度验证和状态指示器。

**运行**:
```bash
pytest tests/python/test_initial_energies.py -v -s
```

**示例输出**:
```
Component              | DFTpy (Ha)         | DFTcu (Ha)         | Abs Diff     |   Status
──────────────────────────────────────────────────────────────────────────────────────────
Thomas-Fermi          |    65.4612345678   |    65.4612345678   |    1.23e-15  |        ✓
von Weizsäcker        |    12.3456789012   |    12.3456789012   |    2.34e-15  |        ✓
Wang-Teter            |    -3.2109876543   |    -3.2109876543   |    1.11e-15  |        ✓
...
TOTAL (Evaluator)     |                    |    98.7654321098   |    3.45e-14  |        ✓
```

#### test_scf.py
自洽场 (SCF) 迭代测试，验证 DIIS 和 Anderson 优化器。

**运行**:
```bash
pytest tests/python/test_scf.py -v -s
```

#### test_scf_comparison.py
SCF 性能对比测试，与 DFTpy 对比收敛速度和精度。

**运行**:
```bash
pytest tests/python/test_scf_comparison.py -v -s
```

### 📊 性能基准测试

#### test_tf_kedf_benchmark.py
全面的性能基准测试，对比 DFTcu 和 DFTpy：
- **精度验证**: 相对误差 < 10⁻¹⁵
- **性能对比**: 多种网格尺寸的速度测试

**运行**:
```bash
# 所有 benchmark
pytest tests/python/test_tf_kedf_benchmark.py -v -s

# 仅性能汇总
pytest tests/python/test_tf_kedf_benchmark.py::TestTFKEDFBenchmark::test_performance_summary -v -s
```

**示例输出**:
```
Grid Size       Points     DFTpy (ms)   DFTcu (ms)   Speedup
----------------------------------------------------------------------
16×16×16        4,096      0.19         0.03         6.36x
32×32×32        32,768     0.66         0.03         24.80x
64×64×64        262,144    5.23         0.33         16.01x

Average speedup: 15.72x
```

### 🔧 基础功能测试

#### test_dftcu.py
基础 API 测试，验证 Grid, Field, FFT 等核心功能。

**运行**:
```bash
pytest tests/python/test_dftcu.py -v
```

## 前置要求

### DFTpy 安装（对比测试需要）

```bash
# 自动安装（推荐）
make setup

# 手动安装
source .venv/bin/activate
uv pip install -e external/DFTpy --no-deps
```

如果 DFTpy 未安装，对比测试会自动跳过。

### 测试数据

所有测试使用标准测试系统：
- **系统**: FCC Al (4 原子)
- **网格**: 32³ 点
- **晶胞**: 7.653 Bohr

## pytest 配置

### Markers

- `@pytest.mark.benchmark` - 性能基准测试
- `@pytest.mark.slow` - 慢速测试
- `@pytest.mark.comparison` - DFTpy 对比测试

### 调试选项

```bash
# 详细输出
pytest tests/python/ -vv

# 显示 print 输出
pytest tests/python/ -s

# 在失败时进入 pdb
pytest tests/python/ --pdb

# 运行特定 marker
pytest tests/python/ -m benchmark

# 跳过慢速测试
pytest tests/python/ -m "not slow"
```

## 测试开发

### 添加新测试

1. 创建 `test_*.py` 文件
2. 使用 `pytest` 风格：函数以 `test_` 开头
3. 添加 docstring 说明测试目的
4. 使用 `assert` 进行断言
5. 添加适当的 markers

### 精度标准

- **KEDF**: 绝对误差 < 10⁻¹⁵ Ha
- **Hartree/Pseudo**: 绝对误差 < 10⁻¹⁴ Ha
- **Ewald**: 绝对误差 < 10⁻¹³ Ha
- **总能量**: 绝对误差 < 10⁻¹³ Ha

### 性能测试

使用 `time.perf_counter()` 测量时间，多次运行取平均值。

```python
import time

start = time.perf_counter()
# ... 测试代码 ...
elapsed = time.perf_counter() - start
```

## 持续集成

所有测试在每次提交时自动运行。确保：
- ✅ 所有测试通过
- ✅ 代码覆盖率 > 80%
- ✅ 无内存泄漏
- ✅ 格式符合规范

---

**快速链接**: [主 README](../../README.md) | [开发指南](../../DEVELOPMENT.md) | [贡献指南](../../CONTRIBUTING.md)
