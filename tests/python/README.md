# DFTcu Python Tests

Python 测试套件，包括功能测试和性能基准测试。

## 测试文件

### test_tf_kedf.py
基础 Thomas-Fermi KEDF 功能测试，验证与 DFTpy 的精度一致性。

**运行**:
```bash
pytest tests/python/test_tf_kedf.py -v
```

### test_tf_kedf_benchmark.py
全面的性能基准测试，对比 DFTcu 和 DFTpy 的：
- **精度**: 验证机器精度一致性（相对误差 < 10⁻¹⁵）
- **性能**: 多种网格尺寸的速度对比

**运行**:
```bash
# 所有 benchmark
pytest tests/python/test_tf_kedf_benchmark.py -v -s

# 性能汇总
pytest tests/python/test_tf_kedf_benchmark.py::TestTFKEDFBenchmark::test_performance_summary -v -s
```

## 测试结果示例

```
Grid Size       Points     DFTpy (ms)   DFTcu (ms)   Speedup
----------------------------------------------------------------------
16×16×16        4,096      0.19         0.03         6.36x
32×32×32        32,768     0.66         0.03         24.80x
64×64×64        262,144    5.23         0.33         16.01x

Average speedup: 15.72x
```

## 前置要求

**DFTpy 安装**:
```bash
# 自动安装（推荐）
make setup

# 手动安装
source .venv/bin/activate
uv pip install -e external/DFTpy --no-deps
```

如果 DFTpy 未安装，对比测试会自动跳过。

## pytest Markers

- `@pytest.mark.benchmark` - 性能基准测试
- `@pytest.mark.slow` - 慢速测试
- `@pytest.mark.comparison` - DFTpy 对比测试

## 调试测试

```bash
# 详细输出
pytest tests/python/ -vv

# 显示 print 输出
pytest tests/python/ -s

# 在失败时进入 pdb
pytest tests/python/ --pdb
```
