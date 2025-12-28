/**
 * \page blog_ls_refactor_cn 线搜索算法重构 (中文)
 */
# 技术笔记：线搜索算法重构 (Line Search Refactor)

## 工作总结

### 完成内容
✅ 创建独立line search模块 (`src/math/linesearch.cu`)
✅ 解耦TNOptimizer，移除~200行耦合代码
✅ 简单函数测试100%匹配scipy
✅ 编译通过，无警告错误

### 当前问题
⚠️ DFT能量景观精度不足：误差3.3e-2 Ha (目标<1e-6 Ha)
⚠️ Zoom函数收敛到局部点(0.567)而非全局最优(0.425)

---

## 常用命令

### 编译
```bash
# 快速重装
CUDACXX=/usr/local/cuda/bin/nvcc uv pip install --force-reinstall -e .

# 完整构建
make clean && make build
```

### 测试
```bash
# C++单元测试
./build/tests/test_linesearch

# Python对比测试
source .venv/bin/activate
python tests/python/test_linesearch.py
python debug_line_search.py
```

### 调试
```bash
# 查看详细输出
python debug_line_search.py 2>&1 | grep -A 50 "Starting Truncated"

# 对比scipy行为
python debug_linesearch_comparison.py
```

---

## 文件清单

### 新增 (6个)
- `src/math/linesearch.cuh` - 接口
- `src/math/linesearch.cu` - 实现
- `tests/test_linesearch.cu` - C++测试
- `tests/python/test_linesearch.py` - Python测试
- `tests/python/test_linesearch_dft.py` - DFT测试
- `docs/TN_LINESEARCH_REFACTOR.md` - 详细文档

### 修改 (4个)
- `CMakeLists.txt` - 添加linesearch.cu
- `tests/CMakeLists.txt` - 添加测试
- `src/solver/optimizer.cu` - 集成新line search
- `src/solver/optimizer.cuh` - 移除旧声明

---

## 下一步

### 方案1：调试Zoom（1-2天）
1. Trace scipy zoom详细行为
2. 对比cubic interpolation计算
3. 修复区间更新逻辑

### 方案2：移植DCSRCH（2-3天）
直接移植scipy的728行MINPACK算法，保证工业级鲁棒性

### 建议
优先尝试方案1，如果快速修复不成功则转方案2

---

## 关键发现

### Zoom收敛问题
```
初始区间: [0, 1]
  dphi(0) = -1.475 (负)
  dphi(1) = +1.850 (正)
最优点应在: 0.425 (dphi=-0.109, 满足Wolfe)
实际返回: 0.567 (dphi=+0.571, 不满足Wolfe)
```

**原因**: Cubic interpolation每次更新a_lo向右移动，远离最优点

---

## 参考
详细技术报告见：`docs/TN_LINESEARCH_REFACTOR.md`
