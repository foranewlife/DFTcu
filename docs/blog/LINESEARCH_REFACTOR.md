/**
 * \page blog_ls_refactor 线搜索算法重构 (英文)
 */
# Technical Note: Line Search Algorithm Refactor

**日期**: 2025-12-27
**目标**: 将line search从TNOptimizer解耦，实现scipy兼容的独立模块

---

## 1. 背景与动机

### 问题描述
TN优化器与DFTpy存在精度差异（误差~1e-5 Ha），用户要求："为什么不实现一个和dftpy一模一样的linesearch呢？"

### 用户策略
采用逐步验证的方法：
1. 首先保证独立DCSRCH和scipy相同
2. 然后验证line search
3. 最后验证TN优化器
4. 同时分离optimize和line search，简化耦合程度

---

## 2. 实施过程

### 2.1 创建独立Line Search模块

#### 文件结构
```
src/math/
├── linesearch.cuh  # 接口定义
└── linesearch.cu   # scipy兼容实现
tests/
├── test_linesearch.cu              # C++单元测试
└── python/
    ├── test_linesearch.py          # Python参考测试
    └── test_linesearch_dft.py      # DFT能量景观测试
```

#### 核心API
```cpp
bool scalar_search_wolfe1(
    std::function<double(double)> phi,      // 目标函数
    std::function<double(double)> derphi,   // 导数函数
    double phi0,                             // 初始值
    double derphi0,                          // 初始导数
    double phi0_old,                         // 前一步值（用于自适应步长）
    double c1,                               // Armijo参数 (1e-4)
    double c2,                               // 曲率参数 (0.2)
    double amax,                             // 最大步长 (π)
    double amin,                             // 最小步长 (0)
    double xtol,                             // 收敛容差 (1e-14)
    double& alpha_star,                      // 输出: 最优步长
    double& phi_star                         // 输出: 最优值
);
```

### 2.2 解耦TNOptimizer

#### 修改前
- TNOptimizer包含私有方法: `line_search()`, `eval_at_theta()`, `zoom()`, `update_phi_at_theta()`
- 强耦合，难以独立测试

#### 修改后
```cpp
// optimizer.cu中使用lambda封装
auto phi_func = [&](double theta) -> double {
    RealField phi_trial(grid_);
    RealField rho_trial(grid_);
    RealField v_trial(grid_);
    v_scale(n, cos(theta), phi_backup.data(), phi_trial.data());
    v_axpy(n, sin(theta), p.data(), phi_trial.data());
    v_mul(n, phi_trial.data(), phi_trial.data(), rho_trial.data());
    return evaluator.compute(rho_trial, v_trial);
};

// 调用独立line search
bool converged = scalar_search_wolfe1(
    phi_func, derphi_func,
    energy, 2.0 * g_dot_p, prev_prev_energy,
    1e-4, 0.2, M_PI, 0.0, 1e-14,
    theta_final, energy_new
);
```

### 2.3 测试验证

#### 简单二次函数测试
```bash
./build/tests/test_linesearch
```
**结果**: ✅ 完美匹配scipy (alpha=2.0, error=0)

#### DFT能量景观测试
```bash
python tests/python/test_linesearch.py
```
**SciPy结果**: alpha=0.424629, E=-2.117703861804 Ha

---

## 3. 使用的命令总结

### 文件操作
```bash
# 读取文件（分析现有实现）
Read src/solver/optimizer.cu
Read src/solver/optimizer.cuh
Read external/DFTpy/src/dftpy/optimization/optimization.py
Read .venv/lib/python3.11/site-packages/scipy/optimize/_linesearch.py
Read .venv/lib/python3.11/site-packages/scipy/optimize/_dcsrch.py

# 创建新文件
Write src/math/linesearch.cuh
Write src/math/linesearch.cu
Write tests/test_linesearch.cu
Write tests/python/test_linesearch.py
Write tests/python/test_linesearch_dft.py

# 编辑文件
Edit CMakeLists.txt                    # 添加linesearch.cu到DFTCU_CORE_SOURCES
Edit tests/CMakeLists.txt              # 添加test_linesearch.cu
Edit src/solver/optimizer.cu          # 移除旧line search，集成新模块
Edit src/solver/optimizer.cuh         # 移除line search相关声明
Edit src/math/linesearch.cu           # 添加调试输出

# 搜索命令
Grep "optimization_options.*=" external/DFTpy/src/dftpy/optimization
Grep "add_library.*dftcu_core" CMakeLists.txt
Glob "**/*.cu" src/
```

### 编译测试命令
```bash
# 清理和重新编译
make clean
make build

# 使用uv快速重装（开发模式）
CUDACXX=/usr/local/cuda/bin/nvcc uv pip install --force-reinstall -e .

# 运行C++测试
./build/tests/test_linesearch

# 运行Python测试
source .venv/bin/activate
python tests/python/test_linesearch.py
python tests/python/test_linesearch_dft.py
python debug_line_search.py
```

### 调试命令
```bash
# 查看编译输出
make build 2>&1 | tail -50

# 实时监控输出
tail -30 /tmp/claude/tasks/[task_id].output

# 比较结果
python compare_tn_steps.py
python debug_linesearch_comparison.py
```

---

## 4. 修改的文件清单

### 新增文件 (6个)
| 文件 | 类型 | 用途 |
|------|------|------|
| `src/math/linesearch.cuh` | 头文件 | Line search接口定义 |
| `src/math/linesearch.cu` | 源文件 | scipy兼容实现（zoom算法） |
| `tests/test_linesearch.cu` | 测试 | C++单元测试 |
| `tests/python/test_linesearch.py` | 测试 | Python参考实现对比 |
| `tests/python/test_linesearch_dft.py` | 测试 | DFT能量景观验证 |
| `docs/TN_LINESEARCH_REFACTOR.md` | 文档 | 本技术报告 |

### 修改文件 (4个)
| 文件 | 主要修改 |
|------|----------|
| `CMakeLists.txt` | 添加`src/math/linesearch.cu`到核心库 |
| `tests/CMakeLists.txt` | 添加`test_linesearch.cu`到测试目标 |
| `src/solver/optimizer.cu` | 1. 移除旧line search实现（~200行）<br>2. 添加`#include "math/linesearch.cuh"`<br>3. 使用lambda封装phi/derphi<br>4. 调用`scalar_search_wolfe1()` |
| `src/solver/optimizer.cuh` | 移除line search相关私有方法声明 |

---

## 5. 当前进度与问题

### ✅ 已完成
1. **模块化架构**
   - Line search完全独立，可单独测试
   - 使用std::function实现回调，降低耦合

2. **简单函数测试通过**
   - 二次函数: f(x) = (x-2)²
   - 结果: alpha=2.0 (100%匹配scipy)

3. **编译集成成功**
   - 无编译错误或警告
   - 所有模块正确链接

### ⚠️ 待解决问题

#### DFT能量景观精度不足

**测试结果对比**:
| 实现 | alpha | Energy (Ha) | 误差 |
|------|-------|-------------|------|
| SciPy DCSRCH | 0.4246 | -2.117704 | 基准 |
| DFTcu (当前) | 0.5674 | -2.084654 | 3.3e-2 Ha (1.56%) |

**问题分析**:
```
Zoom迭代过程（10次全部未收敛）:
iter 0: alpha=0.6211, dphi=+0.816  # 导数仍为正
iter 1: alpha=0.6149, dphi=+0.788
iter 2: alpha=0.6088, dphi=+0.761
...
iter 9: alpha=0.5674, dphi=+0.571  # 最终返回，但不满足Wolfe曲率条件

要求: |dphi| <= c2*|dphi0| = 0.2*1.475 = 0.295
实际: dphi = 0.571 > 0.295 ❌
```

**根本原因**:
Zoom函数的cubic interpolation收敛到局部点而非全局最优：
- 初始区间: [0, 1], dphi_lo=-1.475, dphi_hi=+1.850
- 最优点应在: 0.425 (dphi=-0.109, 满足Wolfe)
- 实际收敛到: 0.567 (dphi=+0.571, 不满足Wolfe)

---

## 6. 技术细节对比

### DFTpy使用的Line Search
```python
# dftpy/math_utils.py:46
values = line_search_wolfe1(
    fobj.f, fobj.fprime, x, pk,
    gfk=func0[1], old_fval=func0[0],
    c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol
)
```

### SciPy实现路径
```
line_search_wolfe1()  [_linesearch.py:37]
    └─> scalar_search_wolfe1()  [_linesearch.py:100]
        └─> DCSRCH()  [_dcsrch.py:17]
            - 728行MINPACK算法
            - More-Thuente方法
            - 复杂状态机
```

### DFTcu实现路径
```
scalar_search_wolfe1()  [linesearch.cu:100]
    ├─> Bracket phase: 寻找包含最优点的区间
    └─> zoom(): 区间细化
        └─> Cubic interpolation: 三次插值求试探点
```

### 关键参数对比
| 参数 | DFTpy | DFTcu | 说明 |
|------|-------|-------|------|
| c1 (Armijo) | 1e-4 | 1e-4 | ✓ 一致 |
| c2 (Curvature) | 0.2 | 0.2 | ✓ 一致 |
| amax | π | π | ✓ 一致 |
| xtol | 1e-14 | 1e-14 | ✓ 一致 |
| 初始步长 | `min(1.0, 1.01*2*(f0-f_old)/df0)` | 相同 | ✓ 一致 |

---

## 7. 调试日志示例

### SciPy行为
```python
# 初始步长
alpha1 = 1.0  # 第一次迭代使用1.0

# 函数求值
phi(1.0) = -1.495  # 能量上升，触发zoom
derphi(1.0) = +1.850

# Zoom结果
alpha = 0.4246  # 满足Strong Wolfe条件
phi = -2.1177
derphi = -0.109  # |derphi| < 0.295 ✓
```

### DFTcu行为（详细trace）
```
[LS iter 0: alpha=1.000, phi=-1.495e+00, dphi=1.850e+00]
[LS: calling zoom (insufficient decrease)]
  [Zoom: a_lo=0.000, a_hi=1.000,
         phi_lo=-1.726, phi_hi=-1.495,
         dphi_lo=-1.475, dphi_hi=+1.850]
  [Zoom iter 0: alpha_j=0.6211, phi_j=-2.0473, dphi_j=0.8158]
  [Zoom iter 1: alpha_j=0.6149, phi_j=-2.0523, dphi_j=0.7882]
  ...
  [Zoom iter 9: alpha_j=0.5674, phi_j=-2.0847, dphi_j=0.5710]
  [Zoom: max iterations, returning a_lo=0.5674, phi_lo=-2.0847]
[Line search: theta=0.5674, E=-2.0847, converged=no]
```

**观察**:
1. Zoom每次更新`a_lo`向右移动（0 → 0.621 → 0.615 → ... → 0.567）
2. 导数始终为正且递减（1.85 → 0.816 → 0.788 → ... → 0.571）
3. 从未跨越最优点（alpha=0.425, dphi=-0.109）
4. 10次迭代后返回局部点

---

## 8. 下一步计划

### 方案A: 调试当前Zoom实现 (推荐短期)
**优点**: 保持代码简洁
**步骤**:
1. 详细trace scipy的zoom行为（添加print到scipy代码）
2. 逐迭代对比cubic interpolation计算
3. 检查区间更新逻辑（尤其是a_lo/a_hi交换条件）

**调试命令**:
```python
# 修改scipy源码添加调试
import scipy.optimize._linesearch as ls
# 在zoom函数中添加print语句

# 运行对比
python debug_linesearch_comparison.py
```

### 方案B: 移植SciPy DCSRCH (推荐长期)
**优点**: 工业级鲁棒性
**工作量**: ~2-3天
**步骤**:
1. 理解DCSRCH状态机（728行Fortran转Python）
2. 创建`src/math/dcsrch.cu`
3. 端口测试：简单函数 → Rosenbrock → DFT能量

**参考文件**:
- `.venv/lib/python3.11/site-packages/scipy/optimize/_dcsrch.py`
- More & Thuente (1994) "Line Search Algorithms with Guaranteed Sufficient Decrease"

### 方案C: 混合方法
1. 先快速修复zoom的明显bug（cubic插值公式错误？）
2. 如果仍不收敛，再移植DCSRCH

---

## 9. 验证检查清单

### 单元测试
- [x] 简单二次函数 (f(x)=(x-2)²)
- [ ] Rosenbrock函数
- [ ] 强非凸函数
- [ ] DFT能量景观

### 集成测试
- [x] TN优化器编译通过
- [x] 第一步优化运行
- [ ] 多步优化收敛
- [ ] 精度匹配DFTpy (<1e-6 Ha)

### 性能测试
- [ ] Line search调用次数对比
- [ ] 单次line search耗时
- [ ] 整体优化速度

---

## 10. 参考文献

1. **SciPy源码**
   - `scipy/optimize/_linesearch.py` - line_search_wolfe1/wolfe2
   - `scipy/optimize/_dcsrch.py` - MINPACK DCSRCH算法

2. **DFTpy源码**
   - `dftpy/optimization/optimization.py` - TN优化器
   - `dftpy/math_utils.py` - Line search封装

3. **学术文献**
   - Nocedal & Wright (2006), "Numerical Optimization"
   - More & Thuente (1994), "Line Search Algorithms with Guaranteed Sufficient Decrease"

4. **项目文档**
   - `TN_ALIGNMENT_REPORT.md` - 初始对齐报告
   - `TN_CONVERGENCE_ANALYSIS.md` - 收敛性分析

---

## 附录A: 关键代码片段

### Cubic Interpolation公式
```cpp
// src/math/linesearch.cu:39
double a = derphi_lo + derphi_hi - 3.0 * (phi_lo - phi_hi) / dalpha;
double b_sq = a * a - derphi_lo * derphi_hi;
double b = (b_sq > 0) ? std::sqrt(b_sq) : 0.0;

if (dalpha > 0) {
    alpha_j = a_lo + dalpha * (1.0 - (derphi_hi + b - a) /
                                (derphi_hi - derphi_lo + 2.0 * b));
} else {
    alpha_j = a_lo - dalpha * (derphi_lo + b - a) /
                                (derphi_lo - derphi_hi + 2.0 * b);
}
```

### 区间更新逻辑
```cpp
// src/math/linesearch.cu:62
if (phi_j > phi0 + c1 * alpha_j * derphi0 || phi_j >= phi_lo) {
    // Armijo条件不满足，更新上界
    a_hi = alpha_j;
    phi_hi = phi_j;
    derphi_hi = derphi_j;
} else {
    // Armijo满足，检查曲率
    if (std::abs(derphi_j) <= -c2 * derphi0) {
        // Strong Wolfe满足！
        return true;
    }

    // 曲率不满足，检查是否需要交换区间
    if (derphi_j * (a_hi - a_lo) >= 0) {
        a_hi = a_lo;
        phi_hi = phi_lo;
        derphi_hi = derphi_lo;
    }

    // 更新下界
    a_lo = alpha_j;
    phi_lo = phi_j;
    derphi_lo = derphi_j;
}
```

---

## 附录B: 问题排查指南

### 症状1: Line search返回alpha=0
**可能原因**:
- 初始步长计算错误
- phi0_old未正确传递

**检查点**:
```cpp
// optimizer.cu中确认
double prev_prev_energy = 1e10;  // 第一次迭代
// 后续迭代
prev_prev_energy = prev_energy;
```

### 症状2: Zoom无限循环
**可能原因**:
- 区间更新逻辑错误
- xtol设置过小

**检查点**:
```cpp
if (std::abs(dalpha) < xtol) {
    return true;  // 确保有退出条件
}
```

### 症状3: 精度不足但收敛
**可能原因**:
- Cubic interpolation公式错误
- 数值精度损失

**调试方法**:
```bash
# 添加详细输出
std::cout << std::setprecision(16) << alpha_j << std::endl;

# 对比scipy
python -c "from scipy.optimize._linesearch import _cubicmin; print(_cubicmin(...))"
```

---

**文档版本**: 1.0
**最后更新**: 2025-12-27
**维护者**: DFTcu开发团队
