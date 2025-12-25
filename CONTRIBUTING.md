# Contributing to DFTcu

Thank you for your interest in contributing to DFTcu! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Building the Project](#building-the-project)
- [Running Tests](#running-tests)
- [Submitting Changes](#submitting-changes)
- [Adding New Functionals](#adding-new-functionals)

---

## Development Setup

### Prerequisites

- **CUDA Toolkit** (12.0+) with nvcc compiler
- **CMake** (3.18+)
- **Python** (3.9+)
- **Git**
- **clang-format** (for code formatting)
- **Doxygen** (optional, for documentation)

### Clone the Repository

```bash
git clone https://github.com/your-org/DFTcu.git
cd DFTcu
git submodule update --init --recursive
```

### Python Environment Setup

We recommend using `uv` for fast Python environment management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install numpy scipy pybind11 pytest black flake8 isort
```

### Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically format and check your code before each commit.

---

## Code Style

### C++/CUDA Code

We use **clang-format** with a custom configuration (`.clang-format`).

**Key conventions:**
- **Indentation**: 4 spaces
- **Line length**: 100 characters maximum
- **Naming**:
  - Classes: `PascalCase` (e.g., `ThomasFermi`)
  - Functions/methods: `snake_case` (e.g., `compute_energy`)
  - Variables: `snake_case` with trailing underscore for members (e.g., `grid_`, `coeff_`)
  - Constants: `UPPER_SNAKE_CASE`
- **Braces**: Same line for classes/functions (Google style)

**Format your code:**
```bash
# Format all C++/CUDA files
find src -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

# Or use the Makefile target
make format
```

### Python Code

We use **black** for formatting and **flake8** for linting.

```bash
# Format Python code
black .

# Check linting
flake8 . --max-line-length=100
```

### Documentation

- **C++/CUDA**: Use Doxygen-style comments
  ```cpp
  /**
   * @brief Compute Thomas-Fermi kinetic energy
   * @param rho Input density field
   * @param v_kedf Output potential field
   * @return Total kinetic energy in Hartree
   */
  double compute(const RealField& rho, RealField& v_kedf);
  ```

- **Python**: Use docstrings (Google style)
  ```python
  def compute_energy(density: np.ndarray) -> float:
      """
      Compute the kinetic energy.

      Args:
          density: Electron density array

      Returns:
          Total energy in Hartree
      """
  ```

---

## Building the Project

### Using CMake (Recommended)

```bash
# Configure
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \  # Adjust for your GPU
    -DBUILD_TESTING=ON \
    -DBUILD_DOCS=ON

# Build
cmake --build build -j$(nproc)

# Install (optional)
cmake --install build --prefix ~/.local
```

### Using Make (Legacy)

```bash
make clean
make -j4
```

### GPU Architecture

Set `CMAKE_CUDA_ARCHITECTURES` based on your GPU:
- **RTX 40 series**: 89
- **RTX 30 series**: 86
- **A100**: 80
- **V100**: 70

Check your GPU: `nvidia-smi`

---

## Running Tests

### C++ Unit Tests (Google Test)

```bash
cd build
ctest --output-on-failure

# Or run individual tests
./tests/test_kedf_tf
```

### Python Tests (pytest)

```bash
export PYTHONPATH=$PWD/build:$PYTHONPATH
pytest tests/ -v
```

### Adding New Tests

**C++ test example** (`tests/test_new_feature.cu`):
```cpp
#include <gtest/gtest.h>
#include "test_utils.h"

namespace dftcu {
namespace testing {

TEST(NewFeatureTest, BasicFunctionality) {
    // Your test code
    EXPECT_EQ(1 + 1, 2);
}

} // namespace testing
} // namespace dftcu
```

**Python test example** (`tests/test_new_feature.py`):
```python
import numpy as np
import dftcu

def test_new_feature():
    # Your test code
    assert True
```

---

## Submitting Changes

### Workflow

1. **Fork** the repository on GitHub
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Make changes** and commit:
   ```bash
   git add .
   git commit -m "Add new feature: ..."
   ```
4. **Run tests** to ensure nothing broke:
   ```bash
   make test  # or ctest in build/
   ```
5. **Push** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
6. **Open a Pull Request** on GitHub

### Commit Messages

Follow conventional commits format:

```
type(scope): Short description

Longer explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```
feat(kedf): Add von Weizsäcker functional

Implements the vW kinetic energy functional with FFT-based
gradient computation. Validated against DFTpy.

Closes #42
```

---

## Adding New Functionals

### Step 1: Create Files

For a new KEDF functional (e.g., von Weizsäcker):

```bash
src/functional/kedf/
├── vw.cuh   # Header file
└── vw.cu    # Implementation
```

### Step 2: Implement the Functional

**Header (`vw.cuh`):**
```cpp
#pragma once
#include "kedf_base.cuh"

namespace dftcu {

class VonWeizsacker : public KEDF_Base {
public:
    VonWeizsacker(double coeff = 1.0);
    double compute(const RealField& rho, RealField& v_kedf) override;
    const char* name() const override { return "von Weizsäcker"; }

private:
    double coeff_;
};

} // namespace dftcu
```

**Implementation (`vw.cu`):**
```cpp
#include "vw.cuh"
#include "utilities/gradient.cuh"

namespace dftcu {

VonWeizsacker::VonWeizsacker(double coeff) : coeff_(coeff) {}

double VonWeizsacker::compute(const RealField& rho, RealField& v_kedf) {
    // Your implementation here
    // 1. Compute gradients
    // 2. Calculate energy and potential
    // 3. Return energy
}

} // namespace dftcu
```

### Step 3: Add to Build System

Edit `CMakeLists.txt`:
```cmake
set(DFTCU_SOURCES
    # ... existing files ...
    src/functional/kedf/vw.cu
)
```

### Step 4: Add Python Bindings

Edit `src/api/dftcu_api.cu`:
```cpp
#include "functional/kedf/vw.cuh"

// In PYBIND11_MODULE:
py::class_<dftcu::VonWeizsacker>(m, "VonWeizsacker")
    .def(py::init<double>(), py::arg("coeff") = 1.0)
    .def("compute", &dftcu::VonWeizsacker::compute);
```

### Step 5: Add Tests

Create `tests/test_kedf_vw.cu` and `tests/test_vw.py`.

### Step 6: Update Documentation

- Add docstrings to your code
- Update `ARCHITECTURE_ANALYSIS.md` if needed
- Mention in your PR description

---

## Documentation

### Building Documentation

```bash
cd build
make doc
```

View in browser:
```bash
firefox docs/html/index.html
```

### Writing Good Documentation

- **Public APIs**: Must have Doxygen comments
- **Complex algorithms**: Add implementation notes
- **Usage examples**: Include in docstrings
- **Math formulas**: Use LaTeX in Doxygen (`\f$ E = mc^2 \f$`)

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/DFTcu/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/DFTcu/discussions)
- **Email**: dftcu-dev@example.com

---

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see `LICENSE` file).

---

Thank you for contributing to DFTcu! 🚀
