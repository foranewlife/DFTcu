# DFTcu: CUDA Acceleration

DFTcu is a high-performance CUDA-accelerated backend for DFT calculations, designed to be compatible with [DFTpy](https://gitlab.com/pavanello-research-group/dftpy). It utilizes C++/CUDA to implement heavy grid-based operations, following the architectural patterns of [GPUMD](https://github.com/brucefan1983/GPUMD).

## Features

- **CUDA-Native Grid Operations**: Efficient handling of 3D grids and fields directly on GPU.
- **FFT Acceleration**: Integration with `cuFFT` for fast reciprocal space transformations.
- **DFT Functionals**:
  - **Hartree Potential**: Fast solver using cuFFT.
  - **Local Pseudopotentials**: Support for local PP calculations in reciprocal space.
- **Seamless Python Integration**: Built with `pybind11` to provide a Pythonic API that mimics DFTpy.
- **Memory Management**: Automatic GPU memory handling using the `GPU_Vector` template (GPUMD-style).

## Directory Structure

```
DFTcu/
├── src/                    # C++/CUDA source code
│   ├── model/             # Grid, Atoms, Field classes
│   ├── fft/               # FFT solver (cuFFT wrapper)
│   ├── functional/        # DFT functionals
│   │   ├── kedf/         # Kinetic energy functionals (TF, vW, etc.)
│   │   └── xc/           # Exchange-correlation (future)
│   ├── utilities/         # Helper functions and kernels
│   └── api/               # Python bindings (pybind11)
├── tests/                  # Unit tests
│   ├── python/            # Python tests
│   └── test_*.cu          # C++ tests (Google Test)
├── docs/                   # Documentation
├── scripts/                # Helper scripts
└── external/               # Dependencies (DFTpy, GPUMD)
```

## Prerequisites

- **NVIDIA GPU**: (Architecture `sm_89` or compatible).
- **CUDA Toolkit**: `nvcc`, `cuFFT`, etc.
- **Python**: 3.9+ (Managed by `uv` recommended).
- **C++ Compiler**: Support for C++14.

## Getting Started

### Quick Setup (Recommended)

**One command to set up everything:**

```bash
make setup
```

This automatically:
- ✅ Checks CUDA, CMake, and Python
- ✅ Installs `uv` (if needed)
- ✅ Creates virtual environment (`.venv`)
- ✅ Installs all dependencies from `pyproject.toml`
- ✅ Sets up pre-commit hooks

Then activate the environment:
```bash
source .venv/bin/activate
```

### Build and Test

```bash
# Build the project
make build

# Run all tests
make test

# Format code
make format
```

See [UV_WORKFLOW.md](UV_WORKFLOW.md) for detailed dependency management.

### Manual Setup (Alternative)

If you prefer manual control:

```bash
# 1. Install dependencies with uv
uv sync --all-extras

# 2. Activate environment
source .venv/bin/activate

# 3. Configure CMake (adjust CUDA architecture for your GPU)
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86

# Build
cmake --build build -j$(nproc)
```

Or use the legacy Makefile:
```bash
make -f Makefile.legacy
```

This will generate a shared library (e.g., `dftcu.cpython-311-x86_64-linux-gnu.so`) in the `build/` directory.

### 3. Run Tests

```bash
# C++ tests
cd build
ctest --output-on-failure

# Python tests
export PYTHONPATH=$PWD/build:$PYTHONPATH
pytest tests/python/ -v
```

## Usage Example

```python
import numpy as np
import dftcu

# Define Grid
lattice = np.eye(3) * 10.0
nr = [32, 32, 32]
grid = dftcu.Grid(lattice.flatten().tolist(), nr)

# Create Field on GPU
rho = dftcu.RealField(grid)
rho.fill(1.0) # Fill with constant value

# Calculate Hartree Potential
hartree = dftcu.Hartree(grid)
vh = dftcu.RealField(grid)
energy = hartree.compute(rho, vh)

print(f"Hartree Energy: {energy}")
```

## Documentation

- **[DOCS_INDEX.md](DOCS_INDEX.md)** - Documentation index and navigation guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference (recommended)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md)** - Detailed architecture analysis

## Acknowledgments

- **DFTpy**: For the original Pythonic DFT architecture.
- **GPUMD**: For the efficient CUDA implementation patterns and macros.
