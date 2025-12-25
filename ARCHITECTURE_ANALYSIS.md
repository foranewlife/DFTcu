# DFTcu Architecture Analysis and Implementation Roadmap

## Executive Summary

This document provides a comprehensive analysis of the DFTpy-to-CUDA porting project (DFTcu), based on architectural patterns from GPUMD. The analysis identifies completed components, missing functionality, and provides a detailed roadmap for full DFTpy CUDA implementation.

---

## 1. DFTpy Core Architecture Analysis

### 1.1 Data Structures

#### **Grid System** (grid.py)
- **BaseGrid**: Base class for grid representation
  - Manages lattice vectors (3x3 matrix)
  - Handles discretization: `nr` (grid points per direction)
  - Computes reciprocal space: `nrG` (for FFT)
  - Supports MPI parallelization
  - **Key properties**: `nnr` (total points), `dV` (volume element), `volume`

- **DirectGrid**: Real-space grid
- **ReciprocalGrid**: k-space grid with:
  - `g`: reciprocal vectors
  - `gg`: |g|^2 for each grid point
  - `invgg`: 1/|g|^2 (regularized at g=0)

#### **Field System** (field.py)
- **BaseField**: Extends `np.ndarray`
  - Attached to a Grid
  - Supports rank (scalar/vector fields)
  - **Methods**: FFT/IFFT operations, integration, gradients
  - Memory layout: Can be C or Fortran order

### 1.2 FFT Operations (fft.py)
- **Backend**: pyfftw (FFTW wrapper)
- **Types**: Real-to-complex (R2C) and complex-to-complex (C2C)
- **Key operations**:
  - `field.fft()` → reciprocal space
  - `field.ifft()` → real space
  - Automatic grid switching between DirectGrid ↔ ReciprocalGrid

### 1.3 DFT Functionals

#### **Hartree Functional** (functional/hartree.py) ✅ **PORTED**
- **Algorithm**:
  1. FFT density: ρ(r) → ρ(G)
  2. Solve Poisson: V_H(G) = 4π ρ(G) / |G|²
  3. IFFT potential: V_H(G) → V_H(r)
- **Energy**: E_H = 0.5 ∫ ρ(r) V_H(r) dr
- **Status**: Already implemented in DFTcu (hartree.cu)

#### **Kinetic Energy Density Functionals (KEDF)** (functional/kedf/) ❌ **MISSING**
DFTpy supports multiple KEDFs for orbital-free DFT:

1. **Thomas-Fermi (TF)**: tf.py
   - Local functional: τ[ρ] = C_TF ρ^(5/3)
   - Simple pointwise operation

2. **von Weizsäcker (vW)**: vw.py
   - Gradient correction: τ[ρ] = (1/8) |∇ρ|²/ρ
   - Requires gradient computation

3. **Wang-Teter (WT)**: wt.py
   - Nonlocal kernel-based KEDF
   - Reciprocal space convolution with Lindhard kernel

4. **LKT (Luo-Karasiev-Trickey)**: lkt.py
   - Advanced nonlocal KEDF
   - Multiple kernel components

5. **MGP (Modified Gradient Approximation)**: mgp.py
6. **GGA-based**: gga.py
7. **HC, SM, FP**: Other variants

**Computational Pattern**:
- Most involve reciprocal space operations
- Kernel convolutions: ∫ K(|r-r'|) f(r') dr' = FFT^-1[K(G) · FFT[f(r)]]

#### **Exchange-Correlation (XC)** (functional/xc/) ❌ **MISSING**
- **LibXC integration**: xc_functional.py
  - LDA, GGA, meta-GGA functionals
  - Requires density and gradients as input
- **rVV10**: Non-local van der Waals correction

#### **Pseudopotentials** (functional/pseudo/) ⚠️ **PARTIAL**
- **LocalPseudo** ✅ Already in DFTcu (pseudo.cu)
  - Local PP in reciprocal space: V_loc(G)
- **NonlocalPseudo** ❌ Missing
  - Requires projectors and atomic orbitals
  - More complex structure factor calculations

---

## 2. GPUMD Architecture Patterns (Successfully Adopted)

### 2.1 Memory Management ✅
- **GPU_Vector<T>**: Template class (gpu_vector.cuh)
  - Automatic allocation/deallocation
  - Host ↔ Device copy methods
  - Fill operations
  - **DFTcu adoption**: Used in Grid, Field classes

### 2.2 Portability Layer ✅
- **gpu_macro.cuh**: Unified CUDA/HIP API
  - Memory: `gpuMalloc`, `gpuMemcpy`
  - FFT: `gpufftHandle`, `gpufftExecC2C`
  - Error handling: `gpuError_t`, `gpuSuccess`
  - **DFTcu adoption**: Fully integrated

### 2.3 Error Handling ✅
- **CHECK macro** (error.cuh):
  ```cpp
  #define CHECK(call) \
    if (call != gpuSuccess) { /* print error & exit */ }
  ```
- **GPU_CHECK_KERNEL**: Validate kernel launches
- **DFTcu adoption**: Applied throughout

---

## 3. Current DFTcu Implementation Status

### ✅ Completed Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Grid | model/grid.cuh | ✅ Done | Lattice, reciprocal space, g-vectors |
| Field | model/field.cuh | ✅ Done | GPU-based field storage |
| GPU_Vector | utilities/gpu_vector.cuh | ✅ Done | From GPUMD |
| FFT Solver | fft/fft_solver.cuh | ✅ Done | cuFFT wrapper |
| Hartree | functional/hartree.cu | ✅ Done | Poisson solver |
| LocalPseudo | functional/pseudo.cu | ✅ Done | Local pseudopotential |
| Python API | api/dftcu_api.cu | ✅ Done | pybind11 bindings |

### ❌ Missing Core Functionality

#### 3.1 KEDF Functionals (HIGH PRIORITY)
Orbital-free DFT requires kinetic energy functionals:
- [ ] Thomas-Fermi (TF)
- [ ] von Weizsäcker (vW)
- [ ] Wang-Teter (WT) nonlocal kernel
- [ ] LKT nonlocal functional
- [ ] Other variants (MGP, HC, SM, FP)

**Implementation needs**:
- Gradient kernels for vW
- Kernel generation and convolution for WT/LKT
- Reciprocal space operations

#### 3.2 XC Functionals (HIGH PRIORITY)
- [ ] LibXC integration or manual LDA/GGA implementation
- [ ] Density gradient computation (∇ρ)
- [ ] Stress tensor calculation

#### 3.3 Advanced Field Operations (MEDIUM PRIORITY)
- [ ] Gradient: `∇f` (real space or via FFT: iFFT[iG · FFT[f]])
- [ ] Laplacian: `∇²f` (via FFT: iFFT[-|G|² · FFT[f]])
- [ ] Divergence and curl operations
- [ ] Integration kernels (already partial in Hartree)

#### 3.4 Optimization Support (MEDIUM PRIORITY)
DFTpy uses scipy optimizers for density minimization:
- [ ] Energy/force evaluation interface
- [ ] Density update mechanisms
- [ ] Convergence checking utilities

#### 3.5 Extended Features (LOW PRIORITY)
- [ ] Nonlocal pseudopotentials (projectors)
- [ ] Spin-polarized calculations
- [ ] Stress tensor computation
- [ ] Forces on atoms (Hellmann-Feynman)

---

## 4. Implementation Roadmap

### Phase 1: Essential KEDF Functionals (Week 1-2)
**Goal**: Enable basic orbital-free DFT calculations

1. **Thomas-Fermi (TF)**
   - Simplest: pointwise `E_TF = C_TF * ρ^(5/3)`
   - Kernel: `compute_tf_energy_potential_kernel`
   - Files: `src/functional/kedf/tf.cu`, `tf.cuh`

2. **von Weizsäcker (vW)**
   - Requires gradient: `|∇ρ|² / ρ`
   - Implement gradient kernel (can use FFT method)
   - Files: `src/functional/kedf/vw.cu`, `vw.cuh`

3. **Basic gradient operations**
   - Add to `utilities/kernels.cu`: `compute_gradient_fft`
   - Input: real field → Output: 3-component vector field

### Phase 2: Nonlocal KEDF (Week 3-4)
**Goal**: Advanced functionals for accurate OF-DFT

1. **Wang-Teter (WT)**
   - Lindhard response kernel in reciprocal space
   - Convolution via FFT
   - Files: `src/functional/kedf/wt.cu`, `wt.cuh`

2. **LKT functional**
   - Multiple kernel components
   - Reference: DFTpy's `lkt.py` for kernel definitions
   - Files: `src/functional/kedf/lkt.cu`, `lkt.cuh`

### Phase 3: XC Functionals (Week 5-6)
**Goal**: Full Kohn-Sham DFT capability

1. **LDA functional**
   - Manual implementation (Perdew-Wang parameterization)
   - Pointwise evaluation
   - Files: `src/functional/xc/lda.cu`, `lda.cuh`

2. **GGA functional (PBE)**
   - Requires ρ and |∇ρ|
   - Files: `src/functional/xc/gga_pbe.cu`, `gga_pbe.cuh`

3. **Optional: LibXC integration**
   - Wrap LibXC library calls in CUDA kernels
   - More effort but provides all functionals

### Phase 4: Optimization & Testing (Week 7-8)
**Goal**: Validate against DFTpy, performance tuning

1. **Validation tests**
   - Port DFTpy examples (examples/ofdft/simple.py)
   - Compare energies, potentials, forces
   - Files: `test/test_hartree.py`, `test/test_kedf.py`

2. **Performance benchmarks**
   - Profile cuFFT usage
   - Optimize kernel launches (block/grid sizes)
   - Memory access patterns (coalescing)

3. **Python API extensions**
   - Add all new functionals to pybind11 interface
   - Match DFTpy's API design

---

## 5. Technical Implementation Details

### 5.1 Gradient Computation (for vW, GGA)

**Option A: FFT-based (recommended)**
```cpp
// Gradient via reciprocal space: ∇f = iFFT[i*G * FFT[f]]
void compute_gradient_fft(const RealField& f, RealField& grad_x,
                          RealField& grad_y, RealField& grad_z) {
    // 1. FFT f(r) → f(G)
    ComplexField f_G(f.grid());
    fft_forward(f, f_G);

    // 2. Multiply by i*G_x, i*G_y, i*G_z
    ComplexField grad_x_G(f.grid()), grad_y_G(f.grid()), grad_z_G(f.grid());
    multiply_by_iG_kernel<<<...>>>(f_G, gx, gy, gz, grad_x_G, grad_y_G, grad_z_G);

    // 3. IFFT back to real space
    fft_backward(grad_x_G, grad_x);
    fft_backward(grad_y_G, grad_y);
    fft_backward(grad_z_G, grad_z);
}
```

**Option B: Finite differences** (less accurate but direct)
- 2nd-order centered: `∂f/∂x ≈ (f[i+1] - f[i-1]) / (2Δx)`
- Handle periodic boundary conditions

### 5.2 KEDF Kernel Structure

All KEDFs follow similar pattern:
```cpp
class KEDF_Base {
public:
    virtual double compute(const RealField& rho, RealField& v_kedf) = 0;
    // Returns energy, fills potential v_kedf = δE/δρ
};

class ThomasFermi : public KEDF_Base {
    double compute(const RealField& rho, RealField& v_kedf) override {
        // Launch kernel: E = ∫ C_TF ρ^(5/3) dV
        //                V = (5/3) C_TF ρ^(2/3)
    }
};
```

### 5.3 Python API Extension Example

```cpp
// In dftcu_api.cu
PYBIND11_MODULE(dftcu, m) {
    // ... existing code ...

    // KEDF functionals
    py::class_<ThomasFermi>(m, "ThomasFermi")
        .def(py::init<const Grid&>())
        .def("compute", &ThomasFermi::compute);

    py::class_<VonWeizsacker>(m, "VonWeizsacker")
        .def(py::init<const Grid&>())
        .def("compute", &VonWeizsacker::compute);
}
```

---

## 6. Key Challenges & Solutions

### Challenge 1: Complex FFT operations
- **Issue**: Multiple FFT/IFFT calls per functional
- **Solution**:
  - Reuse cuFFT plans (already cached in FFTSolver)
  - Consider cuFFT streams for overlap

### Challenge 2: Gradient accuracy
- **Issue**: FFT method may have Gibbs phenomena
- **Solution**:
  - Apply low-pass filter in reciprocal space
  - Match DFTpy's gradient implementation exactly

### Challenge 3: Kernel function complexity (WT, LKT)
- **Issue**: Lindhard kernel has logarithmic singularities
- **Solution**:
  - Copy kernel generation logic from DFTpy
  - Pre-compute kernels on CPU, transfer to GPU
  - Store in Grid object

### Challenge 4: LibXC integration
- **Issue**: LibXC is CPU-based
- **Solution**:
  - Manual LDA/GGA implementation (faster, fewer dependencies)
  - Alternative: Call LibXC on CPU for small systems (batch)

---

## 7. Testing & Validation Strategy

### 7.1 Unit Tests
For each functional:
```python
def test_hartree_vs_dftpy():
    # Create identical grid in DFTpy and DFTcu
    # Compare energies (rtol=1e-6)
    # Compare potentials (rtol=1e-5)
```

### 7.2 Integration Tests
- **Aluminum bulk**: Simple metallic system
  - KEDF: TF + vW combination
  - Compare total energy per atom
- **Water molecule**: Test XC functionals
  - LDA or GGA
  - Compare HOMO-LUMO gap

### 7.3 Performance Benchmarks
- **Grid sizes**: 32³, 64³, 128³, 256³
- **Metrics**: Time per SCF iteration, memory usage
- **Target**: >10x speedup vs CPU DFTpy for 128³ grid

---

## 8. File Structure (Proposed)

```
src/
├── functional/
│   ├── hartree.cu/cuh          ✅ Existing
│   ├── pseudo.cu/cuh           ✅ Existing
│   ├── kedf/
│   │   ├── tf.cu/cuh           ❌ TODO
│   │   ├── vw.cu/cuh           ❌ TODO
│   │   ├── wt.cu/cuh           ❌ TODO
│   │   ├── lkt.cu/cuh          ❌ TODO
│   │   └── kedf_base.cuh       ❌ TODO (abstract class)
│   └── xc/
│       ├── lda.cu/cuh          ❌ TODO
│       ├── gga_pbe.cu/cuh      ❌ TODO
│       └── xc_base.cuh         ❌ TODO
├── utilities/
│   ├── kernels.cu              ✅ Existing
│   ├── gradient.cu/cuh         ❌ TODO (FFT-based gradients)
│   └── reduction.cu/cuh        ❌ TODO (sum/integrate utilities)
└── api/
    └── dftcu_api.cu            ⚠️  Extend with new functionals
```

---

## 9. References & Resources

### DFTpy Key Files to Study:
1. `src/dftpy/functional/kedf/tf.py` - TF implementation
2. `src/dftpy/functional/kedf/wt.py` - WT kernel logic
3. `src/dftpy/functional/xc/xc_functional.py` - LibXC wrapper
4. `examples/ofdft/simple.py` - Usage example

### GPUMD Patterns to Reuse:
1. `src/utilities/gpu_vector.cuh` - Memory management ✅
2. `src/utilities/gpu_macro.cuh` - Portability ✅
3. `src/force/*.cu` - Kernel launch patterns

### CUDA Best Practices:
- cuFFT documentation: [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/)
- Kernel optimization: Occupancy calculator, shared memory usage
- Profiling: `nvprof`, `nsys`, Nsight Compute

---

## 10. Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Code coverage | >80% of DFTpy functionals | 30% (Hartree + LocalPP) |
| Accuracy | <0.1% energy difference vs DFTpy | ✅ Hartree validated |
| Performance | >10x speedup (128³ grid) | Not yet benchmarked |
| API compatibility | Drop-in replacement for key functions | Partial |
| Documentation | Full API docs + examples | In progress |

---

## Conclusion

DFTcu has a solid foundation based on GPUMD's architecture. The next critical steps are:
1. **Implement KEDF functionals** (TF, vW, WT) for orbital-free DFT
2. **Add gradient operations** to support vW and GGA
3. **Validate against DFTpy** examples
4. **Benchmark performance** and optimize

With focused effort, DFTcu can become a complete CUDA backend for DFTpy, enabling large-scale DFT calculations on GPUs.
