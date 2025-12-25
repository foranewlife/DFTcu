# CMake Presets

This directory contains additional preset configurations for specialized builds.

## Using CMake Presets

### List Available Presets

```bash
cmake --list-presets
```

### Configure with a Preset

```bash
# Default configuration
cmake --preset=default

# Debug build
cmake --preset=debug

# Release build (optimized)
cmake --preset=release

# GPU-specific builds
cmake --preset=rtx4090  # RTX 40 series (sm_89)
cmake --preset=rtx3090  # RTX 30 series (sm_86)
cmake --preset=a100     # A100 (sm_80)
cmake --preset=v100     # V100 (sm_70)

# Multi-GPU support (all architectures)
cmake --preset=multi-gpu

# Profiling build
cmake --preset=profile

# CI/CD build
cmake --preset=ci
```

### Build with a Preset

```bash
cmake --build --preset=release
```

### Test with a Preset

```bash
ctest --preset=default
```

## Available Presets

### Build Types

| Preset | Build Type | Description | CUDA Arch |
|--------|------------|-------------|-----------|
| `default` | Release | Default configuration | 86 (RTX 3090) |
| `debug` | Debug | Debug symbols, no optimization | 86 |
| `release` | Release | Maximum optimization | 86 |
| `relwithdebinfo` | RelWithDebInfo | Optimized + debug symbols | 86 |
| `profile` | Release | With profiling enabled | 86 |

### GPU-Specific

| Preset | GPU | Compute Capability | Best For |
|--------|-----|-------------------|----------|
| `rtx4090` | RTX 4090 | 8.9 (sm_89) | Latest Ada Lovelace |
| `rtx3090` | RTX 3090 | 8.6 (sm_86) | Ampere architecture |
| `a100` | A100 | 8.0 (sm_80) | Data center |
| `v100` | V100 | 7.0 (sm_70) | Broad compatibility |
| `multi-gpu` | Multiple | 70;80;86;89 | Universal binary |

### Special Builds

| Preset | Purpose | Features |
|--------|---------|----------|
| `ci` | CI/CD | Tests ON, Docs OFF, sm_70 |
| `profile` | Profiling | Lineinfo for nvprof/Nsight |

## Creating Custom Presets

Add your custom presets to `CMakePresets.json`:

```json
{
  "name": "my-preset",
  "displayName": "My Custom Preset",
  "description": "Custom build configuration",
  "inherits": "release",
  "cacheVariables": {
    "CMAKE_CUDA_ARCHITECTURES": "89",
    "CUSTOM_OPTION": "ON"
  }
}
```

## User Presets (Local)

Create `CMakeUserPresets.json` for local configurations (not tracked by Git):

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "my-local",
      "inherits": "release",
      "cacheVariables": {
        "CMAKE_CUDA_ARCHITECTURES": "89"
      }
    }
  ]
}
```

## Makefile Integration

The Makefile provides shortcuts:

```bash
# Using presets via Makefile
make preset-debug
make preset-release

# Or directly
make CUDA_ARCH=89 build
```

## Tips

1. **Check Your GPU**: Run `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` to find your compute capability
2. **Multiple GPUs**: Use `multi-gpu` preset to support all GPUs
3. **IDE Support**: VSCode and CLion automatically detect presets
4. **Makefile Wrapper**: Use `make` for simpler commands

## Examples

### Quick Development Workflow

```bash
# Configure for debug
cmake --preset=debug

# Build
cmake --build --preset=debug

# Test
ctest --preset=debug
```

### Optimized Release

```bash
cmake --preset=rtx4090
cmake --build --preset=rtx4090 -j
```

### CI/CD Pipeline

```bash
cmake --preset=ci
cmake --build --preset=ci
ctest --preset=ci
```
