# Python Tests

This directory contains Python-level tests for the DFTcu library.

## Test Files

- `test_dftcu.py` - Basic functionality tests (simple smoke tests)
- `test_tf_kedf.py` - Thomas-Fermi KEDF validation against DFTpy

## Running Tests

```bash
# From project root
export PYTHONPATH=$PWD/build:$PYTHONPATH
pytest tests/python/ -v

# Or from build directory
cd build
pytest ../tests/python/ -v

# Run specific test
pytest tests/python/test_tf_kedf.py -v

# With coverage
pytest tests/python/ --cov=dftcu --cov-report=html
```

## Test Organization

- **Unit tests**: Test individual components (Grid, Field, etc.)
- **Integration tests**: Test functional combinations (Hartree + FFT)
- **Validation tests**: Compare against DFTpy reference results

## Adding New Tests

1. Create `test_*.py` file in this directory
2. Import pytest and dftcu
3. Write test functions starting with `test_`
4. Run pytest to verify

Example:
```python
import pytest
import numpy as np
import dftcu

def test_new_feature():
    grid = dftcu.Grid([10, 0, 0, 0, 10, 0, 0, 0, 10], [16, 16, 16])
    # Your test code
    assert grid.nnr() == 16**3
```
