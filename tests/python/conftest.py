"""
Pytest configuration for DFTcu Python tests
"""

import os
import sys

# Add build directory to Python path
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Add DFTpy source to path for comparison tests
dftpy_src = os.path.join(os.path.dirname(__file__), "..", "..", "external", "DFTpy", "src")
if os.path.exists(dftpy_src):
    sys.path.insert(0, dftpy_src)


def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "comparison: mark test as DFTpy comparison test")
