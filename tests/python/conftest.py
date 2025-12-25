"""
Pytest configuration for DFTcu Python tests
"""
import os
import sys

# Add build directory to Python path
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
