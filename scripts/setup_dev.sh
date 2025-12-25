#!/bin/bash
# Development environment setup script
# Uses uv for fast Python package management

set -e

echo "======================================"
echo "DFTcu Development Setup"
echo "======================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA first."
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release | awk '{print $5}')"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.18+."
    exit 1
fi

echo "✓ CMake found: $(cmake --version | head -n1)"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✓ uv installed"
else
    echo "✓ uv found: $(uv --version)"
fi

# Create virtual environment and install all dependencies with uv sync
echo ""
echo "Setting up Python environment with uv..."
echo "This will:"
echo "  - Create/update virtual environment (.venv)"
echo "  - Install all dependencies from pyproject.toml"
echo "  - Create uv.lock for reproducible builds"

# Install all dependencies including dev
uv sync --all-extras

echo ""
echo "Installing DFTpy from submodule for comparison tests..."
if [ -d "external/DFTpy" ]; then
    .venv/bin/pip install -e external/DFTpy --no-deps
    echo "✓ DFTpy installed from submodule"
else
    echo "⚠ DFTpy submodule not found (run: git submodule update --init)"
fi

echo ""
echo "✓ Python environment ready"

# Install pre-commit hooks (only if in a git repo)
if [ -d ".git" ]; then
    echo ""
    echo "Installing pre-commit hooks..."
    .venv/bin/pre-commit install
    echo "✓ Pre-commit hooks installed"
else
    echo ""
    echo "⚠ Not a git repository - skipping pre-commit hooks"
fi

echo ""
echo "======================================"
echo "✓ Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Build the project:"
echo "     make build"
echo ""
echo "  3. Run tests:"
echo "     make test"
echo ""
echo "  4. Format code before committing:"
echo "     make format"
echo ""
echo "Useful uv commands:"
echo "  uv sync              - Sync dependencies"
echo "  uv add <package>     - Add a new dependency"
echo "  uv remove <package>  - Remove a dependency"
echo "  uv lock              - Update uv.lock"
echo ""
