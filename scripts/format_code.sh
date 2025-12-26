#!/bin/bash
# Code formatting script
# Unifies behavior between local development and CI

set -e

# Use virtual environment if available
if [ -d ".venv" ]; then
    BLACK=".venv/bin/black"
    ISORT=".venv/bin/isort"
    FLAKE8=".venv/bin/flake8"
else
    BLACK="black"
    ISORT="isort"
    FLAKE8="flake8"
fi

# Target directories as an array
TARGETS=("src" "tests")

echo "Formatting C++/CUDA files with clang-format..."
# We explicitly target src and tests to match other tools
find "${TARGETS[@]}" \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) \
    -exec clang-format -i -style=file {} + 2>/dev/null || echo "Note: No C++/CUDA files found"

echo "Formatting Python files with black..."
$BLACK "${TARGETS[@]}"

echo "Sorting Python imports with isort..."
$ISORT "${TARGETS[@]}"

echo "Running flake8 lint check..."
# Uses configuration from .flake8
$FLAKE8 "${TARGETS[@]}" || echo "Warning: Linting issues found"

echo "✓ Code formatting complete!"
