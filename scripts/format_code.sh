#!/bin/bash
# Code formatting script

set -e

# Use virtual environment if available
if [ -d ".venv" ]; then
    BLACK=".venv/bin/black"
    ISORT=".venv/bin/isort"
else
    BLACK="black"
    ISORT="isort"
fi

echo "Formatting C++/CUDA files with clang-format..."
find src tests \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) \
    -exec clang-format -i -style=file {} + 2>/dev/null || echo "Note: Some files may not exist yet"

echo "Formatting Python files with black..."
$BLACK . 2>/dev/null || echo "Warning: black not found or no Python files to format"

echo "Sorting Python imports with isort..."
$ISORT --profile black . 2>/dev/null || echo "Warning: isort not found or no Python files"

echo "✓ Code formatting complete!"
