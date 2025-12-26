# DFTcu Makefile - Modern build system wrapper
PROJECT_NAME := DFTcu
BUILD_DIR := build
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python3
PYTEST := $(VENV_DIR)/bin/pytest
UV := uv

# Use CUDACXX if provided, otherwise fallback to system nvcc
CUDACXX ?= /usr/local/cuda/bin/nvcc
CMAKE := cmake
CMAKE_BUILD_TYPE ?= Release

# Architecture 52 triggers auto-detection in our CMakeLists.txt
CUDA_ARCH ?= 52
STRIPPED_ARCH := $(strip $(CUDA_ARCH))
DISPLAY_ARCH := $(STRIPPED_ARCH)
ifeq ($(STRIPPED_ARCH),52)
  DISPLAY_ARCH := auto
endif

# CMAKE FLAGS
# Use Ninja for faster builds and compatibility with scikit-build-core
CMAKE_FLAGS := -GNinja \
               -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
               -DCMAKE_CUDA_COMPILER=$(CUDACXX) \
               -DCMAKE_CUDA_ARCHITECTURES=$(STRIPPED_ARCH) \
               -DBUILD_TESTING=ON \
               -DBUILD_DOCS=ON

# Use venv Python if it exists
ifneq (,$(wildcard $(VENV_DIR)/bin/python3))
CMAKE_FLAGS += -DPython3_EXECUTABLE=$(CURDIR)/$(VENV_DIR)/bin/python3
endif

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)$(PROJECT_NAME) - CUDA-accelerated DFT calculations$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

.PHONY: setup
setup: ## Complete development environment setup
	@./scripts/setup_dev.sh

.PHONY: sync
sync: ## Sync dependencies with uv
	@echo "$(YELLOW)Syncing dependencies with uv...$(NC)"
	@$(UV) sync --all-extras
	@echo "$(GREEN)✓ Dependencies synced$(NC)"

.PHONY: install
install: ## Install to venv using uv (compiles with CUDACXX)
	@echo "$(YELLOW)Installing to venv using uv with $(CUDACXX)...$(NC)"
	@CUDACXX=$(CUDACXX) $(UV) pip install .

.PHONY: install-dev
install-dev: ## Install in editable mode for incremental builds (RECOMMENDED for development)
	@echo "$(YELLOW)Installing in editable mode with $(CUDACXX) for incremental builds...$(NC)"
	@CUDACXX=$(CUDACXX) $(UV) pip install -e .
	@echo "$(GREEN)✓ Editable install complete - changes to .cu files will trigger incremental rebuilds$(NC)"

.PHONY: rebuild
rebuild: ## Quick rebuild after source changes (only works with editable install)
	@echo "$(YELLOW)Rebuilding (incremental)...$(NC)"
	@CUDACXX=$(CUDACXX) $(UV) pip install -e . --no-deps
	@echo "$(GREEN)✓ Incremental rebuild complete$(NC)"

##@ Building

.PHONY: configure
configure: ## Configure CMake build
	@echo "$(YELLOW)Configuring CMake with $(CUDACXX) (arch=$(DISPLAY_ARCH))...$(NC)"
	$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@echo "$(GREEN)✓ Configuration complete$(NC)"

.PHONY: build
build: configure ## Build the project (C++ only)
	@echo "$(YELLOW)Building $(PROJECT_NAME)...$(NC)"
	@$(CMAKE) --build $(BUILD_DIR) -j$$(nproc)
	@echo "$(GREEN)✓ Build complete$(NC)"

.PHONY: build-install
build-install: ## Build C++ and install Python package (shares artifacts)
	@echo "$(YELLOW)Building and installing $(PROJECT_NAME)...$(NC)"
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS) -DAUTO_PIP_INSTALL=ON
	@$(CMAKE) --build $(BUILD_DIR) -j$$(nproc)
	@echo "$(GREEN)✓ Build and install complete$(NC)"

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Build artifacts cleaned$(NC)"

##@ Testing

.PHONY: test
test: build ## Run all tests
	@echo "$(YELLOW)Running C++ tests...$(NC)"
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "$(YELLOW)Running Python tests (venv version)...$(NC)"
	@unset PYTHONPATH && $(PYTEST) tests/python/ -v

.PHONY: test-python
test-python: ## Run Python tests only
	@echo "$(YELLOW)Running Python tests (using installed package in venv)...$(NC)"
	@unset PYTHONPATH && $(PYTEST) tests/python/ -v

##@ Code Quality

.PHONY: format
format: ## Format all code
	@echo "$(YELLOW)Formatting code...$(NC)"
	@./scripts/format_code.sh
	@echo "$(GREEN)✓ Code formatted$(NC)"

.PHONY: lint
lint: ## Run linters
	@echo "$(YELLOW)Running linters...$(NC)"
	@$(VENV_DIR)/bin/flake8 . --max-line-length=100 --ignore=E203,W503 || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

##@ Documentation

.PHONY: doc
doc: configure ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@cd $(BUILD_DIR) && $(CMAKE) --build . --target doc
	@echo "$(GREEN)✓ Documentation generated$(NC)"

##@ Advanced

.PHONY: benchmark
benchmark: build ## Run performance benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	@if [ -d "benchmarks" ]; then \
		cd benchmarks && $(PYTHON) run_all.py; \
	else \
		echo "$(YELLOW)No benchmarks directory found$(NC)"; \
	fi

.PHONY: debug
debug: ## Build in debug mode
	@echo "$(YELLOW)Building in debug mode...$(NC)"
	@CUDACXX=$(CUDACXX) SKBUILD_CMAKE_BUILD_TYPE=Debug $(UV) pip install .
	@echo "$(GREEN)✓ Debug build installed$(NC)"
