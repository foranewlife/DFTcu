# DFTcu Makefile - Modern build system wrapper
# This Makefile provides convenient targets for common tasks

# Project configuration
PROJECT_NAME := DFTcu
BUILD_DIR := build
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python3
PYTEST := $(VENV_DIR)/bin/pytest
UV := uv

# CMake configuration
CMAKE := cmake
CMAKE_BUILD_TYPE ?= Release
CUDA_ARCH ?= 52 # 52 triggers auto-detection in CMakeLists.txt
STRIPPED_ARCH := $(strip $(CUDA_ARCH))
DISPLAY_ARCH := $(STRIPPED_ARCH)
ifeq ($(STRIPPED_ARCH),52)
  DISPLAY_ARCH := auto
endif
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
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
setup: ## Complete development environment setup (runs setup_dev.sh)
	@./scripts/setup_dev.sh

.PHONY: sync
sync: ## Sync dependencies from pyproject.toml using uv
	@echo "$(YELLOW)Syncing dependencies with uv...$(NC)"
	@$(UV) sync --all-extras
	@echo "$(GREEN)✓ Dependencies synced$(NC)"

.PHONY: add
add: ## Add a dependency (usage: make add PKG=<package>)
	@if [ -z "$(PKG)" ]; then \
		echo "$(RED)Error: PKG not specified$(NC)"; \
		echo "Usage: make add PKG=<package>"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Adding $(PKG)...$(NC)"
	@$(UV) add $(PKG)
	@echo "$(GREEN)✓ $(PKG) added$(NC)"

.PHONY: remove
remove: ## Remove a dependency (usage: make remove PKG=<package>)
	@if [ -z "$(PKG)" ]; then \
		echo "$(RED)Error: PKG not specified$(NC)"; \
		echo "Usage: make remove PKG=<package>"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Removing $(PKG)...$(NC)"
	@$(UV) remove $(PKG)
	@echo "$(GREEN)✓ $(PKG) removed$(NC)"

.PHONY: lock
lock: ## Update uv.lock file
	@echo "$(YELLOW)Updating uv.lock...$(NC)"
	@$(UV) lock
	@echo "$(GREEN)✓ Lock file updated$(NC)"

##@ Building

.PHONY: configure
configure: ## Configure CMake build
	@echo "$(YELLOW)Configuring CMake ($(CMAKE_BUILD_TYPE), CUDA arch=$(DISPLAY_ARCH))...$(NC)"
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@echo "$(GREEN)✓ Configuration complete$(NC)"

.PHONY: build
build: configure ## Build the project
	@echo "$(YELLOW)Building $(PROJECT_NAME)...$(NC)"
	@$(CMAKE) --build $(BUILD_DIR) -j$$(nproc)
	@echo "$(GREEN)✓ Build complete$(NC)"

.PHONY: rebuild
rebuild: clean build ## Clean and rebuild

.PHONY: install
install: build ## Install the project
	@echo "$(YELLOW)Installing $(PROJECT_NAME)...$(NC)"
	@$(CMAKE) --install $(BUILD_DIR)
	@echo "$(GREEN)✓ Installation complete$(NC)"

##@ Testing

.PHONY: test
test: build ## Run all tests (C++ and Python)
	@echo "$(YELLOW)Running C++ tests...$(NC)"
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "$(GREEN)✓ C++ tests passed$(NC)"
	@echo ""
	@echo "$(YELLOW)Running Python tests...$(NC)"
	@export PYTHONPATH=$$(pwd)/$(BUILD_DIR):$$PYTHONPATH && \
		$(PYTEST) tests/python/ -v
	@echo "$(GREEN)✓ Python tests passed$(NC)"

.PHONY: test-cpp
test-cpp: build ## Run C++ tests only
	@echo "$(YELLOW)Running C++ tests...$(NC)"
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "$(GREEN)✓ C++ tests passed$(NC)"

.PHONY: test-python
test-python: build ## Run Python tests only
	@echo "$(YELLOW)Running Python tests...$(NC)"
	@export PYTHONPATH=$$(pwd)/$(BUILD_DIR):$$PYTHONPATH && \
		$(PYTEST) tests/python/ -v
	@echo "$(GREEN)✓ Python tests passed$(NC)"

.PHONY: test-verbose
test-verbose: build ## Run tests with verbose output
	@echo "$(YELLOW)Running tests (verbose)...$(NC)"
	@cd $(BUILD_DIR) && ctest -V
	@export PYTHONPATH=$$(pwd)/$(BUILD_DIR):$$PYTHONPATH && \
		$(PYTEST) tests/python/ -vv

.PHONY: test-cov
test-cov: build ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@export PYTHONPATH=$$(pwd)/$(BUILD_DIR):$$PYTHONPATH && \
		$(PYTEST) tests/python/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

##@ Code Quality

.PHONY: format
format: ## Format all code (C++, CUDA, Python)
	@echo "$(YELLOW)Formatting code...$(NC)"
	@./scripts/format_code.sh
	@echo "$(GREEN)✓ Code formatted$(NC)"

.PHONY: format-cpp
format-cpp: ## Format C++/CUDA code only
	@echo "$(YELLOW)Formatting C++/CUDA code...$(NC)"
	@find src tests -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | \
		xargs clang-format -i -style=file
	@echo "$(GREEN)✓ C++/CUDA code formatted$(NC)"

.PHONY: format-python
format-python: ## Format Python code only
	@echo "$(YELLOW)Formatting Python code...$(NC)"
	@$(VENV_DIR)/bin/black .
	@$(VENV_DIR)/bin/isort --profile black .
	@echo "$(GREEN)✓ Python code formatted$(NC)"

.PHONY: lint
lint: ## Run linters
	@echo "$(YELLOW)Running linters...$(NC)"
	@$(VENV_DIR)/bin/flake8 . --max-line-length=100 --ignore=E203,W503 || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

.PHONY: check
check: format lint ## Format and lint code
	@echo "$(GREEN)✓ Code quality check complete$(NC)"

##@ Documentation

.PHONY: doc
doc: configure ## Generate documentation with Doxygen
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@cd $(BUILD_DIR) && $(MAKE) doc
	@echo "$(GREEN)✓ Documentation generated in $(BUILD_DIR)/docs/html/$(NC)"

.PHONY: doc-view
doc-view: doc ## Generate and view documentation
	@echo "$(YELLOW)Opening documentation...$(NC)"
	@if command -v xdg-open &> /dev/null; then \
		xdg-open $(BUILD_DIR)/docs/html/index.html; \
	elif command -v open &> /dev/null; then \
		open $(BUILD_DIR)/docs/html/index.html; \
	else \
		echo "$(YELLOW)Please open $(BUILD_DIR)/docs/html/index.html manually$(NC)"; \
	fi

##@ Cleaning

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Build artifacts cleaned$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean everything including venv and caches
	@echo "$(YELLOW)Cleaning everything...$(NC)"
	@./scripts/clean_outdated.sh
	@rm -rf $(VENV_DIR)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Everything cleaned$(NC)"

.PHONY: distclean
distclean: clean-all ## Complete clean (including dependencies)
	@echo "$(YELLOW)Removing all generated files...$(NC)"
	@rm -rf htmlcov/ .coverage .pytest_cache/
	@echo "$(GREEN)✓ Distribution clean complete$(NC)"

##@ Development

.PHONY: dev
dev: setup configure build test ## Full development setup and test
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

.PHONY: quick
quick: ## Quick rebuild and test
	@echo "$(YELLOW)Quick rebuild and test...$(NC)"
	@$(CMAKE) --build $(BUILD_DIR) -j$$(nproc)
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "$(GREEN)✓ Quick build complete$(NC)"

.PHONY: watch
watch: ## Watch for changes and rebuild (requires entr)
	@if ! command -v entr &> /dev/null; then \
		echo "$(RED)Error: entr not installed. Install with: apt install entr$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Watching for changes (Ctrl+C to stop)...$(NC)"
	@find src -name "*.cu" -o -name "*.cuh" | entr -c make quick

##@ Information

.PHONY: info
info: ## Show project information
	@echo "$(BLUE)$(PROJECT_NAME) Project Information$(NC)"
	@echo ""
	@echo "Build configuration:"
	@echo "  Build directory:  $(BUILD_DIR)"
	@echo "  Build type:       $(CMAKE_BUILD_TYPE)"
	@echo "  CUDA arch:        $(CUDA_ARCH)"
	@echo ""
	@echo "Tools:"
	@echo "  CMake:            $$(command -v cmake || echo 'not found')"
	@echo "  nvcc:             $$(command -v nvcc || echo 'not found')"
	@echo "  Python:           $$(command -v python3 || echo 'not found')"
	@echo "  uv:               $$(command -v uv || echo 'not found')"
	@echo ""
	@echo "Versions:"
	@if command -v cmake &> /dev/null; then \
		echo "  CMake version:    $$(cmake --version | head -n1)"; \
	fi
	@if command -v nvcc &> /dev/null; then \
		echo "  CUDA version:     $$(nvcc --version | grep release | awk '{print $$5}')"; \
	fi
	@if command -v python3 &> /dev/null; then \
		echo "  Python version:   $$(python3 --version)"; \
	fi
	@if command -v nvidia-smi &> /dev/null; then \
		echo ""; \
		echo "GPU Information:"; \
		nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | \
			awk '{print "  GPU:              " $$0}'; \
	fi

.PHONY: status
status: ## Show project status
	@cat PROJECT_STATUS.txt 2>/dev/null || echo "$(YELLOW)Run 'make info' for project information$(NC)"

##@ Advanced

.PHONY: benchmark
benchmark: build ## Run performance benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	@if [ -d "benchmarks" ]; then \
		cd benchmarks && $(PYTHON) run_all.py; \
	else \
		echo "$(YELLOW)No benchmarks directory found$(NC)"; \
	fi

.PHONY: profile
profile: ## Build with profiling enabled
	@echo "$(YELLOW)Building with profiling...$(NC)"
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS) -DENABLE_PROFILING=ON
	@$(CMAKE) --build $(BUILD_DIR) -j$$(nproc)
	@echo "$(GREEN)✓ Profiling build complete$(NC)"

.PHONY: debug
debug: ## Build in debug mode
	@echo "$(YELLOW)Building in debug mode...$(NC)"
	@$(CMAKE) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) -j$$(nproc)
	@echo "$(GREEN)✓ Debug build complete$(NC)"

##@ Docker (Future)

.PHONY: docker-build
docker-build: ## Build Docker image (placeholder)
	@echo "$(YELLOW)Docker support coming soon...$(NC)"

.PHONY: docker-run
docker-run: ## Run in Docker container (placeholder)
	@echo "$(YELLOW)Docker support coming soon...$(NC)"

##@ CMake Presets

.PHONY: list-presets
list-presets: ## List available CMake presets
	@$(CMAKE) --list-presets 2>/dev/null || echo "$(YELLOW)No presets configured$(NC)"

.PHONY: preset-debug
preset-debug: ## Configure with debug preset
	@$(CMAKE) --preset=debug
	@echo "$(GREEN)✓ Configured with debug preset$(NC)"

.PHONY: preset-release
preset-release: ## Configure with release preset
	@$(CMAKE) --preset=release
	@echo "$(GREEN)✓ Configured with release preset$(NC)"

# Special targets
.PHONY: all
all: build test doc ## Build, test, and generate docs

.PHONY: ci
ci: check build test ## CI pipeline (format check, build, test)
	@echo "$(GREEN)✓ CI pipeline complete$(NC)"

# Legacy Makefile support
.PHONY: legacy
legacy: ## Build with legacy Makefile
	@if [ -f "Makefile.legacy" ]; then \
		echo "$(YELLOW)Building with legacy Makefile...$(NC)"; \
		$(MAKE) -f Makefile.legacy; \
	else \
		echo "$(RED)Makefile.legacy not found$(NC)"; \
		exit 1; \
	fi
