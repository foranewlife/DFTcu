cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project definition
project(DFTcu
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "CUDA-accelerated DFT calculations compatible with DFTpy"
)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)  # More flexible
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

# CUDA architecture (adjust based on your GPU)
# RTX 40 series: 89
# RTX 30 series: 86
# A100: 80
# V100: 70
# Default to 70 for broader compatibility
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_DOCS "Build documentation" ON)
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)

# Find dependencies
find_package(CUDAToolkit REQUIRED)

# Python for pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Core library sources (without Python bindings)
set(DFTCU_CORE_SOURCES
    src/model/grid.cu
    src/functional/hartree.cu
    src/functional/pseudo.cu
    src/functional/kedf/tf.cu
    src/functional/kedf/vw.cu
    src/functional/kedf/wt.cu
    src/functional/xc/lda_pz.cu
    src/solver/evaluator.cu
    src/solver/optimizer.cu
    src/utilities/kernels.cu
)

# Create static library for core functionality
add_library(dftcu_core STATIC ${DFTCU_CORE_SOURCES})

target_include_directories(dftcu_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(dftcu_core
    PUBLIC
        CUDA::cufft
        CUDA::cudart
)

target_compile_options(dftcu_core
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
        >
)

set_target_properties(dftcu_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create Python module
pybind11_add_module(dftcu src/api/dftcu_api.cu)

# Link Python module to core library
target_link_libraries(dftcu PRIVATE dftcu_core)

# Set output directory for Python module
set_target_properties(dftcu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Profiling flags
if(ENABLE_PROFILING)
    target_compile_options(dftcu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# Installation
install(TARGETS dftcu
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

# Testing
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    else()
        message(WARNING "Doxygen not found. Documentation will not be built.")
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "DFTcu Configuration Summary:")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA version:         ${CUDAToolkit_VERSION}")
message(STATUS "  Python version:       ${Python3_VERSION}")
message(STATUS "  Build testing:        ${BUILD_TESTING}")
message(STATUS "  Build documentation:  ${BUILD_DOCS}")
message(STATUS "  Enable profiling:     ${ENABLE_PROFILING}")
message(STATUS "")
