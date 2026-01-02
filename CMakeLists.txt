cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Initialize C++ first to ensure feature detection works correctly for GTest
project(DFTcu
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "CUDA-accelerated DFT calculations compatible with DFTpy"
)

option(BUILD_WITH_CUDA "Enable CUDA build" ON)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_DOCS "Build documentation" ON)
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)

if(BUILD_WITH_CUDA)
    # 2. Optimized CUDA Architecture Detection
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES MATCHES "^(52|70)$")
        find_program(DETECT_NVCC NAMES nvcc PATHS /usr/local/cuda/bin /usr/bin)

        if(DETECT_NVCC)
            message(STATUS "Probing GPU architecture using: ${DETECT_NVCC}")

            # Try nvidia-smi first
            execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                            OUTPUT_VARIABLE SMI_ARCH RESULT_VARIABLE SMI_RES ERROR_QUIET)
            if(SMI_RES EQUAL 0)
                string(REPLACE "." "" CLEAN_SMI_ARCH "${SMI_ARCH}")
                string(REGEX MATCHALL "[0-9]+" CLEAN_SMI_ARCH "${CLEAN_SMI_ARCH}")
                if(CLEAN_SMI_ARCH)
                    list(GET CLEAN_SMI_ARCH 0 PROBED_ARCH)
                endif()
            endif()

            # Fallback to probe script
            if(NOT PROBED_ARCH)
                set(arch_file "${CMAKE_BINARY_DIR}/detect_arch.cu")
                file(WRITE ${arch_file} [[
                    #include <cuda_runtime.h>
                    #include <stdio.h>
                    int main() {
                        int count = 0;
                        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 1;
                        cudaDeviceProp prop;
                        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 1;
                        printf("%d%d", prop.major, prop.minor);
                        return 0;
                    }
                ]])
                execute_process(COMMAND ${DETECT_NVCC} -arch=sm_${PROBED_ARCH} ${arch_file} -o "${arch_file}.exe"
                                RESULT_VARIABLE PROBE_COMPILE_RES ERROR_QUIET)
                if(PROBE_COMPILE_RES EQUAL 0)
                    execute_process(COMMAND "${arch_file}.exe" OUTPUT_VARIABLE PROBED_ARCH
                                    RESULT_VARIABLE PROBE_EXE_RES ERROR_QUIET)
                endif()
            endif()

            # Verify arch support
            if(PROBED_ARCH)
                set(test_file "${CMAKE_BINARY_DIR}/verif_arch.cu")
                file(WRITE ${test_file} "int main(){return 0;}")
                execute_process(COMMAND ${DETECT_NVCC} -arch=sm_${PROBED_ARCH} ${test_file} -o "${test_file}.exe"
                                RESULT_VARIABLE VERIF_RES OUTPUT_QUIET ERROR_QUIET)
                if(VERIF_RES EQUAL 0)
                    set(DETECTED_OPTIMAL_ARCH ${PROBED_ARCH})
                endif()
            endif()
        endif()
    endif()

    # Set the architecture before enabling CUDA language
    if(DETECTED_OPTIMAL_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_OPTIMAL_ARCH} CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Optimized CUDA architecture detected and verified: ${DETECTED_OPTIMAL_ARCH}")
    elseif(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
    endif()

    # 3. Enable CUDA language
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

    find_package(CUDAToolkit REQUIRED)

    # Python for pybind11
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 REQUIRED)
else()
    # Still need basic Python3 for Doxygen or other scripts if necessary
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    message(STATUS "CUDA build disabled. Skipping CUDA-related targets and tests.")
endif()


if(BUILD_WITH_CUDA)
    # Core library sources
    set(DFTCU_CORE_SOURCES
        src/model/grid.cu
        src/model/density_builder.cu
        src/model/wavefunction.cu
        src/model/wavefunction_builder.cu
        src/functional/hartree.cu
        src/functional/ewald.cu
        src/functional/pseudo.cu
        src/functional/nonlocal_pseudo.cu
        src/functional/kedf/tf.cu
        src/functional/kedf/vw.cu
        src/functional/kedf/wt.cu
        src/functional/kedf/hc.cu
        src/functional/xc/lda_pz.cu
        src/functional/xc/pbe.cu
        src/math/linesearch.cu
        src/math/dcsrch.cu
        src/solver/evaluator.cu
        src/solver/hamiltonian.cu
        src/solver/davidson.cu
        src/solver/subspace_solver.cu
        src/solver/scf.cu
        src/solver/mixer.cu
        src/solver/optimizer.cu
        src/solver/tn_optimizer_legacy.cu
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
            CUDA::nvrtc
            CUDA::cuda_driver
            CUDA::cublas
            CUDA::curand
            CUDA::cusolver
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
    pybind11_add_module(_dftcu src/api/dftcu_api.cu)

    # Link Python module to core library
    target_link_libraries(_dftcu PRIVATE dftcu_core)

    # Set output directory for Python module
    set_target_properties(_dftcu PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dftcu
        OUTPUT_NAME _dftcu
    )

    # Profiling flags
    if(ENABLE_PROFILING)
        target_compile_options(_dftcu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()

    # Installation
    install(TARGETS _dftcu
        LIBRARY DESTINATION dftcu
    )

    # Option to auto-install Python package after build
    option(AUTO_PIP_INSTALL "Automatically run 'uv pip install -e .[dev]' after build" OFF)

    if(AUTO_PIP_INSTALL)
        # Find uv executable
        find_program(UV_EXECUTABLE uv PATHS $ENV{HOME}/.cargo/bin)
        if(UV_EXECUTABLE)
            add_custom_target(pip_install ALL
                COMMAND ${CMAKE_COMMAND} -E echo "Installing Python package with uv..."
                COMMAND ${UV_EXECUTABLE} pip install -e ".[dev]" --no-deps --no-build-isolation
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                DEPENDS _dftcu
                COMMENT "Running uv pip install --no-build-isolation -e .[dev]"
            )
            message(STATUS "AUTO_PIP_INSTALL enabled - will run 'uv pip install -e .[dev]' after build")
        else()
            message(WARNING "uv executable not found - skipping auto pip install")
        endif()
    endif()
endif() # END BUILD_WITH_CUDA

# Testing
if(BUILD_TESTING AND BUILD_WITH_CUDA)
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
if(BUILD_WITH_CUDA)
    message(STATUS "  CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA version:         ${CUDAToolkit_VERSION}")
endif()
message(STATUS "  Python version:       ${Python3_VERSION}")
message(STATUS "  Build with CUDA:      ${BUILD_WITH_CUDA}")
message(STATUS "  Build testing:        ${BUILD_TESTING}")
message(STATUS "  Build documentation:  ${BUILD_DOCS}")
if(BUILD_WITH_CUDA)
    message(STATUS "  Enable profiling:     ${ENABLE_PROFILING}")
endif()
message(STATUS "")
