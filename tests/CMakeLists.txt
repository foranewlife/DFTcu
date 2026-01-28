include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
# For Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Helper function to add a CUDA test
function(add_cuda_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            dftcu_core
            GTest::gtest_main
    )
    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # Set CUDA properties
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    # Run tests from project root directory so relative paths work
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

# ════════════════════════════════════════════════════════════════════════════════
# Unit Tests
# ════════════════════════════════════════════════════════════════════════════════

# 1. Functional Layer Tests
add_cuda_test(unit.functional.test_local_pseudo
    unit/functional/test_local_pseudo.cu
)

# 2. LDA_PZ XC Functional Tests
add_cuda_test(unit.functional.test_lda_pz
    unit/functional/test_lda_pz.cu
)

# 3. Hartree Functional Tests
add_cuda_test(unit.functional.test_hartree
    unit/functional/test_hartree.cu
)

# 4. V_loc Kernel Tests (Solver Layer)
add_cuda_test(unit.solver.test_vloc_kernel
    unit/solver/test_vloc_kernel.cu
)

# ════════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ════════════════════════════════════════════════════════════════════════════════

# 1. LocalPseudoOperator Integration Test
add_cuda_test(integration.functional.test_local_pseudo_operator
    integration/functional/test_local_pseudo_operator.cu
)

# 2. LDA_PZ Integration Test
add_cuda_test(integration.functional.test_lda_pz
    integration/functional/test_lda_pz.cu
)

# 3. Hartree Integration Test
add_cuda_test(integration.functional.test_hartree
    integration/functional/test_hartree.cu
)

# 4. Hamiltonian V_loc Integration Test
add_cuda_test(integration.solver.test_hamiltonian_vloc
    integration/solver/test_hamiltonian_vloc.cu
)

# 5. V_loc Index Mapping Diagnostic Test
add_cuda_test(integration.solver.test_vloc_index_mapping
    integration/solver/test_vloc_index_mapping.cu
)

# 6. nl_d Convention Diagnostic Test
add_cuda_test(integration.solver.test_nl_d_convention
    integration/solver/test_nl_d_convention.cu
)

# ════════════════════════════════════════════════════════════════════════════════
# Regression Tests
# ════════════════════════════════════════════════════════════════════════════════

# Future regression tests go here
