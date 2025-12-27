# Google Test for C++ unit tests
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()
include(GoogleTest)

# Test utilities library
add_library(test_utils STATIC
    test_utils.cu
)

target_include_directories(test_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_utils
    PUBLIC
        CUDA::cudart
        CUDA::cufft
)

# Test executables - only build if source files exist
set(TEST_SOURCES
    test_kedf_tf.cu
    test_grid_pseudo.cu
    test_linesearch.cu
    test_ewald_fcc.cu
)

foreach(test_source ${TEST_SOURCES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_source}")
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})

        target_link_libraries(${test_name}
            PRIVATE
                GTest::gtest_main
                test_utils
                dftcu_core
                CUDA::cufft
                CUDA::cudart
        )

        target_include_directories(${test_name}
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src
        )

        gtest_discover_tests(${test_name})
    else()
        message(STATUS "Skipping ${test_source} (file not found)")
    endif()
endforeach()

# Note: Add more test files here as they are created:
# - test_grid.cu
# - test_field.cu
# - test_hartree.cu

# Python tests using pytest
# First try to find pytest in the project's venv
find_program(PYTEST_EXECUTABLE
    NAMES pytest
    PATHS ${CMAKE_SOURCE_DIR}/.venv/bin
    NO_DEFAULT_PATH
)
# If not found in venv, search system-wide
if(NOT PYTEST_EXECUTABLE)
    find_program(PYTEST_EXECUTABLE pytest)
endif()

if(PYTEST_EXECUTABLE)
    add_test(
        NAME pytest_all
        COMMAND ${PYTEST_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests -v
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(pytest_all PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
    )
endif()
