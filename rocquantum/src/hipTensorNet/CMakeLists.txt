# CMakeLists.txt for hipTensorNet component

# Find ROCm and HIP
find_package(ROCM REQUIRED) # Ensures ROCM_PATH is set
find_package(HIP REQUIRED)

# Find rocBLAS (needed for rocTensorUtil's conceptual contraction wrapper)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED) # Find rocSOLVER

# Define the library for hipTensorNet and rocTensorUtil components
add_library(rocqsim_tensornet STATIC
    rocTensorUtil.cpp
    hipTensorNet.cpp
    rocTensorUtil_kernels.hip # Ensure kernels are compiled
    PermutationKernels.hip # Add the new permutation kernel
)

# Include directories
# The rocquantum root include directory should provide rocquantum/*.h
# The hipTensorNet include directory is the current one (if headers were placed here, but they are in root include)
target_include_directories(rocqsim_tensornet PUBLIC
    
    ${HIP_INCLUDE_DIRS}
    ${rocblas_INCLUDE_DIRS}
    ${rocsolver_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(rocqsim_tensornet PUBLIC
    ${HIP_LIBRARIES}
    roc::rocblas         # Using rocBLAS imported target
    roc::rocsolver       # Using rocSOLVER imported target
)

# If rocTensorUtil_kernels.hip needs to be compiled as part of this static library:
# Check if hipcc is the compiler, necessary for .hip files
if(CMAKE_CXX_COMPILER_ID STREQUAL "HIP" OR CMAKE_HIP_COMPILER_ID) # Check if HIP compiler is in use
    set_source_files_properties(rocTensorUtil_kernels.hip PROPERTIES LANGUAGE HIP)
    set_source_files_properties(PermutationKernels.hip PROPERTIES LANGUAGE HIP) # Set language for new kernel
    message(STATUS "rocTensorUtil_kernels.hip and PermutationKernels.hip will be compiled with HIP compiler.")
else()
    # This case should ideally not happen if project is configured for HIP
    message(WARNING "rocTensorUtil_kernels.hip and PermutationKernels.hip might not be compiled correctly as HIP compiler is not detected as CXX compiler.")
endif()

# Installation rules (optional for now, can be added later)
# install(TARGETS rocquantum_tensornet DESTINATION lib)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/rocquantum/rocTensorUtil.h # Incorrect path, fix later
#               ${CMAKE_CURRENT_SOURCE_DIR}/../include/rocquantum/hipTensorNet.h
#         DESTINATION include/rocquantum)

# Ensure that the rocTensorUtil.h and hipTensorNet.h headers are accessible
# The main rocquantum library will depend on this.
# The include path for rocquantum/* should be handled by the parent CMakeLists.txt
# when setting up include directories for targets linking against rocquantum.
# This target_include_directories makes sure that if something links ONLY to rocqsim_tensornet,
# it gets the necessary include paths.
# The $<INSTALL_getRootRuntimeDependenciesസ്റ്റ്rocquantum_DIR}/include> is a bit of a placeholder;
# it should correctly point to where rocquantum/rocTensorUtil.h and rocquantum/hipTensorNet.h are found
# relative to the build directory or install prefix. Often this is ${CMAKE_SOURCE_DIR}/rocquantum/include.
# For an internal library component, it's often just:
# target_include_directories(rocqsim_tensornet PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>")
# Or handled by the top-level CMakeLists.txt for all rocquantum components.
# For now, relying on the main CMakeLists to provide the include path for "rocquantum/*.h"
# The target_include_directories above for rocblas and HIP are correct.
# The one for rocquantum_DIR is problematic. Let's use a relative path for now.
# This assumes rocTensorUtil.h and hipTensorNet.h are in rocquantum/include/rocquantum/
target_include_directories(rocqsim_tensornet PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" # Path to rocquantum/include from rocquantum/src/hipTensorNet
)

# Add an alias if we want to link with a simpler name like roc::rocqsim_tensornet
# add_library(roc::rocqsim_tensornet ALIAS rocqsim_tensornet) # Optional

# --- Unit Tests ---
# Enable testing if it's not enabled at a higher level
if(NOT CMAKE_PROJECT_NAME STREQUAL "rocQuantumTests")
    enable_testing()
endif()

# Add the test executable for rocTensorUtil
add_executable(test_rocTensorUtil test_hipTensorNet_rocTensorUtil.cpp)
target_link_libraries(test_rocTensorUtil PRIVATE rocqsim_tensornet)

# Add the new test executable for slicing
add_executable(test_slicing test_hipTensorNet_slicing.cpp)
target_link_libraries(test_slicing PRIVATE rocqsim_tensornet)

# Add the new test executable for the permutation kernel
add_executable(test_permutation ../../tests/hipTensorNet/test_PermutationKernels.cpp)
target_link_libraries(test_permutation PRIVATE rocqsim_tensornet)


# Add tests to CTest
add_test(NAME RocTensorUtilTest COMMAND test_rocTensorUtil)
add_test(NAME SlicingLogicTest COMMAND test_slicing)
add_test(NAME PermutationKernelTest COMMAND test_permutation)
