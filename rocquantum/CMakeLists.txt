cmake_minimum_required(VERSION 3.18)
project(rocQuantum LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HIP
find_package(HIP REQUIRED)
if(NOT HIP_FOUND)
    message(FATAL_ERROR "HIP not found. Please set HIP_PATH or ensure amdhip64/bin is in PATH.")
endif()
message(STATUS "Found HIP: ${HIP_ROOT_DIR}")
message(STATUS "HIP Language: ${HIP_LANG}") # Should be HIP or CUDA

# Set compiler flags for HIP
# With CMake 3.21+ and ROCm 5.2+, HIP language support is more native.
# For older versions or more control, you might need to set CMAKE_HIP_ARCHITECTURES
# Example: set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100")
# Or let it auto-detect if possible.
if(DEFINED ENV{CMAKE_HIP_ARCHITECTURES})
    set(CMAKE_HIP_ARCHITECTURES $ENV{CMAKE_HIP_ARCHITECTURES})
    message(STATUS "Using CMAKE_HIP_ARCHITECTURES from environment: ${CMAKE_HIP_ARCHITECTURES}")
else()
    # Attempt to get current GPU architecture if only one is present or use a default
    # This is a simplified approach. A robust build system might require user to specify.
    # execute_process(COMMAND ${HIP_ROOT_DIR}/bin/rocminfo COMMAND grep "Name: gfx" OUTPUT_VARIABLE ROCM_GPU_NAME ERROR_QUIET)
    # if(ROCM_GPU_NAME)
    #   string(REGEX REPLACE ".*Name: *(gfx[0-9a-f]+).*" "\1" CMAKE_HIP_ARCHITECTURES "${ROCM_GPU_NAME}")
    #   message(STATUS "Auto-detected HIP Architecture: ${CMAKE_HIP_ARCHITECTURES}")
    # else()
    #   message(WARNING "Could not auto-detect HIP architecture. Set CMAKE_HIP_ARCHITECTURES. Using default gfx906.")
    #   set(CMAKE_HIP_ARCHITECTURES "gfx906") # MI100 as a default example
    # endif()
    # For now, let's not set it explicitly and rely on ROCm's default or environment variable
    message(STATUS "CMAKE_HIP_ARCHITECTURES not set explicitly in CMakeLists.txt. Relying on ROCm default or environment variable.")
endif()


# Find rocBLAS
# ROCm CMake modules should provide FindrocBLAS.cmake
# Ensure CMAKE_MODULE_PATH includes path to ROCm CMake files if not in standard location
# e.g., list(APPEND CMAKE_MODULE_PATH "${HIP_ROOT_DIR}/../lib/cmake/rocblas")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_ROOT_DIR}/lib/cmake/rocblas" "${HIP_ROOT_DIR}/../lib/cmake/rocblas" "/opt/rocm/lib/cmake/rocblas")
find_package(rocBLAS REQUIRED)
if(NOT rocBLAS_FOUND)
    message(FATAL_ERROR "rocBLAS not found.")
endif()
message(STATUS "Found rocBLAS: ${rocBLAS_LIBRARIES}")

# Find hipRAND (needed for rocsvMeasure, though current uses basic hiprand)
# Similar to rocBLAS for finding the package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_ROOT_DIR}/lib/cmake/hiprand" "${HIP_ROOT_DIR}/../lib/cmake/hiprand" "/opt/rocm/lib/cmake/hiprand")
find_package(hiprand REQUIRED) # hipRAND provides hiprand::hiprand target
if(NOT hiprand_FOUND)
    message(FATAL_ERROR "hiprand not found.")
endif()
message(STATUS "Found hiprand: ${hiprand_LIBRARIES}")


# Add include directory for the project's public API
include_directories(
    ${PROJECT_SOURCE_DIR}/include # For rocquantum/hipStateVec.h
    ${HIP_INCLUDE_DIRS}
    ${rocBLAS_INCLUDE_DIRS}
    ${hiprand_INCLUDE_DIRS}
)

add_subdirectory(src/hipStateVec)

# Example for tests (can be expanded later)
# enable_testing()
# add_subdirectory(tests/hipStateVec) # If you add tests later

message(STATUS "rocQuantum project configured. HIP compiler: ${CMAKE_HIP_COMPILER}")

add_subdirectory(python/rocq)
