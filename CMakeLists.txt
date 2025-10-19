cmake_minimum_required(VERSION 3.18)
project(rocQuantum LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Locate toolchains and dependencies
# ------------------------------------------------------------------
find_package(HIP REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------------
# Component libraries
# ------------------------------------------------------------------
add_subdirectory(rocquantum/src/hipStateVec)
add_subdirectory(rocquantum/src/hipTensorNet)

# Core static library that aggregates higher-level logic
add_library(rocquantum STATIC
    rocquantum/src/simulator.cpp
    rocquantum/src/kernels.hip.cpp
    rocqCompiler/HipStateVecBackend.cpp
)

target_include_directories(rocquantum
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include
        ${CMAKE_CURRENT_SOURCE_DIR} # for bindings that include project headers
)

target_link_libraries(rocquantum
    PUBLIC
        hipStateVec
        rocqsim_tensornet
)

# ------------------------------------------------------------------
# Python binding module
# ------------------------------------------------------------------
pybind11_add_module(rocquantum_bind bindings.cpp)

target_include_directories(rocquantum_bind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include
)

target_link_libraries(rocquantum_bind
    PRIVATE
        rocquantum
        pybind11::module
        Python3::Python
)
