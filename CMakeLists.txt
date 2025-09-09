cmake_minimum_required(VERSION 3.18)
project(rocQuantum LANGUAGES CXX)

# Find HIP from ROCm
# Assumes ROCM_PATH is set, or cmake can find it.
find_package(HIP REQUIRED)

# Find Python
find_package(PythonLibs 3 REQUIRED)

# Find pybind11
# This assumes pybind11 is installed in a location findable by CMake.
# If not, you may need to provide -Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11
find_package(pybind11 REQUIRED)

# ------------------------------------------------------------------
# 1. Define the core C++ static library
# ------------------------------------------------------------------
add_library(rocquantum STATIC
    rocquantum/src/simulator.cpp
)

# Add include directories for the library
target_include_directories(rocquantum PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include
)

# Link rocquantum against HIP
target_link_libraries(rocquantum PRIVATE hip::host)

# ------------------------------------------------------------------
# 2. Define the Python binding module
# ------------------------------------------------------------------
pybind11_add_module(rocquantum_bind
    bindings.cpp
)

# Link the binding module against our static library and Python
target_link_libraries(rocquantum_bind PRIVATE
    rocquantum
    ${PYTHON_LIBRARIES}
)

# Add include directories for the binding module
target_include_directories(rocquantum_bind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${pybind11_INCLUDE_DIRS}
)
