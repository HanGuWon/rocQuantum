cmake_minimum_required(VERSION 3.18) # Pybind11 typically needs a modern CMake

# Find Pybind11
# This assumes Pybind11 is installed in a way CMake can find it (e.g., via pip install pybind11, or from source build)
# Or, if using Pybind11 as a submodule:
# add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11) # Adjust path if needed
find_package(pybind11 REQUIRED)

# Define the Python extension module
# The name given here will be the name of the .so file (e.g., _rocq_hip_backend.cpython-38-x86_64-linux-gnu.so)
# and can be imported in Python as `import _rocq_hip_backend` (or whatever name is chosen)
pybind11_add_module(_rocq_hip_backend # Module name
    SHARED # Create a shared library
    bindings.cpp # Source file for the bindings
)

# Link against the hipStateVec library
# Assumes hipStateVec is a target defined in a parent CMake scope (e.g., in src/hipStateVec/CMakeLists.txt)
# and that the root CMakeLists.txt has `add_subdirectory(src/hipStateVec)` before `add_subdirectory(python/rocq)`
target_link_libraries(_rocq_hip_backend PRIVATE hipStateVec)

# Add include directories
# Include directory for hipStateVec API (hipStateVec.h)
target_include_directories(_rocq_hip_backend PRIVATE
    ${PROJECT_SOURCE_DIR}/include # For rocquantum/hipStateVec.h
    # Pybind11 include directory is usually handled by pybind11_add_module or find_package(pybind11)
    # ${pybind11_INCLUDE_DIRS} # Explicitly if needed
)

# Optional: Specify C++ standard if not inherited or if specific version needed for bindings
# target_compile_features(_rocq_hip_backend PRIVATE cxx_std_17) # Example for C++17

# Optional: Installation rules for the Python module
# This helps if you want to install your package system-wide or in a virtual environment
# pybind11_install(_rocq_hip_backend) # Simplified install
# Or more detailed:
# install(TARGETS _rocq_hip_backend LIBRARY DESTINATION rocq) # Installs to <prefix>/lib/pythonX.Y/site-packages/rocq/
# The destination should match the package structure you want.
# For now, we'll rely on the build directory for testing.

message(STATUS "rocQuantum Python bindings module configured: _rocq_hip_backend")
