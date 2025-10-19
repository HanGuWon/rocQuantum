# setup.py

import os
import sys
from setuptools import setup

# pybind11-specific build helpers
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("pybind11 is required to build this package. Please install it with 'pip install pybind11'", file=sys.stderr)
    sys.exit(1)


# --- Developer Customization Section ---
# Set these environment variables to point to the correct locations
# or modify the paths directly here.

# Path to the ROCm installation directory
# On Windows, this might not be a standard path. Adjust as needed.
ROCM_PATH = os.environ.get("ROCM_PATH", "C:/Program Files/AMD/ROCm")

# Path to the rocQuantum-1 library project root
# We assume this setup.py is in the root of the rocQuantum-1 project.
ROCQUANTUM_PATH = os.path.dirname(os.path.abspath(__file__))
# --- End Customization Section ---

# Check if the paths are valid
if not os.path.exists(ROCM_PATH):
    print(f"Warning: ROCM_PATH '{ROCM_PATH}' does not exist.", file=sys.stderr)
    print("Please set the ROCM_PATH environment variable or edit setup.py.", file=sys.stderr)
    # On Windows, we might not fail immediately, but compilation will likely fail.

if not os.path.exists(ROCQUANTUM_PATH):
    print(f"Error: Could not determine ROCQUANTUM_PATH '{ROCQUANTUM_PATH}'.", file=sys.stderr)
    sys.exit(1)

# Define include and library paths
include_dirs = [
    os.path.join(ROCQUANTUM_PATH, "include"),
    os.path.join(ROCQUANTUM_PATH, "rocquantum", "include"),
    os.path.join(ROCM_PATH, "include"),
]

library_dirs = [
    os.path.join(ROCQUANTUM_PATH, "build"),
    os.path.join(ROCQUANTUM_PATH, "build", "rocquantum", "src", "hipStateVec"),
    os.path.join(ROCQUANTUM_PATH, "build", "rocquantum", "src", "hipTensorNet"),
    os.path.join(ROCM_PATH, "lib"),
]

# Add pybind11's include path
import pybind11
pybind11_include = pybind11.get_include()
if pybind11_include not in include_dirs:
    include_dirs.append(pybind11_include)


ext_modules = [
    Pybind11Extension(
        "rocquantum_bind",
        ["bindings.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[
            "rocquantum",
            "hipStateVec",
            "rocqsim_tensornet",
            "amdhip64",
            "hiprand",
            "rocblas",
            "rocsolver",
        ],
        extra_compile_args=["/EHsc", "/std:c++17"] if sys.platform == "win32" else ["-std=c++17"],
        extra_link_args=[],
    ),
]

setup(
    name="rocquantum_bind",
    version="0.1.0",
    author="Gemini",
    description="Python bindings for the rocQuantum simulator",
    long_description="This package provides the Python interface to the C++/HIP based rocQuantum library.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.10",
        "numpy"
    ],
)
