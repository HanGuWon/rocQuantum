# rocQuantum-1 Framework Main Entry Point
# This file makes the `rocq` directory a Python package and exposes the public API.

"""
rocQuantum-1 (rocq)

A high-performance quantum computing simulation framework for AMD GPUs.
"""

# Public API Imports
from .noise import NoiseModel
from .kernel import kernel, execute
from .operator import QuantumOperator, PauliOperator, HermitianOperator, get_expectation_value
from .qvec import qvec
from .gates import h, ry, rz, cnot, x, y, z
