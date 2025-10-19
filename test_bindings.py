# test_bindings.py

import pytest
import numpy as np
import sys
import os

# Add the build directory to the path to find the compiled module
# This is a common pattern for testing with setuptools
# Note: The exact path depends on the build process (e.g., build/lib.win-amd64-3.9)
# A more robust way is to run tests after installation.
# For now, we assume the user will run `pip install .` before `pytest`.

try:
    import rocquantum_bind
except ImportError as e:
    # Provide a helpful error message if the module is not found.
    pytest.skip(
        f"rocquantum_bind module not found. Have you built it? Run 'pip install .' in the project root. Error: {e}",
        allow_module_level=True,
    )


def test_simulator_initialization():
    """
    Tests if the QSim object can be instantiated correctly.
    """
    try:
        sim = rocquantum_bind.QSim(num_qubits=2)
        assert sim is not None, "Simulator object should not be None."
    except Exception as e:
        pytest.fail(f"Failed to instantiate QSim: {e}")


def test_bell_state_simulation():
    """
    Create a Bell state |Φ+⟩ using the hipStateVec-backed simulator and verify the
    resulting statevector matches the analytic expectation.
    """
    num_qubits = 2

    # 1. Instantiate the simulator
    sim = rocquantum_bind.QSim(num_qubits)

    # 2. Apply gates to create a Bell state: |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
    sim.ApplyGate("H", target_qubit=0)
    sim.ApplyGate("CNOT", control_qubit=0, target_qubit=1)

    # 3. Execute the circuit
    sim.Execute()

    # 4. Retrieve the final state vector
    final_state_vector = sim.GetStateVector()

    # 5. Verify the result
    expected_state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128)

    assert final_state_vector is not None, "GetStateVector should return a NumPy array, not None."
    assert isinstance(final_state_vector, np.ndarray), "State vector should be a NumPy array."
    assert final_state_vector.dtype == np.complex128, (
        f"State vector should have dtype complex128, but has {final_state_vector.dtype}."
    )
    assert final_state_vector.shape == (2**num_qubits,), (
        f"State vector shape should be {(2**num_qubits,)}, but is {final_state_vector.shape}."
    )

    # Use np.allclose for robust floating-point comparison
    assert np.allclose(final_state_vector, expected_state), (
        f"State vector {final_state_vector} does not match expected {expected_state}"
    )


def test_custom_gate_application():
    """
    Tests the binding for applying a custom gate matrix from NumPy.
    """
    sim = rocquantum_bind.QSim(num_qubits=1)

    # Define a Pauli-X gate as a NumPy array
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    try:
        # This call just checks if the binding works without crashing.
        sim.ApplyGate(pauli_x, target_qubit=0)
    except Exception as e:
        pytest.fail(f"ApplyGate with a NumPy array failed: {e}")
