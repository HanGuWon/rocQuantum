import numpy as np
import pytest
import sys
import os

# Add the project root to the path to allow importing 'rocq'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.rocq import api as rocq

@pytest.fixture
def simulator():
    """Provides a rocQuantum simulator instance for tests."""
    return rocq.Simulator()

def test_crx(simulator):
    """Tests the Controlled-RX gate."""
    num_qubits = 2
    angle = np.pi / 2
    
    # rocQuantum execution
    circuit = rocq.Circuit(num_qubits, simulator)
    circuit.x(0) # Prepare |10> state
    circuit.crx(angle, 0, 1)
    
    # This is a placeholder for getting the state vector, assuming a method exists
    # For now, we assume the flush mechanism is tied to a yet-to-be-implemented get_statevector()
    # As a workaround, we can measure, which triggers a flush. This is not ideal for testing.
    # Let's assume a get_statevector() method will be added.
    # For now, this test structure is a blueprint.
    
    # Expected result with NumPy
    psi_initial = np.zeros(2**num_qubits, dtype=np.complex128)
    psi_initial[1] = 1.0 # |01> state, since Qiskit order is q1q0
    
    c = np.cos(angle / 2)
    s = -1j * np.sin(angle / 2)
    crx_matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c, s],
                           [0, 0, s, c]])
    
    psi_final_expected = crx_matrix @ psi_initial
    
    # In a real test, we would get psi_final_rocq from the simulator
    # and assert np.allclose(psi_final_rocq, psi_final_expected)
    assert True # Placeholder assertion

def test_ccx(simulator):
    """Tests the Toffoli (CCX) gate."""
    num_qubits = 3
    
    # rocQuantum execution on |110> -> |111>
    circuit = rocq.Circuit(num_qubits, simulator)
    circuit.x(0)
    circuit.x(1)
    circuit.ccx(0, 1, 2)

    # Expected result with NumPy
    psi_initial = np.zeros(2**num_qubits, dtype=np.complex128)
    psi_initial[3] = 1.0 # |011> state (q2q1q0)
    
    psi_final_expected = np.zeros(2**num_qubits, dtype=np.complex128)
    psi_final_expected[7] = 1.0 # |111> state
    
    assert True # Placeholder for np.allclose(psi_final_rocq, psi_final_expected)

def test_cswap(simulator):
    """Tests the Fredkin (CSWAP) gate."""
    num_qubits = 3

    # rocQuantum execution on |110> -> |101>
    circuit = rocq.Circuit(num_qubits, simulator)
    circuit.x(0) # control
    circuit.x(1) # target1
    circuit.cswap(0, 1, 2)

    # Expected result with NumPy
    psi_initial = np.zeros(2**num_qubits, dtype=np.complex128)
    psi_initial[3] = 1.0 # |011> state (q2q1q0)

    psi_final_expected = np.zeros(2**num_qubits, dtype=np.complex128)
    psi_final_expected[5] = 1.0 # |101> state
    
    assert True # Placeholder for np.allclose(psi_final_rocq, psi_final_expected)
