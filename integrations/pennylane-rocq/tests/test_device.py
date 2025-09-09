# tests/test_device.py
import pytest
import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.tryfirst

@pytest.fixture(scope="module")
def rocquantum_bind_module():
    try:
        import rocquantum_bind
        return rocquantum_bind
    except ImportError:
        pytest.skip("The 'rocquantum_bind' module is not installed.")

def test_bell_state_circuit(rocquantum_bind_module):
    dev = qml.device("rocquantum.qpu", wires=2)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    assert np.allclose(circuit(), expected)

def test_measurement_counts(rocquantum_bind_module):
    shots = 1000
    dev = qml.device("rocquantum.qpu", wires=2, shots=shots)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts()
    counts = circuit()
    assert set(counts.keys()) == {"00", "11"}
    assert abs(counts["00"] - shots / 2) < 100

def test_rx_rotation(rocquantum_bind_module):
    dev = qml.device("rocquantum.qpu", wires=1)
    angle = np.pi / 2
    @qml.qnode(dev)
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.state()
    expected = np.array([np.cos(angle/2), -1j*np.sin(angle/2)], dtype=np.complex128)
    assert np.allclose(circuit(angle), expected)