# cirq_rocm/tests/test_simulator.py
import pytest
import cirq
import numpy as np
from collections import Counter

@pytest.fixture(scope="module")
def RocQuantumSimulator():
    try:
        from cirq_rocm import RocQuantumSimulator
        return RocQuantumSimulator
    except ImportError:
        pytest.skip("Could not import RocQuantumSimulator.")

def test_bell_state_simulation(RocQuantumSimulator):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    sim = RocQuantumSimulator()
    result = sim.simulate(circuit)
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    assert np.allclose(result.final_state_vector, expected)

def test_bell_state_run(RocQuantumSimulator):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key='m'))
    sim = RocQuantumSimulator()
    reps = 1000
    result = sim.run(circuit, repetitions=reps)
    outcomes = result.measurements['m'][:, 0] * 2 + result.measurements['m'][:, 1]
    counts = Counter(outcomes)
    assert set(counts.keys()) == {0, 3}
    assert abs(counts[0] - reps / 2) < 100

def test_ry_rotation(RocQuantumSimulator):
    q0 = cirq.LineQubit(0)
    angle = np.pi / 2
    circuit = cirq.Circuit(cirq.ry(rads=angle).on(q0))
    sim = RocQuantumSimulator()
    result = sim.simulate(circuit)
    expected = np.array([np.cos(angle/2), np.sin(angle/2)], dtype=np.complex128)
    assert np.allclose(result.final_state_vector, expected)