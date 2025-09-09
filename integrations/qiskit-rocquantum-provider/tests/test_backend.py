# tests/test_backend.py
import unittest
import numpy as np
from qiskit import QuantumCircuit

try:
    from qiskit_rocquantum_provider.backend import RocQuantumBackend
    QISKIT_ROCQ_AVAILABLE = True
except ImportError:
    QISKIT_ROCQ_AVAILABLE = False

@unittest.skipIf(not QISKIT_ROCQ_AVAILABLE, "qiskit-rocquantum-provider is not installed")
class TestRocQuantumBackend(unittest.TestCase):
    def setUp(self):
        self.backend = RocQuantumBackend()

    def test_bell_state_statevector(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.save_statevector()
        statevector = self.backend.run(qc).result().get_statevector()
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        self.assertTrue(np.allclose(statevector, expected))

    def test_bell_state_counts(self):
        shots = 1000
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        counts = self.backend.run(qc, shots=shots).result().get_counts()
        self.assertEqual(set(counts.keys()), {"00", "11"})
        self.assertAlmostEqual(counts.get("00", 0) / shots, 0.5, delta=0.1)

    def test_rz_rotation(self):
        angle = np.pi / 2
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(angle, 0)
        qc.save_statevector()
        statevector = self.backend.run(qc).result().get_statevector()
        expected = (1/np.sqrt(2)) * np.array([np.exp(-1j*angle/2), np.exp(1j*angle/2)])
        self.assertTrue(np.allclose(statevector, expected))