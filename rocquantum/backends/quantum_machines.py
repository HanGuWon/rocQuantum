# rocquantum/backends/quantum_machines.py
# TODO: Implement the Quantum Machines backend client.
from .base import RocqBackend
class QuantumMachinesBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def _build_payload(self, circuit, shots): pass
