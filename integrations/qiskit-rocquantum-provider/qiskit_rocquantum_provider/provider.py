# qiskit-rocquantum-provider/qiskit_rocquantum_provider/provider.py
from qiskit.providers import ProviderV1

from .backend import RocQuantumBackend

class RocQuantumProvider(ProviderV1):
    """
    Provides access to the rocQuantum simulator backend.
    """
    def __init__(self):
        super().__init__()
        self.name = "rocquantum_provider"
        self._backends = {"rocquantum_simulator": RocQuantumBackend()}

    def backends(self, name=None, **kwargs):
        return list(self._backends.values())
