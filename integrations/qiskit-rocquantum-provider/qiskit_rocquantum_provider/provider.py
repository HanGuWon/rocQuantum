from qiskit.providers import ProviderV1
from .backend import RocQuantumBackend

class RocQuantumProvider(ProviderV1):
    """
    Provider for the rocQuantum simulator backend.
    """
    def __init__(self):
        super().__init__()
        self.name = 'rocquantum_provider'
        # Instantiate and store the backend instance
        self._backends = {"rocq_simulator": RocQuantumBackend(provider=self)}

    def backends(self, name=None, **kwargs):
        """Return a list of backends."""
        if name:
            return [self._backends[name]]
        return list(self._backends.values())