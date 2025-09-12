# rocquantum/backends/orca.py
# TODO: Implement the ORCA Computing backend client.
from .base import RocqBackend
class OrcaBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def _build_payload(self, circuit, shots): pass
