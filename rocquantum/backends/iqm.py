# rocquantum/backends/iqm.py
# TODO: Implement the IQM backend client.
from .base import RocqBackend
class IQMBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def _build_payload(self, circuit, shots): pass
