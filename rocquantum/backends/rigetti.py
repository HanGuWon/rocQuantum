# rocquantum/backends/rigetti.py
# TODO: Implement the Rigetti backend client.
from .base import RocqBackend
class RigettiBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def _build_payload(self, circuit, shots): pass
