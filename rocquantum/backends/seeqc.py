# rocquantum/backends/seeqc.py
# TODO: Implement the SEEQC backend client.
from .base import RocqBackend
class SeeqcBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def _build_payload(self, circuit, shots): pass
