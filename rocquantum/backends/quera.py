# rocquantum/backends/quera.py
# TODO: Implement the QuEra backend client.
from .base import RocqBackend
class QuEraBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def _build_payload(self, circuit, shots): pass
