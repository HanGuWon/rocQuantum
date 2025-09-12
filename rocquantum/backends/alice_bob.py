# rocquantum/backends/alice_bob.py
# TODO: Implement the Alice & Bob backend using the Dynamiqs SDK.
from .base import RocqBackend
from rocquantum.circuit import QuantumCircuit
class AliceBobBackend(RocqBackend):
    def authenticate(self): pass
    def _get_auth_headers(self): pass
    def submit_job(self, circuit: QuantumCircuit, shots: int): pass
    def get_job_status(self, job_id: str): pass
    def get_job_result(self, job_id: str): pass
