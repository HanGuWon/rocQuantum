# rocquantum/backends/base.py

"""
This module defines the foundational abstract base class (ABC) for all
third-party QPU backend implementations within the rocQuantum framework.
"""

import abc
import requests
from typing import Dict, Any, Callable, Union

# Forward declaration to avoid circular imports
if False:
    from rocquantum.circuit import QuantumCircuit

# ==============================================================================
#  Custom Exception Classes
# ==============================================================================

class BackendAuthenticationError(Exception):
    """Raised when authentication with a third-party backend API fails."""
    pass

class JobSubmissionError(Exception):
    """Raised when a job submission to the backend fails."""
    pass

class ResultRetrievalError(Exception):
    """Raised when fetching the result of a completed job fails."""
    pass

# ==============================================================================
#  Refactored Abstract Base Class
# ==============================================================================

class RocqBackend(abc.ABC):
    """
    An abstract base class that defines the required interface and provides
    shared functionality for all rocQuantum hardware backend clients.
    """

    def __init__(self, backend_name: str, api_endpoint: str):
        """Initializes the backend client."""
        self.backend_name = backend_name
        self.api_endpoint = api_endpoint

    @abc.abstractmethod
    def authenticate(self) -> None:
        """Handles the provider-specific authentication flow."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Constructs provider-specific authorization headers."""
        raise NotImplementedError

    def _build_payload(self, circuit_representation: str, shots: int) -> Dict[str, Any]:
        """Constructs the provider-specific JSON payload for a job request."""
        raise NotImplementedError("This method is for Type A backends and must be overridden.")

    def submit_job(self, circuit: Union["QuantumCircuit", str], shots: int) -> str:
        """
        Submits a quantum circuit to the backend for execution.
        This is the default implementation for Type A (API-based) backends.
        Type B (local SDK) backends must override this method.
        """
        if not isinstance(circuit, str):
            raise JobSubmissionError(
                "This backend requires a pre-compiled QASM string. "
                "To submit a QuantumCircuit object, use a different backend."
            )
        
        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"
        
        payload = self._build_payload(circuit, shots)

        try:
            response = requests.post(
                f"{self.api_endpoint}/jobs", headers=headers, json=payload
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise JobSubmissionError(f"Job submission failed due to a network error: {e}")

        response_data = response.json()
        job_id = response_data.get("id")

        if not job_id:
            raise JobSubmissionError(f"API response did not contain a job ID.")
        return job_id

    def get_job_status(self, job_id: str) -> str:
        """Retrieves the current status of a job from the backend API."""
        try:
            response = requests.get(
                f"{self.api_endpoint}/jobs/{job_id}", headers=self._get_auth_headers()
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ResultRetrievalError(f"Failed to get job status for job '{job_id}': {e}")

        response_data = response.json()
        status = response_data.get("status")

        if not status:
            raise ResultRetrievalError(f"API response for job '{job_id}' did not contain a status.")
        return status

    def get_job_result(self, job_id: str) -> Dict[str, int]:
        """Retrieves the final measurement counts for a completed job."""
        status = self.get_job_status(job_id)
        if status != "completed":
            raise ResultRetrievalError(
                f"Cannot retrieve results for job '{job_id}' because its status is '{status}'."
            )

        try:
            response = requests.get(
                f"{self.api_endpoint}/jobs/{job_id}", headers=self._get_auth_headers()
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ResultRetrievalError(f"Failed to retrieve results for job '{job_id}': {e}")

        response_data = response.json()
        histogram = response_data.get("data", {}).get("histogram")

        if histogram is None:
            raise ResultRetrievalError(f"API response for job '{job_id}' did not contain a histogram.")
        return histogram
