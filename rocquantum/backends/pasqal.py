# rocquantum/backends/pasqal.py

"""
This module provides a concrete implementation of the RocqBackend for the 
Pasqal quantum computing platform.

It enables communication with the Pasqal REST API to manage
the lifecycle of quantum jobs, including authentication, submission, 
status monitoring, and result retrieval.
"""

import os
import requests
from typing import Dict

from .base import (
    RocqBackend,
    BackendAuthenticationError,
    JobSubmissionError,
    ResultRetrievalError,
)

# The base URL for the Pasqal API
PASQAL_API_ENDPOINT = "https://api.pasqal.cloud"


class PasqalBackend(RocqBackend):
    """
    A client for interacting with the Pasqal quantum computing hardware.

    This class implements the RocqBackend interface and provides a concrete
    method for executing quantum circuits on Pasqal's QPUs.
    """

    def __init__(self, backend_name: str = "pasqal", api_endpoint: str = PASQAL_API_ENDPOINT):
        """
        Initializes the Pasqal backend client.

        Args:
            backend_name (str): The specific name of the Pasqal backend to target.
            api_endpoint (str): The base URL for the Pasqal API.
        """
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
        self.api_key: str | None = None

    def authenticate(self) -> None:
        """
        Authenticates with the Pasqal API using an API key.

        This method reads the API key from the `PASQAL_API_KEY` environment
        variable.

        Raises:
            BackendAuthenticationError: If the `PASQAL_API_KEY` environment
                                        variable is not set or is empty.
        """
        api_key = os.getenv("PASQAL_API_KEY")
        if not api_key:
            raise BackendAuthenticationError(
                "Authentication failed: The 'PASQAL_API_KEY' environment variable "
                "is not set. Please set it to your Pasqal API key."
            )
        self.api_key = api_key
        print("Authentication successful.")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Constructs the authorization headers required for API requests."""
        if not self.api_key:
            raise BackendAuthenticationError(
                "Client is not authenticated. Please call authenticate() first."
            )
        return {"Authorization": f"ApiKey {self.api_key}"}

    def submit_job(self, circuit_representation: str, shots: int) -> str:
        """
        Submits a quantum circuit in OpenQASM format to the Pasqal backend.

        Args:
            circuit_representation (str): A string containing the OpenQASM 3.0
                                          representation of the quantum circuit.
            shots (int): The number of times the circuit will be executed.

        Returns:
            str: The unique job ID assigned by Pasqal for tracking.

        Raises:
            JobSubmissionError: If the API request fails or returns an
                                unsuccessful status code.
        """
        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"

        payload = {
            "target": self.backend_name,
            "shots": shots,
            "body": {
                "language": "OPENQASM",
                "program": circuit_representation,
            },
        }

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
            raise JobSubmissionError(
                f"Job submission failed: API response did not contain a job ID. "
                f"Response: {response_data}"
            )

        return job_id

    def get_job_status(self, job_id: str) -> str:
        """
        Retrieves the current status of a job from the Pasqal API.

        Args:
            job_id (str): The ID of the job to check.

        Returns:
            str: The status of the job.

        Raises:
            ResultRetrievalError: If the API request to get the job status fails.
        """
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
            raise ResultRetrievalError(
                f"API response for job '{job_id}' did not contain a status. "
                f"Response: {response_data}"
            )
            
        return status

    def get_job_result(self, job_id: str) -> Dict[str, int]:
        """
        Retrieves the results for a completed job from the Pasqal API.

        Args:
            job_id (str): The ID of the job to retrieve results for.

        Returns:
            Dict[str, int]: A dictionary containing the measurement histogram.

        Raises:
            ResultRetrievalError: If the job is not yet complete, or if the
                                  API request to get the results fails.
        """
        status = self.get_job_status(job_id)
        if status != "completed":
            raise ResultRetrievalError(
                f"Cannot retrieve results for job '{job_id}' because its "
                f"status is '{status}'. Results are only available for "
                "'completed' jobs."
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
             raise ResultRetrievalError(
                f"API response for job '{job_id}' did not contain a histogram. "
                f"Response: {response_data}"
            )

        return histogram
