# rocquantum/backends/ionq.py

"""
This module provides a concrete implementation of the RocqBackend for the 
IonQ quantum computing platform.

It enables communication with the IonQ REST API (Version 0.3) to manage
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

# The base URL for the IonQ API, version 0.3
IONQ_API_V0_3_ENDPOINT = "https://api.ionq.co/v0.3"


class IonQBackend(RocqBackend):
    """
    A client for interacting with the IonQ quantum computing hardware.

    This class implements the RocqBackend interface and provides a concrete
    method for executing quantum circuits on IonQ's QPUs through their
    public REST API.
    """

    def __init__(self, backend_name: str = "qpu", api_endpoint: str = IONQ_API_V0_3_ENDPOINT):
        """
        Initializes the IonQ backend client.

        Args:
            backend_name (str): The specific name of the IonQ backend to target.
                                Defaults to 'qpu'. Other examples include 
                                'qpu.aria-1' or 'simulator'.
            api_endpoint (str): The base URL for the IonQ API. Defaults to the
                                standard v0.3 endpoint.
        """
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
        self.api_key: str | None = None

    def authenticate(self) -> None:
        """
        Authenticates with the IonQ API using an API key.

        This method reads the API key from the `IONQ_API_KEY` environment
        variable.

        Raises:
            BackendAuthenticationError: If the `IONQ_API_KEY` environment
                                        variable is not set or is empty.
        """
        api_key = os.getenv("IONQ_API_KEY")
        if not api_key:
            raise BackendAuthenticationError(
                "Authentication failed: The 'IONQ_API_KEY' environment variable "
                "is not set. Please set it to your IonQ API key."
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
        Submits a quantum circuit in OpenQASM format to the IonQ backend.

        Args:
            circuit_representation (str): A string containing the OpenQASM 3.0
                                          representation of the quantum circuit.
            shots (int): The number of times the circuit will be executed.

        Returns:
            str: The unique job ID assigned by IonQ for tracking.

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
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
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
        Retrieves the current status of a job from the IonQ API.

        Args:
            job_id (str): The ID of the job to check.

        Returns:
            str: The status of the job (e.g., 'ready', 'running', 'completed').

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
        Retrieves the results for a completed job from the IonQ API.

        Args:
            job_id (str): The ID of the job to retrieve results for.

        Returns:
            Dict[str, int]: A dictionary containing the measurement histogram,
                            where keys are measurement outcomes (as strings
                            representing integers) and values are the counts.

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
        
        # Note: IonQ returns histogram keys as strings of integers. For true
        # bitstring representation, one would need to know the number of
        # measured qubits to format them (e.g., '1' -> '01'). Here, we return
        # the direct output from the API.
        histogram = response_data.get("data", {}).get("histogram")

        if histogram is None:
             raise ResultRetrievalError(
                f"API response for job '{job_id}' did not contain a histogram. "
                f"Response: {response_data}"
            )

        return histogram
