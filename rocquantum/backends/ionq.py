# rocquantum/backends/ionq.py

"""
This module provides a concrete implementation of the RocqBackend for the 
IonQ quantum computing platform.

It enables communication with the IonQ REST API (Version 0.3) to manage
the lifecycle of quantum jobs, including authentication, submission, 
status monitoring, and result retrieval.
"""

import os
from typing import Dict, Any

from .base import (
    RocqBackend,
    BackendAuthenticationError,
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

    def _build_payload(self, circuit_representation: str, shots: int) -> Dict[str, Any]:
        """
        Constructs the provider-specific JSON payload for an IonQ job request.

        Args:
            circuit_representation (str): The OpenQASM 3.0 string for the circuit.
            shots (int): The number of execution shots.

        Returns:
            Dict[str, Any]: The JSON payload for the API request.
        """
        return {
            "target": self.backend_name,
            "shots": shots,
            "body": {
                "language": "OPENQASM",
                "program": circuit_representation,
            },
        }