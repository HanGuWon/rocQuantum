# rocquantum/backends/infleqtion.py

"""
This module provides a concrete implementation of the RocqBackend for the 
Infleqtion quantum computing platform using the Superstaq API.

It enables communication with the Infleqtion REST API to manage
the lifecycle of quantum jobs, including authentication, submission, 
status monitoring, and result retrieval.
"""

import os
from typing import Dict, Any

from .base import (
    RocqBackend,
    BackendAuthenticationError,
)

# The base URL for the Infleqtion Superstaq API
INFLEQTION_API_ENDPOINT = "https://api.superstaq.infleqtion.com"


class InfleqtionBackend(RocqBackend):
    """
    A client for interacting with the Infleqtion quantum computing hardware
    via the Superstaq API.

    This class implements the RocqBackend interface and provides a concrete
    method for executing quantum circuits on Infleqtion's QPUs.
    """

    def __init__(self, backend_name: str = "infleqtion", api_endpoint: str = INFLEQTION_API_ENDPOINT):
        """
        Initializes the Infleqtion backend client.

        Args:
            backend_name (str): The specific name of the Infleqtion backend to target.
            api_endpoint (str): The base URL for the Superstaq API.
        """
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
        self.api_key: str | None = None

    def authenticate(self) -> None:
        """
        Authenticates with the Superstaq API using an API key.

        This method reads the API key from the `SUPERSTAQ_API_KEY` environment
        variable.

        Raises:
            BackendAuthenticationError: If the `SUPERSTAQ_API_KEY` environment
                                        variable is not set or is empty.
        """
        api_key = os.getenv("SUPERSTAQ_API_KEY")
        if not api_key:
            raise BackendAuthenticationError(
                "Authentication failed: The 'SUPERSTAQ_API_KEY' environment variable "
                "is not set. Please set it to your Superstaq API key."
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
        Constructs the provider-specific JSON payload for an Infleqtion job request.

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