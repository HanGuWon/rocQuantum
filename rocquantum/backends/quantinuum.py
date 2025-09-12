# rocquantum/backends/quantinuum.py

"""
This module provides a concrete implementation of the RocqBackend for the
Quantinuum quantum computing platform.

Following the DRY principle, this class now only implements the logic unique
to Quantinuum: file-based authentication, Bearer token header construction,
and the specific JSON payload structure. All common HTTP logic is inherited
from the RocqBackend base class.
"""

import os
import json
from typing import Dict, Any, Optional

from .base import (
    RocqBackend,
    BackendAuthenticationError,
)

QUANTINUUM_API_ENDPOINT = "https://api.quantinuum.com"

class QuantinuumBackend(RocqBackend):
    """
    A client for interacting with Quantinuum hardware, featuring a file-based
    authentication system.
    """

    def __init__(self, backend_name: str = "quantinuum", api_endpoint: str = QUANTINUUM_API_ENDPOINT):
        """Initializes the Quantinuum backend client."""
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
        self.auth_credentials: Optional[Dict[str, Any]] = None

    def authenticate(self) -> None:
        """
        Authenticates with the Quantinuum API by loading credentials from a
        JSON file specified by the `CUDAQ_QUANTINUUM_CREDENTIALS` env var.
        """
        credentials_path = os.getenv("CUDAQ_QUANTINUUM_CREDENTIALS")
        if not credentials_path:
            raise BackendAuthenticationError(
                "Authentication failed: The 'CUDAQ_QUANTINUUM_CREDENTIALS' "
                "environment variable is not set."
            )
        try:
            with open(credentials_path, 'r') as f:
                self.auth_credentials = json.load(f)
        except FileNotFoundError:
            raise BackendAuthenticationError(
                f"Authentication failed: Credentials file not found at '{credentials_path}'"
            )
        except json.JSONDecodeError:
            raise BackendAuthenticationError(
                f"Authentication failed: File at '{credentials_path}' is not valid JSON."
            )
        print("Quantinuum authentication successful.")

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Constructs the Bearer token authorization headers from the loaded credentials.
        """
        if not self.auth_credentials:
            raise BackendAuthenticationError(
                "Client is not authenticated. Please call authenticate() first."
            )
        access_token = self.auth_credentials.get("access_token")
        if not access_token:
            raise BackendAuthenticationError(
                "Authentication failed: 'access_token' not found in credentials file."
            )
        return {"Authorization": f"Bearer {access_token}"}

    def _build_payload(self, circuit_representation: str, shots: int) -> Dict[str, Any]:
        """
        Constructs the specific JSON payload required by the Quantinuum API.
        """
        # TODO: Implement QIR payload generation for enhanced interoperability.
        return {
            "target": self.backend_name,
            "shots": shots,
            "body": {
                "language": "OPENQASM",
                "program": circuit_representation,
            },
        }