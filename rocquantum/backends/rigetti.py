# rocquantum/backends/rigetti.py

"""
This module provides a concrete implementation of the RocqBackend for the 
Rigetti quantum computing platform, accessed via Amazon Braket.

It uses the boto3 library to interact with the AWS Braket service, 
encapsulating the logic for authentication, job submission, and result
retrieval for Rigetti QPUs.
"""

import boto3
from botocore.exceptions import ClientError
from typing import Dict, Any

from .base import (
    RocqBackend,
    BackendAuthenticationError,
    JobSubmissionError,
)

# Default AWS region for Braket if not specified
DEFAULT_AWS_REGION = "us-west-1"

# ARN for the Rigetti Aspen-M-3 QPU on AWS Braket
RIGETTI_ASPEN_M_3_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"


class RigettiBackend(RocqBackend):
    """
    A client for interacting with Rigetti quantum hardware via AWS Braket.

    This class implements the RocqBackend interface but overrides the default
    API-based methods to use the boto3 SDK for AWS communication.
    """

    def __init__(self, backend_name: str = "rigetti", aws_region: str = DEFAULT_AWS_REGION):
        """
        Initializes the Rigetti backend client.

        Args:
            backend_name (str): The name of the backend.
            aws_region (str): The AWS region where the Braket service is hosted.
        """
        super().__init__(backend_name=backend_name, api_endpoint="")  # API endpoint not used
        self.aws_region = aws_region
        self.braket_client = None

    def authenticate(self) -> None:
        """
        Authenticates with AWS Braket using credentials configured for boto3.

        boto3 automatically searches for credentials in environment variables,
        shared credential files (~/.aws/credentials), or IAM roles.

        Raises:
            BackendAuthenticationError: If the AWS session cannot be created
                                        due to missing credentials or other
                                        configuration issues.
        """
        try:
            self.braket_client = boto3.client("braket", region_name=self.aws_region)
            # A simple check to confirm client was created
            if self.braket_client is None:
                 raise ClientError({}, "Boto3 client creation failed")
            print("AWS Braket client created successfully.")
        except ClientError as e:
            raise BackendAuthenticationError(
                f"AWS Braket authentication failed: {e}. Ensure your AWS "
                "credentials (e.g., AWS_ACCESS_KEY_ID) are configured correctly."
            )

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Returns an empty dictionary as authentication is handled by the SDK.
        """
        return {}

    def submit_job(self, circuit: str, shots: int) -> str:
        """
        Submits a quantum circuit (as a QASM string) to a Rigetti QPU on AWS Braket.

        This method overrides the base class's REST API implementation to use
        the boto3 SDK.

        Args:
            circuit (str): The OpenQASM 3.0 representation of the circuit.
            shots (int): The number of times to execute the circuit.

        Returns:
            str: The `quantumTaskArn` which serves as the unique job ID.

        Raises:
            JobSubmissionError: If the job submission via boto3 fails.
        """
        if self.braket_client is None:
            raise JobSubmissionError("Braket client not initialized. Please call authenticate() first.")

        # This requires the user to have an S3 bucket for Braket results.
        # For this example, we'll assume a placeholder. In a real scenario,
        # this should be configurable.
        s3_output_location = f"s3://amazon-braket-{self.aws_region}-<YOUR_ACCOUNT_ID>/rigetti-tasks"

        try:
            response = self.braket_client.create_quantum_task(
                action={
                    "type": "OPENQASM",
                    "source": circuit,
                },
                shots=shots,
                deviceArn=RIGETTI_ASPEN_M_3_ARN,
                s3OutputLocation=s3_output_location,
            )
        except ClientError as e:
            raise JobSubmissionError(f"Failed to submit job to AWS Braket: {e}")

        quantum_task_arn = response.get("quantumTaskArn")
        if not quantum_task_arn:
            raise JobSubmissionError("AWS Braket response did not contain a quantumTaskArn.")
            
        return quantum_task_arn