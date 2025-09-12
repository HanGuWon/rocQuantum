# rocquantum/backends/base.py

"""
This module defines the foundational abstract base class (ABC) for all 
third-party QPU backend implementations within the rocQuantum framework.

It establishes a mandatory, unified interface for the rocQuantum runtime 
to interact with any quantum hardware provider, ensuring a consistent 
execution lifecycle: Authentication -> Job Submission -> Result Retrieval.
"""

import abc
from typing import Dict

# ==============================================================================
#  Custom Exception Classes
# ==============================================================================

class BackendAuthenticationError(Exception):
    """
    Raised when authentication with a third-party backend API fails.
    
    This could be due to invalid credentials, expired tokens, or network issues
    preventing a successful connection to the authentication endpoint.
    """
    pass

class JobSubmissionError(Exception):
    """
    Raised when a job submission to the backend fails.
    
    This can occur if the circuit representation is invalid, the specified
    backend is unavailable, or the user's account has insufficient credits.
    """
    pass

class ResultRetrievalError(Exception):
    """
    Raised when fetching the result of a completed job fails.
    
    This may happen if the job ID is not found, the job has failed internally
    on the provider's end, or if there is a network error during data retrieval.
    """
    pass

# ==============================================================================
#  Abstract Base Class for Backend Implementations
# ==============================================================================

class RocqBackend(abc.ABC):
    """
    An abstract base class that defines the required interface for all
    rocQuantum hardware backend clients.
    
    This class ensures that every backend implementation adheres to a standard
    contract, allowing the rocQuantum runtime to manage them in a hardware-
    agnostic manner. It formalizes the essential operations of authentication,
    job submission, status checking, and result retrieval.
    """

    def __init__(self, backend_name: str, api_endpoint: str):
        """
        Initializes the backend client.

        Args:
            backend_name (str): The specific name of the provider's backend
                                (e.g., 'ionq_qpu', 'quantinuum_h1').
            api_endpoint (str): The base URL for the provider's API.
        """
        self.backend_name = backend_name
        self.api_endpoint = api_endpoint

    @abc.abstractmethod
    def authenticate(self) -> None:
        """
        Handles authentication with the provider's API.
        
        Implementations should securely handle credentials, which may be
        sourced from environment variables, local configuration files, or
        other secure stores. This method is called before any other
        interaction with the backend.

        Raises:
            BackendAuthenticationError: If the authentication process fails for
                                        any reason (e.g., invalid API key,
                                        network error).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_job(self, circuit_representation: str, shots: int) -> str:
        """
        Submits a quantum circuit to the backend for execution.

        The circuit is provided in a standardized string format (e.g., OpenQASM 3.0)
        and is sent to the provider's API endpoint for processing.

        Args:
            circuit_representation (str): A string representing the quantum
                                          circuit (e.g., in OpenQASM 3.0 format).
            shots (int): The number of times to execute the circuit to gather
                         measurement statistics.

        Returns:
            str: A unique job identifier (job_id) returned by the backend,
                 which can be used to track the job's status and retrieve
                 its results.

        Raises:
            JobSubmissionError: If the job submission fails (e.g., due to an
                                invalid circuit, insufficient credits, or an
                                API error).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_job_status(self, job_id: str) -> str:
        """
        Checks the status of a previously submitted job.

        This method polls the backend to determine the current state of the
        job, such as whether it is queued, running, or completed.

        Args:
            job_id (str): The unique identifier of the job to check.

        Returns:
            str: A string indicating the job's current status. Common values
                 include 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_job_result(self, job_id: str) -> Dict[str, int]:
        """
        Retrieves the final measurement counts for a completed job.

        This method should only be called after `get_job_status` indicates
        that the job is 'COMPLETED'.

        Args:
            job_id (str): The unique identifier of the completed job.

        Returns:
            Dict[str, int]: A dictionary where keys are the measurement outcomes
                            (bitstrings, e.g., '0101') and values are the
                            corresponding counts (number of times that outcome
                            was observed). Example: {'01': 512, '10': 488}.

        Raises:
            ResultRetrievalError: If the results cannot be fetched, for instance,
                                  if the job has not completed, the job_id is
                                  invalid, or an API error occurs.
        """
        raise NotImplementedError
