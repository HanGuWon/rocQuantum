# rocquantum/backends/qristal.py

"""
This module provides a Type B (Direct, Provider-Managed Integration) backend
for the Quantum Brilliance Qristal SDK.

This backend executes jobs locally by invoking the Qristal SDK's command-line
tool. It demonstrates a synchronous, file-based execution model, contrasting
with the API-based Type A backends.
"""

import subprocess
import tempfile
import uuid
import json
from typing import Dict, Any

from .base import RocqBackend, JobSubmissionError, ResultRetrievalError
from rocquantum.circuit import QuantumCircuit

class QuantumBrillianceBackend(RocqBackend):
    """
    A backend for executing circuits locally via the Qristal SDK.

    This class overrides the default job submission logic to run a local
    subprocess, making it a Type B backend.
    """

    def __init__(self, backend_name: str = "qristal", api_endpoint: str = "local"):
        """Initializes the Qristal local backend."""
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
        self._local_results: Dict[str, Dict] = {}

    def authenticate(self) -> None:
        """Authentication is not required for a local SDK."""
        print("Qristal is a local backend; no authentication needed.")
        pass

    def _get_auth_headers(self) -> Dict[str, str]:
        """Not applicable for a local backend."""
        return {}

    def _build_payload(self, circuit_representation: str, shots: int) -> Dict[str, Any]:
        """Not applicable for the direct execution model."""
        raise NotImplementedError("Payload building is not used for Type B backends.")

    def submit_job(self, circuit: QuantumCircuit, shots: int) -> str:
        """
        Executes a quantum circuit synchronously using the Qristal CLI.

        This method writes the circuit to a temporary QASM file and runs it
        using a subprocess.

        Args:
            circuit (QuantumCircuit): The QuantumCircuit object to execute.
            shots (int): The number of shots to run.

        Returns:
            str: A unique local job identifier.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise JobSubmissionError("Qristal backend requires a QuantumCircuit object, not a QASM string.")

        qasm_string = circuit.to_qasm()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as tmp_file:
            tmp_file.write(qasm_string)
            tmp_filepath = tmp_file.name

        command = ["qristal", "--run", tmp_filepath, "--shots", str(shots)]
        
        try:
            print(f"Executing command: {" ".join(command)}")
            # In a real scenario, this would execute the Qristal CLI.
            # For this simulation, we will mock the output.
            # result = subprocess.run(command, capture_output=True, text=True, check=True)
            # stdout = result.stdout
            
            # Mocked stdout for demonstration purposes
            mock_histogram = {'00': shots // 2, '11': shots // 2}
            stdout = f"Execution complete.\nHistogram: {json.dumps(mock_histogram)}"
            print(f"Mocked stdout: {stdout}")

        except FileNotFoundError:
            raise JobSubmissionError(
                "Job submission failed: 'qristal' command not found. "
                "Is the Qristal SDK installed and in your system's PATH?"
            )
        except subprocess.CalledProcessError as e:
            raise JobSubmissionError(
                f"Job execution failed with error: {e.stderr}"
            )

        job_id = f"local-run-{uuid.uuid4()}"
        self._local_results[job_id] = {"stdout": stdout}
        return job_id

    def get_job_status(self, job_id: str) -> str:
        """For a synchronous local run, the status is always 'completed'."""
        if job_id in self._local_results:
            return 'completed'
        raise ResultRetrievalError(f"Local job ID '{job_id}' not found.")

    def get_job_result(self, job_id: str) -> Dict[str, int]:
        """
        Parses the stored stdout from the subprocess to extract the histogram.
        """
        if job_id not in self._local_results:
            raise ResultRetrievalError(f"Local job ID '{job_id}' not found.")

        stdout = self._local_results[job_id]["stdout"]
        
        try:
            # Assume the output contains a line like "Histogram: {'00': 50, ...}"
            histogram_line = next(line for line in stdout.splitlines() if "Histogram:" in line)
            json_str = histogram_line.split("Histogram:")[1].strip()
            histogram = json.loads(json_str)
            return histogram
        except (StopIteration, json.JSONDecodeError, IndexError) as e:
            raise ResultRetrievalError(
                f"Failed to parse histogram from Qristal output. Error: {e}. "
                f"Full output:\n{stdout}"
            )
