# tests/test_ionq_backend.py

"""
End-to-End test script for the IonQ backend integration.

This script verifies the complete workflow:
1. Setting the IonQ backend as the target.
2. Authenticating with the IonQ API.
3. Submitting a quantum job (a simple Bell State circuit).
4. Polling for the job's status until completion.
5. Retrieving and displaying the final results.

!! IMPORTANT !!
To run this test successfully, you MUST set the `IONQ_API_KEY` environment
variable to your personal IonQ API key before executing the script.

Example:
- On Linux/macOS: export IONQ_API_KEY="your_api_key_here"
- On Windows:    set IONQ_API_KEY="your_api_key_here"
"""

import sys
import time
import os

# Add the project root to the Python path to allow for direct imports
# of the rocquantum modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rocquantum.core import set_target, get_active_backend
from rocquantum.backends.base import \
    BackendAuthenticationError,
    JobSubmissionError,
    ResultRetrievalError

# A simple Bell State circuit in OpenQASM 3.0 format.
# This circuit creates a maximally entangled state between two qubits.
BELL_STATE_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

def run_ionq_test():
    """Executes the full end-to-end test for the IonQ backend."""
    job_id = None
    try:
        # 1. Set the target backend. We use the IonQ simulator to avoid
        #    consuming QPU credits during testing.
        print("--> Attempting to set 'ionq' backend (using simulator)...")
        set_target('ionq', backend_name='simulator')
        print("--> Backend set and authenticated successfully.")

        # 2. Get the active backend and submit the job.
        backend = get_active_backend()
        print("\n--> Submitting Bell State circuit job...")
        job_id = backend.submit_job(
            circuit_representation=BELL_STATE_QASM,
            shots=100
        )
        print(f"--> Job submitted successfully. Job ID: {job_id}")

        # 3. Poll for job status.
        print("\n--> Polling for job status...")
        while True:
            status = backend.get_job_status(job_id)
            print(f"    Current status: '{status}'")
            if status in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(5)  # Wait 5 seconds between checks

        # 4. Retrieve and print the results if the job completed.
        if status == 'completed':
            print("\n--> Job completed. Retrieving results...")
            results = backend.get_job_result(job_id)
            print("--> Final Histogram:")
            print(results)
            # Expected output for a Bell state is roughly 50% '00' and 50% '11'.
            # IonQ's simulator represents this as {'0': 50, '3': 50}.
        else:
            print(f"\n--> Job finished with status: '{status}'. No results to display.")

    except (BackendAuthenticationError, JobSubmissionError, ResultRetrievalError) as e:
        print(f"\n[ERROR] An error occurred during the rocQuantum workflow: {e}")
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    if not os.getenv("IONQ_API_KEY"):
        print("\n[FATAL] The 'IONQ_API_KEY' environment variable is not set.")
        print("Please set it to your IonQ API key to run this test.")
    else:
        run_ionq_test()
