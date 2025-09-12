# tests/test_quantinuum_backend.py

"""
Pytest-based end-to-end test for the Quantinuum backend integration.

This test verifies the full user workflow against the Quantinuum API,
ensuring that the refactored QuantinuumBackend class functions correctly.

!! IMPORTANT !!
This test will be automatically skipped if the `CUDAQ_QUANTINUUM_CREDENTIALS`
environment variable is not set to the path of a valid credentials file.
"""

import os
import sys
import time
import pytest

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rocquantum.core import set_target, get_active_backend

# --- Test Setup ---

# A simple Bell State circuit in OpenQASM 3.0 format.
BELL_STATE_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

# Condition to skip the test if credentials are not configured.
CREDENTIALS_NOT_CONFIGURED = not os.getenv("CUDAQ_QUANTINUUM_CREDENTIALS")
SKIP_REASON = "CUDAQ_QUANTINUUM_CREDENTIALS environment variable not set."

# --- Test Function ---

@pytest.mark.skipif(CREDENTIALS_NOT_CONFIGURED, reason=SKIP_REASON)
def test_quantinuum_full_workflow():
    """
    Tests the full workflow: set_target -> submit_job -> poll status -> get_result.
    """
    print("--> Attempting to set 'quantinuum' backend...")
    # 1. Set the target backend.
    set_target('quantinuum')
    print("--> Backend set and authenticated successfully.")

    # 2. Get the active backend and submit the job.
    backend = get_active_backend()
    shots = 100
    print(f"--> Submitting Bell State circuit job with {shots} shots...")
    job_id = backend.submit_job(
        circuit_representation=BELL_STATE_QASM,
        shots=shots
    )
    print(f"--> Job submitted successfully. Job ID: {job_id}")
    assert job_id is not None

    # 3. Poll for job status.
    print("--> Polling for job status...")
    while True:
        status = backend.get_job_status(job_id)
        print(f"    Current status: '{status}'")
        if status in ['completed', 'failed', 'cancelled']:
            break
        time.sleep(5)
    
    assert status == 'completed'

    # 4. Retrieve and validate the results.
    print("--> Job completed. Retrieving results...")
    results = backend.get_job_result(job_id)
    print(f"--> Final Histogram: {results}")
    
    assert isinstance(results, dict)
    assert sum(results.values()) == shots