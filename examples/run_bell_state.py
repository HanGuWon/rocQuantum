# examples/run_bell_state.py

"""
"Hello, Quantum World!" - A rocQuantum Showcase

This script demonstrates the core functionality and vision of the rocQuantum
framework: to define a quantum circuit once and execute it on vastly different
backend architectures with minimal code changes.
"""

import os
import sys
import time

# Add the project root to the Python path to allow for direct imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rocquantum.circuit import QuantumCircuit
from rocquantum.core import set_target, get_active_backend

def main():
    """Defines and runs a Bell State circuit on multiple backends."""

    # 1. Define the Quantum Circuit Programmatically
    # Using the QuantumCircuit class, we can build our circuit without
    # writing any raw QASM. This provides a clean, Pythonic interface.
    print("--> Step 1: Building the quantum circuit for a Bell State...")
    bell_circuit = QuantumCircuit(num_qubits=2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    # Measurement is added automatically by the to_qasm() method if not present.
    print("--> Circuit built successfully.")
    print(f"    QASM representation:\n{bell_circuit.to_qasm()}\n")

    # --- Execution on a Type A Backend (Remote API) ---
    print("--- Running on a Type A Backend (IonQ Simulator) ---")
    try:
        # 2. Set the Target Backend
        # This single line of code is all that's needed to switch hardware.
        # Here, we target the IonQ simulator.
        # NOTE: This requires the IONQ_API_KEY environment variable to be set.
        if not os.getenv("IONQ_API_KEY"):
            raise EnvironmentError("IONQ_API_KEY not set. Skipping IonQ execution.")
        
        set_target('ionq', backend_name='simulator')
        backend = get_active_backend()

        # 3. Submit the Job
        # For a Type A backend, we submit the QASM string representation.
        print("--> Submitting job to IonQ simulator...")
        job_id = backend.submit_job(bell_circuit.to_qasm(), shots=100)
        print(f"--> Job submitted. ID: {job_id}")

        # 4. Retrieve Results
        print("--> Polling for results...")
        while True:
            status = backend.get_job_status(job_id)
            print(f"    Job status: {status}")
            if status == 'completed':
                results = backend.get_job_result(job_id)
                print(f"--> Results received: {results}\n")
                break
            elif status in ['failed', 'cancelled']:
                print("--> Job did not complete successfully.\n")
                break
            time.sleep(2)

    except Exception as e:
        print(f"[ERROR] Could not run on IonQ backend: {e}\n")

    # --- Execution on a Type B Backend (Local SDK) ---
    print("--- Running on a Type B Backend (Qristal Simulator) ---")
    try:
        # 2. Set the Target Backend
        # Now, we switch to a completely different architecture.
        set_target('qristal')
        backend = get_active_backend()

        # 3. Submit the Job
        # For this Type B backend, we submit the QuantumCircuit object directly.
        # The job runs synchronously.
        print("--> Submitting job to local Qristal simulator...")
        job_id = backend.submit_job(bell_circuit, shots=100)
        
        # 4. Retrieve Results
        # Since it's a synchronous run, we can get results immediately.
        status = backend.get_job_status(job_id)
        print(f"    Job status: {status}")
        if status == 'completed':
            results = backend.get_job_result(job_id)
            print(f"--> Results received: {results}\n")

    except Exception as e:
        print(f"[ERROR] Could not run on Qristal backend: {e}\n")

    # --- Execution on a Type C Backend (Cloud Intermediary SDK) ---
    print("--- Running on a Type C Backend (Rigetti via AWS Braket) ---")
    try:
        # 2. Set the Target Backend
        # This demonstrates using a cloud provider's SDK.
        # NOTE: Requires AWS credentials (e.g., AWS_ACCESS_KEY_ID) to be set.
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            raise EnvironmentError("AWS_ACCESS_KEY_ID not set. Skipping Rigetti execution.")

        set_target('rigetti')
        backend = get_active_backend()

        # 3. Submit the Job
        # Like Type A, this backend expects a QASM string.
        print("--> Submitting job to Rigetti on AWS Braket...")
        # NOTE: This will fail if your S3 bucket location in rigetti.py is not configured.
        # This is a demonstration of the submission call.
        job_id = backend.submit_job(bell_circuit.to_qasm(), shots=100)
        print(f"--> Job submitted. AWS Braket Task ARN: {job_id}")
        print("--> NOTE: AWS Braket is asynchronous. You would typically poll for results.")
        print("    (Result polling not shown in this example for brevity.)\n")

    except Exception as e:
        print(f"[ERROR] Could not run on Rigetti backend: {e}\n")


if __name__ == "__main__":
    main()