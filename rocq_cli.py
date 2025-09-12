# rocq_cli.py

"""
rocQuantum Command-Line Interface (CLI)

A simple tool for demonstrating the rocQuantum framework's capabilities
by running a standard Bell State circuit on a specified backend.
"""

import os
import argparse
import sys
import time

from rocquantum.circuit import QuantumCircuit
from rocquantum.core import set_target, get_active_backend
from rocquantum.backends.base import \
    BackendAuthenticationError,
    JobSubmissionError,
    ResultRetrievalError

def create_bell_circuit() -> QuantumCircuit:
    """Creates a standard 2-qubit Bell State circuit."""
    qc = QuantumCircuit(num_qubits=2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def check_environment_vars(backend_name: str):
    """Checks for required environment variables for a given backend."""
    if backend_name == 'ionq' and not os.getenv('IONQ_API_KEY'):
        raise EnvironmentError("Error: IONQ_API_KEY environment variable not set.")
    if backend_name == 'quantinuum' and not os.getenv('CUDAQ_QUANTINUUM_CREDENTIALS'):
        raise EnvironmentError("Error: CUDAQ_QUANTINUUM_CREDENTIALS environment variable not set.")
    if backend_name == 'rigetti' and not os.getenv('AWS_ACCESS_KEY_ID'):
        raise EnvironmentError("Error: AWS credentials (e.g., AWS_ACCESS_KEY_ID) not set.")
    # Add other backend checks here as needed

def main():
    """Parses arguments and runs the quantum circuit."""
    parser = argparse.ArgumentParser(
        description="Run a Bell State circuit on a rocQuantum backend."
    )
    parser.add_argument(
        "command",
        choices=["run"],
        help="The command to execute.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="The name of the backend to run on (e.g., 'ionq', 'qristal', 'rigetti').",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help="The number of shots to execute.",
    )
    args = parser.parse_args()

    if args.command == "run":
        try:
            print(f"--> Preparing to run on backend: {args.backend}")
            
            # 1. Check for credentials
            check_environment_vars(args.backend)
            
            # 2. Create circuit
            bell_circuit = create_bell_circuit()
            print("--> Bell State circuit created.")

            # 3. Set target and get backend
            if args.backend == 'ionq':
                set_target(args.backend, backend_name='simulator')
            else:
                set_target(args.backend)
            backend = get_active_backend()
            print(f"--> Target set to '{backend.backend_name}'.")

            # 4. Submit job
            print(f"--> Submitting job with {args.shots} shots...")
            
            # Handle different submission requirements
            if args.backend in ['ionq', 'rigetti', 'pasqal', 'infleqtion']: # Type A/C
                job_id = backend.submit_job(bell_circuit.to_qasm(), shots=args.shots)
            else: # Type B (local SDK)
                job_id = backend.submit_job(bell_circuit, shots=args.shots)
            
            print(f"--> Job submitted successfully. Job ID: {job_id}")

            # 5. Poll for and display results
            print("--> Waiting for results...")
            while True:
                status = backend.get_job_status(job_id)
                print(f"    Current job status: {status}")
                if status == 'completed':
                    results = backend.get_job_result(job_id)
                    print("\n--> Execution complete!")
                    print(f"    Results: {results}")
                    break
                elif status in ['failed', 'cancelled']:
                    print(f"\n--> Job {status}. Cannot retrieve results.")
                    break
                # For synchronous backends like Qristal, this loop runs once.
                if status == 'completed':
                    break
                time.sleep(2)

        except (EnvironmentError, BackendAuthenticationError, JobSubmissionError, ResultRetrievalError, ImportError) as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
