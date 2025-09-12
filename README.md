# rocQuantum Version 1.0

A hardware-agnostic quantum computing framework, inspired by the architecture of NVIDIA's CUDA-Q, designed to provide a unified interface for executing quantum circuits across a wide range of third-party quantum processing units (QPUs).

## Description

The core goal of rocQuantum is to offer a seamless user experience where a quantum circuit can be defined once and then executed on different hardware backends—from remote cloud platforms to local simulators—with minimal changes to the code. This is achieved through a robust backend abstraction layer that supports multiple integration models:

*   **Type A (Remote API):** For API-based services (e.g., IonQ, Pasqal), rocQuantum manages the HTTP client, authentication, and job lifecycle internally.
*   **Type B (Local SDK):** For providers that require a local SDK (e.g., Quantum Brilliance's Qristal), rocQuantum interfaces directly with the provider's tools for synchronous execution.
*   **Type C (Cloud Intermediary):** For platforms accessed via a cloud provider's SDK (e.g., Rigetti on AWS Braket), rocQuantum handles the interaction with the intermediary service.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd rocQuantum-1
    ```

2.  Install the required Python packages:
    ```bash
    pip install requests boto3
    ```
    *(Note: Specific provider SDKs like `qiskit` or `cirq` should be installed as needed).*

## Configuration

To access third-party backends, you must configure your credentials using environment variables.

*   **For IonQ:**
    ```bash
    export IONQ_API_KEY="YOUR_IONQ_API_KEY"
    ```

*   **For Quantinuum:**
    ```bash
    export CUDAQ_QUANTINUUM_CREDENTIALS="YOUR_USERNAME,YOUR_PASSWORD"
    ```

*   **For Rigetti (via AWS Braket):**
    The Rigetti backend uses the `boto3` library, which automatically finds AWS credentials. Configure them using standard AWS methods, such as:
    *   Environment Variables:
        ```bash
        export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
        export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
        export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN" # (Optional)
        ```
    *   Shared Credentials File (`~/.aws/credentials`)

*   **For Pasqal & Infleqtion:**
    Set `PASQAL_API_KEY` and `SUPERSTAQ_API_KEY` respectively.

## CLI Usage

A simple Command-Line Interface is provided for quick demonstrations. It runs a standard Bell State circuit on the specified backend.

**Example:**

```bash
python rocq_cli.py run --backend ionq --shots 100
```

This command will:
1.  Check for the `IONQ_API_KEY`.
2.  Target the IonQ simulator.
3.  Submit a Bell State circuit for 100 shots.
4.  Poll for and print the final measurement results.

## Supported Backends

The framework is designed for extensibility. The following backends are currently integrated:

| Provider           | Backend Name | Status      | Type              |
| ------------------ | ------------ | ----------- | ----------------- |
| **IonQ**           | `ionq`       | Implemented | A (Remote API)    |
| **Quantinuum**     | `quantinuum` | Implemented | A (Remote API)    |
| **Pasqal**         | `pasqal`     | Implemented | A (Remote API)    |
| **Infleqtion**     | `infleqtion` | Implemented | A (Remote API)    |
| **Quantum Brilliance** | `qristal`    | Implemented | B (Local SDK)     |
| **Rigetti**        | `rigetti`    | Implemented | C (Cloud Intermediary)|
| **Alice & Bob**    | `alice_bob`  | Skeleton    | -                 |
| **IQM**            | `iqm`        | Skeleton    | -                 |
| **ORCA Computing** | `orca`       | Skeleton    | -                 |
| **Quantum Machines**| `qm`         | Skeleton    | -                 |
| **QuEra**          | `quera`      | Skeleton    | -                 |
| **SEEQC**          | `seeqc`      | Skeleton    | -                 |
| **Xanadu**         | `xanadu`     | Skeleton    | -                 |

## Full Example

The script in `examples/run_bell_state.py` demonstrates the power of the framework by defining a single circuit and running it on three different backend types: IonQ (API), Qristal (local SDK), and Rigetti (cloud intermediary).
