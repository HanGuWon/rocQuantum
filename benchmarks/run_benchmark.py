import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Setup Python Path ---
def setup_paths():
    """Adds the necessary project directories to the Python path."""
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        integrations_path = os.path.join(project_root, '..', 'integrations')
        build_path = os.path.join(project_root, '..', 'build')
        sys.path.insert(0, os.path.abspath(integrations_path))
        sys.path.insert(0, os.path.abspath(build_path))
        print("Project paths successfully added.")
    except Exception as e:
        print(f"Error setting up paths: {e}")
        sys.exit(1)

setup_paths()

# --- Import Frameworks ---
try:
    import pennylane as qml
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_rocquantum_provider.provider import RocQuantumProvider
    print("Successfully imported all frameworks and providers.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all dependencies are installed (`pip install pennylane qiskit qiskit-aer matplotlib`).")
    sys.exit(1)

# --- Benchmark Configuration ---
QUBIT_RANGE = range(10, 22, 2)
NUM_TRIALS = 5

# --- Circuit Generation ---
def generate_pennylane_qft(num_qubits):
    """Creates a PennyLane QNode for the QFT circuit."""
    def qft_rotations(wires):
        for i in range(len(wires)):
            for j in range(i):
                qml.CRZ(np.pi / 2**(i - j), wires=[wires[j], wires[i]])
    
    def swap_qubits(wires):
        for i in range(len(wires) // 2):
            qml.SWAP(wires=[wires[i], wires[len(wires) - 1 - i]])

    @qml.qnode(qml.device('default.qubit', wires=num_qubits))
    def circuit():
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        qft_rotations(wires=range(num_qubits))
        swap_qubits(wires=range(num_qubits))
        return qml.state()
    return circuit

def generate_qiskit_qft(num_qubits):
    """Creates a Qiskit QuantumCircuit for the QFT."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            qc.cp(np.pi / 2**(j - i), j, i)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    return qc

# --- Benchmarking Functions ---
def run_pennylane_benchmark():
    print("\n" + "="*40)
    print(" PennyLane Performance Benchmark: QFT ")
    print("="*40)
    
    results = {"qubits": [], "rocq_time": [], "cpu_time": []}
    
    for n_qubits in QUBIT_RANGE:
        print(f"\nRunning benchmark for {n_qubits} qubits...")
        
        # 1. rocQuantum Device
        rocq_device = qml.device('rocq.pennylane', wires=n_qubits)
        rocq_circuit = generate_pennylane_qft(n_qubits)
        rocq_circuit.device = rocq_device
        
        times = []
        for _ in range(NUM_TRIALS):
            start_time = time.perf_counter()
            rocq_circuit()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        rocq_avg_time = np.mean(times)
        
        # 2. Default CPU Device
        cpu_device = qml.device('default.qubit', wires=n_qubits)
        cpu_circuit = generate_pennylane_qft(n_qubits)
        cpu_circuit.device = cpu_device
        
        times = []
        for _ in range(NUM_TRIALS):
            start_time = time.perf_counter()
            cpu_circuit()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        cpu_avg_time = np.mean(times)
        
        print(f"  Qubits: {n_qubits:2} | rocQuantum: {rocq_avg_time:7.3f}s | CPU (default.qubit): {cpu_avg_time:7.3f}s")
        results["qubits"].append(n_qubits)
        results["rocq_time"].append(rocq_avg_time)
        results["cpu_time"].append(cpu_avg_time)
        
    return results

def run_qiskit_benchmark():
    print("\n" + "="*40)
    print(" Qiskit Performance Benchmark: QFT ")
    print("="*40)
    
    results = {"qubits": [], "rocq_time": [], "cpu_time": []}
    
    rocq_backend = RocQuantumProvider().get_backend("rocq_simulator")
    cpu_backend = AerSimulator()
    
    for n_qubits in QUBIT_RANGE:
        print(f"\nRunning benchmark for {n_qubits} qubits...")
        circuit = generate_qiskit_qft(n_qubits)
        
        # 1. rocQuantum Backend
        t_qc_rocq = transpile(circuit, rocq_backend)
        times = []
        for _ in range(NUM_TRIALS):
            start_time = time.perf_counter()
            rocq_backend.run(t_qc_rocq, shots=1).result() # Using shots=1 for statevector-like simulation
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        rocq_avg_time = np.mean(times)
        
        # 2. Aer CPU Backend
        t_qc_cpu = transpile(circuit, cpu_backend)
        times = []
        for _ in range(NUM_TRIALS):
            start_time = time.perf_counter()
            cpu_backend.run(t_qc_cpu, shots=1).result()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        cpu_avg_time = np.mean(times)
        
        print(f"  Qubits: {n_qubits:2} | rocQuantum: {rocq_avg_time:7.3f}s | CPU (AerSimulator): {cpu_avg_time:7.3f}s")
        results["qubits"].append(n_qubits)
        results["rocq_time"].append(rocq_avg_time)
        results["cpu_time"].append(cpu_avg_time)
        
    return results

# --- Plotting Function ---
def plot_results(results, framework_name):
    """Generates and saves a plot of the benchmark results."""
    plt.figure(figsize=(10, 6))
    plt.plot(results["qubits"], results["rocq_time"], 'o-', label='rocQuantum-1 Simulator (GPU)')
    plt.plot(results["qubits"], results["cpu_time"], 's-', label=f'Default {framework_name} CPU Simulator')
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Execution Time (seconds)')
    plt.yscale('log')
    plt.title(f'{framework_name} QFT Benchmark: rocQuantum-1 vs. CPU')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    filename = f'benchmarks/benchmark_results_{framework_name.lower()}.png'
    plt.savefig(filename)
    print(f"\nBenchmark plot saved to '{filename}'")

# --- Main Execution ---
if __name__ == "__main__":
    pennylane_results = run_pennylane_benchmark()
    plot_results(pennylane_results, "PennyLane")
    
    qiskit_results = run_qiskit_benchmark()
    plot_results(qiskit_results, "Qiskit")
    
    print("\nBenchmark run complete.")
