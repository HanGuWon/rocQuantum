import numpy as np
import rocq
# Import the low-level backend directly to access TensorNetwork features
from rocq import _rocq_hip_backend as backend

def run_network_test(dtype):
    """
    Runs a tensor network contraction test for a specific data type.
    """
    print(f"\n--- Running Tensor Network Test for dtype: {dtype.__name__} ---")

    # 1. Define Tensors (metadata)
    # Network: T0(a,b), T1(b,c), T2(c,a) -> scalar
    print("Defining a simple tensor network: T0(a,b) * T1(b,c) * T2(c,a)")
    
    # Create RocTensor metadata objects.
    # In a full implementation, these would be associated with device data.
    tensor0 = backend.RocTensor([2, 2])
    tensor0.labels = ["a", "b"]

    tensor1 = backend.RocTensor([2, 2])
    tensor1.labels = ["b", "c"]

    tensor2 = backend.RocTensor([2, 2])
    tensor2.labels = ["c", "a"]

    # This will be the final result (a scalar)
    result_tensor = backend.RocTensor([])

    # 2. Create and populate the Tensor Network
    try:
        # The RocsvHandle is needed for stream/resource management in a full implementation
        simulator = rocq.Simulator()

        # **MODIFIED**: Create the RocTensorNetwork handle using the new API.
        # The data type is passed via the 'dtype_source' argument.
        print(f"Creating RocTensorNetwork with dtype: {dtype.__name__}")
        tn_handle = backend.RocTensorNetwork(simulator.handle, dtype_source=dtype)
        
        # **MODIFIED**: Use the add_tensor method on the handle object.
        tn_handle.add_tensor(tensor0)
        tn_handle.add_tensor(tensor1)
        tn_handle.add_tensor(tensor2)

        # 3. Contract the network
        # The C++ side currently only performs pathfinding and returns NOT_IMPLEMENTED.
        # We expect a printout and a warning, not a crash.
        optimizer_config = {
            "memory_limit": 1024 * 1024 * 512  # 512 MB
        }
        print(f"Attempting contraction with optimizer config: {optimizer_config}")
        
        tn_handle.contract(optimizer_config, result_tensor)

        print(f"Pathfinding simulation for {dtype.__name__} completed.")
        print("Check console output for C++-side messages.")

    except Exception as e:
        print(f"An unexpected error occurred during the test for {dtype.__name__}: {e}")
        print("Please ensure the rocQuantum project is built and installed correctly.")


def run_all_examples():
    """
    Demonstrates the use of the tensor network interface with various data types.
    """
    print("======================================================")
    print("= hipTensorNet Multi-Data Type Example Demonstration =")
    print("======================================================")
    
    # Run the same network contraction with different data types
    # The C++ backend will be instantiated with the correct template type.
    run_network_test(np.float32)
    run_network_test(np.float64)
    run_network_test(np.complex64)
    run_network_test(np.complex128)

    print("\nAll tensor network tests completed.")


if __name__ == "__main__":
    run_all_examples()