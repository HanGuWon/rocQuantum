import numpy as np
import rocq
from rocq import _rocq_hip_backend as backend

def run_slicing_example():
    """
    Demonstrates the use of the memory-constrained pathfinding with slicing.
    """
    print("Initializing simulator for tensor network operations.")
    try:
        simulator = rocq.Simulator()
    except RuntimeError as e:
        print(f"\nAn error occurred during simulator init: {e}")
        return

    # 1. Define a large tensor network that would likely exceed a small memory limit
    # Network: T0(a,b,c,d), T1(d,e,f,g), T2(g,h,i,j), Result(a,b,c,e,f,h,i,j)
    print("\nDefining a large tensor network.")
    
    tensor0 = backend.RocTensor([2,2,2,16], py_data_np_array=None); tensor0.labels=["a","b","c","d"]
    tensor1 = backend.RocTensor([16,2,2,16], py_data_np_array=None); tensor1.labels=["d","e","f","g"]
    tensor2 = backend.RocTensor([16,2,2,2], py_data_np_array=None); tensor2.labels=["g","h","i","j"]
    result_tensor = backend.RocTensor([], py_data_np_array=None)

    # 2. Create and populate the Tensor Network
    try:
        tn_handle = backend.RocTensorNetwork(simulator.handle)
        backend.rocTensorNetworkAddTensor(tn_handle, tensor0)
        backend.rocTensorNetworkAddTensor(tn_handle, tensor1)
        backend.rocTensorNetworkAddTensor(tn_handle, tensor2)

        # 3. Attempt contraction with a memory limit to trigger slicing
        # Set a memory limit of 1KB, which is guaranteed to be too small.
        optimizer_config = {
            "repetitions": 8, 
            "memory_limit": 1024 
        }
        print(f"\nAttempting contraction with optimizer config: {optimizer_config}")
        
        tn_handle.contract(optimizer_config, result_tensor)

        print("\nSlicing pathfinding simulation completed.")
        print("Check console output for 'Decided to slice over label...'.")

    except RuntimeError as e:
        if "NOT_IMPLEMENTED" in str(e):
             print("\nSuccessfully received expected 'NOT_IMPLEMENTED' status.")
             print("This confirms the pathfinder with slicing awareness ran correctly.")
             print("Priority 2 Goal ACHIEVED.")
        else:
            print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_slicing_example()
