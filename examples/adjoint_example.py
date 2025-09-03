import numpy as np
# Assuming rocq.api is in the python path
import rocq.api as rocq

def run_adjoint_example():
    """
    Demonstrates creating a kernel and generating its adjoint via the compiler.
    """
    print("=======================================")
    print("= Adjoint Generation Example          =")
    print("=======================================")

    # 1. Define a simple quantum kernel as an MLIR function string.
    # This kernel applies S gate, then H gate to qubit 0.
    # The adjoint should be H, then S-dagger.
    kernel_name = "sh_kernel"
    # Note: This is just the function, not a full module.
    # The rocq.adjoint function will wrap it in a module for compilation.
    mlir_func_string = f"""
  func.func @{kernel_name}() {{
    %q0 = quantum.alloc_qubit
    quantum.gate "s" (%q0)
    quantum.gate "h" (%q0)
    quantum.dealloc_qubit %q0
    return
  }}
"""
    
    print("\nOriginal Kernel MLIR Function:")
    print(mlir_func_string)

    # Create a Kernel object
    sh_kernel = rocq.Kernel(name=kernel_name, mlir_string=mlir_func_string)

    # 2. Generate the adjoint of the kernel by calling the Python API.
    # This invokes the C++ compiler, runs the AdjointGenerationPass,
    # and returns a new kernel with the transformed MLIR.
    print("\nCalling rocq.adjoint() to generate the adjoint kernel...")
    
    try:
        adjoint_kernel_module = rocq.adjoint(sh_kernel)
        
        print("\nSUCCESS: Adjoint kernel module generated.")
        print("The MLIR module below now contains both the original and the new .adj function.")
        print("Note the reversed gate order and the 'is_adjoint' attribute in the .adj function.")
        print("----------------------------------------------------")
        print(adjoint_kernel_module.mlir_string)
        print("----------------------------------------------------")

    except Exception as e:
        print(f"An error occurred during adjoint generation: {e}")
        print("This may be due to a missing build step for the C++ compiler and bindings.")


if __name__ == "__main__":
    run_adjoint_example()
