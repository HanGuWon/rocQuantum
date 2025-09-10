import rocq

# 1. Define a simple quantum kernel
@rocq.kernel
def simple_kernel():
    q = rocq.qvec(2)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])

# 2. Instantiate the kernel
kernel_instance = simple_kernel()

# 3. Compile the kernel to QIR and print the result
qir_output = kernel_instance.qir()

print("\n--- Generated QIR (LLVM IR) ---")
print(qir_output)
print("---------------------------------")

# --- Verification ---
print("\n--- Verifying QIR Output ---")
if "__quantum__qis__h__body" in qir_output and \
   "__quantum__qis__cnot__body" in qir_output and \
   "call void @__quantum__qis__h__body" in qir_output:
    print("Verification PASSED: QIR contains the correct function calls.")
else:
    print("Verification FAILED: QIR output is missing expected content.")
