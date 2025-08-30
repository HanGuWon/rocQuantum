import rocq
import numpy as np

# Create a simulator
sim = rocq.Simulator()

# Create a tensor network
tn = rocq.TensorNetwork(simulator=sim)

# Create tensors
tensor_a = np.random.rand(2, 2).astype(np.complex64)
tensor_b = np.random.rand(2, 2).astype(np.complex64)

# Add tensors to the network
tn.add_tensor(tensor_a, ["a", "b"])
tn.add_tensor(tensor_b, ["b", "c"])

# Contract the network
result = tn.contract()

# Print the result
print(result)
