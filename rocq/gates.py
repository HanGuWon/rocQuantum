# Defines the gate operations available in the rocq programming model.
# When a kernel is being "recorded", these functions do not execute;
# they merely register themselves and their arguments in the kernel's context.

from .kernel import _KernelBuildContext

# Gate functions
def h(target):
    _KernelBuildContext.add_gate("h", [target])

def x(target):
    _KernelBuildContext.add_gate("x", [target])

def y(target):
    _KernelBuildContext.add_gate("y", [target])

def z(target):
    _KernelBuildContext.add_gate("z", [target])

def ry(angle, target):
    _KernelBuildContext.add_gate("ry", [target], params={"theta": angle})

def rz(angle, target):
    _KernelBuildContext.add_gate("rz", [target], params={"phi": angle})

def cnot(control, target):
    _KernelBuildContext.add_gate("cnot", [control, target])
