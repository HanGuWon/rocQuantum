#ifndef QUANTUM_OPS_TD
#define QUANTUM_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Quantum_Dialect : Dialect {
  let name = "quantum";
  let cppNamespace = "quantum";
  let summary = "A dialect for quantum computing.";
}

// Base class for Quantum dialect operations
class Quantum_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quantum_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def QubitType : DialectType<Quantum_Dialect, "qubit", "a single qubit">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def QallocOp : Quantum_Op<"qalloc", []> {
  let summary = "Allocate a register of qubits";
  let arguments = (ins I64Attr:$size);
  let results = (outs Variadic<QubitType>:$qubits);
}

def HOp : Quantum_Op<"h", [NoSideEffect]>;
def XOp : Quantum_Op<"x", [NoSideEffect]>;
def YOp : Quantum_Op<"y", [NoSideEffect]>;
def ZOp : Quantum_Op<"z", [NoSideEffect]>;
def CnotOp : Quantum_Op<"cnot", [NoSideEffect]>;

// New conditional operation
def IfOp : Quantum_Op<"if", [RecursiveSideEffects]> {
    let summary = "A conditional operation with then and else regions.";
    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$thenRegion,
                   region SizedRegion<1>:$elseRegion);
}

#endif // QUANTUM_OPS_TD