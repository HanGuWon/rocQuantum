#ifndef ROCQUANTUM_QUANTUM_OPS_TD
#define ROCQUANTUM_QUANTUM_OPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// Base class for Quantum dialect operations
class Quantum_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quantum_Dialect, mnemonic, traits>;

// --- Custom Types ---
def QubitType : DialectType<Quantum_Dialect, "qubit"> {
  let summary = "A quantum bit (qubit) type.";
  let description = [{
    Represents a single qubit. Operations like allocation, deallocation,
    and gates operate on values of this type.
  }];
  let cppNamespace = "::rocquantum::quantum";
}


// --- Operations ---

def AllocQubitOp : Quantum_Op<"alloc_qubit"> {
  let summary = "Allocate a new qubit.";
  let description = [{
    Allocates a single qubit, which is initially in the |0> state.
    The qubit must be deallocated eventually using `quantum.dealloc_qubit`.
  }];

  let results = (outs QubitType:$qubit);

  // Side effects: Memory allocation
  let  effects = [DeclareResourceEffect<Resource@"Qubit">];

  let assemblyFormat = "attr-dict";
}

def DeallocQubitOp : Quantum_Op<"dealloc_qubit"> {
  let summary = "Deallocate a qubit.";
  let description = [{
    Deallocates a qubit previously allocated by `quantum.alloc_qubit`.
    The qubit should typically be in the |0> state before deallocation,
    though this might not be strictly enforced by all backends.
  }];

  let arguments = (ins QubitType:$qubit);

  // Side effects: Memory deallocation
  let  effects = [FreeResourceEffect<Resource@"Qubit">];

  let assemblyFormat = "$qubit attr-dict";
}

def GenericGateOp : Quantum_Op<"gate", [AttrSizedOperandSegments]> {
  let summary = "Apply a generic quantum gate.";
  let description = [{
    Applies a named quantum gate to a list of qubit operands.
    The specific unitary matrix corresponding to the gate name is defined
    by the backend or a gate library.
    This op can represent single-qubit or multi-qubit gates.
    An 'is_adjoint' attribute can be set to true to apply the adjoint of the gate.
  }];

  let arguments = (ins StrAttr:$gate_name, Variadic<QubitType>:$qubits);
  // For parameterized gates, add: Variadic<F64Type>:$params

  // Add an optional boolean attribute to the op itself.
  let optionalAttributes = (ins BoolAttr:$is_adjoint);

  // Results could be the same qubits if ops are defined to return their inputs,
  // or void if gates are purely side-effecting on the state vector.
  // For now, assume side-effecting.
  // let results = (outs Variadic<QubitType>:$result_qubits);

  let assemblyFormat = "$gate_name `(` $qubits `)` attr-dict";
  // Example with params: $gate_name `(` $params `)` `on` $qubits attr-dict
}

def MeasureOp : Quantum_Op<"measure"> {
  let summary = "Measure a qubit in the computational basis.";
  let description = [{
    Measures a single qubit in the Z (computational) basis.
    Returns a classical bit (i1) representing the outcome (0 or 1).
    This operation may collapse the quantum state associated with the qubit.
  }];

  let arguments = (ins QubitType:$qubit);
  let results = (outs I1Type:$result); // Classical measurement outcome (0 or 1)

  // Side effects: May collapse state, interacts with classical world
  let effects = [SideEffect<MemoryEffects::Read, Resource@"QuantumState">,
                 SideEffect<MemoryEffects::Write, Resource@"QuantumState">];


  let assemblyFormat = "$qubit attr-dict `->` type($result)";
}

def IfOp : Quantum_Op<"if", [SingleBlockImplicitTerminator<"YieldOp">, NoSideEffect]> {
    let summary = "Conditional execution operation";
    let description = [{
        The `if` operation represents an if-then style conditional. It has a
        single region for the 'then' body. The operation takes a condition value,
        which must be a 1-bit integer (i1). If the condition is true (1),
        the 'then' region is executed. There is no 'else' body.
    }];

    let arguments = (ins I1Type:$condition);

    let regions = (region SizedRegion<1>:$thenRegion);

    let assemblyFormat = "$condition `{` $thenRegion `}` attr-dict";
}

def YieldOp : Quantum_Op<"yield", [Terminator]> {
    let summary = "Yield operation for quantum dialect regions";
    let description = [{
        `quantum.yield` is a special terminator operation for blocks inside other
        quantum operations. For example, it terminates the block within a
        `quantum.if` operation.
    }];

    let arguments = (ins);
    let assemblyFormat = "attr-dict";
}

#endif // ROCQUANTUM_QUANTUM_OPS_TD