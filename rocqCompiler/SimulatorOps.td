#ifndef SIMULATOR_OPS_TD
#define SIMULATOR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Simulator_Dialect : Dialect {
  let name = "sim";
  let cppNamespace = "rocq::mlir::sim";
  let summary = "A dialect for targeting a quantum simulator backend.";
}

// Base class for Simulator dialect operations
class Sim_Op<string mnemonic, list<Trait> traits = []> :
    Op<Simulator_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def ApplyGateOp : Sim_Op<"apply_gate", []> {
  let summary = "An operation representing a generic gate call to the backend.";
  let arguments = (ins StringAttr:$gate_name, Variadic<OpaqueType<"quantum", "qubit">>:$targets);
}

def ApplyParamGateOp : Sim_Op<"apply_param_gate", []> {
  let summary = "An operation for a parametrized gate call.";
  let arguments = (ins StringAttr:$gate_name, F64Attr:$parameter, Variadic<OpaqueType<"quantum", "qubit">>:$targets);
}

#endif // SIMULATOR_OPS_TD
