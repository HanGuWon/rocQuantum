#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "rocquantum/Dialect/QuantumOps.h" // Assumes this path, adjust if necessary
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "adjoint-generation"

using namespace mlir;
using namespace rocquantum::quantum;

namespace {
struct AdjointGenerationPass : public PassWrapper<AdjointGenerationPass, OperationPass<ModuleOp>> {
    void runOnOperation() override;
    StringRef getArgument() const final { return "adjoint-generation"; }
    StringRef getDescription() const final { return "Generate adjoint variants of quantum kernels."; }

private:
    // Generates the adjoint of a given function.
    func::FuncOp generateAdjoint(func::FuncOp originalFunc, OpBuilder& builder);
};
} // namespace

func::FuncOp AdjointGenerationPass::generateAdjoint(func::FuncOp originalFunc, OpBuilder& builder) {
    auto module = originalFunc->getParentOfType<ModuleOp>();
    std::string adjFuncName = originalFunc.getName().str() + ".adj";
    
    // Create the new adjoint function declaration
    builder.setInsertionPoint(originalFunc);
    auto adjFunc = builder.create<func::FuncOp>(originalFunc.getLoc(), adjFuncName, originalFunc.getFunctionType());
    adjFunc.addEntryBlock();
    
    IRMapping mapper;
    // Map the arguments of the original function to the arguments of the new adjoint function.
    for (unsigned i = 0; i < originalFunc.getNumArguments(); ++i) {
        mapper.map(originalFunc.getArgument(i), adjFunc.getArgument(i));
    }

    // Set the builder to the start of the new adjoint function's body
    builder.setInsertionPointToStart(&adjFunc.front());

    // Iterate the operations in the original function's body in reverse order
    for (auto& op : llvm::reverse(originalFunc.front().getOperations())) {
        if (auto genericGate = dyn_cast<GenericGateOp>(op)) {
            // Clone the gate and flip the 'is_adjoint' attribute
            auto newGate = builder.clone(op, mapper);
            bool isCurrentlyAdjoint = newGate->getAttr("is_adjoint").dyn_cast_or_null<BoolAttr>();
            if (isCurrentlyAdjoint && isCurrentlyAdjoint.getValue()) {
                // If it was adjoint, remove the attribute.
                newGate->removeAttr("is_adjoint");
            } else {
                // If it was not adjoint, add the attribute.
                newGate->setAttr("is_adjoint", builder.getBoolAttr(true));
            }
        } else {
            // For non-gate operations, just clone them.
            builder.clone(op, mapper);
        }
    }

    return adjFunc;
}

void AdjointGenerationPass::runOnOperation() {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    OpBuilder builder(&getContext());

    std::vector<func::CallOp> callsToProcess;
    module.walk([&](func::CallOp callOp) {
        if (callOp->hasAttr("rocq.adjoint.kernel")) {
            callsToProcess.push_back(callOp);
        }
    });

    for (auto callOp : callsToProcess) {
        // Get the name of the function being called
        StringRef calleeName = callOp.getCallee();
        auto originalFunc = dyn_cast_or_null<func::FuncOp>(symbolTable.lookup(calleeName));
        if (!originalFunc) {
            callOp.emitError() << "call to unknown function '" << calleeName << "'";
            continue;
        }

        // Define the adjoint function's name and check if it already exists
        std::string adjFuncName = std::string(calleeName) + ".adj";
        auto adjFunc = dyn_cast_or_null<func::FuncOp>(symbolTable.lookup(adjFuncName));

        if (!adjFunc) {
            // If it doesn't exist, generate it.
            adjFunc = generateAdjoint(originalFunc, builder);
            symbolTable.insert(adjFunc);
        }

        // Replace the original call with a call to the adjoint function
        builder.setInsertionPoint(callOp);
        auto newCall = builder.create<func::CallOp>(callOp.getLoc(), adjFunc, callOp.getOperands());
        
        // Clean up the old call
        callOp.replaceAllUsesWith(newCall.getResults());
        callOp.erase();
    }
}

// Factory function to create the pass
std::unique_ptr<Pass> createAdjointGenerationPass() {
    return std::make_unique<AdjointGenerationPass>();
}
