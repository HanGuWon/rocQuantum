#include "MLIRCompiler.h"
#include <iostream>
#include <string>

// MLIR Core & Passes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// LLVM
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

// Our Dialects and Passes
#include "QuantumOps.h.inc"
#include "SimulatorOps.h.inc"

// Forward declarations for our custom pass creation functions
std::unique_ptr<mlir::Pass> createQuantumToSimulatorPass();
std::unique_ptr<mlir::Pass> createSimulatorToQIRPass();

namespace rocq {

struct MLIRCompiler::Impl {
    mlir::MLIRContext context;
};

MLIRCompiler::MLIRCompiler(unsigned n_qubits, std::unique_ptr<QuantumBackend> be)
    : num_qubits(n_qubits), backend(std::move(be)), pimpl(new Impl) {
    pimpl->context.getOrLoadDialect<rocq::mlir::quantum::QuantumDialect>();
    pimpl->context.getOrLoadDialect<rocq::mlir::sim::SimulatorDialect>();
    pimpl->context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    mlir::registerLLVMDialectTranslation(pimpl->context);
}

MLIRCompiler::~MLIRCompiler() {
    delete pimpl;
}

std::string MLIRCompiler::emit_qir(const std::string& mlir_string) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_string, &pimpl->context);
    if (!module) {
        return "Error: Failed to parse MLIR string.";
    }

    // --- Run the full lowering pipeline ---
    mlir::PassManager pm(&pimpl->context);
    pm.addPass(createQuantumToSimulatorPass()); // quantum -> sim
    pm.addPass(createSimulatorToQIRPass());     // sim -> llvm
    
    // Add standard passes to convert the rest of the IR to LLVM
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    if (mlir::failed(pm.run(*module))) {
        return "Error: Failed to run lowering passes to LLVM.";
    }

    // --- Translate to LLVM IR ---
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
        return "Error: Failed to translate MLIR to LLVM IR.";
    }

    std::string llvm_ir_string;
    llvm::raw_string_ostream os(llvm_ir_string);
    llvmModule->print(os, nullptr);
    
    return os.str();
}


std::vector<std::complex<double>> MLIRCompiler::compile_and_execute(
    const std::string& mlir_string,
    const std::map<std::string, bool>& args) {
    // This function remains the same as before, using the sim dialect interpreter.
    // ... (implementation from previous step)
    return {}; // Placeholder
}

} // namespace rocq