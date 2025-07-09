#ifndef ROCQUANTUM_MLIRCOMPILER_H
#define ROCQUANTUM_MLIRCOMPILER_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h" // For ModuleOp
#include "mlir/Dialect/Func/IR/FuncOps.h" // For FuncOp and FunctionType
#include "mlir/IR/OwningOpRef.h"
#include "rocquantum/Dialect/QuantumDialect.h" // Include our QuantumDialect

#include <string>
#include <memory> // For std::unique_ptr
#include "llvm/ADT/SmallVector.h" // For TypeRange (mlir::TypeRange is often an alias to this)


namespace rocquantum {
namespace compiler {

class MLIRCompiler {
public:
    MLIRCompiler();
    ~MLIRCompiler();

    // Initializes a new MLIR module. Returns true on success.
    // The module will be owned by this MLIRCompiler instance.
    bool initializeModule(const std::string& moduleName = "rocq_module");

    // Dumps the current module to stderr (for debugging).
    void dumpModule() const;

    // Gets the current module as a string.
    std::string getModuleString() const;

    // Parses an MLIR string and loads it into the current module.
    // Returns true on success, false on parsing error.
    // This will replace the current module if one exists.
    bool loadModuleFromString(const std::string& mlirString);

    // Gets a pointer to the MLIRContext (non-owning).
    // Caution: Lifetime is tied to MLIRCompiler instance.
    mlir::MLIRContext* getContext();

    // Gets a pointer to the ModuleOp (non-owning).
    // Caution: Lifetime is tied to MLIRCompiler instance and its OwningOpRef.
    mlir::ModuleOp getModule() const;

    // Creates a new mlir::func::FuncOp within the current module.
    // Returns the created FuncOp (non-owning, valid as long as module_ is valid).
    // The argument types are MLIR types (e.g., from QubitType::get(context)).
    // Returns an empty FuncOp (evaluates to false in boolean context) on failure.
    mlir::func::FuncOp createFunction(const std::string& funcName,
                                      mlir::TypeRange argTypes,
                                      mlir::TypeRange resultTypes = {});

    // Adds a generic quantum gate to the last block of the given function.
    // Qubit values are passed as block arguments to the function.
    // Returns success if the op was added.
    mlir::LogicalResult addGenericGateOp(mlir::func::FuncOp funcOp,
                                         const std::string& gateName,
                                         mlir::ValueRange qubitOperands,
                                         const std::vector<double>& params = {}); // For parameterized gates


    // MLIRCompiler is non-copyable, non-movable for simplicity with OwningOpRef
    MLIRCompiler(const MLIRCompiler&) = delete;
    MLIRCompiler& operator=(const MLIRCompiler&) = delete;
    MLIRCompiler(MLIRCompiler&&) = delete;
    MLIRCompiler& operator=(MLIRCompiler&&) = delete;

private:
    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_; // Owning reference to the top-level module
};

} // namespace compiler
} // namespace rocquantum

#endif // ROCQUANTUM_MLIRCOMPILER_H
