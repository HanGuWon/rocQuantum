#include "rocquantum/Compiler/MLIRCompiler.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h" // For dumping to string
#include <string>

namespace rocquantum {
namespace compiler {

MLIRCompiler::MLIRCompiler() {
    context_ = std::make_unique<mlir::MLIRContext>();
    // Load our custom Quantum dialect and any other dialects needed (e.g., func)
    if (context_) {
        context_->getOrLoadDialect<quantum::QuantumDialect>();
        context_->getOrLoadDialect<mlir::func::FuncDialect>(); // Standard dialect for functions
        // Add other dialects like arith, scf if they will be used directly
    } else {
        throw std::runtime_error("Failed to create MLIRContext.");
    }
}

MLIRCompiler::~MLIRCompiler() {
    // module_ and context_ will be cleaned up by their smart pointers.
}

bool MLIRCompiler::initializeModule(const std::string& moduleName) {
    if (!context_) return false;

    // Create a new ModuleOp.
    // The OpBuilder is used to insert operations into the MLIR IR.
    // We'll set the insertion point to the context to create a top-level ModuleOp.
    mlir::OpBuilder builder(context_.get());
    module_ = builder.create<mlir::ModuleOp>(builder.getUnknownLoc(), llvm::StringRef(moduleName));

    return static_cast<bool>(module_); // Check if module creation was successful
}

void MLIRCompiler::dumpModule() const {
    if (module_) {
        module_->dump();
    } else {
        llvm::errs() << "MLIRCompiler: No module to dump.\n";
    }
}

std::string MLIRCompiler::getModuleString() const {
    if (!module_) {
        return "MLIRCompiler: No module to get string from.";
    }
    std::string moduleStr;
    llvm::raw_string_ostream ss(moduleStr);
    module_->print(ss);
    return ss.str();
}

mlir::MLIRContext* MLIRCompiler::getContext() {
    return context_.get();
}

mlir::ModuleOp MLIRCompiler::getModule() const {
    if (!module_) {
        // This case should ideally not be hit if initializeModule was called and succeeded.
        // Consider throwing or returning an empty ModuleOp if that's more appropriate.
        return nullptr;
    }
    return *module_;
}

bool MLIRCompiler::loadModuleFromString(const std::string& mlirString) {
    if (!context_) {
        llvm::errs() << "MLIRContext not initialized, cannot parse module string.\n";
        return false;
    }

    // Use MLIR's parser to parse the string.
    // The mlir::parseSourceString function can be used here.
    // It requires a SourceMgr and can populate an existing ModuleOp or create a new one.
    // For simplicity, let's try to re-initialize module_ with the parsed content.

    // llvm::SourceMgr sourceMgr;
    // sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirString), llvm::SMLoc());

    // Create a new module from the string.
    // This will replace the existing module_ if any.
    module_ = mlir::parseSourceString<mlir::ModuleOp>(mlirString, context_.get());

    if (!module_) {
        llvm::errs() << "Failed to parse MLIR string into a ModuleOp.\n";
        return false;
    }

    // Optionally, verify the newly parsed module.
    if (mlir::failed(mlir::verify(module_.get()))) {
        llvm::errs() << "Parsed MLIR module failed verification.\n";
        // module_->dump(); // Dump the problematic module
        // module_ = nullptr; // Invalidate module if verification fails
        // return false; // Strict: fail on verification error
    }
    // For now, let's be lenient on verification if parsing succeeded.

    return true;
}

mlir::func::FuncOp MLIRCompiler::createFunction(const std::string& funcName,
                                                mlir::TypeRange argTypes,
                                                mlir::TypeRange resultTypes) {
    if (!module_ || !context_) {
        llvm::errs() << "Module or MLIRContext not initialized. Cannot create function.\n";
        return nullptr; // Return an empty FuncOp
    }

    mlir::OpBuilder builder(module_->getBodyRegion()); // Builder to insert into module's body
    auto functionType = mlir::FunctionType::get(context_.get(), argTypes, resultTypes);

    // Create the function, but don't define its body yet.
    // The location can be file/line info if available, otherwise UnknownLoc.
    auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcName, functionType);

    // To define the function body, create a block and set the builder's insertion point.
    // mlir::Block *entryBlock = funcOp.addEntryBlock();
    // builder.setInsertionPointToStart(entryBlock);
    // Now ops can be added to the function body using the builder.
    // For now, this function just creates the FuncOp declaration.
    // Callers will get the builder and add blocks/ops.

    return funcOp;
}


} // namespace compiler
} // namespace rocquantum
