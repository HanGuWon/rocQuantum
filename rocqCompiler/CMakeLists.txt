# ... (previous content from CMakeLists.txt)

# --- New Pass for QIR Lowering ---
add_library(SimulatorToQIRPass STATIC passes/SimulatorToQIRPass.cpp)
target_include_directories(SimulatorToQIRPass PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(SimulatorToQIRPass PRIVATE
  MLIRPass MLIRTransforms MLIRDialect MLIRIR
  MLIRLLVMDialect # Depends on the LLVM dialect
  SimulatorDialect
)
add_dependencies(SimulatorToQIRPass gen_Simulator)


# --- Update Main Compiler Library ---
target_link_libraries(rocqCompiler PRIVATE
    # Our own components
    QuantumToSimulatorPass
    SimulatorToQIRPass # Add the new pass
    QuantumDialect
    SimulatorDialect
    rocquantum_lib
    # MLIR components
    MLIRIR MLIRParser MLIRPass MLIRSupport
    MLIRLLVMDialect # Add LLVM dialect support
    MLIRToLLVMIRTranslation # For translating to LLVM
    # LLVM components
    LLVMSupport LLVMCore
)
# ... (rest of the file)