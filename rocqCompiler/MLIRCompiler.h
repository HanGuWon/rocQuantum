#ifndef MLIR_COMPILER_H
#define MLIR_COMPILER_H

#include <string>
#include <vector>
#include <complex>
#include <memory>
#include <map>
#include "QuantumBackend.h"

namespace rocq {

class MLIRCompiler {
public:
    MLIRCompiler(unsigned num_qubits, std::unique_ptr<QuantumBackend> backend);
    ~MLIRCompiler();

    std::vector<std::complex<double>> compile_and_execute(
        const std::string& mlir_string,
        const std::map<std::string, bool>& args);

    // New method to emit QIR
    std::string emit_qir(const std::string& mlir_string);

private:
    struct Impl;
    Impl* pimpl;

    unsigned num_qubits;
    std::unique_ptr<QuantumBackend> backend;
};

} // namespace rocq

#endif // MLIR_COMPILER_H