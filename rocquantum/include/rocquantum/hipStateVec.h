#ifndef HIPSTATEVEC_H
#define HIPSTATEVEC_H

#include <hip/hip_runtime.h> // For hipFloatComplex, hipDoubleComplex, hipError_t

// Define rocComplex based on precision
#ifdef ROCQ_PRECISION_DOUBLE
    typedef hipDoubleComplex rocComplex;
    typedef double real_t;
    const real_t REAL_EPSILON = 1e-12;
#else
    typedef hipFloatComplex rocComplex;
    typedef float real_t;
    const real_t REAL_EPSILON = 1e-6f;
#endif

// Opaque handle for hipStateVec resources
struct rocsvInternalHandle; // Forward declaration
typedef struct rocsvInternalHandle* rocsvHandle_t;

// Status codes for rocQuantum operations
typedef enum {
    ROCQ_STATUS_SUCCESS = 0,
    ROCQ_STATUS_FAILURE = 1,
    ROCQ_STATUS_INVALID_VALUE = 2,
    ROCQ_STATUS_ALLOCATION_FAILED = 3,
    ROCQ_STATUS_HIP_ERROR = 4,
    ROCQ_STATUS_NOT_IMPLEMENTED = 5,
    ROCQ_STATUS_RCCL_ERROR = 6 // Ensure this is present
    // Add more specific error codes as needed
} rocqStatus_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a hipStateVec handle.
 *
 * @param[out] handle Pointer to the handle to be created.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvCreate(rocsvHandle_t* handle);

/**
 * @brief Destroys a hipStateVec handle and releases associated resources.
 *
 * @param[in] handle The handle to be destroyed.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvDestroy(rocsvHandle_t handle);

/**
 * @brief Allocates memory for the state vector on the device.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] numQubits The number of qubits in the state vector.
 * @param[out] d_state Pointer to the device memory for the state vector.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state);

/**
 * @brief Initializes the state vector to the |0...0> state.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] d_state Pointer to the device memory for the state vector.
 * @param[in] numQubits The number of qubits.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits);

/**
 * @brief Allocates memory for a distributed state vector across multiple GPUs.
 *
 * The total number of qubits determines the global state vector size. This size is
 * then divided among the available GPUs managed by the handle.
 * Assumes numGpus in the handle is a power of 2.
 *
 * @param[in] handle The hipStateVec handle, assumed to be initialized for multi-GPU.
 * @param[in] totalNumQubits The total number of qubits for the global state vector.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits);

/**
 * @brief Initializes a distributed state vector to the |0...0> state.
 *
 * This function assumes rocsvAllocateDistributedState has been successfully called.
 * It sets all amplitudes to zero across all GPU slices, then sets the first amplitude
 * (global index 0, on GPU 0) to 1.0.
 *
 * @param[in] handle The hipStateVec handle managing the distributed state.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle);

/**
 * @brief Applies a pre-fused 2x2 matrix to a single target qubit.
 *
 * The matrix represents a sequence of single-qubit gates that have already been
 * multiplied together on the CPU.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] targetQubit The global index of the qubit the fused gate acts upon.
 * @param[in] d_fusedMatrix Pointer to the 2x2 gate matrix (DEVICE memory, column-major).
 *                          The matrix must be unitary.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvApplyFusedSingleQubitMatrix(rocsvHandle_t handle,
                                              unsigned targetQubit,
                                              const rocComplex* d_fusedMatrix);

/**
 * @brief Swaps the roles of two qubit indices in the state vector representation,
 *        potentially requiring data exchange between GPUs if a global slice qubit is involved.
 *
 * This is used to make non-local qubits local for gate application.
 * The state vector is modified in place.
 * Assumes temporary swap buffers are managed by the handle.
 *
 * @param[in] handle The hipStateVec handle, managing all GPU resources.
 * @param[in] qubit_idx1 First global qubit index to swap.
 * @param[in] qubit_idx2 Second global qubit index to swap.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2);

/**
 * @brief Applies a quantum gate (matrix) to the specified qubits in the state vector.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] d_state Pointer to the device state vector.
 * @param[in] numQubits Total number of qubits in the state vector.
 * @param[in] qubitIndices Array of qubit indices the gate acts upon.
 * @param[in] numTargetQubits Number of target qubits (e.g., 1 for single-qubit gate, 2 for two-qubit gate).
 * @param[in] matrixDevice Pointer to the gate matrix (DEVICE memory, column-major).
 * @param[in] matrixDim Dimension of the gate matrix (e.g., 2 for 1-qubit, 4 for 2-qubit).
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state,
                              unsigned numQubits,
                              const unsigned* qubitIndices,
                              unsigned numTargetQubits,
                              const rocComplex* matrixDevice, // Changed from 'matrix'
                              unsigned matrixDim);

/**
 * @brief Measures a single qubit in the computational basis.
 *
 * Collapses the state vector to the measured outcome.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] d_state Pointer to the device state vector.
 * @param[in] numQubits Total number of qubits in the state vector.
 * @param[in] qubitToMeasure The index of the qubit to measure.
 * @param[out] outcome Pointer to store the measurement outcome (0 or 1).
 * @param[out] probability Pointer to store the probability of the measured outcome.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvMeasure(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned qubitToMeasure,
                          int* outcome,
                          double* probability);

// --- Single Qubit Specific Gates ---

/**
 * @brief Applies a Pauli-X gate to the target qubit.
 */
rocqStatus_t rocsvApplyX(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies a Pauli-Y gate to the target qubit.
 */
rocqStatus_t rocsvApplyY(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies a Pauli-Z gate to the target qubit.
 */
rocqStatus_t rocsvApplyZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies a Hadamard gate to the target qubit.
 */
rocqStatus_t rocsvApplyH(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies a Phase (S) gate to the target qubit.
 */
rocqStatus_t rocsvApplyS(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies a T gate to the target qubit.
 */
rocqStatus_t rocsvApplyT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies an S-dagger (conjugate transpose of S) gate to the target qubit.
 */
rocqStatus_t rocsvApplySdg(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit);

/**
 * @brief Applies an Rx rotation gate to the target qubit.
 * @param theta Rotation angle in radians.
 */
rocqStatus_t rocsvApplyRx(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta); // theta stays double for API

/**
 * @brief Applies an Ry rotation gate to the target qubit.
 * @param theta Rotation angle in radians.
 */
rocqStatus_t rocsvApplyRy(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta); // theta stays double for API

/**
 * @brief Applies an Rz rotation gate to the target qubit.
 * @param theta Rotation angle in radians.
 */
rocqStatus_t rocsvApplyRz(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta); // theta stays double for API

// --- Two Qubit Specific Gates ---

/**
 * @brief Applies a CNOT (Controlled-NOT) gate.
 * @param controlQubit The control qubit index.
 * @param targetQubit The target qubit index.
 */
rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit);

/**
 * @brief Applies a CZ (Controlled-Z) gate.
 * @param qubit1 Index of the first qubit.
 * @param qubit2 Index of the second qubit.
 */
rocqStatus_t rocsvApplyCZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2);

/**
 * @brief Applies a SWAP gate between two qubits.
 * @param qubit1 Index of the first qubit.
 * @param qubit2 Index of the second qubit.
 */
rocqStatus_t rocsvApplySWAP(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2);

/**
 * @brief Calculates the expectation value of a single Pauli Z operator on a target qubit.
 * <psi|Z_k|psi> = P(k=0) - P(k=1)
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] d_state Pointer to the device state vector. For multi-GPU, this is legacy and the
 *                    state from the handle's distributed slices is used. For single GPU, this must match
 *                    the allocated state in the handle.
 * @param[in] numQubits Total number of qubits in the state vector.
 * @param[in] targetQubit The global index of the qubit for which to calculate <Z>.
 * @param[out] result Pointer to store the expectation value (a double).
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocsvGetExpectationValueSinglePauliZ(rocsvHandle_t handle,
                                                  rocComplex* d_state,
                                                  unsigned numQubits,
                                                  unsigned targetQubit,
                                                  double* result);

/**
 * @brief Calculates the expectation value of a single Pauli X operator on a target qubit.
 * <psi|X_k|psi>
 * Note: This function MODIFIES the state vector by applying basis change rotations.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] d_state Pointer to the device state vector. (See notes in rocsvGetExpectationValueSinglePauliZ)
 * @param[in] numQubits Total number of qubits.
 * @param[in] targetQubit The global index of the qubit for <X>.
 * @param[out] result Pointer to store the expectation value.
 * @return rocqStatus_t Status.
 */
rocqStatus_t rocsvGetExpectationValueSinglePauliX(rocsvHandle_t handle,
                                                  rocComplex* d_state,
                                                  unsigned numQubits,
                                                  unsigned targetQubit,
                                                  double* result);

/**
 * @brief Calculates the expectation value of a single Pauli Y operator on a target qubit.
 * <psi|Y_k|psi>
 * Note: This function MODIFIES the state vector by applying basis change rotations.
 *
 * @param[in] handle The hipStateVec handle.
 * @param[in] d_state Pointer to the device state vector. (See notes in rocsvGetExpectationValueSinglePauliZ)
 * @param[in] numQubits Total number of qubits.
 * @param[in] targetQubit The global index of the qubit for <Y>.
 * @param[out] result Pointer to store the expectation value.
 * @return rocqStatus_t Status.
 */
rocqStatus_t rocsvGetExpectationValueSinglePauliY(rocsvHandle_t handle,
                                                  rocComplex* d_state,
                                                  unsigned numQubits,
                                                  unsigned targetQubit,
                                                  double* result);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIPSTATEVEC_H
