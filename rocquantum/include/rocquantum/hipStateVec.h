#ifndef HIPSTATEVEC_H
#define HIPSTATEVEC_H

#include <hip/hip_runtime.h> // For hipFloatComplex, hipDoubleComplex, hipError_t

// Define rocComplex based on precision (default to float for now)
// TODO: Add a mechanism to switch precision (e.g., via a compile-time flag)
typedef hipFloatComplex rocComplex;
// typedef hipDoubleComplex rocComplex; // For double precision

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
    ROCQ_STATUS_NOT_IMPLEMENTED = 5
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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIPSTATEVEC_H
