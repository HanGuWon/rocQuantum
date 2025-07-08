#include "rocquantum/hipTensorNet.h"
#include "rocquantum/hipStateVec.h" // For rocqStatus_t, checkHipError etc. (if needed by C-API impl)
#include <vector>
#include <new> // For std::nothrow

// Define the opaque struct that rocTensorNetworkHandle_t points to
struct rocTnStruct {
    rocquantum::TensorNetwork network;
    // May add other resources like rocblas_handle, hipStream_t if managed per network
    // For now, assume they are passed into contract.
};

extern "C" {

rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = new(std::nothrow) rocTnStruct();
    if (!tn_struct) {
        *handle = nullptr;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    *handle = tn_struct;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkDestroy(rocTensorNetworkHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);
    // Before deleting tn_struct, ensure any owned resources within network are handled.
    // Currently, rocTensor within TensorNetwork are views (owned_ = false by default on copy).
    // If rocTensors added to the network could own their data and TensorNetwork
    // was responsible for them, freeing logic would be needed here.
    // For now, TensorNetwork just holds copies of rocTensor metadata.
    delete tn_struct;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkAddTensor(rocTensorNetworkHandle_t handle, const rocquantum::util::rocTensor* tensor) {
    if (!handle || !tensor) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);
    if (!tensor->data_ && tensor->get_element_count() > 0) { // Data pointer is null for non-empty tensor
        return ROCQ_STATUS_INVALID_VALUE;
    }

    // The TensorNetwork::add_tensor makes a copy of the rocTensor struct.
    // It's crucial that the rocComplex* data within the passed 'tensor'
    // remains valid on the device for the lifetime of the network's use of it.
    // The rocTensor copied into the network will also have owned_ = false by default.
    tn_struct->network.add_tensor(*tensor);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkAddContraction(rocTensorNetworkHandle_t handle,
                                            int tensor_idx_A, int mode_idx_A,
                                            int tensor_idx_B, int mode_idx_B) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);
    return tn_struct->network.add_contraction(tensor_idx_A, mode_idx_A, tensor_idx_B, mode_idx_B);
}

rocqStatus_t rocTensorNetworkContract(rocTensorNetworkHandle_t handle,
                                      rocquantum::util::rocTensor* result_tensor,
                                      rocblas_handle blas_handle,
                                      hipStream_t stream) {
    if (!handle || !result_tensor || !blas_handle) { // stream can be 0 (default)
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);

    // The actual contraction logic is within TensorNetwork::contract
    // which is currently a stub.
    // The C-API function here calls the C++ class method.
    // The result_tensor passed in should be populated by the contract method.
    // If contract method allocates memory for result_tensor->data_, it should set owned_ = true.
    return tn_struct->network.contract(result_tensor, blas_handle, stream);
}

} // extern "C"
