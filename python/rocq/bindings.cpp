#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rocquantum/hipStateVec.h"
#include "rocquantum/hipTensorNet.h" // Include new header
#include <complex>                 // For std::complex

namespace py = pybind11;

// Helper to map NumPy dtype to rocDataType_t enum
rocDataType_t get_rocq_dtype_from_numpy(py::dtype dt) {
    if (dt.is(py::dtype::of<float>())) return ROC_DATATYPE_F32;
    if (dt.is(py::dtype::of<double>())) return ROC_DATATYPE_F64;
    if (dt.is(py::dtype::of<std::complex<float>>())) return ROC_DATATYPE_C64;
    if (dt.is(py::dtype::of<std::complex<double>>())) return ROC_DATATYPE_C128;
    throw std::runtime_error("Unsupported NumPy data type for TensorNetwork");
}


// Helper to convert py::array_t<rocComplex> (NumPy array from Python) to rocComplex* on device
// This is a simplified helper. Error handling and memory management need care.
// The caller is responsible for freeing d_matrix if it's allocated by this helper.
// For rocsvApplyMatrix, matrixDevice is already on device, so this helper is for
// a scenario where Python provides a host matrix that needs to be on device.
// However, the rocsvApplyMatrix C-API now expects matrixDevice to *already* be on device.
// So, the Python side will need to manage this (e.g. have a function to create device matrix from numpy).

// Let's define a simple DeviceMemory class for Python to manage GPU buffers for matrices
class DeviceBuffer {
public:
    void* ptr_ = nullptr;
    size_t size_bytes_ = 0;
    bool owned_ = true; // Does this wrapper own the memory (i.e., should it free it)?

    DeviceBuffer() = default;

    DeviceBuffer(size_t num_elements, size_t element_size) : size_bytes_(num_elements * element_size) {
        if (hipMalloc(&ptr_, size_bytes_) != hipSuccess) {
            throw std::runtime_error("Failed to allocate device memory in DeviceBuffer constructor");
        }
    }

    // Constructor to wrap an existing device pointer (e.g., d_state)
    // This wrapper does NOT own the memory.
    DeviceBuffer(void* existing_ptr, size_t size_bytes, bool take_ownership = false) 
        : ptr_(existing_ptr), size_bytes_(size_bytes), owned_(take_ownership) {}


    ~DeviceBuffer() {
        if (owned_ && ptr_) {
            hipFree(ptr_);
        }
    }

    // Disable copy constructor and assignment to prevent double free / shallow copies of owned memory
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allow move construction and assignment
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_bytes_(other.size_bytes_), owned_(other.owned_) {
        other.ptr_ = nullptr;
        other.size_bytes_ = 0;
        other.owned_ = false; // Transferred ownership
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (owned_ && ptr_) {
                hipFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_bytes_ = other.size_bytes_;
            owned_ = other.owned_;
            other.ptr_ = nullptr;
            other.size_bytes_ = 0;
            other.owned_ = false;
        }
        return *this;
    }

    void copy_from_numpy(py::array_t<rocComplex, py::array::c_style | py::array::forcecast> np_array) {
        if (!ptr_ || np_array.nbytes() > size_bytes_) {
            throw std::runtime_error("Device buffer not allocated, null, or NumPy array too large.");
        }
        if (hipMemcpy(ptr_, np_array.data(), np_array.nbytes(), hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("Failed to copy NumPy array to device");
        }
    }
    
    // Method to get the raw pointer (e.g., rocComplex*)
    template<typename T>
    T* get_ptr() const {
        return static_cast<T*>(ptr_);
    }
    
    size_t nbytes() const { return size_bytes_; }
};


// Wrapper for rocsvHandle_t to ensure proper creation and destruction
class RocsvHandleWrapper {
public:
    rocsvHandle_t handle_ = nullptr;

    RocsvHandleWrapper() {
        rocqStatus_t status = rocsvCreate(&handle_);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create rocsvHandle: " + std::to_string(status));
        }
    }

    ~RocsvHandleWrapper() {
        if (handle_) {
            rocsvDestroy(handle_); // Ignoring status on destroy for simplicity in destructor
        }
    }

    // Disable copy constructor and assignment
    RocsvHandleWrapper(const RocsvHandleWrapper&) = delete;
    RocsvHandleWrapper& operator=(const RocsvHandleWrapper&) = delete;

    // Allow move construction
    RocsvHandleWrapper(RocsvHandleWrapper&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    // Allow move assignment
    RocsvHandleWrapper& operator=(RocsvHandleWrapper&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                rocsvDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    rocsvHandle_t get() const { return handle_; }
};


PYBIND11_MODULE(_rocq_hip_backend, m) {
    m.doc() = "Python bindings for rocQuantum hipStateVec library";

    py::enum_<rocqStatus_t>(m, "rocqStatus")
        .value("SUCCESS", ROCQ_STATUS_SUCCESS)
        .value("FAILURE", ROCQ_STATUS_FAILURE)
        .value("INVALID_VALUE", ROCQ_STATUS_INVALID_VALUE)
        .value("ALLOCATION_FAILED", ROCQ_STATUS_ALLOCATION_FAILED)
        .value("HIP_ERROR", ROCQ_STATUS_HIP_ERROR)
        .value("NOT_IMPLEMENTED", ROCQ_STATUS_NOT_IMPLEMENTED)
        .export_values();

    // DeviceBuffer class for managing device memory from Python
    py::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def(py::init<>()) // Default constructor
        .def(py::init<size_t, size_t>(), py::arg("num_elements"), py::arg("element_size"))
        .def("copy_from_numpy", &DeviceBuffer::copy_from_numpy, "Copies data from a NumPy array to the device buffer.")
        .def("nbytes", &DeviceBuffer::nbytes, "Returns the size of the buffer in bytes.");
        // get_ptr is not directly exposed as it's unsafe for general Python.
        // Python code will pass DeviceBuffer objects to wrapped C functions that extract the pointer.


    // Wrapper for the handle
    py::class_<RocsvHandleWrapper>(m, "RocsvHandle")
        .def(py::init<>());
        // The handle itself is mostly opaque to Python users of this direct binding layer.
        // Higher-level Python classes (Simulator) will use it.

    // State management functions
    // rocsvAllocateState: The returned d_state (rocComplex**) is tricky.
    // We'll return a DeviceBuffer that wraps the allocated device pointer.
    m.def("allocate_state_internal", 
        [](const RocsvHandleWrapper& handle_wrapper, unsigned numQubits) {
            rocComplex* d_state_ptr = nullptr;
            rocqStatus_t status = rocsvAllocateState(handle_wrapper.get(), numQubits, &d_state_ptr);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvAllocateState failed: " + std::to_string(status));
            }
            size_t num_elements = 1ULL << numQubits;
            // The DeviceBuffer now owns this d_state_ptr and will hipFree it.
            return DeviceBuffer(static_cast<void*>(d_state_ptr), num_elements * sizeof(rocComplex), true /*owned*/);
        }, py::arg("handle"), py::arg("num_qubits"), "Allocates state vector on device, returns an owning DeviceBuffer.");

    m.def("initialize_state", 
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits) {
            // Basic check, Python side should ensure numQubits matches buffer allocation
            if (d_state_buffer.nbytes() != ( (1ULL << numQubits) * sizeof(rocComplex) ) ) {
                 throw std::runtime_error("DeviceBuffer size mismatch in initialize_state");
            }
            return rocsvInitializeState(handle_wrapper.get(), d_state_buffer.get_ptr<rocComplex>(), numQubits);
        }, py::arg("handle"), py::arg("d_state_buffer"), py::arg("num_qubits"));

    m.def("allocate_distributed_state",
        [](RocsvHandleWrapper& handle_wrapper, unsigned totalNumQubits) {
            rocqStatus_t status = rocsvAllocateDistributedState(handle_wrapper.get(), totalNumQubits);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvAllocateDistributedState failed: " + std::to_string(status));
            }
        }, py::arg("handle"), py::arg("total_num_qubits"), "Allocates a distributed state vector across multiple GPUs.");

    m.def("initialize_distributed_state",
        [](RocsvHandleWrapper& handle_wrapper) {
            rocqStatus_t status = rocsvInitializeDistributedState(handle_wrapper.get());
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvInitializeDistributedState failed: " + std::to_string(status));
            }
        }, py::arg("handle"), "Initializes a distributed state vector to the |0...0> state.");

    // Specific single-qubit gates
    m.def("apply_x", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyX(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies X gate");
    m.def("apply_y", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyY(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies Y gate");
    m.def("apply_z", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyZ(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies Z gate");
    m.def("apply_h", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyH(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies H gate");
    m.def("apply_s", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyS(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies S gate");
    m.def("apply_t", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyT(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies T gate");
    m.def("apply_sdg", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplySdg(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies S dagger gate");

    // Rotation gates
    m.def("apply_rx", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyRx(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies Rx gate");
    m.def("apply_ry", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyRy(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies Ry gate");
    m.def("apply_rz", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyRz(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies Rz gate");

    // Specific two-qubit gates
    m.def("apply_cnot", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned ctrlQ, unsigned tgtQ) {
        return rocsvApplyCNOT(h.get(), d_state.get_ptr<rocComplex>(), nQ, ctrlQ, tgtQ); }, "Applies CNOT gate");
    m.def("apply_cz", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned q1, unsigned q2) {
        return rocsvApplyCZ(h.get(), d_state.get_ptr<rocComplex>(), nQ, q1, q2); }, "Applies CZ gate");
    m.def("apply_swap", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned q1, unsigned q2) {
        return rocsvApplySWAP(h.get(), d_state.get_ptr<rocComplex>(), nQ, q1, q2); }, "Applies SWAP gate");
    
    // rocsvApplyMatrix
    m.def("apply_matrix", 
        [](const RocsvHandleWrapper& handle_wrapper, 
           DeviceBuffer& d_state_buffer, 
           unsigned numQubits, 
           std::vector<unsigned> qubitIndices_vec, // Use std::vector for easy conversion
           DeviceBuffer& matrix_device_buffer, // Matrix already on device
           unsigned matrixDim) {
            // Basic checks
            if (qubitIndices_vec.size() == 0) {
                throw std::runtime_error("qubitIndices must not be empty for apply_matrix");
            }
            unsigned numTargetQubits = qubitIndices_vec.size();
            // matrixDim should be 1U << numTargetQubits
            // matrix_device_buffer.nbytes() should be matrixDim * matrixDim * sizeof(rocComplex)
            
            // The C API expects const unsigned* for qubitIndices.
            // The d_targetIndices for m=3,4,>=5 in C++ code is created on device.
            // Here, qubitIndices is passed from Python as a list/vector, used by C++ to create d_targetIndices.
            // The current C API rocsvApplyMatrix takes const unsigned* qubitIndices (host pointer).
            // This is consistent.

            return rocsvApplyMatrix(handle_wrapper.get(), 
                                    d_state_buffer.get_ptr<rocComplex>(), 
                                    numQubits, 
                                    qubitIndices_vec.data(), // Pass pointer to vector's data
                                    numTargetQubits, 
                                    matrix_device_buffer.get_ptr<rocComplex>(), 
                                    matrixDim);
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), 
           py::arg("qubit_indices"), py::arg("matrix_device"), py::arg("matrix_dim"));

    // rocsvMeasure
    m.def("measure", 
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned qubitToMeasure) {
            int outcome = 0;
            double probability = 0.0;
            rocqStatus_t status = rocsvMeasure(handle_wrapper.get(), 
                                               d_state_buffer.get_ptr<rocComplex>(), 
                                               numQubits, 
                                               qubitToMeasure, 
                                               &outcome, &probability);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvMeasure failed: " + std::to_string(status));
            }
            return py::make_tuple(outcome, probability);
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("qubit_to_measure"),
           "Measures a single qubit. Returns (outcome, probability).");

    m.def("get_expectation_value_z",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned targetQubit) {
            double result = 0.0;
            rocqStatus_t status = rocsvGetExpectationValueSinglePauliZ(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubit,
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValueSinglePauliZ failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubit"),
           "Calculates <Z_k> for the target qubit.");

    m.def("get_expectation_value_x",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned targetQubit) {
            double result = 0.0;
            rocqStatus_t status = rocsvGetExpectationValueSinglePauliX(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubit,
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValueSinglePauliX failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubit"),
           "Calculates <X_k> for the target qubit. Modifies state vector.");

    m.def("get_expectation_value_y",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned targetQubit) {
            double result = 0.0;
            rocqStatus_t status = rocsvGetExpectationValueSinglePauliY(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubit,
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValueSinglePauliY failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubit"),
           "Calculates <Y_k> for the target qubit. Modifies state vector.");

    m.def("get_expectation_value_pauli_product_z",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& targetQubits_vec) {
            double result = 0.0;
            if (targetQubits_vec.empty()) { // Should be handled by Python PauliOperator logic too
                return 1.0; // Expectation of Identity
            }
            rocqStatus_t status = rocsvGetExpectationValuePauliProductZ(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubits_vec.data(), // Pass pointer to vector's data
                                               static_cast<unsigned>(targetQubits_vec.size()),
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValuePauliProductZ failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubits"),
           "Calculates <Z_q0 Z_q1 ...> for the target qubits. Does not modify state.");

    m.def("get_expectation_pauli_string",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::string& pauliString, const std::vector<unsigned>& targetQubits_vec) {
            double result = 0.0;
            if (pauliString.length() != targetQubits_vec.size()) {
                throw std::runtime_error("Pauli string length must match the number of target qubits.");
            }
            if (targetQubits_vec.empty()) {
                return 1.0; // Expectation of Identity
            }

            rocStatus_t status = rocsvGetExpectationPauliString(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               pauliString.c_str(),
                                               targetQubits_vec.data(),
                                               static_cast<unsigned>(targetQubits_vec.size()),
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationPauliString failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("pauli_string"), py::arg("target_qubits"),
           "Calculates expectation value for a generic Pauli string (e.g., \"IXYZ\"). Non-destructive.");

    m.def("sample",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& measuredQubits_vec, unsigned numShots) {
            if (numShots == 0) {
                return py::array_t<uint64_t>(0);
            }

            py::array_t<uint64_t> h_results(numShots);
            
            rocqStatus_t status = rocsvSample(
                                        handle_wrapper.get(),
                                        d_state_buffer.get_ptr<rocComplex>(),
                                        numQubits,
                                        measuredQubits_vec.data(),
                                        static_cast<unsigned>(measuredQubits_vec.size()),
                                        numShots,
                                        h_results.mutable_data());

            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvSample failed: " + std::to_string(status));
            }
            return h_results;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("measured_qubits"), py::arg("num_shots"),
           "Samples from the state vector and returns an array of measurement outcomes (bitstrings).");

    m.def("apply_controlled_matrix",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& controlQubits_vec, const std::vector<unsigned>& targetQubits_vec,
           DeviceBuffer& matrix_device_buffer) {
            
            unsigned numControls = static_cast<unsigned>(controlQubits_vec.size());
            unsigned numTargets = static_cast<unsigned>(targetQubits_vec.size());

            if (numTargets == 0) return ROCQ_STATUS_SUCCESS;
            if (numControls == 0) { // Fallback to regular apply_matrix
                 return rocsvApplyMatrix(handle_wrapper.get(), 
                                    d_state_buffer.get_ptr<rocComplex>(), 
                                    numQubits, 
                                    targetQubits_vec.data(),
                                    numTargets, 
                                    matrix_device_buffer.get_ptr<rocComplex>(), 
                                    1U << numTargets);
            }

            rocqStatus_t status = rocsvApplyControlledMatrix(
                                        handle_wrapper.get(),
                                        d_state_buffer.get_ptr<rocComplex>(),
                                        numQubits,
                                        controlQubits_vec.data(),
                                        numControls,
                                        targetQubits_vec.data(),
                                        numTargets,
                                        matrix_device_buffer.get_ptr<rocComplex>());

            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvApplyControlledMatrix failed: " + std::to_string(status));
            }
            return status;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), 
           py::arg("control_qubits"), py::arg("target_qubits"), py::arg("matrix_device"),
           "Applies a matrix to target qubits, controlled by control qubits.");

    // Add a helper to create a DeviceBuffer and copy a numpy array to it
    m.def("create_device_matrix_from_numpy",
        [](py::array_t<rocComplex, py::array::c_style | py::array::forcecast> np_array) {
            if (np_array.ndim() != 2) throw std::runtime_error("NumPy array must be 2D for matrix.");
            size_t num_elements = np_array.size();
            DeviceBuffer db(num_elements, sizeof(rocComplex)); // Owns memory
            db.copy_from_numpy(np_array);
            return db;
        }, py::arg("numpy_array"), "Creates a DeviceBuffer and copies a NumPy array to it.");

    // --- rocTensorUtil Bindings ---
    py::class_<rocquantum::util::rocTensor>(m, "RocTensor")
        .def(py::init<>(), "Default constructor")
        .def(py::init([](const std::vector<long long>& dims, py::object py_data_np_array) {
            // This constructor is primarily for Python-side creation of metadata.
            // Actual device data allocation should be done via allocate_tensor.
            // If py_data_np_array is a numpy array, we could try to initialize from it,
            // but that complicates ownership. For now, primarily for dimensions.
            auto tensor = rocquantum::util::rocTensor();
            tensor.dimensions_ = dims;
            tensor.calculate_strides(); // Calculate strides based on dimensions
            // Data pointer (tensor.data_) should be set via allocate_tensor or by wrapping existing device memory.
            // tensor.owned_ will be set by allocate_tensor.
            return tensor;
        }), py::arg("dimensions"), py::arg("py_data_np_array") = py::none(), "Constructor with dimensions. Data must be set via allocate_tensor or from existing buffer.")
        .def_property("dimensions",
            [](const rocquantum::util::rocTensor &self) { return self.dimensions_; },
            [](rocquantum::util::rocTensor &self, const std::vector<long long>& dims) {
                self.dimensions_ = dims;
                self.calculate_strides(); // Recalculate strides when dimensions change
            })
        .def_property("labels",
            [](const rocquantum::util::rocTensor &self) { return self.labels_; },
            [](rocquantum::util::rocTensor &self, const std::vector<std::string>& lbls) { self.labels_ = lbls; })
        .def_property_readonly("strides", [](const rocquantum::util::rocTensor &self) { return self.strides_; })
        .def("get_element_count", &rocquantum::util::rocTensor::get_element_count)
        .def("rank", &rocquantum::util::rocTensor::rank)
        // Note: data_ pointer is not directly exposed for safety.
        // owned_ flag is also not directly exposed, managed by allocate/free.
        .def("__repr__", [](const rocquantum::util::rocTensor &t) {
            std::string rep = "<rocq.RocTensor dimensions=[";
            for (size_t i = 0; i < t.dimensions_.size(); ++i) {
                rep += std::to_string(t.dimensions_[i]) + (i == t.dimensions_.size() - 1 ? "" : ", ");
            }
            rep += "], rank=" + std::to_string(t.rank());
            rep += ", elements=" + std::to_string(t.get_element_count());
            rep += (t.data_ ? ", has_data" : ", no_data");
            rep += (t.owned_ ? ", owned" : ", view");
            rep += ">";
            return rep;
        });

    m.def("allocate_tensor", [](rocquantum::util::rocTensor& tensor) {
        // This function will modify tensor in-place to allocate its data
        rocqStatus_t status = rocquantum::util::rocTensorAllocate(&tensor);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("rocTensorAllocate failed: " + std::to_string(status));
        }
        // The tensor passed by reference is modified (data_ pointer set, owned_ set to true)
    }, py::arg("tensor").noconvert(), "Allocates device memory for the given RocTensor object (modifies in-place).");

    m.def("free_tensor", [](rocquantum::util::rocTensor& tensor) {
        rocqStatus_t status = rocquantum::util::rocTensorFree(&tensor);
        if (status != ROCQ_STATUS_SUCCESS) {
            // Perhaps just warn or log if freeing non-owned/null data,
            // but rocTensorFree should handle this gracefully.
            // For now, any error from rocTensorFree is an exception.
            throw std::runtime_error("rocTensorFree failed: " + std::to_string(status));
        }
        // Tensor is modified in-place (data_ to nullptr, owned_ to false)
    }, py::arg("tensor").noconvert(), "Frees device memory for the RocTensor if it's owned (modifies in-place).");

    m.def("permute_tensor",
        [](rocquantum::util::rocTensor& output_tensor,
           const rocquantum::util::rocTensor& input_tensor,
           const std::vector<int>& host_permutation_map) {
        // Ensure output_tensor has its data_ pointer allocated and dimensions/strides set correctly by the caller
        // before calling this. The rocTensorPermute C-function doesn't allocate for output.
        if (!output_tensor.data_ && output_tensor.get_element_count() > 0) {
            throw std::runtime_error("Output tensor for permute_tensor must be pre-allocated.");
        }
        rocqStatus_t status = rocquantum::util::rocTensorPermute(&output_tensor, &input_tensor, host_permutation_map);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("rocTensorPermute failed: " + std::to_string(status));
        }
    }, py::arg("output_tensor").noconvert(), py::arg("input_tensor"), py::arg("permutation_map"));

    // --- hipTensorNet Bindings ---
    // Opaque handle wrapper for rocTensorNetworkHandle_t
    class RocTensorNetworkHandleWrapper {
    public:
        rocTensorNetworkHandle_t handle_ = nullptr;
        RocsvHandleWrapper& sim_handle_ref_; // Keep a reference to the simulator's handle for rocBLAS/stream

        RocTensorNetworkHandleWrapper(RocsvHandleWrapper& sim_handle, py::object dtype_source) 
            : sim_handle_ref_(sim_handle) {
            
            py::dtype dt;
            if (py::isinstance<py::array>(dtype_source)) {
                dt = py::cast<py::array>(dtype_source).dtype();
            } else if (py::isinstance<py::dtype>(dtype_source)) {
                dt = py::cast<py::dtype>(dtype_source);
            } else {
                throw std::invalid_argument("Constructor requires a NumPy array or a NumPy dtype to determine the data type.");
            }

            rocDataType_t rocq_dtype = get_rocq_dtype_from_numpy(dt);
            
            rocqStatus_t status = rocTensorNetworkCreate(&handle_, rocq_dtype);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create rocTensorNetworkHandle: " + std::to_string(status));
            }
        }

        ~RocTensorNetworkHandleWrapper() {
            if (handle_) {
                rocTensorNetworkDestroy(handle_);
            }
        }

        // Disable copy constructor and assignment
        RocTensorNetworkHandleWrapper(const RocTensorNetworkHandleWrapper&) = delete;
        RocTensorNetworkHandleWrapper& operator=(const RocTensorNetworkHandleWrapper&) = delete;

        // Allow move construction
        RocTensorNetworkHandleWrapper(RocTensorNetworkHandleWrapper&& other) noexcept
            : handle_(other.handle_), sim_handle_ref_(other.sim_handle_ref_) {
            other.handle_ = nullptr;
        }
        
        // Note: Move assignment is problematic with reference members and is omitted for simplicity.
        RocTensorNetworkHandleWrapper& operator=(RocTensorNetworkHandleWrapper&& other) = delete;

        rocTensorNetworkHandle_t get() const { return handle_; }
        RocsvHandleWrapper& get_sim_handle() const { return sim_handle_ref_; }
    };

    py::class_<RocTensorNetworkHandleWrapper>(m, "RocTensorNetwork")
        .def(py::init<RocsvHandleWrapper&, py::object>(), py::arg("simulator_handle"), py::arg("dtype_source"),
             "Creates a Tensor Network manager for a specific data type.\n"
             "dtype_source: A NumPy array or np.dtype to specify the precision (e.g., np.float32, np.complex64).")
        
        .def("add_tensor", [](RocTensorNetworkHandleWrapper& self, const rocquantum::util::rocTensor& tensor) {
            rocqStatus_t status = rocTensorNetworkAddTensor(self.get(), &tensor);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocTensorNetworkAddTensor failed: " + std::to_string(status));
            }
        }, py::arg("tensor"))

        .def("contract", [](RocTensorNetworkHandleWrapper& self, 
                                 py::object config_obj, // Can be dict or None
                                 rocquantum::util::rocTensor& result_tensor_py) {
            // This part of the binding remains largely the same, as the C-API signature for contract hasn't changed.
            hipTensorNetContractionOptimizerConfig_t config;
            // Default values
            config.memory_limit = 0; // 0 means no limit

            if (!config_obj.is_none()) {
                py::dict config_dict = py::cast<py::dict>(config_obj);
                if (config_dict.contains("memory_limit")) {
                    config.memory_limit = config_dict["memory_limit"].cast<size_t>();
                }
                // Add other optimizer params here as they are added to the struct
            }

            // TODO: The rocblas_handle and hipStream_t should be retrieved from the simulator handle.
            // The current structure of RocsvHandleWrapper makes this difficult without modifying it.
            // Using placeholders for now.
            rocblas_handle blas_h = nullptr; // Placeholder
            hipStream_t stream = 0; // Placeholder

            rocqStatus_t status = rocTensorNetworkContract(self.get(), &config, &result_tensor_py, blas_h, stream);
            
            if (status != ROCQ_STATUS_SUCCESS && status != ROCQ_STATUS_NOT_IMPLEMENTED) {
                throw std::runtime_error("rocTensorNetworkContract failed: " + std::to_string(status));
            }
            if (status == ROCQ_STATUS_NOT_IMPLEMENTED) {
                py::print("Warning: rocTensorNetworkContract path execution is not fully implemented yet.");
            }
        }, py::arg("optimizer_config"), py::arg("result_tensor").noconvert(), "Contracts the tensor network. Result tensor must be pre-allocated.");

    // --- MLIRCompiler Bindings ---
    py::class_<rocquantum::compiler::MLIRCompiler>(m, "MLIRCompiler")
        .def(py::init<>(), "Initializes the MLIR compiler environment.")
        .def("initialize_module", &rocquantum::compiler::MLIRCompiler::initializeModule,
             py::arg("module_name") = "rocq_module",
             "Initializes a new MLIR module. Returns true on success.")
        .def("dump_module", &rocquantum::compiler::MLIRCompiler::dumpModule,
             "Dumps the current MLIR module to stderr.")
        .def("get_module_string", &rocquantum::compiler::MLIRCompiler::getModuleString,
             "Returns the current MLIR module as a string representation.")
        .def("load_module_from_string", &rocquantum::compiler::MLIRCompiler::loadModuleFromString,
             py::arg("mlir_string"),
             "Parses an MLIR string and loads it into the compiler's module. Returns true on success.")
        .def("run_adjoint_generation_pass", &rocquantum::compiler::MLIRCompiler::runAdjointGenerationPass,
             "Runs the Adjoint Generation compiler pass on the current module.")
        .def("create_function",
             [](rocquantum::compiler::MLIRCompiler &self, const std::string& funcName,
                const std::vector<std::string>& argTypeStrs,
                const std::vector<std::string>& resultTypeStrs) {

                mlir::MLIRContext* context = self.getContext();
                if (!context) throw std::runtime_error("MLIRContext is null in MLIRCompiler.");

                llvm::SmallVector<mlir::Type, 4> argTypes;
                for (const auto& typeStr : argTypeStrs) {
                    // For this to work robustly, types need to be registered or be builtin.
                    // Our QuantumDialect registers "!quantum.qubit".
                    // Other types like "f64", "i32" are builtin.
                    mlir::Type type = mlir::parseAttribute(typeStr, context).dyn_cast_or_null<mlir::TypeAttr>().getValue();
                    if (!type) { // Fallback for simple dialect types like "!quantum.qubit"
                        if (typeStr == "!quantum.qubit") type = rocquantum::quantum::QubitType::get(context);
                        // Add more custom type string parsing here if needed
                    }
                    if (!type) throw std::runtime_error("Failed to parse argument type string: " + typeStr);
                    argTypes.push_back(type);
                }

                llvm::SmallVector<mlir::Type, 4> resultTypes;
                for (const auto& typeStr : resultTypeStrs) {
                    mlir::Type type = mlir::parseAttribute(typeStr, context).dyn_cast_or_null<mlir::TypeAttr>().getValue();
                     if (!type) { // Fallback for simple dialect types
                        if (typeStr == "!quantum.qubit") type = rocquantum::quantum::QubitType::get(context);
                    }
                    if (!type) throw std::runtime_error("Failed to parse result type string: " + typeStr);
                    resultTypes.push_back(type);
                }

                // The C++ createFunction returns a FuncOp, but we don't bind FuncOp directly yet.
                // We'll just call it and rely on it being added to the ModuleOp inside the compiler.
                // Return true/false for success.
                mlir::func::FuncOp funcOp = self.createFunction(funcName, argTypes, resultTypes);
                return static_cast<bool>(funcOp); // True if funcOp is not null
             },
             py::arg("func_name"), py::arg("arg_type_strs"), py::arg("result_type_strs") = std::vector<std::string>(),
             "Creates a new function (FuncOp) in the module. Types are specified as strings.");
        // Note: getContext and getModule (returning raw pointers) are not exposed directly
        // to Python for safety, unless a clear need and management strategy arises.
        // Python will interact via higher-level methods that use these internally.

}
