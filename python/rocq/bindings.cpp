#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // For std::vector, std::map, etc.
#include <pybind11/numpy.h>     // For py::array_t
#include "rocquantum/hipStateVec.h" // Path to the C API header

namespace py = pybind11;

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

        RocTensorNetworkHandleWrapper(RocsvHandleWrapper& sim_handle) : sim_handle_ref_(sim_handle) {
            rocqStatus_t status = rocTensorNetworkCreate(&handle_);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create rocTensorNetworkHandle: " + std::to_string(status));
            }
        }
        ~RocTensorNetworkHandleWrapper() {
            if (handle_) {
                rocTensorNetworkDestroy(handle_);
            }
        }
        RocTensorNetworkHandleWrapper(const RocTensorNetworkHandleWrapper&) = delete;
        RocTensorNetworkHandleWrapper& operator=(const RocTensorNetworkHandleWrapper&) = delete;
        RocTensorNetworkHandleWrapper(RocTensorNetworkHandleWrapper&& other) noexcept
            : handle_(other.handle_), sim_handle_ref_(other.sim_handle_ref_) {
            other.handle_ = nullptr;
        }
        RocTensorNetworkHandleWrapper& operator=(RocTensorNetworkHandleWrapper&& other) noexcept {
            if (this != &other) {
                if (handle_) rocTensorNetworkDestroy(handle_);
                handle_ = other.handle_;
                // sim_handle_ref_ = other.sim_handle_ref_; // This is tricky with references for assignment.
                                                         // For simplicity, ensure sim_handle_ref is const or handle lifetime carefully.
                                                         // Or make it a shared_ptr if simulator handle can go out of scope.
                                                         // For now, assume simulator outlives network handle.
                other.handle_ = nullptr;
            }
            return *this;
        }
        rocTensorNetworkHandle_t get() const { return handle_; }
        RocsvHandleWrapper& get_sim_handle() const { return sim_handle_ref_; }
    };

    py::class_<RocTensorNetworkHandleWrapper>(m, "RocTensorNetwork")
        .def(py::init<RocsvHandleWrapper&>(), py::arg("simulator_handle"), "Creates a Tensor Network manager.")
        .def("add_tensor", [](RocTensorNetworkHandleWrapper& self, const rocquantum::util::rocTensor& tensor) {
            // The C API rocTensorNetworkAddTensor takes a const rocTensor*.
            // pybind11 will pass the rocTensor object by value if not careful.
            // We need to pass a pointer to the tensor object held by Python.
            // However, the C++ TensorNetwork class copies the rocTensor metadata.
            // So, passing by const reference to pybind, which then passes pointer to C API, is fine.
            rocqStatus_t status = rocTensorNetworkAddTensor(self.get(), &tensor);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocTensorNetworkAddTensor failed: " + std::to_string(status));
            }
            // TODO: Return tensor index? The C-API doesn't, but the C++ class does.
            // For now, no return, user manages indices.
        }, py::arg("tensor"))
        .def("add_contraction", [](RocTensorNetworkHandleWrapper& self, int t_idx_a, int m_idx_a, int t_idx_b, int m_idx_b){
            rocqStatus_t status = rocTensorNetworkAddContraction(self.get(), t_idx_a, m_idx_a, t_idx_b, m_idx_b);
             if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocTensorNetworkAddContraction failed: " + std::to_string(status));
            }
        }, py::arg("tensor_idx_a"), py::arg("mode_idx_a"), py::arg("tensor_idx_b"), py::arg("mode_idx_b"))
        .def("contract", [](RocTensorNetworkHandleWrapper& self, rocquantum::util::rocTensor& result_tensor_py) {
            // result_tensor_py must be allocated by Python user with expected final shape.
            // The C++ contract function will fill its data.
            if (!result_tensor_py.data_ && result_tensor_py.get_element_count() > 0) {
                 throw std::runtime_error("Result tensor for contract must be pre-allocated (e.g., using rocq.allocate_tensor).");
            }
            // Get blas_handle and stream from the simulator handle stored in RocTensorNetworkHandleWrapper
            rocblas_handle blas_h = self.get_sim_handle().get()->blasHandles[0]; // Assuming device 0 for now
            hipStream_t stream = self.get_sim_handle().get()->streams[0];     // Assuming device 0 for now
            // A proper multi-GPU tensor network would need more sophisticated handle/stream management.

            rocqStatus_t status = rocTensorNetworkContract(self.get(), &result_tensor_py, blas_h, stream);
            if (status != ROCQ_STATUS_SUCCESS && status != ROCQ_STATUS_NOT_IMPLEMENTED) {
                throw std::runtime_error("rocTensorNetworkContract failed: " + std::to_string(status));
            }
            if (status == ROCQ_STATUS_NOT_IMPLEMENTED) {
                // py::print("Warning: rocTensorNetworkContract is not fully implemented yet.");
                throw std::runtime_error("rocTensorNetworkContract is not fully implemented yet. Status: " + std::to_string(status));
            }
            // result_tensor_py is modified in place by the C function if successful
        }, py::arg("result_tensor").noconvert(), "Contracts the tensor network. Result tensor must be pre-allocated.");

}
