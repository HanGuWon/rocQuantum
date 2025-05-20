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

    // Add a helper to create a DeviceBuffer and copy a numpy array to it
    m.def("create_device_matrix_from_numpy",
        [](py::array_t<rocComplex, py::array::c_style | py::array::forcecast> np_array) {
            if (np_array.ndim() != 2) throw std::runtime_error("NumPy array must be 2D for matrix.");
            // For this simple version, assume square matrix.
            // size_t rows = np_array.shape(0);
            // size_t cols = np_array.shape(1);
            // if (rows != cols) throw std::runtime_error("Matrix must be square.");
            
            size_t num_elements = np_array.size();
            DeviceBuffer db(num_elements, sizeof(rocComplex));
            db.copy_from_numpy(np_array);
            return db;
        }, py::arg("numpy_array"), "Creates a DeviceBuffer and copies a NumPy array to it.");

}
