set(HIPSTATEVEC_SRC
    hipStateVec.cpp
    single_qubit_kernels.hip
    two_qubit_kernels.hip
    measurement_kernels.hip
    multi_qubit_kernels.hip # Added this line
)

add_library(hipStateVec SHARED ${HIPSTATEVEC_SRC})

# Target properties for HIP compilation
# With modern CMake (3.21+) and ROCm (5.2+), set_source_files_properties might not be strictly needed
# if using "HIP" as a language. However, explicitly setting it can be more robust.
# For older CMake/ROCm, you might need CMAKE_HIP_FLAGS or similar.
set_source_files_properties(${HIPSTATEVEC_SRC} PROPERTIES HIP_SOURCE_TYPE "kernels")

target_link_libraries(hipStateVec PUBLIC
    HIP::hip_runtime # Provided by find_package(HIP)
    roc::rocblas     # Provided by find_package(rocBLAS) - modern target name
                     # If older, might be ${rocBLAS_LIBRARIES} directly
    hiprand::hiprand # Provided by find_package(hiprand)
)

# Set install rules if you plan to install the library
# install(TARGETS hipStateVec DESTINATION lib)
# install(FILES ${PROJECT_SOURCE_DIR}/include/rocquantum/hipStateVec.h DESTINATION include/rocquantum)

message(STATUS "hipStateVec library target created.")
