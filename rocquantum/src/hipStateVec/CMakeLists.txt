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

# Find RCCL package (try common names)
find_package(rccl QUIET)
if(NOT rccl_FOUND)
  find_package(RCCL QUIET)
  if(NOT RCCL_FOUND)
    find_package(Hccl REQUIRED) # If Hccl is the one, make it required
    set(rccl_INCLUDE_DIRS ${Hccl_INCLUDE_DIRS})
    set(rccl_LIBRARIES ${Hccl_LIBRARIES})
    set(rccl_FOUND TRUE) # Assume Hccl implies rccl for our purposes
  else()
    set(rccl_INCLUDE_DIRS ${RCCL_INCLUDE_DIRS})
    set(rccl_LIBRARIES ${RCCL_LIBRARIES})
  endif()
elseif(rccl_FOUND)
  # If rccl_FOUND by find_package(rccl), variables should be rccl_INCLUDE_DIRS and rccl_LIBRARIES
  # If using modern CMake, rccl::rccl might be an imported target.
  # For now, assume variables are set.
endif()

if(rccl_FOUND)
  message(STATUS "RCCL/Hccl found.")
  if(rccl_INCLUDE_DIRS)
    target_include_directories(hipStateVec PUBLIC ${rccl_INCLUDE_DIRS})
    message(STATUS "RCCL include dirs: ${rccl_INCLUDE_DIRS}")
  endif()
  if(rccl_LIBRARIES)
    target_link_libraries(hipStateVec PUBLIC ${rccl_LIBRARIES})
    message(STATUS "RCCL libraries: ${rccl_LIBRARIES}")
  elseif(TARGET rccl::rccl) # Check for imported target
     target_link_libraries(hipStateVec PUBLIC rccl::rccl)
     message(STATUS "Linking against RCCL imported target rccl::rccl.")
  else()
     message(WARNING "RCCL found but no library path or imported target (rccl_LIBRARIES or rccl::rccl) defined.")
  endif()
else()
  message(FATAL_ERROR "RCCL (or Hccl) package not found. Please ensure it is installed and discoverable by CMake.")
endif()


# Set install rules if you plan to install the library
# install(TARGETS hipStateVec DESTINATION lib)
# install(FILES ${PROJECT_SOURCE_DIR}/include/rocquantum/hipStateVec.h DESTINATION include/rocquantum)

message(STATUS "hipStateVec library target created.")

# --- Tests ---
if(BUILD_TESTING)
    find_package(GTest REQUIRED) # Assuming GTest is preferred, but simple main() is used in example.
                                 # If not using GTest, this find_package is not strictly needed for the example.

    add_executable(test_hipStateVec_multi_gpu test_hipStateVec_multi_gpu.cpp)
    
    # Link against the hipStateVec library itself (rocquantum_hipStateVec or hipStateVec)
    # The target name for the library is 'hipStateVec' as defined in this CMakeLists.txt
    # Also link against HIP runtime.
    target_link_libraries(test_hipStateVec_multi_gpu PRIVATE 
        hipStateVec 
        HIP::hip_runtime
        # GTest::gtest_main # If using GTest
    )
    # Ensure the test executable can find headers from the library.
    # The hipStateVec target already has PUBLIC include to ../../include
    # so test_hipStateVec_multi_gpu should inherit this.

    # Enable CTest integration
    include(CTest)
    add_test(NAME MultiGPUTests COMMAND test_hipStateVec_multi_gpu)
    
    message(STATUS "Multi-GPU tests for hipStateVec added.")
endif()
