# CMakeLists.txt for the hipDensityMat library

# Find required packages, HIP is essential.
find_package(HIP REQUIRED)

# Define the library target for hipDensityMat.
add_library(rocq_hip_density_mat STATIC
    hipDensityMat.cpp
)

# Specify include directories required by this library.
# It needs access to the main project's include directory and HIP's includes.
target_include_directories(rocq_hip_density_mat PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" # For rocquantum/*.h
    ${HIP_INCLUDE_DIRS}
)

# Specify libraries to link against.
target_link_libraries(rocq_hip_density_mat PUBLIC
    ${HIP_LIBRARIES}
)
