# Find GTest
find_package(GTest REQUIRED)

# Create the test executable
add_executable(test_hipDensityMat test_hipDensityMat.cpp)

# Link against hipDensityMat library, GTest, and HIP runtime
target_link_libraries(test_hipDensityMat PRIVATE
    rocquantum_hipDensityMat
    GTest::gtest_main
    ${HIP_LIBRARIES}
)

# Add the test to CTest
include(GoogleTest)
gtest_discover_tests(test_hipDensityMat)
