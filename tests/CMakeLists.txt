# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

add_executable(flagcv_test_registry cpp/test_registry.cpp)
target_include_directories(flagcv_test_registry PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(flagcv_test_registry PRIVATE flagcv)
flagcv_enable_warnings(flagcv_test_registry)
add_test(NAME flagcv.registry COMMAND flagcv_test_registry)
set_tests_properties(flagcv.registry PROPERTIES LABELS "host")

add_library(flagcv_testfixture MODULE cpp/runtime_test_backend.cpp)
target_include_directories(flagcv_testfixture PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(flagcv_testfixture PRIVATE cxx_std_17)
flagcv_enable_warnings(flagcv_testfixture)
add_executable(flagcv_test_runtime cpp/test_runtime.cpp)
target_link_libraries(flagcv_test_runtime PRIVATE flagcv)
target_compile_features(flagcv_test_runtime PRIVATE cxx_std_17)
flagcv_enable_warnings(flagcv_test_runtime)

set(fixture "${CMAKE_CURRENT_BINARY_DIR}/runtime-fixture")
file(MAKE_DIRECTORY "${fixture}/backends/testfixture/kernels")
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/compiler" "${fixture}/compiler" SYMBOLIC)
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/kernels" "${fixture}/kernels" SYMBOLIC)
configure_file(runtime/compiler.py "${fixture}/backends/testfixture/compiler.py" COPYONLY)
file(WRITE "${fixture}/backends/testfixture/kernels/registry.json" [=[
{"kernels":[{"name":"add","source":"../../../kernels/common/pointwise.py","function":"platform_add"}]}
]=])
add_test(NAME flagcv.runtime COMMAND flagcv_test_runtime "${fixture}")
set_tests_properties(flagcv.runtime PROPERTIES LABELS "host" TIMEOUT 60
  ENVIRONMENT "FLAGCV_PLUGIN_DIR=$<TARGET_FILE_DIR:flagcv_testfixture>;FLAGCV_DATA_DIR=${fixture}")

option(FLAGCV_TEST_NVIDIA "Run public C++ API tests with NVIDIA and OpenCV CPU" OFF)
set(FLAGCV_TEST_NVIDIA_DEVICES "1" CACHE STRING "Number of visible NVIDIA test devices")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/helpers/tolerance.json")
file(READ "${PROJECT_SOURCE_DIR}/helpers/tolerance.json" FLAGCV_TOLERANCE_JSON)
foreach(width IN ITEMS 16 32)
  foreach(tolerance IN ITEMS RTOL ATOL)
    string(TOLOWER "${tolerance}" key)
    string(JSON FLAGCV_FP${width}_${tolerance} GET
      "${FLAGCV_TOLERANCE_JSON}" relative float${width} "${key}")
  endforeach()
endforeach()
configure_file(cpp/numerical_policy.hpp.in numerical_policy.hpp @ONLY)

if(FLAGCV_TEST_NVIDIA)
  if(NOT TARGET flagcv_nvidia)
    message(FATAL_ERROR "FLAGCV_TEST_NVIDIA requires the nvidia backend")
  endif()
  find_package(OpenCV 5 REQUIRED COMPONENTS core)
  find_package(CUDAToolkit REQUIRED)
  find_package(Threads REQUIRED)
  add_executable(flagcv_test_nvidia cpp/test_nvidia.cpp)
  target_include_directories(flagcv_test_nvidia PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(flagcv_test_nvidia PRIVATE flagcv opencv_core
    CUDA::cudart Threads::Threads)
  target_compile_features(flagcv_test_nvidia PRIVATE cxx_std_17)
  target_compile_definitions(flagcv_test_nvidia PRIVATE FLAGCV_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
  flagcv_enable_warnings(flagcv_test_nvidia)
  add_test(NAME flagcv.nvidia.public_api
    COMMAND flagcv_test_nvidia "${FLAGCV_TEST_NVIDIA_DEVICES}")
  set_tests_properties(flagcv.nvidia.public_api PROPERTIES LABELS "nvidia;gpu"
    TIMEOUT 420)

  set(overlay "${CMAKE_CURRENT_BINARY_DIR}/nvidia-test-data")
  file(MAKE_DIRECTORY "${overlay}/backends/nvidia/kernels")
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/compiler" "${overlay}/compiler" SYMBOLIC)
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/kernels" "${overlay}/kernels" SYMBOLIC)
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/backends/nvidia/compiler.py"
    "${overlay}/backends/nvidia/compiler.py" SYMBOLIC)
  file(WRITE "${overlay}/backends/nvidia/kernels/registry.json" [=[
{"kernels":[{"name":"add","source":"../../../kernels/common/pointwise.py",
             "function":"flagcv_missing_nvidia_entry"}]}
]=])
  add_test(NAME flagcv.nvidia.platform_failure
    COMMAND flagcv_test_nvidia --platform-failure)
  set_tests_properties(flagcv.nvidia.platform_failure PROPERTIES
    ENVIRONMENT "FLAGCV_DATA_DIR=${overlay}" LABELS "nvidia;gpu" TIMEOUT 120)
endif()

option(FLAGCV_BENCHMARK_NVIDIA "Build the NVIDIA C++ OpenCV CUDA benchmark" OFF)
if(FLAGCV_BENCHMARK_NVIDIA)
  if(NOT TARGET flagcv_nvidia)
    message(FATAL_ERROR "FLAGCV_BENCHMARK_NVIDIA requires the nvidia backend")
  endif()
  find_package(OpenCV REQUIRED COMPONENTS core cudaarithm)
  find_package(CUDAToolkit REQUIRED)
  add_executable(flagcv_benchmark_nvidia
    "${PROJECT_SOURCE_DIR}/benchmark/cpp/benchmark_nvidia.cpp")
  target_include_directories(flagcv_benchmark_nvidia PRIVATE
    "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(flagcv_benchmark_nvidia PRIVATE flagcv opencv_core
    opencv_cudaarithm CUDA::cudart)
  target_compile_features(flagcv_benchmark_nvidia PRIVATE cxx_std_17)
  flagcv_enable_warnings(flagcv_benchmark_nvidia)
endif()
