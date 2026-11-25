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
add_library(flagcv_readbackfixture MODULE cpp/runtime_test_backend.cpp)
target_include_directories(flagcv_readbackfixture PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(flagcv_readbackfixture PRIVATE cxx_std_17)
target_compile_definitions(flagcv_readbackfixture PRIVATE FLAGCV_TEST_READBACK=1)

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
configure_file(runtime/compiler.py "${fixture}/backends/readbackfixture/compiler.py" COPYONLY)
file(WRITE "${fixture}/backends/testfixture/kernels/registry.json" [=[
{"kernels":[{"name":"add","source":"../../../kernels/common/pointwise.py","function":"platform_add"}]}
]=])
add_test(NAME flagcv.runtime COMMAND flagcv_test_runtime "${fixture}")
set_tests_properties(flagcv.runtime PROPERTIES LABELS "host" TIMEOUT 60
  ENVIRONMENT "FLAGCV_PLUGIN_DIR=$<TARGET_FILE_DIR:flagcv_testfixture>;FLAGCV_DATA_DIR=${fixture}")

function(flagcv_add_resize_test target backend runtime)
  add_executable(${target} cpp/test_resize.cpp cpp/resize_reference.cpp)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}")
  target_link_libraries(${target} PRIVATE flagcv opencv_core opencv_imgproc ${runtime})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)
  flagcv_enable_warnings(${target})
  add_test(NAME flagcv.${backend}.resize COMMAND ${target})
  set_tests_properties(flagcv.${backend}.resize PROPERTIES LABELS "${backend};gpu" TIMEOUT 420)
endfunction()

function(flagcv_add_cvtcolor_test target backend runtime)
  add_executable(${target} cpp/test_cvtcolor.cpp)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${target} PRIVATE flagcv opencv_core opencv_imgproc ${runtime})
  target_compile_features(${target} PRIVATE cxx_std_17)
  flagcv_enable_warnings(${target})
  add_test(NAME flagcv.${backend}.cvtcolor COMMAND ${target})
  set_tests_properties(flagcv.${backend}.cvtcolor PROPERTIES LABELS "${backend};gpu" TIMEOUT 420)
endfunction()

function(flagcv_add_border_threshold_test target backend runtime)
  add_executable(${target} cpp/test_border_threshold.cpp)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${target} PRIVATE flagcv opencv_core opencv_imgproc ${runtime})
  target_compile_features(${target} PRIVATE cxx_std_17)
  flagcv_enable_warnings(${target})
  add_test(NAME flagcv.${backend}.border_threshold COMMAND ${target})
  set_tests_properties(flagcv.${backend}.border_threshold PROPERTIES LABELS "${backend};gpu" TIMEOUT 420)
  add_test(NAME flagcv.${backend}.border_threshold_oracle COMMAND ${target} --oracle)
  set_tests_properties(flagcv.${backend}.border_threshold_oracle PROPERTIES LABELS "host")
endfunction()

function(flagcv_add_spatial_filters_test target backend runtime)
  add_executable(${target} cpp/test_spatial_filters.cpp)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${target} PRIVATE flagcv opencv_core opencv_imgproc ${runtime})
  target_compile_features(${target} PRIVATE cxx_std_17)
  flagcv_enable_warnings(${target})
  add_test(NAME flagcv.${backend}.spatial_filters COMMAND ${target})
  set_tests_properties(flagcv.${backend}.spatial_filters PROPERTIES LABELS "${backend};gpu" TIMEOUT 420)
endfunction()

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

function(flagcv_add_projective_test target backend sdk)
  add_executable(${target} cpp/test_projective.cpp)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${target} PRIVATE flagcv opencv_core opencv_imgproc ${sdk})
  target_compile_features(${target} PRIVATE cxx_std_17)
  flagcv_enable_warnings(${target})
  add_test(NAME flagcv.${backend}.projective COMMAND ${target})
  set_tests_properties(flagcv.${backend}.projective PROPERTIES LABELS "${backend};gpu" TIMEOUT 1200)
endfunction()

if(FLAGCV_TEST_NVIDIA)
  if(NOT TARGET flagcv_nvidia)
    message(FATAL_ERROR "FLAGCV_TEST_NVIDIA requires the nvidia backend")
  endif()
  find_package(OpenCV 5 REQUIRED COMPONENTS core imgproc)
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

  add_executable(flagcv_test_warpaffine cpp/test_warpaffine.cpp)
  target_include_directories(flagcv_test_warpaffine PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(flagcv_test_warpaffine PRIVATE flagcv opencv_core opencv_imgproc CUDA::cudart)
  target_compile_features(flagcv_test_warpaffine PRIVATE cxx_std_17)
  flagcv_enable_warnings(flagcv_test_warpaffine)
  add_test(NAME flagcv.nvidia.warpaffine COMMAND flagcv_test_warpaffine)
  set_tests_properties(flagcv.nvidia.warpaffine PROPERTIES LABELS "nvidia;gpu" TIMEOUT 420)

  flagcv_add_projective_test(flagcv_test_projective nvidia CUDA::cudart)
  flagcv_add_resize_test(flagcv_test_resize nvidia CUDA::cudart)
  flagcv_add_cvtcolor_test(flagcv_test_cvtcolor nvidia CUDA::cudart)
  flagcv_add_border_threshold_test(flagcv_test_border_threshold nvidia CUDA::cudart)
  flagcv_add_spatial_filters_test(flagcv_test_spatial_filters nvidia CUDA::cudart)

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

option(FLAGCV_TEST_HIP "Run the public API matrix on the shared HIP backend" OFF)
set(FLAGCV_TEST_HIP_DEVICES "1" CACHE STRING "Number of visible HIP test devices")
if(FLAGCV_TEST_HIP)
  if(NOT TARGET flagcv_hip)
    message(FATAL_ERROR "FLAGCV_TEST_HIP requires FLAGCV_BACKENDS=hip")
  endif()
  find_package(OpenCV 5 REQUIRED COMPONENTS core)
  find_package(hip REQUIRED)
  find_package(Threads REQUIRED)
  find_package(OpenCV 5 REQUIRED COMPONENTS core imgproc)
  add_executable(flagcv_test_warpaffine_hip cpp/test_warpaffine.cpp)
  target_include_directories(flagcv_test_warpaffine_hip PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(flagcv_test_warpaffine_hip PRIVATE flagcv opencv_core opencv_imgproc hip::host)
  target_compile_features(flagcv_test_warpaffine_hip PRIVATE cxx_std_17)
  target_compile_definitions(flagcv_test_warpaffine_hip PRIVATE FLAGCV_TEST_HIP=1)
  flagcv_enable_warnings(flagcv_test_warpaffine_hip)
  add_test(NAME flagcv.hip.warpaffine COMMAND flagcv_test_warpaffine_hip)
  set_tests_properties(flagcv.hip.warpaffine PROPERTIES LABELS "hip;gpu" TIMEOUT 420)

  flagcv_add_projective_test(flagcv_test_projective_hip hip hip::host)
  target_compile_definitions(flagcv_test_projective_hip PRIVATE FLAGCV_TEST_HIP=1)
  flagcv_add_resize_test(flagcv_test_resize_hip hip hip::host)
  target_compile_definitions(flagcv_test_resize_hip PRIVATE FLAGCV_TEST_HIP=1)
  flagcv_add_cvtcolor_test(flagcv_test_cvtcolor_hip hip hip::host)
  target_compile_definitions(flagcv_test_cvtcolor_hip PRIVATE FLAGCV_TEST_HIP=1)
  flagcv_add_border_threshold_test(flagcv_test_border_threshold_hip hip hip::host)
  target_compile_definitions(flagcv_test_border_threshold_hip PRIVATE FLAGCV_TEST_HIP=1)
  flagcv_add_spatial_filters_test(flagcv_test_spatial_filters_hip hip hip::host)
  target_compile_definitions(flagcv_test_spatial_filters_hip PRIVATE FLAGCV_TEST_HIP=1)

  add_executable(flagcv_test_hip cpp/test_nvidia.cpp)
  target_include_directories(flagcv_test_hip PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(flagcv_test_hip PRIVATE flagcv opencv_core hip::host Threads::Threads)
  target_compile_features(flagcv_test_hip PRIVATE cxx_std_17)
  target_compile_definitions(flagcv_test_hip PRIVATE FLAGCV_TEST_HIP=1
    FLAGCV_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
  flagcv_enable_warnings(flagcv_test_hip)
  add_test(NAME flagcv.hip.public_api COMMAND flagcv_test_hip "${FLAGCV_TEST_HIP_DEVICES}")
  set_tests_properties(flagcv.hip.public_api PROPERTIES LABELS "hip;gpu" TIMEOUT 420)

  set(hip_overlay "${CMAKE_CURRENT_BINARY_DIR}/hip-test-data")
  file(MAKE_DIRECTORY "${hip_overlay}/backends/hip/kernels")
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/compiler" "${hip_overlay}/compiler" SYMBOLIC)
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/kernels" "${hip_overlay}/kernels" SYMBOLIC)
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/backends/hygon/compiler.py"
    "${hip_overlay}/backends/hip/compiler.py" SYMBOLIC)
  file(WRITE "${hip_overlay}/backends/hip/kernels/registry.json" [=[
{"kernels":[{"name":"add","source":"../../../kernels/common/pointwise.py",
             "function":"flagcv_missing_nvidia_entry"}]}
]=])
  add_test(NAME flagcv.hip.platform_failure COMMAND flagcv_test_hip --platform-failure)
  set_tests_properties(flagcv.hip.platform_failure PROPERTIES LABELS "hip;gpu" TIMEOUT 120
    ENVIRONMENT "FLAGCV_DATA_DIR=${hip_overlay}")
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

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(compiler_fixture "${CMAKE_CURRENT_BINARY_DIR}/compiler-fixture")
file(MAKE_DIRECTORY "${compiler_fixture}/backends/testfixture")
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/compiler" "${compiler_fixture}/compiler" SYMBOLIC)
configure_file(compiler/provider.py "${compiler_fixture}/backends/testfixture/compiler.py" COPYONLY)
add_executable(test_compiler_client test_compiler_client.cpp)
target_include_directories(test_compiler_client PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_compiler_client PRIVATE flagcv)
flagcv_enable_warnings(test_compiler_client)
add_test(NAME flagcv.compiler_client COMMAND test_compiler_client "${compiler_fixture}")
set_tests_properties(flagcv.compiler_client PROPERTIES TIMEOUT 10 LABELS host
  ENVIRONMENT "FLAGCV_DATA_DIR=${compiler_fixture};FLAGCV_PYTHON=${Python3_EXECUTABLE};FLAGCV_COMPILER_TIMEOUT_MS=3000")
add_test(NAME flagcv.scratch_abi COMMAND Python3::Interpreter
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/test_scratch_abi.py")
set_tests_properties(flagcv.scratch_abi PROPERTIES LABELS host)

add_library(flagcv_scratch_fixture MODULE compiler/scratch_backend.cpp)
set_target_properties(flagcv_scratch_fixture PROPERTIES OUTPUT_NAME flagcv_testfixture
  LIBRARY_OUTPUT_DIRECTORY "${compiler_fixture}/plugin")
target_compile_features(flagcv_scratch_fixture PRIVATE cxx_std_17)
target_include_directories(flagcv_scratch_fixture PRIVATE "${PROJECT_SOURCE_DIR}")
flagcv_enable_warnings(flagcv_scratch_fixture)
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/kernels" "${compiler_fixture}/kernels" SYMBOLIC)
add_executable(test_scratch_launch test_scratch_launch.cpp)
target_link_libraries(test_scratch_launch PRIVATE flagcv)
flagcv_enable_warnings(test_scratch_launch)
add_test(NAME flagcv.scratch_launch COMMAND test_scratch_launch "${compiler_fixture}")
set_tests_properties(flagcv.scratch_launch PROPERTIES TIMEOUT 10 LABELS host
  ENVIRONMENT "FLAGCV_DATA_DIR=${compiler_fixture};FLAGCV_PLUGIN_DIR=$<TARGET_FILE_DIR:flagcv_scratch_fixture>;FLAGCV_PYTHON=${Python3_EXECUTABLE};FLAGCV_COMPILER_TIMEOUT_MS=3000")
