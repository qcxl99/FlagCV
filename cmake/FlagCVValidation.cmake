# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set(FLAGCV_VALIDATION_HEADERS "${PROJECT_BINARY_DIR}/validation")
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
configure_file("${PROJECT_SOURCE_DIR}/tests/cpp/numerical_policy.hpp.in"
  "${FLAGCV_VALIDATION_HEADERS}/numerical_policy.hpp" @ONLY)

function(flagcv_add_gpu_test vendor sdk devices)
  find_package(Threads REQUIRED)
  set(target flagcv_test_${vendor})
  add_executable(${target} "${PROJECT_SOURCE_DIR}/tests/cpp/test_gpu.cpp")
  target_include_directories(${target} PRIVATE "${FLAGCV_VALIDATION_HEADERS}")
  target_link_libraries(${target} PRIVATE flagcv opencv_core "${sdk}" Threads::Threads)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE
    FLAGCV_TEST_DATA="${PROJECT_SOURCE_DIR}/tests/data" ${ARGN})
  flagcv_enable_warnings(${target})
  add_test(NAME flagcv.${vendor}.public_api COMMAND ${target} "${devices}")
  set_tests_properties(flagcv.${vendor}.public_api PROPERTIES
    LABELS "${vendor};gpu" TIMEOUT 420)

  set(overlay "${CMAKE_CURRENT_BINARY_DIR}/platform-failure")
  file(MAKE_DIRECTORY "${overlay}/backends/${vendor}/kernels")
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/compiler" "${overlay}/compiler" SYMBOLIC)
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/kernels" "${overlay}/kernels" SYMBOLIC)
  file(CREATE_LINK "${PROJECT_SOURCE_DIR}/backends/${vendor}/compiler.py"
    "${overlay}/backends/${vendor}/compiler.py" SYMBOLIC)
  file(WRITE "${overlay}/backends/${vendor}/kernels/registry.json" [=[
{"kernels":[{"name":"add","source":"../../../kernels/common/pointwise.py",
             "function":"flagcv_missing_platform_entry","plan":"pointwise"}]}
]=])
  add_test(NAME flagcv.${vendor}.platform_failure COMMAND ${target} --platform-failure)
  set_tests_properties(flagcv.${vendor}.platform_failure PROPERTIES
    ENVIRONMENT "FLAGCV_DATA_DIR=${overlay}" LABELS "${vendor};gpu" TIMEOUT 120)
endfunction()

function(flagcv_add_gpu_benchmark vendor sdk)
  set(target flagcv_benchmark_${vendor})
  add_executable(${target} "${PROJECT_SOURCE_DIR}/benchmark/cpp/benchmark.cpp")
  add_dependencies(${target} flagcv_core_workload)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}"
    "${FLAGCV_VALIDATION_HEADERS}" "${PROJECT_BINARY_DIR}/benchmark")
  target_link_libraries(${target} PRIVATE flagcv opencv_core "${sdk}")
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE FLAGCV_BENCHMARK_VENDOR="${vendor}" ${ARGN})
  flagcv_enable_warnings(${target})
endfunction()
