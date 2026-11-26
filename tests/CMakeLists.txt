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
find_package(Threads REQUIRED)
target_link_libraries(flagcv_test_runtime PRIVATE flagcv Threads::Threads)
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
