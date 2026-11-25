# One test executable per operator family, named after the family:
#
#   flagcv_add_test(bitwise)   # builds tests/test_bitwise.cpp
function(flagcv_add_test family)
  add_executable(test_${family} test_${family}.cpp)
  target_link_libraries(test_${family} PRIVATE flagcv)
  target_include_directories(test_${family} PRIVATE ${PROJECT_SOURCE_DIR})
  flagcv_enable_warnings(test_${family})
  add_test(NAME ${family} COMMAND test_${family})
endfunction()

# Register C++ operator tests here when interface validation is introduced.

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

add_library(flagcv_testfixture MODULE compiler/scratch_backend.cpp)
target_compile_features(flagcv_testfixture PRIVATE cxx_std_17)
target_include_directories(flagcv_testfixture PRIVATE "${PROJECT_SOURCE_DIR}")
flagcv_enable_warnings(flagcv_testfixture)
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/kernels" "${compiler_fixture}/kernels" SYMBOLIC)
add_executable(test_scratch_launch test_scratch_launch.cpp)
target_link_libraries(test_scratch_launch PRIVATE flagcv)
flagcv_enable_warnings(test_scratch_launch)
add_test(NAME flagcv.scratch_launch COMMAND test_scratch_launch "${compiler_fixture}")
set_tests_properties(flagcv.scratch_launch PROPERTIES TIMEOUT 10 LABELS host
  ENVIRONMENT "FLAGCV_DATA_DIR=${compiler_fixture};FLAGCV_PLUGIN_DIR=$<TARGET_FILE_DIR:flagcv_testfixture>;FLAGCV_PYTHON=${Python3_EXECUTABLE};FLAGCV_COMPILER_TIMEOUT_MS=3000")
