# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(workload_header "${CMAKE_CURRENT_BINARY_DIR}/core_cases.hpp")
add_custom_command(OUTPUT "${workload_header}"
  COMMAND Python3::Interpreter -m workloads.core --cpp-header "${workload_header}"
  DEPENDS "${PROJECT_SOURCE_DIR}/workloads/core.py"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  VERBATIM)
add_custom_target(flagcv_core_workload DEPENDS "${workload_header}")
