# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# HIP plugin implementation shared by the hip and legacy hygon registrations.
# The core never links it (docs/architecture.md, "Plugin boundary").

get_filename_component(_hip_backend "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(_hip_target "flagcv_${_hip_backend}")
add_library(${_hip_target} MODULE "${PROJECT_SOURCE_DIR}/backends/hip/backend.cpp")
find_package(hip REQUIRED)
target_link_libraries(${_hip_target} PRIVATE hip::host)
target_include_directories(${_hip_target} PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(${_hip_target} PRIVATE cxx_std_17)
set_target_properties(${_hip_target} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(${_hip_target})

install(TARGETS ${_hip_target}
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES "${PROJECT_SOURCE_DIR}/backends/hygon/compiler.py"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/${_hip_backend}")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/${_hip_backend}/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/${_hip_backend}/kernels"
  FILES_MATCHING PATTERN "*.py")
