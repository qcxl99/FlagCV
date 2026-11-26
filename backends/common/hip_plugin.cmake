# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

get_filename_component(vendor "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
find_package(hip CONFIG REQUIRED)
get_target_property(hip_host_libraries hip::host INTERFACE_LINK_LIBRARIES)
if(vendor STREQUAL "hygon" AND NOT hip_host_libraries MATCHES "(^|::|/)galaxyhip([.;>/]|$)")
  message(FATAL_ERROR "The hygon backend requires Hygon DTK hip::host (Galaxy HIP); set hip_DIR to the DTK SDK.")
endif()
if(vendor STREQUAL "amd" AND NOT hip_host_libraries MATCHES "(^|::|/)amdhip64([.;>/]|$)")
  message(FATAL_ERROR "The amd backend requires AMD ROCm hip::host (amdhip64); set hip_DIR to the ROCm SDK.")
endif()

set(plugin "flagcv_${vendor}")
add_library(${plugin} MODULE "${PROJECT_SOURCE_DIR}/backends/common/hip_backend.cpp")
target_link_libraries(${plugin} PRIVATE hip::host)
target_include_directories(${plugin} PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(${plugin} PRIVATE cxx_std_17)
target_compile_definitions(${plugin} PRIVATE FLAGCV_HIP_VENDOR="${vendor}")
set_target_properties(${plugin} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(${plugin})

install(TARGETS ${plugin} LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/${vendor}")
install(FILES kernels/registry.json DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/${vendor}/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/${vendor}/kernels"
  FILES_MATCHING PATTERN "*.py")
