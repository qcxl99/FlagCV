# Operator sources are added here as they land: one file per family under
# core/ or imgproc/, plus its lowering under lowering/.
add_library(flagcv
  core/arithmetic.cpp
  core/bitwise.cpp
  core/copy_make_border.cpp
  core/flip.cpp
  image/gpumat.cpp
  imgproc/cvtcolor.cpp
  imgproc/threshold.cpp
  imgproc/resize.cpp
  imgproc/resize_tables.cpp
  imgproc/warpaffine.cpp
  imgproc/spatial_filters.cpp
  lowering/flip.cpp
  lowering/pointwise.cpp
  lowering/cvtcolor.cpp
  lowering/resize.cpp
  lowering/warpaffine.cpp
  lowering/spatial_filters.cpp
  runtime/compiler_client.cpp
  runtime/json.cpp
  runtime/paths.cpp
  runtime/runtime.cpp)

target_include_directories(flagcv
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR})

# Build-tree fallbacks for locating kernels, the compiler and plugins; the
# install tree is found relative to the library itself (src/runtime/paths.hpp).
file(RELATIVE_PATH FLAGCV_DATA_FROM_LIBDIR
  "${CMAKE_INSTALL_FULL_LIBDIR}" "${CMAKE_INSTALL_FULL_DATADIR}/flagcv")
target_compile_definitions(flagcv PRIVATE
  FLAGCV_DATA_FROM_LIBDIR="${FLAGCV_DATA_FROM_LIBDIR}"
  FLAGCV_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  FLAGCV_PLUGIN_BUILD_DIR="${FLAGCV_PLUGIN_OUTPUT_DIR}")

# The core links no vendor SDK and no Python: only the dynamic loader, for
# backend plugins.
target_link_libraries(flagcv PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(flagcv PUBLIC cxx_std_17)
flagcv_enable_warnings(flagcv)
set_property(SOURCE imgproc/resize_tables.cpp imgproc/spatial_filters.cpp APPEND PROPERTY COMPILE_OPTIONS
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)
set_target_properties(flagcv PROPERTIES VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

install(TARGETS flagcv EXPORT FlagCVTargets)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" TYPE INCLUDE)
