# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

option(FLAGCV_TEST_NVIDIA "Run public C++ API tests with NVIDIA and OpenCV CPU" OFF)
set(FLAGCV_TEST_NVIDIA_DEVICES "1" CACHE STRING "Number of visible NVIDIA test devices")
if(FLAGCV_BUILD_TESTS AND FLAGCV_TEST_NVIDIA)
  if(NOT TARGET flagcv_nvidia)
    message(FATAL_ERROR "FLAGCV_TEST_NVIDIA requires the nvidia backend")
  endif()
  find_package(OpenCV 5 REQUIRED COMPONENTS core)
  find_package(CUDAToolkit REQUIRED)
  flagcv_add_gpu_test(nvidia CUDA::cudart "${FLAGCV_TEST_NVIDIA_DEVICES}")
endif()

if(FLAGCV_BUILD_BENCHMARKS AND TARGET flagcv_nvidia)
  find_package(OpenCV 5 REQUIRED COMPONENTS core OPTIONAL_COMPONENTS cudaarithm)
  find_package(CUDAToolkit REQUIRED)
  flagcv_add_gpu_benchmark(nvidia CUDA::cudart)
  if(TARGET opencv_cudaarithm)
    target_compile_definitions(flagcv_benchmark_nvidia PRIVATE FLAGCV_NATIVE_OPENCV_CUDA)
    target_link_libraries(flagcv_benchmark_nvidia PRIVATE opencv_cudaarithm)
  endif()
endif()
