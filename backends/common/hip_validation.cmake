# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

string(TOUPPER "${vendor}" vendor_upper)
option(FLAGCV_TEST_${vendor_upper} "Run public API tests for ${vendor}" OFF)
set(FLAGCV_TEST_${vendor_upper}_DEVICES "1" CACHE STRING "Number of visible ${vendor} test devices")
if(FLAGCV_BUILD_TESTS AND FLAGCV_TEST_${vendor_upper})
  if(NOT TARGET flagcv_${vendor})
    message(FATAL_ERROR "FLAGCV_TEST_${vendor_upper} requires FLAGCV_BACKENDS=${vendor}")
  endif()
  find_package(OpenCV 5 REQUIRED COMPONENTS core)
  find_package(hip CONFIG REQUIRED)
  flagcv_add_gpu_test("${vendor}" hip::host "${FLAGCV_TEST_${vendor_upper}_DEVICES}"
    FLAGCV_TEST_HIP=1 FLAGCV_TEST_VENDOR="${vendor}")
endif()

if(FLAGCV_BUILD_BENCHMARKS AND TARGET flagcv_${vendor})
  find_package(OpenCV 5 REQUIRED COMPONENTS core)
  find_package(hip CONFIG REQUIRED)
  flagcv_add_gpu_benchmark("${vendor}" hip::host FLAGCV_TEST_HIP=1)
endif()
