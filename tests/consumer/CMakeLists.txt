# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)
project(FlagCVInstallConsumer LANGUAGES CXX)

find_package(FlagCV CONFIG REQUIRED)
add_executable(flagcv_install_consumer main.cpp)
target_link_libraries(flagcv_install_consumer PRIVATE FlagCV::flagcv)
option(FLAGCV_CONSUMER_NVIDIA "Exercise the installed NVIDIA backend" ON)
if(FLAGCV_CONSUMER_NVIDIA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(flagcv_install_consumer PRIVATE FLAGCV_CONSUMER_NVIDIA)
  target_link_libraries(flagcv_install_consumer PRIVATE CUDA::cudart)
endif()
