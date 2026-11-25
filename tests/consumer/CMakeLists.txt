# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)
project(FlagCVInstallConsumer LANGUAGES CXX)

find_package(FlagCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_executable(flagcv_install_consumer main.cpp)
target_link_libraries(flagcv_install_consumer PRIVATE FlagCV::flagcv CUDA::cudart)
