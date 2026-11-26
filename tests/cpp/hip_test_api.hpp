// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_runtime_api.h>

// The correctness fixtures use the same host runtime operations on both GPUs.
// Keep these aliases in test code; FlagCV's public headers use no vendor types.
#define CUDART_CB
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaErrorNotReady = hipErrorNotReady;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr auto cudaGetDevice = hipGetDevice;
constexpr auto cudaGetDeviceCount = hipGetDeviceCount;
constexpr auto cudaSetDevice = hipSetDevice;
constexpr auto cudaGetErrorString = hipGetErrorString;
inline cudaError_t cudaMalloc(void** ptr, std::size_t size) { return hipMalloc(ptr, size); }
constexpr auto cudaFree = hipFree;
constexpr auto cudaMemcpy = hipMemcpy;
constexpr auto cudaMemcpyAsync = hipMemcpyAsync;
constexpr auto cudaStreamCreateWithFlags = hipStreamCreateWithFlags;
constexpr auto cudaStreamDestroy = hipStreamDestroy;
constexpr auto cudaStreamQuery = hipStreamQuery;
constexpr auto cudaStreamSynchronize = hipStreamSynchronize;
constexpr auto cudaLaunchHostFunc = hipLaunchHostFunc;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
constexpr auto cudaEventCreate = hipEventCreate;
constexpr auto cudaEventDestroy = hipEventDestroy;
constexpr auto cudaEventRecord = hipEventRecord;
constexpr auto cudaEventSynchronize = hipEventSynchronize;
constexpr auto cudaEventElapsedTime = hipEventElapsedTime;
constexpr auto cudaGetDeviceProperties = hipGetDeviceProperties;
constexpr auto cudaDeviceGetPCIBusId = hipDeviceGetPCIBusId;
constexpr auto cudaDriverGetVersion = hipDriverGetVersion;
constexpr auto cudaRuntimeGetVersion = hipRuntimeGetVersion;
