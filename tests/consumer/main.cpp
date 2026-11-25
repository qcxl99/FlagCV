// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

auto allocate(std::size_t bytes) {
  void* data = nullptr;
  check(cudaMalloc(&data, bytes));
  return std::unique_ptr<void, decltype(&cudaFree)>(data, cudaFree);
}
}  // namespace

int main() {
  try {
    std::array<unsigned char, 8> values{1, 2, 3, 4, 5, 6, 7, 8};
    const auto src = allocate(values.size()), dst = allocate(values.size());
    check(cudaMemcpy(src.get(), values.data(), values.size(), cudaMemcpyHostToDevice));
    flagcv::Handle handle;
    const flagcv::GpuMat src_view(src.get(), 4, 2, flagcv::Depth::k8U, 1);
    const flagcv::GpuMat dst_view(dst.get(), 4, 2, flagcv::Depth::k8U, 1);
    for (int i = 0; i < 3; ++i) flagcv::flip(handle, src_view, dst_view, 1);
    check(cudaMemcpy(values.data(), dst.get(), values.size(), cudaMemcpyDeviceToHost));
    if (values != std::array<unsigned char, 8>{4, 3, 2, 1, 8, 7, 6, 5})
      throw std::runtime_error("installed runtime returned incorrect output");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
