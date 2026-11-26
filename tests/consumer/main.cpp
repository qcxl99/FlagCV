// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>

#if defined(FLAGCV_CONSUMER_VENDOR) || defined(FLAGCV_CONSUMER_NVIDIA)
#ifdef FLAGCV_CONSUMER_VENDOR
#include "hip_test_api.hpp"
constexpr auto backend = FLAGCV_CONSUMER_VENDOR;
#else
#include <cuda_runtime_api.h>
constexpr auto backend = "nvidia";
#endif

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
  auto release = [](void* ptr) { (void)cudaFree(ptr); };
  return std::unique_ptr<void, decltype(release)>(data, release);
}
}  // namespace

int main() {
  try {
    std::array<unsigned char, 8> values{1, 2, 3, 4, 5, 6, 7, 8};
    const auto src = allocate(values.size()), dst = allocate(values.size());
    check(cudaMemcpy(src.get(), values.data(), values.size(), cudaMemcpyHostToDevice));
    flagcv::Handle handle(backend);
    const flagcv::GpuMat src_view(src.get(), 4, 2, flagcv::Depth::k8U, 1);
    const flagcv::GpuMat dst_view(dst.get(), 4, 2, flagcv::Depth::k8U, 1);
    for (int i = 0; i < 3; ++i) flagcv::flip(handle, src_view, dst_view, 1);
    check(cudaMemcpy(values.data(), dst.get(), values.size(), cudaMemcpyDeviceToHost));
    if (values != std::array<unsigned char, 8>{4, 3, 2, 1, 8, 7, 6, 5})
      throw std::runtime_error("installed flip returned incorrect output");
    flagcv::add(handle, src_view, src_view, dst_view);
    check(cudaMemcpy(values.data(), dst.get(), values.size(), cudaMemcpyDeviceToHost));
    if (values != std::array<unsigned char, 8>{2, 4, 6, 8, 10, 12, 14, 16})
      throw std::runtime_error("installed add returned incorrect output");
    flagcv::bitwise_not(handle, src_view, dst_view);
    check(cudaMemcpy(values.data(), dst.get(), values.size(), cudaMemcpyDeviceToHost));
    if (values != std::array<unsigned char, 8>{254, 253, 252, 251, 250, 249, 248, 247})
      throw std::runtime_error("installed bitwise_not returned incorrect output");
    flagcv::setDevice(backend);
    flagcv::flip(src_view, dst_view, -1);
    check(cudaMemcpy(values.data(), dst.get(), values.size(), cudaMemcpyDeviceToHost));
    if (values != std::array<unsigned char, 8>{8, 7, 6, 5, 4, 3, 2, 1})
      throw std::runtime_error("installed flip with setDevice returned incorrect output");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
#else
int main() {
  const flagcv::GpuMat view(nullptr, 4, 2, flagcv::Depth::k16U, 3);
  try {
    flagcv::flip(view, view, 0);
  } catch (const flagcv::Error&) {
    return view.row_pitch() == 24 && view.row_stride() == 12 ? 0 : 1;
  }
  return 1;
}
#endif
