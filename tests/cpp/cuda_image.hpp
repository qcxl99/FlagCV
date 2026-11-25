// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <string>

#include "numerical_policy.hpp"
#include "operators.hpp"

namespace flagcv_test {

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

inline void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class DeviceMemory {
 public:
  explicit DeviceMemory(std::size_t bytes) { cuda_check(cudaMalloc(&ptr, bytes)); }
  ~DeviceMemory() { cudaFree(ptr); }
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  void* ptr = nullptr;
};

class Stream {
 public:
  explicit Stream(bool custom = true) {
    if (custom) cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream() { if (value) cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  void wait() const { cuda_check(cudaStreamSynchronize(value)); }
  cudaStream_t value = nullptr;
};

/* The storage includes guard rows and the column padding outside its ROI. */
class Image {
 public:
  Image(int rows, int cols, int type, int x = 0, int y = 0, int extra = 0)
      : host(rows + y + (y != 0), cols + x + extra, type),
        roi(x, y, cols, rows), device(host.total() * host.elemSize()) {}

  cv::Mat pixels() const { return host(roi); }

  flagcv::GpuMat view() const {
    auto* start = static_cast<unsigned char*>(device.ptr) +
                  roi.y * host.step + roi.x * host.elemSize();
    return {start, roi.width, roi.height, depth_for(host.depth()), host.channels(),
            static_cast<std::int64_t>(host.step)};
  }

  void upload(cudaStream_t stream) const {
    cuda_check(cudaMemcpyAsync(device.ptr, host.data, host.total() * host.elemSize(),
                               cudaMemcpyHostToDevice, stream));
  }

  cv::Mat download(const Stream& stream) const {
    cv::Mat received(host.size(), host.type());
    cuda_check(cudaMemcpyAsync(received.data, device.ptr,
                               received.total() * received.elemSize(),
                               cudaMemcpyDeviceToHost, stream.value));
    stream.wait();
    return received;
  }

  cv::Mat host;
  const cv::Rect roi;
  DeviceMemory device;
};

inline bool bytes_equal(const cv::Mat& a, const cv::Mat& b) {
  require(a.size() == b.size() && a.type() == b.type(), "matrix layout changed");
  for (int y = 0; y < a.rows; ++y)
    if (std::memcmp(a.ptr(y), b.ptr(y), a.cols * a.elemSize())) return false;
  return true;
}

inline void compare(Op op, const cv::Mat& actual, const cv::Mat& expected) {
  if (!arithmetic(op) || expected.depth() == CV_8U || expected.depth() == CV_16U) {
    require(bytes_equal(actual, expected), "result differs at the byte level");
    return;
  }
  cv::Mat lhs, rhs;
  actual.convertTo(lhs, CV_32F);
  expected.convertTo(rhs, CV_32F);
  const bool half = expected.depth() == CV_16F;
  const double rtol = half ? float16_rtol : float32_rtol;
  const double atol = half ? float16_atol : float32_atol;
  for (int y = 0; y < lhs.rows; ++y) {
    for (int x = 0; x < lhs.cols * lhs.channels(); ++x) {
      const float a = lhs.ptr<float>(y)[x], b = rhs.ptr<float>(y)[x];
      if (a == b || (std::isnan(a) && std::isnan(b))) continue;
      require(std::isfinite(a) && std::isfinite(b) &&
                  std::abs(a - b) <= atol + rtol * std::abs(b),
              "floating-point result differs at (" + std::to_string(y) + ", " +
                  std::to_string(x) + "): " + std::to_string(a) + " vs " +
                  std::to_string(b));
    }
  }
}

}  // namespace flagcv_test
