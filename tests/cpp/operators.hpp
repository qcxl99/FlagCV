// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flagcv/core.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace flagcv_test {

inline void configure_cpu_reference() {
  // Portable CPU arithmetic. Optional SIMD dispatch in OpenCV 5.0 can
  // return zero instead of saturation for large unsigned 16-bit products.
  cv::setUseOptimized(false);
}

enum class Op { add, subtract, multiply, bitwise_and, bitwise_not, flip };

inline constexpr std::array operators = {
    Op::add, Op::subtract, Op::multiply, Op::bitwise_and, Op::bitwise_not,
    Op::flip};
inline constexpr const char* names[] = {
    "add", "subtract", "multiply", "bitwise_and", "bitwise_not", "flip"};

inline const char* name(Op op) { return names[static_cast<int>(op)]; }
inline bool unary(Op op) { return op >= Op::bitwise_not; }
inline bool arithmetic(Op op) { return op <= Op::multiply; }
inline bool supports_mask(Op op) { return op != Op::multiply && op != Op::flip; }

inline void call(Op op, flagcv::Handle& handle, const flagcv::GpuMat& a,
                 const flagcv::GpuMat& b, const flagcv::GpuMat& dst,
                 flagcv::Stream stream, int flip_code = 0,
                 const flagcv::GpuMat& mask = {}, double scale = 1.0) {
  switch (op) {
    case Op::add:
      if (mask.empty()) flagcv::add(handle, a, b, dst, stream);
      else flagcv::add(handle, a, b, dst, mask, stream);
      break;
    case Op::subtract:
      if (mask.empty()) flagcv::subtract(handle, a, b, dst, stream);
      else flagcv::subtract(handle, a, b, dst, mask, stream);
      break;
    case Op::multiply:
      if (scale == 1.0) flagcv::multiply(handle, a, b, dst, stream);
      else flagcv::multiply(handle, a, b, dst, scale, stream);
      break;
    case Op::bitwise_and:
      if (mask.empty()) flagcv::bitwise_and(handle, a, b, dst, stream);
      else flagcv::bitwise_and(handle, a, b, dst, mask, stream);
      break;
    case Op::bitwise_not:
      if (mask.empty()) flagcv::bitwise_not(handle, a, dst, stream);
      else flagcv::bitwise_not(handle, a, dst, mask, stream);
      break;
    case Op::flip:
      flagcv::flip(handle, a, dst, flip_code, stream);
      break;
  }
}

inline void reference(Op op, const cv::Mat& a, const cv::Mat& b, cv::Mat& dst,
                      int flip_code = 0, const cv::Mat& mask = {},
                      double scale = 1.0) {
  switch (op) {
    case Op::add: cv::add(a, b, dst, mask); break;
    case Op::subtract: cv::subtract(a, b, dst, mask); break;
    case Op::multiply:
      if (scale == 1.0 && a.type() == CV_16UC1 && b.type() == CV_16UC1 &&
          a.rows == 1 && a.cols == 1 && b.size() == a.size()) {
        // OpenCV 5.0.0 can return zero for this scalar shape on overflow
        // even with CPU optimizations disabled. Use unsigned saturation.
        const auto product = std::uint32_t{a.at<std::uint16_t>(0, 0)} *
                             b.at<std::uint16_t>(0, 0);
        constexpr std::uint32_t max_u16 = std::numeric_limits<std::uint16_t>::max();
        dst.create(1, 1, CV_16UC1);
        dst.at<std::uint16_t>(0, 0) = static_cast<std::uint16_t>(std::min(product, max_u16));
      } else {
        cv::multiply(a, b, dst, scale);
      }
      break;
    case Op::bitwise_and: cv::bitwise_and(a, b, dst, mask); break;
    case Op::bitwise_not: cv::bitwise_not(a, dst, mask); break;
    case Op::flip: cv::flip(a, dst, flip_code); break;
  }
}

inline flagcv::Depth depth_for(int depth) {
  switch (depth) {
    case CV_8U: return flagcv::Depth::k8U;
    case CV_8S: return flagcv::Depth::k8S;
    case CV_16U: return flagcv::Depth::k16U;
    case CV_16S: return flagcv::Depth::k16S;
    case CV_32S: return flagcv::Depth::k32S;
    case CV_16F: return flagcv::Depth::k16F;
    case CV_32F: return flagcv::Depth::k32F;
    case CV_64F: return flagcv::Depth::k64F;
  }
  throw std::invalid_argument("unsupported test depth");
}

}  // namespace flagcv_test
