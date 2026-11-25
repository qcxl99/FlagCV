/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/core.hpp>

#include "src/runtime/runtime.hpp"

namespace flagcv {

void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, nullptr, PointwiseMode::kAnd,
                   stream);
}

void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, const GpuMat& mask, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, &mask, PointwiseMode::kAnd, stream);
}

void bitwise_not(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 Stream stream) {
  launch_pointwise(handle, src, src, dst, nullptr, PointwiseMode::kNot, stream);
}

void bitwise_not(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 const GpuMat& mask, Stream stream) {
  launch_pointwise(handle, src, src, dst, &mask, PointwiseMode::kNot, stream);
}

}  // namespace flagcv
