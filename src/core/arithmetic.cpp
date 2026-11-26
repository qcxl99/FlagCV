/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/core.hpp>

#include "src/runtime/runtime.hpp"

namespace flagcv {

void add(Handle& handle, const GpuMat& src1, const GpuMat& src2,
         const GpuMat& dst, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, nullptr, PointwiseMode::kAdd,
                   stream);
}

void add(Handle& handle, const GpuMat& src1, const GpuMat& src2,
         const GpuMat& dst, const GpuMat& mask, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, &mask, PointwiseMode::kAdd,
                   stream);
}

void subtract(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, nullptr, PointwiseMode::kSubtract,
                   stream);
}

void subtract(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, const GpuMat& mask, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, &mask, PointwiseMode::kSubtract,
                   stream);
}

void multiply(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, Stream stream) {
  multiply(handle, src1, src2, dst, 1.0, stream);
}

void multiply(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, double scale, Stream stream) {
  launch_pointwise(handle, src1, src2, dst, nullptr, PointwiseMode::kMultiply,
                   stream, scale);
}

void add(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst,
         Stream stream) {
  add(detail::currentHandle(), src1, src2, dst, stream);
}

void add(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst,
         const GpuMat& mask, Stream stream) {
  add(detail::currentHandle(), src1, src2, dst, mask, stream);
}

void subtract(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst,
              Stream stream) {
  subtract(detail::currentHandle(), src1, src2, dst, stream);
}

void subtract(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst,
              const GpuMat& mask, Stream stream) {
  subtract(detail::currentHandle(), src1, src2, dst, mask, stream);
}

void multiply(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst,
              Stream stream) {
  multiply(detail::currentHandle(), src1, src2, dst, stream);
}

void multiply(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst,
              double scale, Stream stream) {
  multiply(detail::currentHandle(), src1, src2, dst, scale, stream);
}

}  // namespace flagcv
