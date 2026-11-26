/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_CORE_HPP_
#define FLAGCV_CORE_HPP_

#include <flagcv/gpumat.hpp>
#include <flagcv/handle.hpp>

/*
 * Per-element arithmetic and logic, comparison, reduction, channel handling.
 * Mirrors opencv2/core.hpp, so a declaration added here keeps the OpenCV name
 * and argument order.
 *
 * Signature convention:
 *
 *   void add(Handle& handle,
 *            const GpuMat& src1,
 *            const GpuMat& src2,
 *            const GpuMat& dst,
 *            Stream stream = nullptr);
 *
 * `handle` leads because it carries the backend and the kernel cache;
 * `stream` trails with a default so the common call reads like OpenCV. Calls
 * enqueue on the supplied stream. The caller keeps buffers, stream and Handle
 * alive until the work completes. The allocation and stream must belong to
 * the Handle's device. See docs/cpp-operators.md.
 */
namespace flagcv {

/*
 * These array-array overloads accept matching extents, depths and 1-4
 * channels. Pitches may differ. Arithmetic supports 8U, 16U, 16F and 32F.
 * Bitwise operators act on the underlying bits of every Depth.
 * The caller preallocates dst; an exact source/dst alias is supported.
 * A mask is an optional 8U, single-channel image with matching extents.
 * Only pixels with nonzero mask values are written.
 */
void add(Handle& handle, const GpuMat& src1, const GpuMat& src2,
         const GpuMat& dst, Stream stream = nullptr);
void add(Handle& handle, const GpuMat& src1, const GpuMat& src2,
         const GpuMat& dst, const GpuMat& mask, Stream stream = nullptr);
void subtract(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, Stream stream = nullptr);
void subtract(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, const GpuMat& mask, Stream stream = nullptr);

void multiply(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, Stream stream = nullptr);
/* Explicit finite scale is supported for 32F. Other depths use unit scale. */
void multiply(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, double scale, Stream stream = nullptr);

void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, Stream stream = nullptr);
void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, const GpuMat& mask,
                 Stream stream = nullptr);
void bitwise_not(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 Stream stream = nullptr);
void bitwise_not(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 const GpuMat& mask, Stream stream = nullptr);

/*
 * Mirrors cv::cuda::flip(src, dst, flipCode, stream).
 *
 * flip_code == 0 flips around the x-axis (rows reversed), > 0 around the
 * y-axis (columns reversed), < 0 around both. Every depth and 1-4 channels
 * are supported; pixels move bit-for-bit. `dst` must match `src` in extent,
 * depth and channels. Exact in-place flipping is supported. Distinct views
 * may use different pitches, and their pixel regions must be disjoint.
 */
void flip(Handle& handle, const GpuMat& src, const GpuMat& dst, int flip_code,
          Stream stream = nullptr);

}  // namespace flagcv

#endif  // FLAGCV_CORE_HPP_
