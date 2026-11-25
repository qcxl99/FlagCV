/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_IMGPROC_HPP_
#define FLAGCV_IMGPROC_HPP_

#include <flagcv/gpumat.hpp>
#include <flagcv/handle.hpp>

#include <array>

/*
 * Colour conversion, geometric transforms, filtering, morphology,
 * thresholding, histograms. Mirrors opencv2/imgproc.hpp.
 */
namespace flagcv {

// A row-major 2x3 transform and a constant value per output channel.
using AffineMatrix = std::array<double, 6>;
using BorderValue = std::array<double, 4>;

/*
 * Sample src into the preallocated dst view. Use dst dimensions to set the
 * output size. With inverse_map, matrix maps dst coordinates directly to src.
 * Source and destination must have the same pixel type and disjoint storage.
 *
 * Supports 8U, 16U and 32F with 1, 3 or 4 packed channels and independent
 * row pitches. Area is an alias for linear. With a transparent border,
 * initialized destination pixels supply missing linear taps and wholly
 * outside pixels retain their previous value. Cubic disallows transparency.
 * Keep the handle and views alive until the supplied stream completes.
 */
void warpAffine(Handle& handle, const GpuMat& src, const GpuMat& dst,
                const AffineMatrix& matrix,
                Interpolation interpolation = Interpolation::kLinear,
                Border border = Border::kConstant,
                const BorderValue& border_value = {},
                bool inverse_map = false, Stream stream = nullptr);

}  // namespace flagcv

#endif  // FLAGCV_IMGPROC_HPP_
