/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/lowering/lowering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/lowering/helpers.hpp"

namespace flagcv {
namespace {

AffineMatrix sampling_matrix(const AffineMatrix& matrix, bool inverse_map,
                             const ImageSpec& dst, Interpolation interpolation) {
  for (double entry : matrix) {
    if (!std::isfinite(entry))
      throw std::invalid_argument("warpAffine matrix entries must be finite");
  }

  AffineMatrix sampling = matrix;
  if (!inverse_map) {
    const double det = matrix[0] * matrix[4] - matrix[1] * matrix[3];
    const double inv_det = det != 0 ? 1.0 / det : 0.0;
    sampling[0] = matrix[4] * inv_det;
    sampling[1] = -matrix[1] * inv_det;
    sampling[3] = -matrix[3] * inv_det;
    sampling[4] = matrix[0] * inv_det;
    sampling[2] = -sampling[0] * matrix[2] - sampling[1] * matrix[5];
    sampling[5] = -sampling[3] * matrix[2] - sampling[4] * matrix[5];
  }

  constexpr double kFloatLimit = std::numeric_limits<float>::max();
  constexpr double kRoundingMargin = 1.0 + 8.0 * std::numeric_limits<float>::epsilon();
  // Nearest and linear round or truncate to i32, then linear adds one tap.
  constexpr double kCoordinateLimit = 2147483392.0;
  for (double entry : sampling) {
    if (!std::isfinite(entry) || std::abs(entry) > kFloatLimit)
      throw std::invalid_argument("warpAffine sampling matrix exceeds finite float32 range");
  }
  const double limit = interpolation == Interpolation::kCubic ? kFloatLimit : kCoordinateLimit;
  for (int row : {0, 3}) {
    const double bound = std::abs(sampling[row]) * (dst.width - 1) +
                         std::abs(sampling[row + 1]) * (dst.height - 1) +
                         std::abs(sampling[row + 2]);
    if (bound > limit / kRoundingMargin)
      throw std::invalid_argument("warpAffine sampling coordinates exceed kernel index range");
  }
  return sampling;
}

void validate_span(const ImageSpec& image) {
  constexpr auto limit = std::numeric_limits<std::int32_t>::max();
  const auto row_values = std::int64_t{image.width} * image.channels;
  if (row_values > limit || image.row_stride > limit ||
      (image.height - 1) * image.row_stride + row_values > limit)
    throw std::invalid_argument("warpAffine view exceeds int32 element indexing");
}

int border_code(Border border) {
  // The common kernel uses the OpenCV border numbers.
  switch (border) {
    case Border::kConstant: return 0;
    case Border::kReplicate: return 1;
    case Border::kReflect: return 2;
    case Border::kWrap: return 3;
    case Border::kReflect101: return 4;
    case Border::kTransparent: return 5;
  }
  throw std::invalid_argument("warpAffine unsupported border mode");
}

}  // namespace

LoweredOp lower_warp_affine(const GpuMat& src, const GpuMat& dst,
                           const AffineMatrix& matrix, Interpolation interpolation,
                           Border border, const BorderValue& border_value,
                           bool inverse_map) {
  if ((src.depth() != Depth::k8U && src.depth() != Depth::k16U && src.depth() != Depth::k32F) ||
      (src.channels() != 1 && src.channels() != 3 && src.channels() != 4))
    throw std::invalid_argument("warpAffine supports 8U/16U/32F and 1, 3 or 4 channels");
  if (src.depth() != dst.depth() || src.channels() != dst.channels())
    throw std::invalid_argument("warpAffine source and destination pixel types must match");
  const auto s = to_spec(src, "src");
  const auto d = to_spec(dst, "dst");
  validate_span(s);
  validate_span(d);
  if (images_overlap(src, dst))
    throw std::invalid_argument("warpAffine does not support overlapping views");

  const char* kernel = nullptr;
  switch (interpolation) {
    case Interpolation::kNearest: kernel = "warp_affine_nearest"; break;
    case Interpolation::kArea:
    case Interpolation::kLinear: kernel = "warp_affine_linear"; break;
    case Interpolation::kCubic: kernel = "warp_affine_cubic"; break;
    default: throw std::invalid_argument("warpAffine unsupported interpolation");
  }
  const int mode = border_code(border);
  const bool transparent = border == Border::kTransparent;
  if (transparent && interpolation == Interpolation::kCubic)
    throw std::invalid_argument("warpAffine cubic interpolation does not support transparent border");

  const auto sampling = sampling_matrix(matrix, inverse_map, d, interpolation);
  LoweredOp op;
  op.kernel = kernel;
  op.images = {s, d};
  op.params.reserve(24);
  constexpr const char* matrix_names[] = {"M00", "M01", "M02", "M10", "M11", "M12"};
  for (std::size_t i = 0; i < sampling.size(); ++i)
    op.add_fp32_param(matrix_names[i], static_cast<float>(sampling[i]));
  op.add_param("src_h", s.height);
  op.add_param("src_w", s.width);
  op.add_param("dst_h", d.height);
  op.add_param("dst_w", d.width);
  op.add_param("src_rs", s.row_stride);
  op.add_param("dst_rs", d.row_stride);

  constexpr const char* border_names[] = {"bv0", "bv1", "bv2", "bv3"};
  for (std::size_t i = 0; i < border_value.size(); ++i) {
    double value = border_value[i];
    if (!std::isfinite(value) ||
        (s.depth == Depth::k32F && std::abs(value) > std::numeric_limits<float>::max()))
      throw std::invalid_argument("warpAffine border values must be finite and representable");
    if (s.depth != Depth::k32F) {
      const double maximum = s.depth == Depth::k8U ? 255.0 : 65535.0;
      value = std::clamp(value, 0.0, maximum);
      const double lower = std::floor(value);
      const double fraction = value - lower;
      value = lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0));
    }
    op.add_fp32_param(border_names[i], static_cast<float>(value));
  }

  op.add_constexpr("cn", s.channels);
  if (interpolation != Interpolation::kNearest) {
    op.add_constexpr("U8", s.depth == Depth::k8U);
    op.add_constexpr("U16", s.depth == Depth::k16U);
  }
  op.add_constexpr("BORDER", mode);
  const bool replicate_single_axis = border != Border::kConstant && !transparent;
  const int border_x = replicate_single_axis && s.width == 1 ? 1 : mode;
  const int border_y = replicate_single_axis && s.height == 1 ? 1 : mode;
  if (interpolation != Interpolation::kNearest)
    op.add_constexpr("TRANSPARENT", transparent);
  op.add_constexpr("BORDER_X", border_x);
  op.add_constexpr("BORDER_Y", border_y);
  // Fixed OpenCV 5.0.0 SSE3 reference profile verified with tie-coordinate probes.
  op.add_constexpr("FUSED", false);
  if (interpolation == Interpolation::kNearest)
    op.add_constexpr("TRANSPARENT", transparent);
  return op;
}

}  // namespace flagcv
