/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/lowering/lowering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/lowering/helpers.hpp"

namespace flagcv {
namespace {

int interpolation_code(Interpolation value) {
  switch (value) {
    case Interpolation::kNearest: return 0;
    case Interpolation::kLinear:
    case Interpolation::kArea: return 1;
    case Interpolation::kCubic: return 2;
  }
  throw std::invalid_argument("unsupported geometric interpolation");
}

int border_code(Border value) {
  switch (value) {
    case Border::kConstant: return 0;
    case Border::kReplicate: return 1;
    case Border::kReflect: return 2;
    case Border::kWrap: return 3;
    case Border::kReflect101: return 4;
    case Border::kTransparent: return 5;
  }
  throw std::invalid_argument("unsupported geometric border");
}

LoweredOp geometry(const char* kernel, const GpuMat& src, const GpuMat& dst) {
  LoweredOp op;
  op.kernel = kernel;
  op.images = {to_spec(src, "src"), to_spec(dst, "dst")};
  if (src.depth() != dst.depth() || src.channels() != dst.channels())
    throw std::invalid_argument("geometric source and destination pixel types must match");
  if ((src.depth() != Depth::k8U && src.depth() != Depth::k16U &&
       src.depth() != Depth::k32F) ||
      (src.channels() != 1 && src.channels() != 3 && src.channels() != 4))
    throw std::invalid_argument("geometry supports 8U/16U/32F with 1, 3 or 4 channels");
  if (images_overlap(src, dst))
    throw std::invalid_argument("geometric source and destination views overlap");
  return op;
}

void add_sampling(LoweredOp& op, Interpolation interpolation, Border border,
                  const BorderValue& values, bool hint_strides = false) {
  const auto& s = op.images[0];
  const auto& d = op.images[1];
  op.add_param("src_h", s.height);
  op.add_param("src_w", s.width);
  op.add_param("dst_h", d.height);
  op.add_param("dst_w", d.width);
  if (hint_strides) {
    op.add_hinted_param("src_rs", s.row_stride);
    op.add_hinted_param("dst_rs", d.row_stride);
  } else {
    op.add_param("src_rs", s.row_stride);
    op.add_param("dst_rs", d.row_stride);
  }
  constexpr const char* names[] = {"bv0", "bv1", "bv2", "bv3"};
  for (std::size_t channel = 0; channel < values.size(); ++channel) {
    double value = values[channel];
    if (!std::isfinite(value) ||
        (s.depth == Depth::k32F && std::abs(value) > std::numeric_limits<float>::max()))
      throw std::invalid_argument("geometric border values must be finite and representable");
    if (s.depth != Depth::k32F) {
      const double floor = std::floor(value);
      const double fraction = value - floor;
      value = floor + (fraction > 0.5 || (fraction == 0.5 && std::fmod(floor, 2.0) != 0));
      value = value > std::numeric_limits<std::int32_t>::max()
                  ? 0.0 : std::clamp(value, 0.0, s.depth == Depth::k8U ? 255.0 : 65535.0);
    }
    op.add_fp32_param(names[channel], static_cast<float>(value));
  }
  op.add_constexpr("CN", s.channels);
  op.add_constexpr("INTER", interpolation_code(interpolation));
  op.add_constexpr("BORDER", border_code(border));
}

PerspectiveMatrix sampling_matrix(const PerspectiveMatrix& m, bool inverse_map) {
  for (const double entry : m) {
    if (!std::isfinite(entry))
      throw std::invalid_argument("warpPerspective matrix entries must be finite");
  }
  if (inverse_map) return m;
  PerspectiveMatrix adjugate = {
      m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
      m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
      m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]};
  const double determinant = m[0] * adjugate[0] -
                             m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * adjugate[6];
  if (determinant == 0) return {};
  if (!std::isfinite(determinant))
    throw std::invalid_argument("warpPerspective determinant overflows");
  const double reciprocal = 1.0 / determinant;
  for (double& entry : adjugate) entry *= reciprocal;
  return adjugate;
}

bool map_type(const GpuMat& map, Depth depth, int channels) {
  return !map.empty() && map.depth() == depth && map.channels() == channels;
}

}  // namespace

LoweredOp lower_warp_perspective(const GpuMat& src, const GpuMat& dst,
                                 const PerspectiveMatrix& matrix,
                                 Interpolation interpolation, Border border,
                                 const BorderValue& border_value, bool inverse_map) {
  // Perspective has no signed-short map dimension limit. The fixed CPU's
  // SIMD source byte offsets and cubic signed coordinates are int32.
  constexpr auto index_limit = std::numeric_limits<std::int32_t>::max();
  constexpr int axis_limit = index_limit / 2;
  auto op = geometry("warp_perspective", src, dst);
  if (src.width() >= axis_limit || src.height() >= axis_limit ||
      dst.width() > (index_limit - 128) / dst.channels())
    throw std::invalid_argument("warpPerspective view exceeds coordinate index range");
  const auto source_span = (src.height() - 1) * src.row_pitch() +
                           std::int64_t{src.width()} * src.channels() * depth_size(src.depth());
  if (source_span > index_limit)
    throw std::invalid_argument("warpPerspective source exceeds SIMD byte offset range");
  const auto& source = op.images[0];
  const auto& destination = op.images[1];
  const auto destination_row_bytes =
      std::int64_t{dst.width()} * dst.channels() * depth_size(dst.depth());
  const bool bounded_destination = destination_row_bytes <= index_limit &&
      dst.height() - 1 <= (index_limit - destination_row_bytes) / dst.row_pitch();
  const bool vector_c4 = source.depth == Depth::k32F && source.channels == 4 &&
                         interpolation == Interpolation::kCubic && border == Border::kReplicate &&
                         destination.width == 3840 && destination.height == 2160 &&
                         bounded_destination &&
                         source.address % 16 == 0 && destination.address % 16 == 0 &&
                         source.row_stride % 16 == 0 && destination.row_stride % 16 == 0;
  if (vector_c4) op.kernel = "warp_perspective_c4";
  const auto sampling = sampling_matrix(matrix, inverse_map);
  constexpr const char* names[] = {
      "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22"};
  for (std::size_t i = 0; i < sampling.size(); ++i) {
    const double value = sampling[i];
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
      throw std::invalid_argument("warpPerspective sampling coefficients exceed finite float range");
    op.add_fp32_param(names[i], static_cast<float>(value));
  }
  add_sampling(op, interpolation, border, border_value, vector_c4);
  return op;
}

LoweredOp lower_remap(const GpuMat& src, const GpuMat& dst, const GpuMat& map1,
                      const GpuMat& map2, Interpolation interpolation,
                      Border border, const BorderValue& border_value, bool relative_map) {
  auto op = geometry("remap", src, dst);
  if (src.width() >= 32767 || src.height() >= 32767 ||
      dst.width() >= 32767 || dst.height() >= 32767)
    throw std::invalid_argument("remap image dimensions must be less than 32767");
  const GpuMat* first = &map1;
  const GpuMat* second = &map2;
  const int inter = interpolation_code(interpolation);
  if (map1.empty()) throw std::invalid_argument("remap first map must not be empty");
  if (inter != 2 && map_type(map2, Depth::k16S, 2)) std::swap(first, second);

  int layout;
  if (map_type(*first, Depth::k32F, 2) && second->empty()) {
    layout = 0;
  } else if (map_type(*first, Depth::k32F, 1) && map_type(*second, Depth::k32F, 1)) {
    layout = 1;
  } else if (map_type(*first, Depth::k16S, 2) &&
             (map_type(*second, Depth::k16U, 1) || map_type(*second, Depth::k16S, 1))) {
    layout = 2;
  } else if (map_type(*first, Depth::k16S, 2) && second->empty() && inter == 0) {
    layout = 3;
  } else {
    throw std::invalid_argument("remap map types do not match an accepted format");
  }

  // A single-map kernel specialization removes every read through map2.
  if (second->empty()) second = first;
  op.add_image(to_spec(*first, "map1"));
  op.add_image(to_spec(*second, "map2"));
  for (auto map : {first, second}) {
    if (map->width() != dst.width() || map->height() != dst.height())
      throw std::invalid_argument("remap maps must match the destination extent");
    if (images_overlap(*map, dst))
      throw std::invalid_argument("remap map and destination views overlap");
  }
  op.add_param("map1_rs", op.images[2].row_stride);
  op.add_param("map2_rs", op.images[3].row_stride);
  add_sampling(op, interpolation, border, border_value);
  op.add_constexpr("LAYOUT", layout);
  op.add_constexpr("RELATIVE", relative_map);
  return op;
}

}  // namespace flagcv
