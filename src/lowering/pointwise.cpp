/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "src/lowering/helpers.hpp"
#include "src/lowering/lowering.hpp"

namespace flagcv {
namespace {
void require_safe_alias(const GpuMat& src, const GpuMat& dst) {
  if (!same_view(src, dst) && images_overlap(src, dst)) {
    throw std::invalid_argument("pointwise does not support partial overlap");
  }
}

const char* kernel_name(PointwiseMode mode, bool masked) {
  switch (mode) {
    case PointwiseMode::kAdd:
      return masked ? "add_masked" : "add";
    case PointwiseMode::kSubtract:
      return masked ? "subtract_masked" : "subtract";
    case PointwiseMode::kMultiply:
      return "multiply";
    case PointwiseMode::kAnd:
      return masked ? "bitwise_and_masked" : "bitwise_and";
    case PointwiseMode::kNot:
      return masked ? "bitwise_not_masked" : "bitwise_not";
  }
  throw std::invalid_argument("unknown pointwise mode");
}
}  // namespace

LoweredOp lower_pointwise(const GpuMat& src1, const GpuMat& src2,
                          const GpuMat& dst, const GpuMat* mask,
                          PointwiseMode mode, double scale) {
  const ImageSpec a = to_spec(src1, "src1");
  const ImageSpec b = to_spec(src2, "src2");
  const ImageSpec d = to_spec(dst, "dst");
  require_same_layout(a, b);
  require_same_layout(a, d);
  if (mode <= PointwiseMode::kMultiply && a.depth != Depth::k8U &&
      a.depth != Depth::k16U && a.depth != Depth::k16F &&
      a.depth != Depth::k32F) {
    throw std::invalid_argument("arithmetic supports 8U, 16U, 16F and 32F");
  }
  if (scale != 1.0 &&
      (a.depth != Depth::k32F || !std::isfinite(scale) ||
       std::abs(scale) > std::numeric_limits<float>::max())) {
    throw std::invalid_argument("multiply supports finite 32F scale or unit scale");
  }
  if (a.channels > 4 || std::int64_t{a.width} * a.channels >
                            std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument(
        "pointwise requires 1-4 channels and an i32 row width");
  }
  require_safe_alias(src1, dst);
  require_safe_alias(src2, dst);

  const bool has_mask = mask != nullptr && !mask->empty();
  LoweredOp op;
  op.kernel = kernel_name(mode, has_mask);
  op.images.reserve(has_mask ? 4 : 3);
  op.params.reserve(9);
  op.add_image(a);
  op.add_image(b);
  op.add_image(d);
  ImageSpec m;
  if (has_mask) {
    m = to_spec(*mask, "mask");
    require_mask(m, d);
    if (images_overlap(*mask, dst)) {
      throw std::invalid_argument("mask must be disjoint from the destination");
    }
    op.add_image(m);
  }
  op.add_hinted_param("width", a.width);
  op.add_param("height", a.height);
  op.add_hinted_param("src1_row_stride", a.row_stride);
  op.add_hinted_param("src2_row_stride", b.row_stride);
  op.add_hinted_param("dst_row_stride", d.row_stride);
  if (has_mask) op.add_param("mask_row_stride", m.row_stride);
  if (mode == PointwiseMode::kMultiply) {
    const float value = static_cast<float>(scale);
    std::int32_t bits;
    static_assert(sizeof(value) == sizeof(bits), "32-bit float required");
    std::memcpy(&bits, &value, sizeof(bits));
    op.add_param("scale_bits", bits);
  }
  op.add_constexpr("CHANNELS", a.channels);
  if (mode != PointwiseMode::kMultiply)
    op.add_constexpr("MODE", static_cast<int>(mode));
  if (!has_mask) {
    const auto row_elements = std::int64_t{a.width} * a.channels;
    op.add_constexpr("PACKED", a.row_stride == row_elements &&
                                   b.row_stride == row_elements &&
                                   d.row_stride == row_elements);
    if (mode != PointwiseMode::kMultiply)
      op.add_constexpr("VEC4", mode > PointwiseMode::kMultiply &&
                                   can_pack_u8x4(a) && can_pack_u8x4(b) &&
                                   can_pack_u8x4(d));
  }
  return op;
}

}  // namespace flagcv
