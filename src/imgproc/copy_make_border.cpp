/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <flagcv/imgproc.hpp>

#include "src/lowering/helpers.hpp"
#include "src/lowering/scalar_bits.hpp"
#include "src/runtime/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace flagcv {
namespace {

int border_mode(Border border) {
  switch (border) {
    case Border::kConstant: return 0;
    case Border::kReplicate: return 1;
    case Border::kReflect: return 2;
    case Border::kReflect101: return 3;
    case Border::kWrap: return 4;
    case Border::kTransparent: break;
  }
  throw std::invalid_argument("copyMakeBorder unsupported border mode");
}

GpuMat expand_view(const GpuMat& src, const BorderParent& parent, BorderExtents& padding) {
  const auto p = to_spec(parent.image, "parent");
  if (p.depth != src.depth() || p.channels != src.channels() ||
      parent.image.row_pitch() != src.row_pitch() ||
      parent.x < 0 || parent.y < 0 || parent.x > p.width - src.width() ||
      parent.y > p.height - src.height())
    throw std::invalid_argument("copyMakeBorder source does not fit the declared parent");

  const auto pixel = depth_size(p.depth) * p.channels;
  const auto pitch = static_cast<std::uintptr_t>(parent.image.row_pitch());
  const auto expected = p.address + std::uintptr_t(parent.y) * pitch + std::uintptr_t(parent.x) * pixel;
  if (expected != reinterpret_cast<std::uintptr_t>(src.data()))
    throw std::invalid_argument("copyMakeBorder parent origin does not identify source pixels");

  const int top = std::min(parent.y, padding.top);
  const int bottom = std::min(p.height - src.height() - parent.y, padding.bottom);
  const int left = std::min(parent.x, padding.left);
  const int right = std::min(p.width - src.width() - parent.x, padding.right);
  padding.top -= top;
  padding.bottom -= bottom;
  padding.left -= left;
  padding.right -= right;
  const auto start = expected - std::uintptr_t(top) * pitch - std::uintptr_t(left) * pixel;
  return {reinterpret_cast<void*>(start), src.width() + left + right, src.height() + top + bottom,
          src.depth(), src.channels(), src.row_pitch()};
}

}  // namespace

void copyMakeBorder(Handle& handle, const GpuMat& src, const GpuMat& dst, BorderExtents padding,
                    Border border, const BorderValue& value, const BorderParent* parent,
                    Stream stream) {
  const auto source = to_spec(src, "src");
  const auto output = to_spec(dst, "dst");
  const int mode = border_mode(border);
  const auto expected_w = std::int64_t{source.width} + padding.left + padding.right;
  const auto expected_h = std::int64_t{source.height} + padding.top + padding.bottom;
  if (padding.top < 0 || padding.bottom < 0 || padding.left < 0 || padding.right < 0 ||
      expected_w != output.width || expected_h != output.height ||
      source.depth != output.depth || source.channels != output.channels || source.channels > 4)
    throw std::invalid_argument("copyMakeBorder invalid padding or destination view");

  const auto row_values = std::int64_t{output.width} * output.channels;
  if (row_values > std::numeric_limits<std::int32_t>::max() ||
      output.height > std::numeric_limits<std::int32_t>::max() / row_values)
    throw std::invalid_argument("copyMakeBorder output exceeds int32 element indexing");

  const GpuMat effective = parent ? expand_view(src, *parent, padding) : src;
  if (same_view(effective, dst) && effective.width() == dst.width() &&
      effective.height() == dst.height())
    return;
  if (images_overlap(effective, dst))
    throw std::invalid_argument("copyMakeBorder destination overlaps the sampled source");

  LoweredOp op;
  op.kernel = "copy_make_border";
  op.images = {to_spec(effective, "src"), output};
  op.add_param("src_w", effective.width());
  op.add_param("src_h", effective.height());
  op.add_param("dst_h", output.height);
  op.add_param("row_values", row_values);
  op.add_param("src_rs", effective.row_stride());
  op.add_param("dst_rs", output.row_stride);
  op.add_param("top", padding.top);
  op.add_param("left", padding.left);
  constexpr const char* lows[] = {"lo0", "lo1", "lo2", "lo3"};
  constexpr const char* highs[] = {"hi0", "hi1", "hi2", "hi3"};
  for (std::size_t i = 0; i < value.size(); ++i)
    add_scalar_words(op, lows[i], highs[i], mode == 0 ? copy_border_scalar(output.depth, value[i]) : 0);
  op.add_constexpr("CHANNELS", output.channels);
  op.add_constexpr("MODE", mode);
  op.add_constexpr("BITS", 8 * depth_size(output.depth));
  launch(handle, std::move(op), stream);
}

}  // namespace flagcv
