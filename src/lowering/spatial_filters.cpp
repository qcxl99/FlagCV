/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */

#include "src/lowering/lowering.hpp"

#include "src/imgproc/spatial_filters.hpp"
#include "src/lowering/helpers.hpp"

#include <limits>
#include <stdexcept>

namespace flagcv {
namespace {

int border_code(Border border) {
  switch (border) {
    case Border::kConstant: return 0;
    case Border::kReplicate: return 1;
    case Border::kReflect: return 2;
    case Border::kReflect101: return 4;
    default: throw std::invalid_argument("unsupported spatial filter border");
  }
}

std::uintptr_t end_address(const ImageSpec& image) {
  constexpr auto limit = std::numeric_limits<std::int32_t>::max();
  if (image.row_stride > limit)
    throw std::invalid_argument("spatial filter row stride exceeds int32");
  const auto area = (image.height - 1) * image.row_stride +
                    std::int64_t{image.width} * image.channels;
  if (area > limit)
    throw std::invalid_argument("spatial filter image span exceeds int32");
  return image.address + area * depth_size(image.depth);
}

LoweredOp prepare(const char* name, const GpuMat& src, const GpuMat& dst,
                 Depth src_depth, Depth dst_depth, DeviceWorkspace workspace,
                 std::size_t coefficient_bytes) {
  const auto s = to_spec(src, "src");
  const auto d = to_spec(dst, "dst");
  if (s.width != d.width || s.height != d.height || s.channels != d.channels ||
      s.channels > 4 || s.depth != src_depth || d.depth != dst_depth)
    throw std::invalid_argument("spatial filter requires matching image extents/channels and plan depths");
  const auto s_end = end_address(s);
  const auto d_end = end_address(d);
  if (images_overlap(src, dst))
    throw std::invalid_argument("spatial filter source and destination overlap");
  const auto begin = reinterpret_cast<std::uintptr_t>(workspace.data);
  if (!begin || begin % alignof(double) || coefficient_bytes > workspace.bytes ||
      coefficient_bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
    throw std::invalid_argument("spatial filter coefficient workspace is missing, unaligned or too small");
  const auto end = begin + coefficient_bytes;
  if ((begin < s_end && s.address < end) || (begin < d_end && d.address < end))
    throw std::invalid_argument("spatial filter coefficient workspace overlaps an image");

  LoweredOp op;
  op.kernel = name;
  op.images = {s, d};
  ImageSpec coefficients;
  coefficients.name = "coeff";
  coefficients.address = begin;
  coefficients.depth = Depth::k64F;
  coefficients.channels = 1;
  op.add_image(coefficients);
  op.add_param("width", s.width);
  op.add_param("height", s.height);
  op.add_param("src_stride", s.row_stride);
  op.add_param("dst_stride", d.row_stride);
  return op;
}

void common_constants(LoweredOp& op, int channels, Border border, bool work_double) {
  op.add_constexpr("CHANNELS", channels);
  op.add_constexpr("BORDER", border_code(border));
  op.add_constexpr("FP64", work_double);
}

}  // namespace

LoweredOp lower_gaussian(const GpuMat& src, const GpuMat& dst,
                         const GaussianPlan& plan, DeviceWorkspace workspace) {
  const auto& data = SpatialFilterAccess::get(plan);
  if (src.width() != data.source.width || src.height() != data.source.height)
    throw std::invalid_argument("GaussianBlur source extents differ from plan");
  auto op = prepare("gaussian_direct", src, dst, data.depth, data.depth, workspace, plan.table().bytes);
  op.add_param("kernel_width", data.kernel.width);
  op.add_param("kernel_height", data.kernel.height);
  common_constants(op, src.channels(), data.border, data.depth == Depth::k64F);
  op.add_constexpr("FIXED_BITS", data.depth == Depth::k8U ? 8 : data.depth == Depth::k16U ? 16 : 0);
  return op;
}

std::array<LoweredOp, 2> lower_gaussian_separable(
    const GpuMat& src, const GpuMat& dst, const GaussianPlan& plan,
    DeviceWorkspace workspace, const GpuMat& scratch) {
  const auto& data = SpatialFilterAccess::get(plan);
  if (src.width() != data.source.width || src.height() != data.source.height)
    throw std::invalid_argument("GaussianBlur source extents differ from plan");
  const auto depth = plan.scratch_depth();
  auto first = prepare("gaussian_horizontal", src, scratch, data.depth, depth,
                       workspace, plan.table().bytes);
  auto second = prepare("gaussian_vertical", scratch, dst, depth, data.depth,
                        workspace, plan.table().bytes);
  if (images_overlap(src, dst))
    throw std::invalid_argument("GaussianBlur source and destination overlap");
  first.add_param("kernel_width", data.kernel.width);
  second.add_param("kernel_width", data.kernel.width);
  second.add_param("kernel_height", data.kernel.height);
  const int bits = data.depth == Depth::k8U ? 8 : data.depth == Depth::k16U ? 16 : 0;
  for (auto* op : {&first, &second}) {
    common_constants(*op, src.channels(), data.border, data.depth == Depth::k64F);
    op->add_constexpr("FIXED_BITS", bits);
  }
  return {std::move(first), std::move(second)};
}

LoweredOp lower_filter2d(const GpuMat& src, const GpuMat& dst,
                         const Filter2DPlan& plan, DeviceWorkspace workspace) {
  const auto& data = SpatialFilterAccess::get(plan);
  auto op = prepare("filter2d_direct", src, dst, data.source_depth, data.destination_depth,
                    workspace, plan.table().bytes);
  op.add_param("kernel_width", data.kernel.width);
  op.add_param("kernel_height", data.kernel.height);
  op.add_param("anchor_x", data.anchor.x);
  op.add_param("anchor_y", data.anchor.y);
  common_constants(op, src.channels(), data.border, data.destination_depth == Depth::k64F);
  op.add_constexpr("ZERO_FILTER", data.zero_filter);
  op.add_constexpr("DEST_INTEGER", is_integer(data.destination_depth));
  op.add_constexpr("DEST_MIN", data.destination_depth == Depth::k16S ? -32768 : 0);
  int maximum = 0;
  switch (data.destination_depth) {
    case Depth::k8U: maximum = 255; break;
    case Depth::k16S: maximum = 32767; break;
    case Depth::k16U: maximum = 65535; break;
    default: break;
  }
  op.add_constexpr("DEST_MAX", maximum);
  return op;
}

}  // namespace flagcv
