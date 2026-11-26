/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */

#include "src/lowering/lowering.hpp"

#include "src/imgproc/spatial_filters.hpp"
#include "src/lowering/helpers.hpp"

#include <algorithm>
#include <cmath>
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

struct DftAxis {
  int transform;
  int block;
};

DftAxis dft_axis(int extent, int kernel, int minimum) {
  // The fixed CPU crossCorr uses 4.5 taps rounded to even, a
  // 256-element minimum-block heuristic, and a 2/3/5-smooth transform length.
  const std::int64_t twice = std::int64_t{9} * kernel;
  const auto rounded = twice / 2 + ((twice & 1) && ((twice / 2) & 1));
  const auto requested = std::min<std::int64_t>(extent, std::max(rounded, std::int64_t{257} - kernel));
  for (auto size = std::max<std::int64_t>(minimum, requested + kernel - 1);
       size <= std::numeric_limits<int>::max(); ++size) {
    auto remainder = size;
    for (int factor : {2, 3, 5})
      while (remainder % factor == 0) remainder /= factor;
    if (remainder == 1)
      return {static_cast<int>(size), static_cast<int>(std::min<std::int64_t>(extent, size - kernel + 1))};
  }
  return {};
}

Size dft_vector_extent(const GpuMat& dst, Size kernel) {
  const auto x = dft_axis(dst.width(), kernel.width, 2);
  if (x.block == 0) return {-1, -1};
  const int edge = dst.width() % x.block;
  // For >=8 elements the final SSE vector starts at tile width minus eight,
  // including the final columns even when the tile width is not a multiple of eight.
  if (x.block >= 8) return {edge > 0 && edge < 8 ? dst.width() - edge : dst.width(), dst.height()};
  // A cropped DFT plane cannot flatten unless the tile spans its full row.
  if (x.block != x.transform || dst.width() != x.block ||
      dst.row_pitch() != std::int64_t{x.block} * std::int64_t{sizeof(std::uint16_t)})
    return {0, 0};
  const auto y = dft_axis(dst.height(), kernel.height, 1);
  if (y.block == 0) return {-1, -1};
  const int min_rows = (8 + x.block - 1) / x.block;
  if (y.block < min_rows) return {0, 0};
  const int bottom = dst.height() % y.block;
  return {dst.width(), bottom > 0 && bottom < min_rows ? dst.height() - bottom : dst.height()};
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
  const bool local_f32 = data.depth == Depth::k32F && (src.channels() == 3 || src.channels() == 4) &&
                         data.kernel.width == data.kernel.height &&
                         (data.kernel.width == 7 || data.kernel.width == 15) &&
                         src.width() >= 1280 && src.width() <= 3840 &&
                         src.height() >= 720 && src.height() <= 2160;
  auto op = prepare(local_f32 ? "gaussian_f32_local" : "gaussian_direct",
                    src, dst, data.depth, data.depth, workspace, plan.table().bytes);
  op.add_param("kernel_width", data.kernel.width);
  op.add_param("kernel_height", data.kernel.height);
  common_constants(op, src.channels(), data.border, data.depth == Depth::k64F);
  op.add_constexpr("FIXED_BITS", data.depth == Depth::k8U ? 8 : data.depth == Depth::k16U ? 16 : 0);
  if (local_f32) op.add_constexpr("TAPS", data.kernel.width);
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
  // GpuMat denotes an isolated view. Match the fixed CPU DFT eligibility
  // only for depth pairs whose accumulation and conversion were qualified.
  const auto channels = src.channels();
  const auto area = std::int64_t{data.kernel.width} * data.kernel.height;
  const bool u16 = data.source_depth == Depth::k16U;
  const bool dst_u16 = data.destination_depth == Depth::k16U;
  const bool dst_f32 = data.destination_depth == Depth::k32F;
  const double delta = data.coefficients.front();
  // With zero delta, multichannel U16 DFT converts the double correlation
  // directly to U16. Only its nonzero-delta C3 path uses the F32 temporary.
  const bool u16_route = u16 && ((dst_u16 && (channels == 1 || !data.has_nonzero_delta)) ||
                                 (channels == 3 && data.has_nonzero_delta && (dst_u16 || dst_f32)));
  const bool dft = area >= 50 && !data.fp32_integer_exact && u16_route &&
                   (data.fp64_u16_exact || data.fp64_u16_pair_exact);
  const bool use_u8_wide = channels == 3 && data.u8_dominant_exact;
  const bool scaled = use_u8_wide ? data.has_nonzero_delta :
      channels == 1 && std::abs(delta) >= std::numeric_limits<double>::epsilon();
  const Size vector = dft && scaled ? dft_vector_extent(dst, data.kernel) : Size{0, 0};
  const bool use_dft = use_u8_wide || (dft && vector.width >= 0);
  const bool use_exact = !use_dft && area >= 50 && u16 && dst_u16 && !data.has_nonzero_delta &&
                         !data.fp32_integer_exact && data.u16_exact_unit != std::numeric_limits<int>::max();
  const bool dense_pair_candidate =
      data.source_depth == Depth::k32F && data.destination_depth == Depth::k32F &&
      data.border == Border::kReflect101 && data.all_normal_f32_taps && src.channels() == 3 &&
      src.width() >= 256 && src.height() >= 128 &&
      data.kernel.width >= 3 && data.kernel.width <= 9 &&
      data.kernel.height >= 3 && data.kernel.height <= 9;
  const bool interior_candidate =
      data.source_depth == Depth::k32F && data.destination_depth == Depth::k32F &&
      data.border == Border::kReflect101 && !data.zero_filter && src.channels() == 4 &&
      src.width() >= 1920 && src.height() >= 1080 &&
      data.kernel.width == 7 && data.kernel.height == 7;
  const char* kernel = use_dft ? "filter2d_dft_precision" :
                       use_exact ? "filter2d_u16_exact" :
                       dense_pair_candidate ? "filter2d_dense" :
                       interior_candidate ? "filter2d_interior" : "filter2d_direct";
  auto op = prepare(kernel,
                    src, dst, data.source_depth, data.destination_depth, workspace, plan.table().bytes);
  if (dense_pair_candidate)
    op.add_param("row_values", std::int64_t{src.width()} * src.channels());
  op.add_param("kernel_width", data.kernel.width);
  op.add_param("kernel_height", data.kernel.height);
  op.add_param("anchor_x", data.anchor.x);
  op.add_param("anchor_y", data.anchor.y);
  if (use_exact) {
    op.add_param("unit_exponent", data.u16_exact_unit);
    op.add_constexpr("CHANNELS", channels);
    op.add_constexpr("BORDER", border_code(data.border));
    return op;
  }
  if (use_dft) {
    op.add_param("vector_columns", vector.width);
    op.add_param("vector_rows", vector.height);
    op.add_constexpr("CHANNELS", channels);
    op.add_constexpr("BORDER", border_code(data.border));
    op.add_constexpr("ADD_F32_DELTA", !use_u8_wide && channels != 1 && data.has_nonzero_delta);
    op.add_constexpr("SCALED", scaled);
    op.add_constexpr("DEST_U16", dst_u16);
    op.add_constexpr("COMPENSATED", !use_u8_wide && !data.fp64_u16_exact);
    return op;
  }
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
