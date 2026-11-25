/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/lowering/lowering.hpp"

#include "src/imgproc/resize_data.hpp"
#include "src/lowering/helpers.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace flagcv {
namespace {

struct BatchSpan {
  GpuMat first;
  std::uintptr_t begin;
  std::uintptr_t end;
  std::uint64_t image_bytes;
  std::uint64_t pitch;
};

BatchSpan batch_span(const GpuMat& view, const ImageSpec& spec, int count,
                     std::int64_t pitch) {
  const auto elem = depth_size(spec.depth);
  const auto row_bytes = static_cast<std::uint64_t>(spec.width) * spec.channels * elem;
  const auto image_bytes = (spec.height - 1) * static_cast<std::uint64_t>(view.row_pitch()) + row_bytes;
  // The source kernels admit at most int32 packed elements per image view.
  constexpr auto limit = std::numeric_limits<std::int32_t>::max();
  if (spec.row_stride > limit || image_bytes / elem > limit)
    throw std::invalid_argument("resize image span exceeds int32 element indexing");

  if (pitch < 0 || pitch % elem != 0 ||
      (count > 1 && static_cast<std::uint64_t>(pitch) < image_bytes))
    throw std::invalid_argument("resize batch pitch is misaligned or overlaps items");
  const auto address = static_cast<std::uintptr_t>(spec.address);
  const auto available = std::min<std::uint64_t>(
      std::numeric_limits<std::uintptr_t>::max() - address,
      std::numeric_limits<std::int64_t>::max());
  const auto stride = static_cast<std::uint64_t>(pitch);
  if (image_bytes > available ||
      (count > 1 && stride > (available - image_bytes) / (count - 1)))
    throw std::invalid_argument("resize batch address range overflows");
  return {view, address, address + (count - 1) * stride + image_bytes, image_bytes, stride};
}

bool intervals_overlap(std::uintptr_t left, std::uintptr_t left_end,
                       std::uintptr_t right, std::uintptr_t right_end) {
  return left < right_end && right < left_end;
}

GpuMat item_view(const BatchSpan& batch, std::uintptr_t address) {
  const auto& view = batch.first;
  return {reinterpret_cast<void*>(address), view.width(), view.height(), view.depth(),
          view.channels(), view.row_pitch()};
}

bool batches_overlap(const BatchSpan& src, const BatchSpan& dst, int count) {
  if (!intervals_overlap(src.begin, src.end, dst.begin, dst.end))
    return false;
  int i = 0, j = 0;
  while (i < count && j < count) {
    const auto src_address = src.begin + i * src.pitch;
    const auto dst_address = dst.begin + j * dst.pitch;
    const auto src_end = src_address + src.image_bytes;
    const auto dst_end = dst_address + dst.image_bytes;
    if (intervals_overlap(src_address, src_end, dst_address, dst_end) &&
        images_overlap(item_view(src, src_address), item_view(dst, dst_address)))
      return true;
    if (src_end <= dst_end) ++i;
    if (dst_end <= src_end) ++j;
  }
  return false;
}

}  // namespace

LoweredOp lower_resize(const GpuMat& src, const GpuMat& dst, const ResizePlan& plan,
                        DeviceWorkspace workspace, int count,
                        std::int64_t src_pitch, std::int64_t dst_pitch, bool batch) {
  using namespace resize_tables;
  const auto& data = ResizeAccess::data(plan);
  if (count <= 0)
    throw std::invalid_argument("resize batch count must be positive");
  if (batch) {
    if (data.depth != Depth::k8U)
      throw std::invalid_argument("resizeBatch supports 8U only");
    const auto size = data.requested_size;
    if ((size.width <= 0 || size.height <= 0) && (size.width != 0 || size.height != 0))
      throw std::invalid_argument("resizeBatch needs a positive size or {0, 0} with scales");
  }
  if (src.depth() != data.depth || dst.depth() != data.depth ||
      src.channels() != data.channels || dst.channels() != data.channels ||
      src.width() != data.source.width || src.height() != data.source.height ||
      dst.width() != data.destination.width || dst.height() != data.destination.height)
    throw std::invalid_argument("resize image views do not match the plan");
  const auto s = to_spec(src, "src");
  const auto d = to_spec(dst, "dst");
  const auto s_batch = batch_span(src, s, count, src_pitch);
  const auto d_batch = batch_span(dst, d, count, dst_pitch);
  if (batches_overlap(s_batch, d_batch, count))
    throw std::invalid_argument("resize source and destination overlap");

  LoweredOp op;
  op.add_image(s);
  op.add_image(d);
  const auto table_bytes = plan.table_bytes();
  if (table_bytes) {
    const auto start = reinterpret_cast<std::uintptr_t>(workspace.data);
    if (!start || start % sizeof(std::uint32_t) != 0 || workspace.bytes < table_bytes ||
        table_bytes > std::numeric_limits<std::uintptr_t>::max() - start)
      throw std::invalid_argument("resize workspace is missing, unaligned or too small");
    const auto end = start + table_bytes;
    if (intervals_overlap(start, end, s_batch.begin, s_batch.end) ||
        intervals_overlap(start, end, d_batch.begin, d_batch.end))
      throw std::invalid_argument("resize workspace overlaps an image batch");

    for (const auto& table : data.tables) {
      ImageSpec pointer;
      pointer.name = table.name;
      pointer.address = start + table.offset;
      pointer.depth = table.depth;
      pointer.channels = 1;
      op.add_image(pointer);
    }
  }

  const auto route = data.route.kind;
  switch (route) {
    case ROUTE_COPY: op.kernel = "resize_copy"; break;
    case ROUTE_NEAREST: op.kernel = "resize_nearest"; break;
    case ROUTE_LINEAR:
    case ROUTE_CUBIC: op.kernel = "resize_separable"; break;
    case ROUTE_AREA_FAST: op.kernel = "resize_area_fast"; break;
    case ROUTE_AREA:
      // Classify from the prepared table, including scale-derived phases.
      if (data.area_polyphase)
        op.kernel = "resize_area_polyphase";
      else
        op.kernel = data.kx == 2 && data.ky == 2 ? "resize_area_2tap" : "resize_area";
      break;
  }
  if (route == ROUTE_LINEAR || route == ROUTE_CUBIC || route == ROUTE_AREA_FAST) {
    op.add_param("src_h", s.height);
    op.add_param("src_w", s.width);
  }
  if (route == ROUTE_AREA) {
    op.add_param("KX", data.kx);
    op.add_param("KY", data.ky);
  }
  op.add_param("dst_h", d.height);
  op.add_param("W", std::int64_t{d.width} * d.channels);
  if (route != ROUTE_COPY)
    op.add_constexpr("cn", d.channels);
  if (route == ROUTE_LINEAR || route == ROUTE_CUBIC)
    op.add_param("xmax", data.xmax);
  if (route == ROUTE_AREA_FAST) {
    const int ix = data.route.ix;
    const int iy = data.route.iy;
    op.add_param("dwidth1", (s.width / ix) * d.channels);
    op.add_fp32_param("scale", 1.0f / static_cast<float>(std::int64_t{ix} * iy));
    op.add_param("IX", ix);
    op.add_param("IY", iy);
  }

  const auto elem = static_cast<std::int64_t>(depth_size(d.depth));
  op.add_param("src_rs", s.row_stride);
  op.add_param("dst_rs", d.row_stride);
  op.add_param("src_bs", src_pitch / elem);
  op.add_param("dst_bs", dst_pitch / elem);
  // The shared compiler reads the batch and pixel extents for launch size.
  // They remain runtime arguments and never specialize a binary by size.
  op.add_param("batch_count", count);
  if (route == ROUTE_AREA || route == ROUTE_AREA_FAST)
    op.add_param("dst_w", d.width);
  if (route == ROUTE_LINEAR || route == ROUTE_CUBIC)
    op.add_constexpr("KSIZE", route == ROUTE_CUBIC ? 4 : 2);
  if (route == ROUTE_AREA_FAST)
    op.add_constexpr("TWO_BY_TWO", data.route.ix == 2 && data.route.iy == 2);
  if (route == ROUTE_LINEAR || route == ROUTE_CUBIC || route == ROUTE_AREA_FAST || route == ROUTE_AREA)
    op.add_constexpr("U8", d.depth == Depth::k8U);
  return op;
}

}  // namespace flagcv
