/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/lowering/lowering.hpp"

#include "src/imgproc/cvtcolor_spec.hpp"
#include "src/lowering/helpers.hpp"

#include <limits>
#include <stdexcept>

namespace flagcv {
namespace {

std::uintptr_t image_end(const ImageSpec& view) {
  const auto span = (view.height - 1) * view.row_stride + std::int64_t{view.width} * view.channels;
  if (view.row_stride > std::numeric_limits<std::int32_t>::max() ||
      span > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("cvtColor image span exceeds int32 element indexing");
  return view.address + span * depth_size(view.depth);
}

void add_pointer(LoweredOp& op, const char* name, std::uintptr_t address, Depth depth) {
  ImageSpec pointer;
  pointer.name = name;
  pointer.address = address;
  pointer.depth = depth;
  pointer.channels = 1;
  op.add_image(pointer);
}

}  // namespace

LoweredOp lower_cvtcolor(const GpuMat& src, const GpuMat& dst, int code,
                          DeviceWorkspace workspace) {
  using namespace color;
  const auto& spec = find(code);
  const auto expected = resolve({src.width(), src.height()}, src.depth(), src.channels(), spec);
  if (dst.width() != expected.size.width || dst.height() != expected.size.height ||
      dst.channels() != expected.channels || dst.depth() != src.depth())
    throw std::invalid_argument("cvtColor destination does not match the conversion output");
  const auto s = to_spec(src, "src");
  const auto d = to_spec(dst, "dst");
  const auto s_end = image_end(s);
  const auto d_end = image_end(d);
  if (images_overlap(src, dst))
    throw std::invalid_argument("cvtColor source and destination overlap");

  const auto family = tables(spec);
  const auto table_bytes = bytes(family);
  const auto start = reinterpret_cast<std::uintptr_t>(workspace.data);
  if (table_bytes) {
    const auto alignment = family == ColorTables::kLabLuvReverse ? 8u : 4u;
    if (!start || start % alignment || workspace.bytes < table_bytes ||
        table_bytes > std::numeric_limits<std::uintptr_t>::max() - start)
      throw std::invalid_argument("cvtColor workspace is missing, unaligned or too small");
    const auto end = start + table_bytes;
    if ((s.address < end && start < s_end) || (d.address < end && start < d_end))
      throw std::invalid_argument("cvtColor workspace overlaps an image");
  }
  // Inactive constexpr branches never dereference their null slots.
  const bool hsv = family == ColorTables::kHsv || family == ColorTables::kHsvFull;
  const bool lab = family == ColorTables::kLabLuvReverse;
  const bool forward_luv = family == ColorTables::kLuvForward;

  LoweredOp op;
  op.kernel = "cvtcolor";
  op.images.reserve(7);
  op.add_image(s);
  op.add_image(d);
  add_pointer(op, "sdiv", hsv ? start : 0, Depth::k32S);
  add_pointer(op, "hdiv", hsv ? start + kHsvTableLength * sizeof(std::int32_t) : 0, Depth::k32S);
  add_pointer(op, "t32", lab ? start : 0, Depth::k32S);
  add_pointer(op, "t16", forward_luv ? start : 0, Depth::k16S);
  add_pointer(op, "t64", lab ? start + kLabLuvI32Count * sizeof(std::int32_t) : 0, Depth::k32S);
  op.add_param("cols", d.width);
  op.add_param("src_rs", s.row_stride);
  op.add_param("dst_rs", d.row_stride);
  op.add_param("uv_rs", spec.op == Op::Yuv420 ? d.height * s.row_stride : 0);
  op.add_param("rows", d.height);
  op.add_constexpr("SCN", spec.scn);
  op.add_constexpr("DCN", spec.dcn);
  op.add_constexpr("OP", static_cast<int>(spec.op));
  op.add_constexpr("BLUE", spec.blue);
  op.add_constexpr("I0", spec.perm[0]);
  op.add_constexpr("I1", spec.perm[1]);
  op.add_constexpr("I2", spec.perm[2]);
  op.add_constexpr("I3", spec.perm[3]);
  op.add_constexpr("INV", spec.inverse);
  op.add_constexpr("HR", spec.hrange);
  op.add_constexpr("GB", spec.gb);
  op.add_constexpr("U16", s.depth == Depth::k16U);
  op.add_constexpr("F32", s.depth == Depth::k32F);
  // OpenCV 5.0.0 SSE3, separate mul/add and groups of four.
  op.add_constexpr("F32_CPU_SEPARATE4", s.depth == Depth::k32F);
  op.add_constexpr("SWG", spec.swg);
  op.add_constexpr("ORD", spec.ord);
  constexpr const char* coefficients[] = {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"};
  for (std::size_t i = 0; i < std::size(coefficients); ++i)
    op.add_constexpr(coefficients[i], spec.coeffs[i]);
  op.add_constexpr("SRGB", spec.srgb);
  return op;
}

}  // namespace flagcv
