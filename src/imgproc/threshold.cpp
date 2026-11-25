/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <flagcv/imgproc.hpp>

#include "src/lowering/helpers.hpp"
#include "src/lowering/scalar_bits.hpp"
#include "src/runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace flagcv {
namespace {

int threshold_kind(Depth depth) {
  switch (depth) {
    case Depth::k8U: return 0;
    case Depth::k16U: return 1;
    case Depth::k16S: return 2;
    case Depth::k32F: return 3;
    case Depth::k64F: return 4;
    default: throw std::invalid_argument("threshold unsupported depth");
  }
}

int threshold_mode(ThresholdMode mode) {
  switch (mode) {
    case ThresholdMode::kBinary: return 0;
    case ThresholdMode::kBinaryInv: return 1;
    case ThresholdMode::kTrunc: return 2;
    case ThresholdMode::kToZero: return 3;
    case ThresholdMode::kToZeroInv: return 4;
  }
  throw std::invalid_argument("threshold unsupported mode");
}

int histogram_bins(Depth depth, AutomaticThreshold method) {
  if (method == AutomaticThreshold::kOtsu) {
    if (depth == Depth::k8U) return 256;
    if (depth == Depth::k16U) return 65536;
  } else if (method == AutomaticThreshold::kTriangle && depth == Depth::k8U) {
    return 256;
  }
  throw std::invalid_argument("threshold unsupported automatic method or depth");
}

std::int64_t row_values(const ImageSpec& source) {
  const auto values = std::int64_t{source.width} * source.channels;
  if (source.channels > 4 || values > std::numeric_limits<std::int32_t>::max() ||
      source.height > std::numeric_limits<std::int32_t>::max() / values)
    throw std::invalid_argument("threshold supports up to four channels and int32 element counts");
  return values;
}

ImageSpec destination(const GpuMat& src, const GpuMat& dst, const ImageSpec& source) {
  const auto result = to_spec(dst, "dst");
  require_same_layout(source, result);
  if (!same_view(src, dst) && images_overlap(src, dst))
    throw std::invalid_argument("threshold does not support partial overlap");
  return result;
}

std::uint64_t pack_threshold_scalar(int kind, double value) {
  if (kind == 3) return scalar_bits<std::uint32_t>(static_cast<float>(value));
  if (kind == 4) return scalar_bits<std::uint64_t>(value);
  return scalar_bits<std::uint32_t>(static_cast<std::int32_t>(value));
}

double fixed_cutoff(int kind, double cutoff) {
  if (!std::isfinite(cutoff))
    throw std::invalid_argument("threshold cutoff must be finite");
  if (kind < 3) {
    cutoff = std::floor(cutoff);
    if (cutoff < std::numeric_limits<std::int32_t>::min() ||
        cutoff > std::numeric_limits<std::int32_t>::max())
      throw std::invalid_argument("threshold integer cutoff floor must fit int32");
  }
  return cutoff;
}

double working_maximum(int kind, double maximum) {
  if (!std::isfinite(maximum))
    throw std::invalid_argument("threshold maxval must be finite");
  if (kind >= 3) return maximum;
  const auto value = cpu_round_i32(maximum);
  if (kind == 0) return std::clamp(value, 0, 255);
  if (kind == 1) return std::clamp(value, 0, 65535);
  return std::clamp(value, -32768, 32767);
}

void apply(Handle& handle, const ImageSpec& source, const ImageSpec& dest, int mode,
           int kind, double cutoff, double maximum, std::uintptr_t result, Stream stream) {
  LoweredOp op;
  op.kernel = "threshold_apply";
  ImageSpec cutoff_image{};
  cutoff_image.name = "cutoff";
  cutoff_image.depth = Depth::k32S;
  cutoff_image.channels = 1;
  cutoff_image.address = result;
  op.images = {source, dest, cutoff_image};
  op.add_param("rows", source.height);
  op.add_param("row_values", row_values(source));
  op.add_param("src_rs", source.row_stride);
  op.add_param("dst_rs", dest.row_stride);
  add_scalar_words(op, "cut_lo", "cut_hi", pack_threshold_scalar(kind, cutoff));
  add_scalar_words(op, "max_lo", "max_hi", pack_threshold_scalar(kind, maximum));
  op.add_constexpr("MODE", mode);
  op.add_constexpr("KIND", kind);
  op.add_constexpr("AUTO", result != 0);
  launch(handle, std::move(op), stream);
}

bool intersects(const ImageSpec& image, std::uintptr_t start, std::uintptr_t end) {
  const auto element = depth_size(image.depth);
  const auto span = (image.height - 1) * std::uint64_t(image.row_stride) * element +
                    std::uint64_t(image.width) * image.channels * element;
  return image.address < end && start < image.address + span;
}

}  // namespace

double threshold(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 double cutoff, double maxval, ThresholdMode mode, bool dry_run, Stream stream) {
  const auto source = to_spec(src, "src");
  const int kind = threshold_kind(source.depth);
  const int code = threshold_mode(mode);
  (void)row_values(source);
  const auto selected = fixed_cutoff(kind, cutoff);
  if (!dry_run) {
    const auto dest = destination(src, dst, source);
    apply(handle, source, dest, code, kind, selected, working_maximum(kind, maxval), 0, stream);
  }
  return selected;
}

std::size_t thresholdWorkspaceBytes(Depth depth, AutomaticThreshold method) {
  return (std::size_t(histogram_bins(depth, method)) + 1) * sizeof(std::int32_t);
}

void thresholdAuto(Handle& handle, const GpuMat& src, const GpuMat& dst, double maxval,
                   ThresholdMode mode, AutomaticThreshold method, ThresholdScratch scratch,
                   bool dry_run, Stream stream) {
  const auto source = to_spec(src, "src");
  const auto values = row_values(source);
  const int bins = histogram_bins(source.depth, method);
  const int code = threshold_mode(mode);
  if (source.channels != 1)
    throw std::invalid_argument("automatic threshold requires one channel");
  const auto start = reinterpret_cast<std::uintptr_t>(scratch.data);
  const auto needed = thresholdWorkspaceBytes(source.depth, method);
  if (!start || start % alignof(std::int32_t) != 0 || scratch.bytes < needed ||
      needed > std::numeric_limits<std::uintptr_t>::max() - start)
    throw std::invalid_argument("threshold workspace is missing, unaligned, or too small");

  ImageSpec dest{};
  double maximum = 0;
  if (!dry_run) {
    dest = destination(src, dst, source);
    maximum = working_maximum(threshold_kind(source.depth), maxval);
  }
  if (intersects(source, start, start + needed) ||
      (!dry_run && intersects(dest, start, start + needed)))
    throw std::invalid_argument("threshold workspace overlaps image storage");

  ImageSpec buffer{};
  buffer.name = "workspace";
  buffer.address = start;
  buffer.depth = Depth::k32S;
  buffer.channels = 1;

  LoweredOp clear;
  clear.kernel = "threshold_clear";
  clear.images = {buffer};
  clear.add_param("bins", bins);
  launch(handle, std::move(clear), stream);

  LoweredOp histogram;
  const bool large_u8_otsu = source.depth == Depth::k8U && method == AutomaticThreshold::kOtsu &&
                             values * source.height >= 1920 * 1080;
  histogram.kernel = large_u8_otsu ? "threshold_histogram_u8_large" : "threshold_histogram";
  histogram.images = {source, buffer};
  histogram.add_param("rows", source.height);
  histogram.add_param("row_values", values);
  histogram.add_param("src_rs", source.row_stride);
  launch(handle, std::move(histogram), stream);

  LoweredOp select;
  select.kernel = "threshold_select";
  select.images = {buffer};
  select.add_param("pixels", values * source.height);
  select.add_constexpr("BINS", bins);
  select.add_constexpr("TRIANGLE", method == AutomaticThreshold::kTriangle);
  launch(handle, std::move(select), stream);

  if (!dry_run)
    apply(handle, source, dest, code, threshold_kind(source.depth), 0, maximum, start, stream);
}

double thresholdAutoHost(Handle& handle, const GpuMat& src, const GpuMat& dst, double maxval,
                         ThresholdMode mode, AutomaticThreshold method, ThresholdScratch scratch,
                         bool dry_run, Stream stream) {
  require_readback(handle);
  thresholdAuto(handle, src, dst, maxval, mode, method, scratch, dry_run, stream);
  std::int32_t selected = 0;
  read_device(handle, scratch.data, &selected, sizeof(selected), stream);
  return selected;
}

}  // namespace flagcv
