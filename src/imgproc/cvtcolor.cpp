/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cvtcolor_spec.hpp"

#include "src/runtime/paths.hpp"
#include "src/runtime/runtime.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flagcv {
namespace color {
namespace {

constexpr CodeSpec codes[] = {
#include "cvtcolor_codes.inc"
};

void validate_size(Size size, int channels) {
  if (size.width <= 0 || size.height <= 0)
    throw std::invalid_argument("cvtColor image dimensions must be positive");
  const auto row = std::int64_t{size.width} * channels;
  constexpr auto limit = std::numeric_limits<std::int32_t>::max();
  if (row > limit || size.height > limit / row)
    throw std::invalid_argument("cvtColor image dimensions exceed int32 element indexing");
}

using HsvTable = std::array<std::int32_t, 2 * kHsvTableLength>;
HsvTable build_hsv_table(int hue_range) {
  const auto divide_round_even = [](int numerator, int divisor) {
    const auto quotient = numerator / divisor;
    const auto twice_remainder = 2 * (numerator % divisor);
    return quotient + (twice_remainder > divisor ||
                       (twice_remainder == divisor && quotient % 2 != 0));
  };
  HsvTable result{};
  for (std::size_t i = 1; i < kHsvTableLength; ++i) {
    result[i] = divide_round_even(255 << 12, static_cast<int>(i));
    result[i + kHsvTableLength] =
        divide_round_even(hue_range << 12, 6 * static_cast<int>(i));
  }
  return result;
}

std::vector<std::uint64_t> load_table(const char* name, std::size_t size,
                                       std::uint64_t checksum) {
  const std::string path = data_dir() + "/tables/cvtcolor/" + name;
  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint64_t> result(size / sizeof(std::uint64_t));
  auto* bytes = reinterpret_cast<unsigned char*>(result.data());
  if (!in.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size)) ||
      in.peek() != std::ifstream::traits_type::eof())
    throw Error("flagcv: invalid cvtColor table size in " + path);

  // Generated FNV-1a checksum catches wrong versions and damaged package data.
  std::uint64_t actual = 14695981039346656037ULL;
  for (std::size_t i = 0; i < size; ++i)
    actual = (actual ^ bytes[i]) * 1099511628211ULL;
  if (actual != checksum)
    throw Error("flagcv: cvtColor table checksum mismatch in " + path);
  return result;
}

}  // namespace

const CodeSpec& find(int code) {
  for (const auto& spec : codes)
    if (spec.code == code)
      return spec;
  throw std::invalid_argument("cvtColor unsupported conversion code " + std::to_string(code));
}

ColorOutput resolve(Size source, Depth depth, int channels, const CodeSpec& spec) {
  if (channels != spec.scn)
    throw std::invalid_argument(std::string("cvtColor source channels do not match ") + spec.name);
  const bool arithmetic = spec.op == Op::Gray || spec.op == Op::Yuv || spec.op == Op::YCrCb;
  const bool u16 = arithmetic || spec.op == Op::Bayer ||
                   (spec.op == Op::Permute && spec.scn == 1);
  const bool f32 = arithmetic || (spec.op == Op::Permute && spec.scn != 2);
  if (depth != Depth::k8U && !(depth == Depth::k16U && u16) &&
      !(depth == Depth::k32F && f32))
    throw std::invalid_argument(std::string("cvtColor does not support this depth for ") + spec.name);
  validate_size(source, channels);

  Size output = source;
  if ((spec.op == Op::Yuv422 || spec.op == Op::Yuv420) && source.width % 2 != 0)
    throw std::invalid_argument("cvtColor packed video color output requires even width");
  if (spec.op == Op::Yuv420) {
    if (source.height % 3 != 0)
      throw std::invalid_argument("cvtColor NV12/NV21 source height must be divisible by three");
    output.height = (source.height / 3) * 2;
  }
  validate_size(output, spec.dcn);
  return {output, spec.dcn};
}

ColorTables tables(const CodeSpec& spec) {
  if (spec.op == Op::Hsv)
    return spec.hrange == 180 ? ColorTables::kHsv : ColorTables::kHsvFull;
  if (spec.op == Op::Luv && !spec.inverse) return ColorTables::kLuvForward;
  if (spec.op == Op::Lab || spec.op == Op::Luv) return ColorTables::kLabLuvReverse;
  return ColorTables::kNone;
}

std::size_t bytes(ColorTables kind) {
  switch (kind) {
    case ColorTables::kNone: return 0;
    case ColorTables::kHsv:
    case ColorTables::kHsvFull: return 2 * kHsvTableLength * sizeof(std::int32_t);
    case ColorTables::kLabLuvReverse:
      return kLabLuvI32Count * sizeof(std::int32_t) + kLuvI64Count * sizeof(std::int64_t);
    case ColorTables::kLuvForward: return kLuvLutCount * sizeof(std::int16_t);
  }
  throw std::invalid_argument("cvtColor unknown table family");
}

}  // namespace color

ColorOutput cvtColorOutput(Size source, Depth depth, int channels, int code) {
  return color::resolve(source, depth, channels, color::find(code));
}

ColorTables cvtColorTableKind(int code) { return color::tables(color::find(code)); }

HostTableData cvtColorTableData(ColorTables kind) {
  const auto size = color::bytes(kind);
  switch (kind) {
    case ColorTables::kNone: return {nullptr, 0};
    case ColorTables::kHsv: {
      static const auto table = color::build_hsv_table(180);
      return {table.data(), size};
    }
    case ColorTables::kHsvFull: {
      static const auto table = color::build_hsv_table(256);
      return {table.data(), size};
    }
    case ColorTables::kLabLuvReverse: {
      static const auto table =
          color::load_table("lab-luv-reverse.bin", size, 0xf26d148e3aaf884fULL);
      return {table.data(), size};
    }
    case ColorTables::kLuvForward: {
      static const auto table =
          color::load_table("luv-forward.bin", size, 0x3477b93482cf9a2bULL);
      return {table.data(), size};
    }
  }
  throw std::invalid_argument("cvtColor unknown table family");
}

void cvtColor(Handle& handle, const GpuMat& src, const GpuMat& dst, int code,
              DeviceWorkspace workspace, Stream stream) {
  launch(handle, lower_cvtcolor(src, dst, code, workspace), stream);
}

}  // namespace flagcv
