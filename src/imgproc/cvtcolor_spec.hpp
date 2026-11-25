/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <flagcv/imgproc.hpp>

namespace flagcv::color {

enum class Op {
  Gray, Permute, Yuv, YCrCb, Hsv, Packed, Bayer, Xyz, Lab, Luv, Yuv422, Yuv420,
};

struct CodeSpec {
  const char* name;
  int code;
  Op op;
  int scn, dcn;
  int blue;
  bool inverse;
  int hrange;
  int perm[4];
  int gb;
  int swg;
  bool srgb;
  int coeffs[9];
  int ord;
};

const CodeSpec& find(int code);
ColorOutput resolve(Size source, Depth depth, int channels, const CodeSpec& spec);
ColorTables tables(const CodeSpec& spec);

constexpr std::size_t kHsvTableLength = 256;
constexpr std::size_t kLabLuvI32Count = 256 + 3072 + 512 + 36864 + 4096 + 2 * 65536;
constexpr std::size_t kLuvI64Count = 65536;
constexpr std::size_t kLuvLutCount = 33 * 33 * 33 * 3 * 8;
std::size_t bytes(ColorTables kind);

}  // namespace flagcv::color
