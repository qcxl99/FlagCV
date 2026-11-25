/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FLAGCV_SRC_LOWERING_SCALAR_BITS_HPP_
#define FLAGCV_SRC_LOWERING_SCALAR_BITS_HPP_

#include "src/lowering/lowering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace flagcv {

template<class To, class From>
To scalar_bits(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline std::int32_t cpu_round_i32(double value) {
  const double floor = std::floor(value);
  const double fraction = value - floor;
  const double rounded =
      floor + (fraction > 0.5 || (fraction == 0.5 && std::fmod(floor, 2.0) != 0));
  constexpr auto low = std::numeric_limits<std::int32_t>::min();
  constexpr auto high = std::numeric_limits<std::int32_t>::max();
  return rounded >= low && rounded <= high ? static_cast<std::int32_t>(rounded) : low;
}

inline void add_scalar_words(LoweredOp& op, const char* lo, const char* hi,
                             std::uint64_t bits) {
  op.add_param(lo, scalar_bits<std::int32_t>(static_cast<std::uint32_t>(bits)));
  op.add_param(hi, scalar_bits<std::int32_t>(static_cast<std::uint32_t>(bits >> 32)));
}

inline std::uint16_t cpu_half_bits(float value) {
  const std::uint32_t input = scalar_bits<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((input >> 16) & 0x8000);
  const std::uint32_t absolute = input & 0x7fffffff;
  if (absolute >= 0x7f800000)
    return sign | (absolute == 0x7f800000 ? 0x7c00 : 0x7e00);
  const int exponent = int(absolute >> 23) - 127;
  if (exponent > 15) return sign | 0x7c00;
  if (exponent < -25) return sign;
  const auto mantissa = absolute & 0x7fffff;
  const int shift = exponent < -14 ? -1 - exponent : 13;
  const auto significand = exponent < -14 ? mantissa | 0x800000 : mantissa;
  const auto unit = std::uint32_t{1} << shift;
  const auto quotient = significand >> shift;
  const auto residue = significand & (unit - 1);
  const auto rounded = quotient + (residue > unit / 2 || (residue == unit / 2 && quotient % 2));
  const auto base = exponent < -14 ? 0 : (exponent + 15) << 10;
  return sign | static_cast<std::uint16_t>(base + rounded);
}

inline std::uint64_t copy_border_scalar(Depth depth, double value) {
  switch (depth) {
    case Depth::k8U: return std::clamp(cpu_round_i32(value), 0, 255);
    case Depth::k8S: return static_cast<std::uint8_t>(std::clamp(cpu_round_i32(value), -128, 127));
    case Depth::k16U: return std::clamp(cpu_round_i32(value), 0, 65535);
    case Depth::k16S: return static_cast<std::uint16_t>(std::clamp(cpu_round_i32(value), -32768, 32767));
    case Depth::k32S: return static_cast<std::uint32_t>(cpu_round_i32(value));
    case Depth::k16F: return cpu_half_bits(static_cast<float>(value));
    case Depth::k32F: return scalar_bits<std::uint32_t>(static_cast<float>(value));
    case Depth::k64F: return scalar_bits<std::uint64_t>(value);
  }
  return 0; // The view has already gone through depth_size.
}

}  // namespace flagcv
#endif
