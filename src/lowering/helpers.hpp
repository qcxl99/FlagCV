/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_LOWERING_HELPERS_HPP_
#define FLAGCV_SRC_LOWERING_HELPERS_HPP_

#include <flagcv/gpumat.hpp>

#include "src/image/types.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagcv {

/*
 * Builds the internal spec, converting the byte pitch to an element stride.
 *
 * Also rejects layouts no kernel can address safely: non-positive extents,
 * rows that overlap because the pitch is shorter than a row, and a base
 * pointer that is not aligned to the pixel depth (a typed device load from it
 * would fault).
 */
inline ImageSpec to_spec(const GpuMat& image, const char* name) {
  if (image.empty()) {
    throw std::invalid_argument(std::string("operand '") + name + "' must not be empty");
  }
  if (image.width() <= 0 || image.height() <= 0 || image.channels() <= 0) {
    throw std::invalid_argument(std::string("operand '") + name +
                                "' must have positive width, height and "
                                "channels");
  }
  const auto address = reinterpret_cast<std::uintptr_t>(image.data());
  if (address % depth_size(image.depth()) != 0) {
    throw std::invalid_argument(std::string("operand '") + name +
                                "' data is not aligned to its pixel depth");
  }
  ImageSpec spec;
  spec.name = name;
  spec.address = address;
  spec.width = image.width();
  spec.height = image.height();
  spec.depth = image.depth();
  spec.channels = image.channels();
  spec.row_stride = image.row_stride();
  if (spec.row_stride < std::int64_t{spec.width} * spec.channels) {
    throw std::invalid_argument(std::string("operand '") + name +
                                "' row pitch is shorter than one row");
  }
  const auto element_size = static_cast<std::int64_t>(depth_size(image.depth()));
  const auto pitch = image.row_pitch();
  const auto row_bytes = std::int64_t{spec.width} * spec.channels * element_size;
  const auto rows_before_last = spec.height - 1;
  if (rows_before_last > (std::numeric_limits<std::int64_t>::max() - row_bytes) / pitch ||
      static_cast<std::uint64_t>(rows_before_last * pitch + row_bytes) >
          std::numeric_limits<std::uintptr_t>::max() - address) {
    throw std::invalid_argument(std::string("operand '") + name + "' address range overflows");
  }
  return spec;
}

inline bool same_view(const GpuMat& a, const GpuMat& b) {
  return a.data() == b.data() && a.row_pitch() == b.row_pitch();
}

inline bool can_pack_u8x4(const ImageSpec& image) {
  return image.depth == Depth::k8U && image.channels == 4 &&
         image.address % 4 == 0 && image.row_stride % 4 == 0;
}

/* Only pixels count as overlap. An ROI may occupy another view's row gaps. */
inline bool images_overlap(const GpuMat& a, const GpuMat& b) {
  using Address = std::uintptr_t;
  auto row_bytes = [](const GpuMat& image) {
    return static_cast<Address>(image.width()) * image.channels() *
           depth_size(image.depth());
  };
  Address a_row = reinterpret_cast<Address>(a.data());
  Address b_row = reinterpret_cast<Address>(b.data());
  const Address a_width = row_bytes(a), b_width = row_bytes(b);
  const auto a_pitch = static_cast<Address>(a.row_pitch());
  const auto b_pitch = static_cast<Address>(b.row_pitch());
  const Address a_end = a_row + (a.height() - 1) * a_pitch + a_width;
  const Address b_end = b_row + (b.height() - 1) * b_pitch + b_width;
  if (a_end <= b_row || b_end <= a_row) return false;
  if (a_pitch == b_pitch) {
    if (a_row > b_row) {
      const auto offset = (a_row - b_row) % a_pitch;
      return offset < b_width || a_pitch - offset < a_width;
    }
    const auto offset = (b_row - a_row) % a_pitch;
    return offset < a_width || a_pitch - offset < b_width;
  }
  while (a_row < a_end && b_row < b_end) {
    if (a_row + a_width <= b_row) {
      const auto advance = (b_row - (a_row + a_width)) / a_pitch + 1;
      if (advance > (a_end - 1 - a_row) / a_pitch) return false;
      a_row += advance * a_pitch;
    } else if (b_row + b_width <= a_row) {
      const auto advance = (a_row - (b_row + b_width)) / b_pitch + 1;
      if (advance > (b_end - 1 - b_row) / b_pitch) return false;
      b_row += advance * b_pitch;
    } else {
      return true;
    }
  }
  return false;
}

inline void require_same_layout(const ImageSpec& a, const ImageSpec& b) {
  if (a.width != b.width || a.height != b.height) {
    throw std::invalid_argument("image extents must match");
  }
  if (a.depth != b.depth) {
    throw std::invalid_argument("image pixel depths must match");
  }
  if (a.channels != b.channels) {
    throw std::invalid_argument("image channel counts must match");
  }
}

inline void require_integer(const ImageSpec& image) {
  if (!is_integer(image.depth)) {
    throw std::invalid_argument("operation requires an integer pixel depth");
  }
}

/*
 * An operation mask is single-channel 8-bit, as in OpenCV.
 *
 * Mask semantics are "do not write", not "write zero": where the mask is zero
 * the destination keeps its prior contents, so a masked call must pass an
 * initialized dst. Kernels get this by folding the mask into the store
 * predicate. See docs/operator-development.md.
 */
inline void require_mask(const ImageSpec& mask, const ImageSpec& dst) {
  if (mask.width != dst.width || mask.height != dst.height) {
    throw std::invalid_argument("mask extent must match the destination");
  }
  if (mask.channels != 1 || mask.depth != Depth::k8U) {
    throw std::invalid_argument("mask must be 8-bit single-channel");
  }
}

}  // namespace flagcv

#endif  // FLAGCV_SRC_LOWERING_HELPERS_HPP_
