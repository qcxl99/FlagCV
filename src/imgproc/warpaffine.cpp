/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/imgproc.hpp>

#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

namespace flagcv {

void warpAffine(Handle& handle, const GpuMat& src, const GpuMat& dst,
                const AffineMatrix& matrix, Interpolation interpolation,
                Border border, const BorderValue& border_value,
                bool inverse_map, Stream stream) {
  launch(handle, lower_warp_affine(src, dst, matrix, interpolation,
                                 border, border_value, inverse_map), stream);
}

}  // namespace flagcv
