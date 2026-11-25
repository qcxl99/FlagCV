/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/imgproc.hpp>

#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

namespace flagcv {

void warpPerspective(Handle& handle, const GpuMat& src, const GpuMat& dst,
                     const PerspectiveMatrix& matrix, Interpolation interpolation,
                     Border border, const BorderValue& border_value,
                     bool inverse_map, Stream stream) {
  launch(handle, lower_warp_perspective(src, dst, matrix, interpolation, border,
                                       border_value, inverse_map), stream);
}

void remap(Handle& handle, const GpuMat& src, const GpuMat& dst, const GpuMat& map1,
           const GpuMat& map2, Interpolation interpolation, Border border,
           const BorderValue& border_value, bool relative_map, Stream stream) {
  launch(handle, lower_remap(src, dst, map1, map2, interpolation, border,
                             border_value, relative_map), stream);
}

}  // namespace flagcv
