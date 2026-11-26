/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/core.hpp>

#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

namespace flagcv {

void flip(Handle& handle, const GpuMat& src, const GpuMat& dst, int flip_code,
          Stream stream) {
  launch(handle, lower_flip(src, dst, flip_code), stream);
}

void flip(const GpuMat& src, const GpuMat& dst, int flip_code,
          Stream stream) {
  flip(detail::currentHandle(), src, dst, flip_code, stream);
}

}  // namespace flagcv
