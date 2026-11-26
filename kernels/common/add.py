# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton addition with OpenCV uint8 saturation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(a, b, dst, count, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = True if EVEN else offsets < count
    left = tl.load(a + offsets, valid)
    right = tl.load(b + offsets, valid)
    if a.dtype.element_ty == tl.uint8:
        result = tl.minimum(left.to(tl.uint16) + right.to(tl.uint16), 255)
    else:
        result = left + right
    tl.store(dst + offsets, result, valid)


def add(a, b, dst):
    count = a.numel()
    block = 2048 if a.dtype == torch.uint8 else 1024
    _add_kernel[(triton.cdiv(count, block),)](
        a, b, dst, count, BLOCK=block, EVEN=count % block == 0, num_warps=4
    )
    return dst
