# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mirror an image around its horizontal axis, vertical axis, or both."""

from operator import index

import triton
import triton.language as tl


@triton.jit
def flip_kernel(
    src_ptr,
    dst_ptr,
    width,
    height,
    src_row_stride,
    dst_row_stride,
    CHANNELS: tl.constexpr,
    FLIP_X: tl.constexpr,
    FLIP_Y: tl.constexpr,
    IN_PLACE: tl.constexpr,
    VEC4: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    ROW_GRID_LIMIT: tl.constexpr,
):
    """dst(y, x, c) = src(y', x', c), y' and x' mirrored when FLIP_Y / FLIP_X.

    The x axis of the grid walks channel values, not pixels: a row holds
    width * CHANNELS elements, so interleaved channels stay contiguous in
    memory and the kernel needs no per-channel loop. Row strides are in
    elements (see docs/architecture.md), and row offsets are widened to int64
    so large padded images cannot overflow.
    """
    if VEC4:
        src_ptr = src_ptr.to(tl.pointer_type(tl.uint32))
        dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint32))
        src_row_stride = src_row_stride // 4
        dst_row_stride = dst_row_stride // 4
    VALUES_PER_PIXEL: tl.constexpr = 1 if VEC4 else CHANNELS
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1).to(tl.int64)
    if ROW_GRID_LIMIT:
        pid_y += tl.program_id(2).to(tl.int64) * ROW_GRID_LIMIT

    elem = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    row = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    in_bounds = (row[:, None] < height) & (elem[None, :] < width * VALUES_PER_PIXEL)

    if FLIP_X:
        pixel = elem // VALUES_PER_PIXEL
        src_elem = (width - 1 - pixel) * VALUES_PER_PIXEL + (
            elem - pixel * VALUES_PER_PIXEL
        )
    else:
        src_elem = elem
    if FLIP_Y:
        src_row = height - 1 - row
    else:
        src_row = row

    src_offsets = src_row[:, None] * src_row_stride + src_elem[None, :]
    dst_offsets = row[:, None] * dst_row_stride + elem[None, :]
    if IN_PLACE:
        # Exactly one lane owns each mirrored pair. Fixed points need no write.
        in_bounds = in_bounds & (dst_offsets < src_offsets)
    values = tl.load(src_ptr + src_offsets, mask=in_bounds)
    if IN_PLACE:
        opposite = tl.load(src_ptr + dst_offsets, mask=in_bounds)
        tl.store(dst_ptr + src_offsets, opposite, mask=in_bounds)
    tl.store(dst_ptr + dst_offsets, values, mask=in_bounds)


@triton.jit
def _flip_kernel(
    src,
    dst,
    count,
    height,
    WIDTH: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // WIDTH
    col = offsets % WIDTH
    if MODE <= 0:
        row = height - 1 - row
    if MODE != 0:
        col = WIDTH - 1 - col
    valid = offsets < count
    values = tl.load(src + row * WIDTH + col, valid)
    tl.store(dst + offsets, values, valid)


def flip(src, dst, flip_code):
    height, width = src.shape
    flip_code = index(flip_code)
    mode = (flip_code > 0) - (flip_code < 0)
    block = 1024
    count = src.numel()
    _flip_kernel[(triton.cdiv(count, block),)](
        src, dst, count, height, WIDTH=width, MODE=mode, BLOCK=block, num_warps=4
    )
    return dst
