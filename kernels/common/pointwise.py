# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pitched and interleaved array-array kernels for the C++ runtime."""

import triton
import triton.language as tl


@triton.jit
def _bitwise(a, b, MODE: tl.constexpr, DEPTH: tl.constexpr):
    if DEPTH == tl.float16:
        bits = tl.uint16
    elif DEPTH == tl.float32:
        bits = tl.uint32
    elif DEPTH == tl.float64:
        bits = tl.uint64
    else:
        bits = DEPTH
    left = a.to(bits, bitcast=True)
    if MODE == 3:
        result = left & b.to(bits, bitcast=True)
    else:
        result = ~left
    return result.to(DEPTH, bitcast=True)


@triton.jit
def _evaluate(a, b, scale_bits, MODE: tl.constexpr, DEPTH: tl.constexpr):
    if MODE > 2:
        result = _bitwise(a, b, MODE, DEPTH)
    else:
        if DEPTH == tl.float16:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        elif DEPTH == tl.uint8 or DEPTH == tl.uint16:
            a = a.to(tl.int32)
            b = b.to(tl.int32)
        if MODE == 0:
            result = a + b
        elif MODE == 1:
            result = tl.maximum(a - b, 0) if DEPTH.is_int() else a - b
        elif DEPTH.is_int():
            result = a.to(tl.uint32) * b.to(tl.uint32)
        else:
            result = a * b
            if DEPTH == tl.float32:
                result *= scale_bits.to(tl.float32, bitcast=True)
        if DEPTH == tl.uint8:
            result = tl.minimum(result, 255)
        elif DEPTH == tl.uint16:
            result = tl.minimum(result, 65535)
    return result


@triton.jit
def _pointwise(
    src1_ptr,
    src2_ptr,
    dst_ptr,
    mask_ptr,
    width,
    height,
    src1_row_stride,
    src2_row_stride,
    dst_row_stride,
    mask_row_stride,
    scale_bits,
    CHANNELS: tl.constexpr,
    MODE: tl.constexpr,
    PACKED: tl.constexpr,
    VEC4: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    ROW_GRID_LIMIT: tl.constexpr,
):
    if VEC4:
        src1_ptr = src1_ptr.to(tl.pointer_type(tl.uint32))
        src2_ptr = src2_ptr.to(tl.pointer_type(tl.uint32))
        dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint32))
        src1_row_stride = src1_row_stride // 4
        src2_row_stride = src2_row_stride // 4
        dst_row_stride = dst_row_stride // 4
    VALUES_PER_PIXEL: tl.constexpr = 1 if VEC4 else CHANNELS
    if PACKED:
        x = tl.program_id(0).to(tl.int64) * BLOCK_X + tl.arange(0, BLOCK_X)
        valid = x < width.to(tl.int64) * VALUES_PER_PIXEL * height
        lhs = x
        rhs = x
        output = x
    else:
        x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
        row_group = tl.program_id(1).to(tl.int64)
        if ROW_GRID_LIMIT:
            row_group += tl.program_id(2).to(tl.int64) * ROW_GRID_LIMIT
        y = row_group * BLOCK_Y + tl.arange(0, BLOCK_Y)
        valid = (y[:, None] < height) & (x[None, :] < width * VALUES_PER_PIXEL)
        if mask_ptr is not None:
            mask_offsets = y[:, None] * mask_row_stride + x[None, :] // CHANNELS
            mask = tl.load(mask_ptr + mask_offsets, valid, 0)
            valid = valid & (mask != 0)
        lhs = y[:, None] * src1_row_stride + x[None, :]
        rhs = y[:, None] * src2_row_stride + x[None, :]
        output = y[:, None] * dst_row_stride + x[None, :]
    a = tl.load(src1_ptr + lhs, valid, 0)
    b = a if MODE == 4 else tl.load(src2_ptr + rhs, valid, 0)
    result = _evaluate(a, b, scale_bits, MODE, src1_ptr.dtype.element_ty)
    tl.store(dst_ptr + output, result, valid)


@triton.jit
def pointwise_kernel(
    src1_ptr,
    src2_ptr,
    dst_ptr,
    width,
    height,
    src1_row_stride,
    src2_row_stride,
    dst_row_stride,
    CHANNELS: tl.constexpr,
    MODE: tl.constexpr,
    PACKED: tl.constexpr,
    VEC4: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    ROW_GRID_LIMIT: tl.constexpr,
):
    _pointwise(
        src1_ptr, src2_ptr, dst_ptr, None, width, height,
        src1_row_stride, src2_row_stride, dst_row_stride, 0, 0,
        CHANNELS, MODE, PACKED, VEC4, BLOCK_X, BLOCK_Y, ROW_GRID_LIMIT,
    )


@triton.jit
def masked_pointwise_kernel(
    src1_ptr,
    src2_ptr,
    dst_ptr,
    mask_ptr,
    width,
    height,
    src1_row_stride,
    src2_row_stride,
    dst_row_stride,
    mask_row_stride,
    CHANNELS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    ROW_GRID_LIMIT: tl.constexpr,
):
    _pointwise(
        src1_ptr, src2_ptr, dst_ptr, mask_ptr, width, height,
        src1_row_stride, src2_row_stride, dst_row_stride, mask_row_stride, 0,
        CHANNELS, MODE, False, False, BLOCK_X, BLOCK_Y, ROW_GRID_LIMIT,
    )


@triton.jit
def multiply_kernel(
    src1_ptr,
    src2_ptr,
    dst_ptr,
    width,
    height,
    src1_row_stride,
    src2_row_stride,
    dst_row_stride,
    scale_bits,
    CHANNELS: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    ROW_GRID_LIMIT: tl.constexpr,
):
    _pointwise(
        src1_ptr, src2_ptr, dst_ptr, None, width, height,
        src1_row_stride, src2_row_stride, dst_row_stride, 0, scale_bits,
        CHANNELS, 2, PACKED, False, BLOCK_X, BLOCK_Y, ROW_GRID_LIMIT,
    )
