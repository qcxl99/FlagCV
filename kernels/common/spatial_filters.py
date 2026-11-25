# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Portable spatial filters over one isolated pitched image view."""
import triton
import triton.language as tl


@triton.jit
def _coordinates(position, size, BORDER: tl.constexpr):
    """Map an axis and return the load predicate for constant-zero padding."""
    if BORDER == 0:
        valid = (position >= 0) & (position < size)
        return tl.maximum(0, tl.minimum(position, size - 1)), valid
    if BORDER == 1:
        return tl.maximum(0, tl.minimum(position, size - 1)), tl.full(position.shape, True, tl.int1)
    if BORDER == 2:
        period = 2 * tl.cast(size, tl.int64)
        folded = position % period
        folded = tl.where(folded < 0, folded + period, folded)
        return tl.where(folded >= size, period - 1 - folded, folded), tl.full(position.shape, True, tl.int1)
    period = tl.maximum(1, 2 * (tl.cast(size, tl.int64) - 1))
    folded = position % period
    folded = tl.where(folded < 0, folded + period, folded)
    return tl.where(folded >= size, period - folded, folded), tl.full(position.shape, True, tl.int1)


@triton.jit
def _pixel_indices(width, height, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    linear = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    pixel = linear // CHANNELS
    return pixel % width, pixel // width, linear % CHANNELS, pixel < tl.cast(width, tl.int64) * height


@triton.jit
def _load_pixel(src, x, y, channel, width, height, stride, mask,
                CHANNELS: tl.constexpr, BORDER: tl.constexpr):
    px, valid_x = _coordinates(x, width, BORDER)
    py, valid_y = _coordinates(y, height, BORDER)
    return tl.load(src + py * stride + px * CHANNELS + channel, mask & valid_x & valid_y, 0)


@triton.jit
def _rounded_i32(value):
    # On the fixed SSE3 reference, an invalid signed conversion yields INT_MIN.
    floor = tl.floor(value)
    safe = (floor >= -2147483648.0) & (floor < 2147483648.0)
    base = tl.where(safe, floor, 0).to(tl.int64)
    fraction = value - floor
    up = (fraction > 0.5) | ((fraction == 0.5) & ((base & 1) != 0))
    rounded = base + up.to(tl.int64)
    return tl.where(safe & (rounded <= 2147483647), rounded, -2147483648)


@triton.jit
def filter2d_direct(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                    kernel_width, kernel_height, anchor_x, anchor_y,
                    CHANNELS: tl.constexpr, BORDER: tl.constexpr, FP64: tl.constexpr,
                    ZERO_FILTER: tl.constexpr, DEST_INTEGER: tl.constexpr,
                    DEST_MIN: tl.constexpr, DEST_MAX: tl.constexpr,
                    BLOCK: tl.constexpr):
    x, y, channel, mask = _pixel_indices(width, height, CHANNELS, BLOCK)
    if FP64:
        work: tl.constexpr = tl.float64
    else:
        work: tl.constexpr = tl.float32
    accum = tl.full((BLOCK,), 0, work) + tl.load(coeff_ptr).to(work)
    for ky in range(kernel_height):
        for kx in range(kernel_width):
            weight = tl.load(coeff_ptr + 1 + ky * kernel_width + kx).to(work)
            # OpenCV preserves one zero tap when every coefficient is zero.
            if weight != 0 or (ZERO_FILTER and ky == 0 and kx == 0):
                value = _load_pixel(src_ptr, x + kx - anchor_x, y + ky - anchor_y,
                                    channel, width, height, src_stride, mask, CHANNELS, BORDER)
                accum += value.to(work) * weight
    if DEST_INTEGER:
        rounded = _rounded_i32(accum)
        output = tl.minimum(tl.maximum(rounded, DEST_MIN), DEST_MAX)
        tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, output, mask)
    else:
        tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, accum, mask)


@triton.jit
def gaussian_direct(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                    kernel_width, kernel_height, CHANNELS: tl.constexpr, BORDER: tl.constexpr,
                    FP64: tl.constexpr, FIXED_BITS: tl.constexpr, BLOCK: tl.constexpr):
    x, y, channel, mask = _pixel_indices(width, height, CHANNELS, BLOCK)
    address = dst_ptr + y * dst_stride + x * CHANNELS + channel
    if kernel_width == 1 and kernel_height == 1:
        original = tl.load(src_ptr + y * src_stride + x * CHANNELS + channel, mask, 0)
        tl.store(address, original, mask)
    else:
        if FIXED_BITS == 16:
            row_type: tl.constexpr = tl.uint32
            work: tl.constexpr = tl.uint64
        elif FIXED_BITS == 8:
            row_type: tl.constexpr = tl.uint32
            work: tl.constexpr = tl.uint32
        elif FP64:
            row_type: tl.constexpr = tl.float64
            work: tl.constexpr = tl.float64
        else:
            row_type: tl.constexpr = tl.float32
            work: tl.constexpr = tl.float32
        accum = tl.full((BLOCK,), 0, work)
        for ky in range(kernel_height):
            row = tl.full((BLOCK,), 0, row_type)
            for kx in range(kernel_width):
                value = _load_pixel(src_ptr, x + kx - kernel_width // 2,
                                    y + ky - kernel_height // 2, channel, width, height,
                                    src_stride, mask, CHANNELS, BORDER)
                row += value.to(row_type) * tl.load(coeff_ptr + kx).to(row_type)
            weight = tl.load(coeff_ptr + kernel_width + ky).to(work)
            accum += row.to(work) * weight
        if FIXED_BITS:
            shift: tl.constexpr = 2 * FIXED_BITS
            result = (accum + (1 << (shift - 1))) >> shift
            tl.store(address, result, mask)
        else:
            tl.store(address, accum, mask)


@triton.jit
def gaussian_horizontal(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                        kernel_width, CHANNELS: tl.constexpr, BORDER: tl.constexpr,
                        FP64: tl.constexpr, FIXED_BITS: tl.constexpr, BLOCK: tl.constexpr):
    x, y, channel, mask = _pixel_indices(width, height, CHANNELS, BLOCK)
    if FIXED_BITS:
        work: tl.constexpr = tl.uint32
    elif FP64:
        work: tl.constexpr = tl.float64
    else:
        work: tl.constexpr = tl.float32
    accum = tl.full((BLOCK,), 0, work)
    for k in range(kernel_width):
        value = _load_pixel(src_ptr, x + k - kernel_width // 2, y, channel,
                            width, height, src_stride, mask, CHANNELS, BORDER)
        accum += value.to(work) * tl.load(coeff_ptr + k).to(work)
    # The integer scratch view has i32 storage and carries unsigned row bits.
    tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, accum, mask)


@triton.jit
def gaussian_vertical(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                      kernel_width, kernel_height, CHANNELS: tl.constexpr,
                      BORDER: tl.constexpr, FP64: tl.constexpr, FIXED_BITS: tl.constexpr,
                      BLOCK: tl.constexpr):
    x, y, channel, mask = _pixel_indices(width, height, CHANNELS, BLOCK)
    if FIXED_BITS == 16:
        work: tl.constexpr = tl.uint64
    elif FIXED_BITS == 8:
        work: tl.constexpr = tl.uint32
    elif FP64:
        work: tl.constexpr = tl.float64
    else:
        work: tl.constexpr = tl.float32
    accum = tl.full((BLOCK,), 0, work)
    for k in range(kernel_height):
        value = _load_pixel(src_ptr, x, y + k - kernel_height // 2, channel,
                            width, height, src_stride, mask, CHANNELS, BORDER)
        if FIXED_BITS:
            value = value.to(tl.uint32)
        accum += value.to(work) * tl.load(coeff_ptr + kernel_width + k).to(work)
    if FIXED_BITS:
        shift: tl.constexpr = 2 * FIXED_BITS
        accum = (accum + (1 << (shift - 1))) >> shift
    tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, accum, mask)


@triton.jit
def filter2d_dense(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                   row_values, kernel_width, kernel_height, anchor_x, anchor_y,
                   CHANNELS: tl.constexpr, BORDER: tl.constexpr, FP64: tl.constexpr,
                   ZERO_FILTER: tl.constexpr, DEST_INTEGER: tl.constexpr,
                   DEST_MIN: tl.constexpr, DEST_MAX: tl.constexpr,
                   BLOCK: tl.constexpr, ROW_PAIR: tl.constexpr):
    if not ROW_PAIR:
        filter2d_direct(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                        kernel_width, kernel_height, anchor_x, anchor_y, CHANNELS,
                        BORDER, FP64, ZERO_FILTER, DEST_INTEGER, DEST_MIN, DEST_MAX, BLOCK)
    else:
        tiles = tl.cdiv(tl.cast(row_values, tl.int64), BLOCK)
        tile = tl.program_id(0).to(tl.int64)
        element = (tile % tiles) * BLOCK + tl.arange(0, BLOCK)
        y = 2 * (tile // tiles)
        x = element // CHANNELS
        channel = element % CHANNELS
        mask = element < row_values
        delta = tl.load(coeff_ptr).to(tl.float32)
        top = tl.full((BLOCK,), 0, tl.float32) + delta
        bottom = tl.full((BLOCK,), 0, tl.float32) + delta
        # r is a virtual source row relative to the top output. Every
        # overlapping row is loaded once. Each accumulator still sees its
        # coefficients in increasing row/column order, starting at delta.
        for r in range(kernel_height + 1):
            sy, valid_y = _coordinates(y + r - anchor_y, height, BORDER)
            for kx in range(kernel_width):
                sx, valid_x = _coordinates(x + kx - anchor_x, width, BORDER)
                pixel = tl.load(src_ptr + sy * src_stride + sx * CHANNELS + channel,
                                mask & valid_x & valid_y, 0)
                if r < kernel_height:
                    weight = tl.load(coeff_ptr + 1 + r * kernel_width + kx).to(tl.float32)
                    top += pixel * weight
                if r > 0:
                    weight = tl.load(coeff_ptr + 1 + (r - 1) * kernel_width + kx).to(tl.float32)
                    bottom += pixel * weight
        tl.store(dst_ptr + y * dst_stride + element, top, mask)
        tl.store(dst_ptr + (y + 1) * dst_stride + element, bottom, mask & (y + 1 < height))
