# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def _positive_mod(value, period):
    residue = value % period
    return tl.where(residue < 0, residue + period, residue)


@triton.jit
def _border_coordinate(value, extent, MODE: tl.constexpr):
    if MODE == 1:
        return tl.minimum(tl.maximum(value, 0), extent - 1)
    extent = extent.to(tl.int64)
    if MODE == 2:
        period = 2 * extent
        residue = _positive_mod(value, period)
        return tl.where(residue < extent, residue, period - 1 - residue)
    if MODE == 3:
        period = tl.maximum(2 * (extent - 1), 1)
        residue = _positive_mod(value, period)
        return tl.where(residue < extent, residue, period - residue)
    return _positive_mod(value, extent)


@triton.jit
def copy_make_border(src_ptr, dst_ptr, src_w, src_h, dst_h, row_values, src_rs, dst_rs,
                     top, left, lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3,
                     CHANNELS: tl.constexpr, MODE: tl.constexpr, BITS: tl.constexpr,
                     BLOCK: tl.constexpr):
    if BITS == 8:
        Raw: tl.constexpr = tl.uint8
    elif BITS == 16:
        Raw: tl.constexpr = tl.uint16
    elif BITS == 32:
        Raw: tl.constexpr = tl.uint32
    else:
        Raw: tl.constexpr = tl.uint64
    src = src_ptr.to(tl.pointer_type(Raw))
    dst = dst_ptr.to(tl.pointer_type(Raw))
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row = index // row_values
    element = index % row_values
    x = element // CHANNELS - left
    y = row - top
    channel = element % CHANNELS
    valid = row < dst_h
    if MODE == 0:
        inside = (x >= 0) & (x < src_w) & (y >= 0) & (y < src_h)
        value = tl.load(src + y * src_rs + x * CHANNELS + channel,
                        mask=valid & inside, other=0)
        lo = tl.where(channel == 0, lo0, tl.where(channel == 1, lo1,
                      tl.where(channel == 2, lo2, lo3))).to(tl.uint32)
        if BITS == 64:
            hi = tl.where(channel == 0, hi0, tl.where(channel == 1, hi1,
                          tl.where(channel == 2, hi2, hi3))).to(tl.uint32)
            fill = lo.to(tl.uint64) | (hi.to(tl.uint64) << 32)
        else:
            fill = lo.to(Raw)
        value = tl.where(inside, value, fill)
    else:
        x = _border_coordinate(x, src_w, MODE)
        y = _border_coordinate(y, src_h, MODE)
        value = tl.load(src + y * src_rs + x * CHANNELS + channel, mask=valid, other=0)
    tl.store(dst + row * dst_rs + element, value, mask=valid)
