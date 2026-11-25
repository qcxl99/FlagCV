# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def _round_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    odd = (lower.to(tl.int32) & 1) != 0
    return lower + ((fraction > 0.5) | ((fraction == 0.5) & odd)).to(value.dtype)


@triton.jit
def _round_i32(value):
    valid = (value >= -2147483648.0) & (value < 2147483648.0)
    safe = tl.where(valid, value, 0.0)
    return tl.where(valid, _round_even(safe).to(tl.int32), -2147483648)


@triton.jit
def _inside(x, y, width, height):
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


@triton.jit
def _border_index(index, size, BORDER: tl.constexpr):
    if BORDER == 1:
        return tl.minimum(tl.maximum(index, 0), size - 1)
    if BORDER == 2:
        period = 2 * size
    elif BORDER == 3:
        period = size
    else:
        period = tl.maximum(2 * (size - 1), 1)
    if BORDER == 3:
        # SSE3's borderInterpolate_fast uses signed 32-bit arithmetic.
        # Its negative wrap branch also accepts cvRound's INT_MIN sentinel.
        index = tl.where(index < 0, index - ((index - size + 1) // size) * size, index)
    folded = index % period
    folded = tl.where(folded < 0, folded + period, folded)
    if BORDER == 2:
        return tl.where(folded < size, folded, period - 1 - folded)
    if BORDER == 4:
        return tl.where(folded < size, folded, period - folded)
    return folded


@triton.jit
def _tap(src, ix, iy, src_w, src_h, src_rs, mask, other,
         CN: tl.constexpr, C: tl.constexpr, BORDER: tl.constexpr):
    if BORDER == 0:
        mask = mask & _inside(ix, iy, src_w, src_h)
    else:
        ix = _border_index(ix, src_w, BORDER)
        iy = _border_index(iy, src_h, BORDER)
    return tl.load(src + iy.to(tl.int64) * src_rs + ix.to(tl.int64) * CN + C, mask, other=other)


@triton.jit
def _rounded_pixel(acc, DTYPE: tl.constexpr):
    if DTYPE == tl.uint8 or DTYPE == tl.uint16:
        maximum: tl.constexpr = 255 if DTYPE == tl.uint8 else 65535
        rounded = _round_i32(acc)
        return tl.minimum(tl.maximum(rounded, 0), maximum).to(DTYPE)
    else:
        return acc


@triton.jit
def _sample_float(src, dst, x, y, sx, sy, src_h, src_w, dst_w, src_rs, dst_rs,
                  bv0, bv1, bv2, bv3, active, CN: tl.constexpr,
                  INTER: tl.constexpr, BORDER: tl.constexpr, BLOCK: tl.constexpr):
    dtype: tl.constexpr = src.dtype.element_ty
    if INTER == 0:
        ix = _round_i32(sx)
        iy = _round_i32(sy)
    else:
        # SSE3 cvFloor uses an int32 overflow sentinel and a wrapped decrement.
        ix = _round_i32(tl.floor(sx))
        iy = _round_i32(tl.floor(sy))
        ix = tl.where(sx < -2147483648.0, 2147483647, ix)
        iy = tl.where(sy < -2147483648.0, 2147483647, iy)
        ax = sx - ix.to(tl.float32)
        ay = sy - iy.to(tl.float32)
    store = active
    sample = active
    if BORDER == 0 or BORDER == 5:
        if INTER == 0:
            within = _inside(ix, iy, src_w, src_h)
        else:
            within = (ix >= -1) & (iy >= -1) & (ix < src_w) & (iy < src_h)
            lane = tl.arange(0, BLOCK)
            group = lane // 8 * 8
            has_sample = tl.full((BLOCK,), False, tl.int1)
            for peer in tl.static_range(8):
                has_sample = has_sample | tl.gather(within.to(tl.int32), group + peer, 0).to(tl.int1)
            within = tl.where(x < dst_w // 8 * 8, has_sample, within)
        sample = sample & within
        if BORDER == 5:
            store = sample

    tap_border: tl.constexpr = 0 if BORDER == 5 else BORDER
    for channel in tl.static_range(CN):
        address = dst + y.to(tl.int64) * dst_rs + x * CN + channel
        if BORDER == 5:
            bv = tl.load(address, sample, other=0)
        else:
            bv = (bv0, bv1, bv2, bv3)[channel].to(dtype)
        out = _tap(src, ix, iy, src_w, src_h, src_rs, sample, bv, CN, channel, tap_border)
        if INTER == 1:
            a = out.to(tl.float32)
            b = _tap(src, ix + 1, iy, src_w, src_h, src_rs, sample, bv,
                     CN, channel, tap_border).to(tl.float32)
            c = _tap(src, ix, iy + 1, src_w, src_h, src_rs, sample, bv,
                     CN, channel, tap_border).to(tl.float32)
            d = _tap(src, ix + 1, iy + 1, src_w, src_h, src_rs, sample, bv,
                     CN, channel, tap_border).to(tl.float32)
            top = a + ax * (b - a)
            bottom = c + ax * (d - c)
            out = _rounded_pixel(top + ay * (bottom - top), dtype)
            if BORDER == 0:
                out = tl.where(within, out, bv)
        tl.store(address, out, store)


@triton.jit
def _sample_table(src, dst, x, y, ix, iy, qx, qy, src_h, src_w, src_rs,
                  dst_rs, bv0, bv1, bv2, bv3, mask, CN: tl.constexpr,
                  INTER: tl.constexpr, BORDER: tl.constexpr):
    dtype: tl.constexpr = src.dtype.element_ty
    inside = _inside(ix, iy, src_w, src_h)
    if BORDER == 5:
        sample_mask = mask & inside
        store_mask = sample_mask
    else:
        sample_mask = mask
        store_mask = mask

    if INTER == 1:
        if dtype == tl.uint8:
            # 15-bit table, with the unit coefficient saturated to int16.
            w0 = tl.minimum((32 - qx) * (32 - qy) * 32, 32767)
            w1 = qx * (32 - qy) * 32
            w2 = (32 - qx) * qy * 32
            w3 = qx * qy * 32
        else:
            w0 = ((32 - qx) * (32 - qy)).to(tl.float32) * (1.0 / 1024)
            w1 = (qx * (32 - qy)).to(tl.float32) * (1.0 / 1024)
            w2 = ((32 - qx) * qy).to(tl.float32) * (1.0 / 1024)
            w3 = (qx * qy).to(tl.float32) * (1.0 / 1024)
        if BORDER == 5:
            right = ix + 1 < src_w
            bottom = iy + 1 < src_h
            weights = w0 + w1 + w2 + w3
            present = w0 + tl.where(right, w1, 0) + tl.where(bottom, w2, 0)
            present = present + tl.where(right & bottom, w3, 0)
            sample_mask = sample_mask & (present != 0)
            store_mask = store_mask & (~inside | (present != 0))

    tap_border: tl.constexpr = 0 if BORDER == 5 else BORDER
    for channel in tl.static_range(CN):
        bv = (bv0, bv1, bv2, bv3)[channel].to(dtype)
        if BORDER == 5:
            bv = tl.full((), 0, dtype)
        out = _tap(src, ix, iy, src_w, src_h, src_rs, sample_mask, bv, CN, channel, tap_border)
        if INTER == 1:
            b = _tap(src, ix + 1, iy, src_w, src_h, src_rs, sample_mask, bv, CN, channel, tap_border)
            c = _tap(src, ix, iy + 1, src_w, src_h, src_rs, sample_mask, bv, CN, channel, tap_border)
            d = _tap(src, ix + 1, iy + 1, src_w, src_h, src_rs, sample_mask, bv, CN, channel, tap_border)
            acc = out * w0 + b * w1 + c * w2 + d * w3
            if BORDER == 5:
                denominator = tl.where(present == 0, 1, present).to(tl.float32)
                partial = tl.div_rn(acc.to(tl.float32) * weights, denominator)
                acc = tl.where(right & bottom, acc, partial.to(acc.dtype))
            if dtype == tl.uint8:
                out = ((acc + 16384) >> 15).to(dtype)
            else:
                out = _rounded_pixel(acc, dtype)
        if BORDER == 0 and INTER == 1:
            entirely_outside = (ix < -1) | (iy < -1) | (ix >= src_w) | (iy >= src_h)
            out = tl.where(entirely_outside, bv, out)
        address = dst + y.to(tl.int64) * dst_rs + x * CN + channel
        tl.store(address, out, store_mask)


@triton.jit
def _cubic_weights(a):
    aa = a * a
    b = 1.0 - a
    bb = b * b
    w0 = -0.75 * (a * bb)
    w1 = aa * (1.25 * a - 2.25) + 1.0
    w3 = -0.75 * (aa * b)
    w2 = 1.0 - w0 - w1 - w3
    return w0, w1, w2, w3


@triton.jit
def _cubic_axes(sx, sy, width, height):
    bw = tl.maximum(width, 16).to(tl.float32)
    bh = tl.maximum(height, 16).to(tl.float32)
    sx = tl.minimum(tl.maximum(sx, -bw), 2.0 * bw, propagate_nan=tl.PropagateNan.NONE)
    sy = tl.minimum(tl.maximum(sy, -bh), 2.0 * bh, propagate_nan=tl.PropagateNan.NONE)
    fx = tl.floor(sx)
    fy = tl.floor(sy)
    return fx.to(tl.int32) - 1, fy.to(tl.int32) - 1, sx - fx, sy - fy


@triton.jit
def _cubic_outside(ix, iy, width, height):
    return (ix >= width) | (ix + 4 < 0) | (iy >= height) | (iy + 4 < 0)


@triton.jit
def _cubic_inlier(ix, iy, width, height):
    return (ix >= 0) & (ix < width - 3) & (iy >= 0) & (iy < height - 5)


@triton.jit
def _cubic_float_border_tap(src, ix, iy, src_w, src_h, src_rs, mask, bv, inner,
                            CN: tl.constexpr, C: tl.constexpr):
    valid = _inside(ix, iy, src_w, src_h)
    x = tl.minimum(tl.maximum(ix, 0), src_w - 1)
    y = tl.minimum(tl.maximum(iy, 0), src_h - 1)
    value = tl.load(src + y.to(tl.int64) * src_rs + x.to(tl.int64) * CN + C, mask, other=0)
    # The fixed CPU fetches the clamped source even for missing taps.
    # Masking via arithmetic preserves its NaN/Inf behavior in mixed groups.
    edge = value * valid.to(tl.float32) + bv * (~valid).to(tl.float32)
    return tl.where(inner, value, edge)


@triton.jit
def _sample_cubic(src, dst, x, y, ix, iy, ax, ay, all_out, inner, src_h, src_w, src_rs, dst_rs,
                  bv0, bv1, bv2, bv3, mask, CN: tl.constexpr, BORDER: tl.constexpr, BLOCK: tl.constexpr):
    wx = _cubic_weights(ax)
    wy = _cubic_weights(ay)
    dtype: tl.constexpr = src.dtype.element_ty
    tap_border: tl.constexpr = 0 if BORDER == 5 else BORDER
    for channel in tl.static_range(CN):
        address = dst + y.to(tl.int64) * dst_rs + x * CN + channel
        if BORDER == 5:
            # OpenCV's cubic edge helper uses the first pixel of this
            # four-wide group as its constant for all missing taps.
            old = tl.load(address, mask, other=0)
            group = tl.arange(0, BLOCK) // 4 * 4
            bv = tl.gather(old, group, 0)
        else:
            bv = (bv0, bv1, bv2, bv3)[channel].to(dtype)
        acc = tl.full((), 0, tl.float32)
        for r in tl.static_range(4):
            row = tl.full((), 0, tl.float32)
            for c in tl.static_range(4):
                if dtype == tl.float32 and (BORDER == 0 or BORDER == 5):
                    v = _cubic_float_border_tap(src, ix + c, iy + r, src_w, src_h, src_rs,
                                                mask & ~all_out, bv, inner, CN, channel)
                else:
                    v = _tap(src, ix + c, iy + r, src_w, src_h, src_rs,
                             mask & ~all_out, bv, CN, channel, tap_border)
                row = row + v.to(tl.float32) * wx[c]
            acc = acc + row * wy[r]
        out = _rounded_pixel(acc, dtype)
        if BORDER == 0:
            out = tl.where(all_out, bv, out)
        tl.store(address, out, mask & ~all_out if BORDER == 5 else mask)


@triton.jit
def _project_cubic(x, y, m00, m01, m02, m10, m11, m12, m20, m21, m22):
    yd = y.to(tl.float32)
    xd = x.to(tl.float64)
    rx = (yd * m01.to(tl.float32) + m02.to(tl.float32)).to(tl.float64)
    ry = (yd * m11.to(tl.float32) + m12.to(tl.float32)).to(tl.float64)
    rw = (yd * m21.to(tl.float32) + m22.to(tl.float32)).to(tl.float64)
    ratio = 1.0 / (rw + m20.to(tl.float32).to(tl.float64) * xd)
    return ((rx + m00.to(tl.float32).to(tl.float64) * xd) * ratio).to(tl.float32), (
            (ry + m10.to(tl.float32).to(tl.float64) * xd) * ratio).to(tl.float32)


@triton.jit
def _cubic_group(outside, active, BLOCK: tl.constexpr):
    groups = tl.reshape((outside & active).to(tl.int32), (BLOCK // 4, 4))
    all_outside = tl.min(groups, axis=1)[:, None] != 0
    return tl.reshape(tl.broadcast_to(all_outside, (BLOCK // 4, 4)), (BLOCK,))


@triton.jit
def _shifted_cubic_tail(width):
    chunk_tail = width % 128
    return (width % 4 != 0) & (chunk_tail >= 4)


@triton.jit
def _perspective_column(column, width, BLOCK: tl.constexpr):
    # A rectangular split grid may pad many programs past the last valid tile.
    # Bound the program index before multiplication can overflow signed i32.
    valid = column <= (width - 1) // BLOCK
    return tl.where(valid, column, 0), valid


@triton.jit
def perspective_kernel(src_ptr, dst_ptr, m00, m01, m02, m10, m11, m12, m20, m21, m22,
                       src_h, src_w, dst_h, dst_w, src_rs, dst_rs, bv0, bv1, bv2, bv3,
                       CN: tl.constexpr, INTER: tl.constexpr, BORDER: tl.constexpr,
                       BLOCK: tl.constexpr, COL_GRID_LIMIT: tl.constexpr):
    column = tl.program_id(1)
    if COL_GRID_LIMIT:
        column += tl.program_id(2) * COL_GRID_LIMIT
    column, valid = _perspective_column(column, dst_w, BLOCK)
    x = column * BLOCK + tl.arange(0, BLOCK)
    y = tl.program_id(0)
    active = valid & (x < dst_w)
    a, b, c = m00, m01, m02
    d, e, f = m10, m11, m12
    g, h, i = m20, m21, m22
    if INTER == 2:
        sx, sy = _project_cubic(x, y, a, b, c, d, e, f, g, h, i)
        ix, iy, ax, ay = _cubic_axes(sx, sy, src_w, src_h)
        inner = tl.full((BLOCK,), False, tl.int1)
        fp32: tl.constexpr = src_ptr.dtype.element_ty == tl.float32
        if BORDER == 0 or BORDER == 5:
            if fp32:
                inner = _cubic_group(_cubic_inlier(ix, iy, src_w, src_h), active, BLOCK)
            outside = _cubic_group(_cubic_outside(ix, iy, src_w, src_h), active, BLOCK)
            if BORDER == 0:
                if _shifted_cubic_tail(dst_w) and (column + 1) * BLOCK > dst_w - 4 and column * BLOCK < dst_w:
                    px, py = _project_cubic(dst_w - 4 + tl.arange(0, 4), y, a, b, c, d, e, f, g, h, i)
                    tx, ty, _, _ = _cubic_axes(px, py, src_w, src_h)
                    tail = tl.min(_cubic_outside(tx, ty, src_w, src_h).to(tl.int32), 0) != 0
                    outside = tl.where(x >= dst_w - 4, tail, outside)
                    if fp32:
                        tail_inner = tl.min(_cubic_inlier(tx, ty, src_w, src_h).to(tl.int32), 0) != 0
                        inner = tl.where(x >= dst_w - 4, tail_inner, inner)
        else:
            outside = tl.full((), False, tl.int1)
        _sample_cubic(src_ptr, dst_ptr, x, y, ix, iy, ax, ay, outside, inner, src_h, src_w,
                      src_rs, dst_rs, bv0, bv1, bv2, bv3, active, CN, BORDER, BLOCK)
    else:
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        vector = x < dst_w // 8 * 8
        nx = tl.where(vector, xf * a + (yf * b + c), xf * a + yf * b + c)
        ny = tl.where(vector, xf * d + (yf * e + f), xf * d + yf * e + f)
        nw = tl.where(vector, xf * g + (yf * h + i), xf * g + yf * h + i)
        sx, sy = tl.div_rn(nx, nw), tl.div_rn(ny, nw)
        _sample_float(src_ptr, dst_ptr, x, y, sx, sy, src_h, src_w, dst_w, src_rs,
                      dst_rs, bv0, bv1, bv2, bv3, active, CN, INTER, BORDER, BLOCK)


@triton.jit
def _maps(map1, map2, x, y, map1_rs, map2_rs, mask, LAYOUT: tl.constexpr):
    first = y.to(tl.int64) * map1_rs
    second = y.to(tl.int64) * map2_rs
    if LAYOUT == 1:
        sx = tl.load(map1 + first + x, mask, other=0)
        sy = tl.load(map2 + second + x, mask, other=0)
    else:
        sx = tl.load(map1 + first + 2 * x, mask, other=0)
        sy = tl.load(map1 + first + 2 * x + 1, mask, other=0)
    if LAYOUT == 2:
        frac = tl.load(map2 + second + x, mask, other=0).to(tl.int32)
    else:
        frac = tl.full((), 0, tl.int32)
    return sx, sy, frac


@triton.jit
def _map_cubic(map1, map2, x, y, map1_rs, map2_rs, mask,
               LAYOUT: tl.constexpr, RELATIVE: tl.constexpr):
    sx, sy, frac = _maps(map1, map2, x, y, map1_rs, map2_rs, mask, LAYOUT)
    if LAYOUT == 2:
        sx = sx.to(tl.float32) + (frac & 31).to(tl.float32) * (1.0 / 32)
        sy = sy.to(tl.float32) + ((frac >> 5) & 31).to(tl.float32) * (1.0 / 32)
    if RELATIVE:
        sx = sx + x.to(tl.float32)
        sy = sy + y.to(tl.float32)
    return sx, sy


@triton.jit
def remap_kernel(src_ptr, dst_ptr, map1_ptr, map2_ptr, map1_rs, map2_rs,
                 src_h, src_w, dst_h, dst_w, src_rs, dst_rs, bv0, bv1, bv2, bv3,
                 CN: tl.constexpr, INTER: tl.constexpr, BORDER: tl.constexpr,
                 LAYOUT: tl.constexpr, RELATIVE: tl.constexpr,
                 BLOCK: tl.constexpr, COL_GRID_LIMIT: tl.constexpr):
    column = tl.program_id(1)
    if COL_GRID_LIMIT:
        column += tl.program_id(2) * COL_GRID_LIMIT
    x = column * BLOCK + tl.arange(0, BLOCK)
    y = tl.program_id(0)
    active = (x < dst_w) & (y < dst_h)
    if INTER == 2:
        sx, sy = _map_cubic(map1_ptr, map2_ptr, x, y, map1_rs, map2_rs, active, LAYOUT, RELATIVE)
        ix, iy, ax, ay = _cubic_axes(sx, sy, src_w, src_h)
        inner = tl.full((BLOCK,), False, tl.int1)
        fp32: tl.constexpr = src_ptr.dtype.element_ty == tl.float32
        if BORDER == 0 or BORDER == 5:
            if fp32:
                inner = _cubic_group(_cubic_inlier(ix, iy, src_w, src_h), active, BLOCK)
            outside = _cubic_group(_cubic_outside(ix, iy, src_w, src_h), active, BLOCK)
            if BORDER == 0:
                if _shifted_cubic_tail(dst_w) and (column + 1) * BLOCK > dst_w - 4 and column * BLOCK < dst_w:
                    xp = dst_w - 4 + tl.arange(0, 4)
                    px, py = _map_cubic(map1_ptr, map2_ptr, xp, y, map1_rs, map2_rs,
                                        xp < dst_w, LAYOUT, RELATIVE)
                    tx, ty, _, _ = _cubic_axes(px, py, src_w, src_h)
                    tail = tl.min(_cubic_outside(tx, ty, src_w, src_h).to(tl.int32), 0) != 0
                    outside = tl.where(x >= dst_w - 4, tail, outside)
                    if fp32:
                        tail_inner = tl.min(_cubic_inlier(tx, ty, src_w, src_h).to(tl.int32), 0) != 0
                        inner = tl.where(x >= dst_w - 4, tail_inner, inner)
        else:
            outside = tl.full((), False, tl.int1)
        _sample_cubic(src_ptr, dst_ptr, x, y, ix, iy, ax, ay, outside, inner, src_h, src_w,
                      src_rs, dst_rs, bv0, bv1, bv2, bv3, active, CN, BORDER, BLOCK)
    else:
        sx, sy, frac = _maps(map1_ptr, map2_ptr, x, y, map1_rs, map2_rs, active, LAYOUT)
        if LAYOUT < 2:
            if RELATIVE:
                sx, sy = sx + x.to(tl.float32), sy + y.to(tl.float32)
            _sample_float(src_ptr, dst_ptr, x, y, sx, sy, src_h, src_w, dst_w, src_rs,
                          dst_rs, bv0, bv1, bv2, bv3, active, CN, INTER, BORDER, BLOCK)
        else:
            fx, fy = frac & 31, (frac >> 5) & 31
            if LAYOUT == 2 and INTER == 0:
                sx = (sx.to(tl.int32) + (fx < 16).to(tl.int32)).to(tl.int16)
                sy = (sy.to(tl.int32) + (fy < 16).to(tl.int32)).to(tl.int16)
            ix, iy = sx.to(tl.int32), sy.to(tl.int32)
            if RELATIVE:
                ix, iy = ix + x, iy + y
            _sample_table(src_ptr, dst_ptr, x, y, ix, iy, fx, fy, src_h, src_w, src_rs,
                          dst_rs, bv0, bv1, bv2, bv3, active, CN, INTER, BORDER)
