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
def _limit_finite_u16(value):
    return tl.where((value > 65535.0) & (value < float("inf")), 65535.0, value)


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
        if DEST_MAX == 65535:
            accum = _limit_finite_u16(accum)
        rounded = _rounded_i32(accum)
        output = tl.minimum(tl.maximum(rounded, DEST_MIN), DEST_MAX)
        tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, output, mask)
    else:
        tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, accum, mask)


@triton.jit
def _two_sum(a, b):
    total = a + b
    offset = total - a
    return total, (a - (total - offset)) + (b - offset)


@triton.jit
def _compensated_add(high, low, x):
    summed = high + x
    error = tl.where(tl.abs(high) >= tl.abs(x), (high - summed) + x, (x - summed) + high)
    return summed, low + error


@triton.jit
def _round_pair_f32(high, tail):
    nearest = high.to(tl.float32)
    nearest_wide = nearest.to(tl.float64)
    bits = tl.cast(nearest, tl.uint32, bitcast=True)
    direction_up = high > nearest_wide
    signed = (bits & 0x80000000) != 0
    bits_next = tl.where(direction_up != signed, bits + 1, bits - 1)
    bits_next = tl.where(nearest == 0.0, tl.where(direction_up, 1, 0x80000001), bits_next)
    neighbor = tl.cast(bits_next.to(tl.uint32), tl.float32, bitcast=True)
    neighbor_wide = neighbor.to(tl.float64)
    # The finite FP32 overflow midpoint is between MAX and 2^128.
    mag = tl.abs(nearest_wide)
    neighbor_mag = tl.abs(neighbor_wide)
    overflow = mag > 1.0e300
    native = tl.where(overflow, tl.where(signed, -340282366920938463463374607431768211456.0, 340282366920938463463374607431768211456.0), nearest_wide)
    other = tl.where(neighbor_mag > 1.0e300,
                     tl.where(signed, -340282366920938463463374607431768211456.0, 340282366920938463463374607431768211456.0), neighbor_wide)
    midpoint = (native + other) * 0.5
    crosses = (high == midpoint) & tl.where(direction_up, tail > 0.0, tail < 0.0)
    return tl.where(crosses, neighbor, nearest)


@triton.jit
def _round_pair_i32(high, tail):
    floor = tl.floor(high)
    safe = (floor >= -2147483648.0) & (floor < 2147483648.0)
    base = tl.where(safe, floor, 0).to(tl.int64)
    fraction = high - floor
    up = (fraction > 0.5) | ((fraction == 0.5) & ((tail > 0.0) | ((tail == 0.0) & ((base & 1) != 0))))
    rounded = base + up.to(tl.int64)
    return tl.where(safe & (rounded <= 2147483647), rounded, -2147483648)


@triton.jit
def filter2d_dft_precision(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                           kernel_width, kernel_height, anchor_x, anchor_y,
                           vector_columns, vector_rows, CHANNELS: tl.constexpr,
                           BORDER: tl.constexpr,
                           ADD_F32_DELTA: tl.constexpr, SCALED: tl.constexpr,
                           DEST_U16: tl.constexpr, COMPENSATED: tl.constexpr, BLOCK: tl.constexpr):
    x, y, channel, mask = _pixel_indices(width, height, CHANNELS, BLOCK)
    work: tl.constexpr = tl.float64
    value = tl.full((BLOCK,), 0, work)
    low = tl.full((BLOCK,), 0, work)
    for ky in range(kernel_height):
        for kx in range(kernel_width):
            weight = tl.load(coeff_ptr + 1 + ky * kernel_width + kx).to(tl.float32)
            if weight != 0:
                pixel = _load_pixel(src_ptr, x + kx - anchor_x, y + ky - anchor_y,
                                    channel, width, height, src_stride, mask, CHANNELS, BORDER)
                product = pixel.to(work) * weight.to(work)
                if COMPENSATED:
                    value, low = _compensated_add(value, low, product)
                else:
                    value += product
    if COMPENSATED:
        value, low = _two_sum(value, low)
    tail = tl.full((BLOCK,), 0, work)
    if ADD_F32_DELTA:
        # The multichannel CPU path stores correlation in a float temporary,
        # performs the float scalar add, then optionally converts to U16.
        if COMPENSATED:
            narrowed = _round_pair_f32(value, low)
        else:
            narrowed = value.to(tl.float32)
        value_f32 = narrowed + tl.load(coeff_ptr).to(tl.float32)
        result = value_f32.to(work)
    elif SCALED:
        # Fixed SSE3 convertScale narrows correlation in vector tiles.
        delta = tl.load(coeff_ptr).to(tl.float32)
        if COMPENSATED:
            narrowed = _round_pair_f32(value, low)
        else:
            narrowed = value.to(tl.float32)
        vector = (narrowed + delta).to(work)
        vector_lane = (x < vector_columns) & (y < vector_rows)
        if COMPENSATED:
            high, low = _compensated_add(value, low, delta.to(work))
            scalar, scalar_tail = _two_sum(high, low)
            result = tl.where(vector_lane, vector, scalar)
            tail = tl.where(vector_lane, 0.0, scalar_tail)
        else:
            result = tl.where(vector_lane, vector, value + delta.to(work))
    else:
        result = value
        tail = low
    if DEST_U16:
        result = _limit_finite_u16(result)
        if COMPENSATED:
            signed = _round_pair_i32(result, tail)
        else:
            signed = _rounded_i32(result)
        output = tl.minimum(tl.maximum(signed, 0), 65535)
        tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, output, mask)
    else:
        tl.store(dst_ptr + y * dst_stride + x * CHANNELS + channel, result, mask)


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
def _local_gaussian_axis(position, size, BORDER: tl.constexpr):
    # Tiled lowering restricts extents to more than the halo radius.
    if BORDER == 0:
        return tl.minimum(tl.maximum(position, 0), size - 1), (position >= 0) & (position < size)
    if BORDER == 1:
        return tl.minimum(tl.maximum(position, 0), size - 1), tl.full(position.shape, True, tl.int1)
    if BORDER == 2:
        folded = tl.where(position < 0, -1 - position, position)
        return tl.where(folded >= size, 2 * size - 1 - folded, folded), tl.full(position.shape, True, tl.int1)
    folded = tl.abs(position)
    return tl.where(folded >= size, 2 * (size - 1) - folded, folded), tl.full(position.shape, True, tl.int1)


@triton.jit
def _gaussian_local_rows(src, dst, coeff, width, height, src_stride, dst_stride,
                         CHANNELS: tl.constexpr, BORDER: tl.constexpr, TAPS: tl.constexpr,
                         BLOCK: tl.constexpr, TILE_Y: tl.constexpr):
    # With 16 output rows and 7 or 15 taps, 32 slots hold the tile and its halo.
    ROWS: tl.constexpr = 32
    radius: tl.constexpr = TAPS // 2
    start = tl.program_id(0) * BLOCK
    elements = start + tl.arange(0, BLOCK)
    row_elements = width * CHANNELS
    start_y = tl.program_id(1) * TILE_Y
    stored = tl.arange(0, ROWS)
    y = start_y + stored - radius
    real_y = (y < height + radius) & (stored < TILE_Y + TAPS - 1)
    mapped_y, valid_y = _local_gaussian_axis(tl.minimum(y, height + radius - 1), height, BORDER)
    src_row = mapped_y[:, None] * src_stride
    valid_y = valid_y & real_y
    horizontal = tl.full((ROWS, BLOCK), 0, tl.float32)
    if start >= radius * CHANNELS and start + BLOCK + radius * CHANNELS <= row_elements:
        # Unit-stride center loads preserve coalescing for interleaved C3.
        for tap in tl.static_range(TAPS):
            value = tl.load(src + src_row + elements[None, :] + (tap - radius) * CHANNELS,
                            valid_y[:, None], 0)
            horizontal += value * tl.load(coeff + tap).to(tl.float32)
    else:
        x = tl.minimum(elements // CHANNELS, width - 1)
        channel = elements % CHANNELS
        for tap in tl.static_range(TAPS):
            mapped_x, valid_x = _local_gaussian_axis(x + tap - radius, width, BORDER)
            value = tl.load(src + src_row + (mapped_x * CHANNELS + channel)[None, :],
                            valid_y[:, None] & (valid_x & (elements < row_elements))[None, :], 0)
            horizontal += value * tl.load(coeff + tap).to(tl.float32)

    local = tl.arange(0, TILE_Y)
    output = tl.full((TILE_Y, BLOCK), 0, tl.float32)
    # Keep the same sequential FP32 sum of taps used by the vertical pass.
    for tap in tl.static_range(TAPS):
        slot = tl.broadcast_to((local + tap)[:, None], (TILE_Y, BLOCK))
        output += tl.gather(horizontal, slot, axis=0) * tl.load(coeff + TAPS + tap).to(tl.float32)
    out_y = start_y + local
    address = tl.minimum(out_y, height - 1)[:, None] * dst_stride + tl.minimum(elements, row_elements - 1)[None, :]
    tl.store(dst + address, output, (out_y < height)[:, None] & (elements < row_elements)[None, :])


@triton.jit
def gaussian_f32_local(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                       kernel_width, kernel_height, CHANNELS: tl.constexpr, BORDER: tl.constexpr,
                       FP64: tl.constexpr, FIXED_BITS: tl.constexpr, TAPS: tl.constexpr,
                       BLOCK: tl.constexpr, TILE_Y: tl.constexpr):
    if TILE_Y:
        _gaussian_local_rows(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                             CHANNELS, BORDER, TAPS, BLOCK, TILE_Y)
    else:
        gaussian_direct(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                        kernel_width, kernel_height, CHANNELS, BORDER, FP64, FIXED_BITS, BLOCK)


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
def _filter2d_interior_tile(src, dst, coefficients, width, height, src_stride, dst_stride,
                            kw, kh, ax, ay, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    columns: tl.constexpr = BLOCK // CHANNELS
    lane = tl.arange(0, BLOCK)
    start = tl.program_id(1).to(tl.int64) * columns
    x = start + lane // CHANNELS
    y = tl.program_id(0).to(tl.int64)
    channel = lane % CHANNELS
    valid = (lane < columns * CHANNELS) & (x < width)
    top = y - ay
    left = start - ax
    # One uniform branch per output tile. The boundary path retains reflect101.
    interior = (top >= 0) & (top + kh <= height) & (left >= 0) & (left + columns + kw - 1 <= width)
    value = tl.full((BLOCK,), 0, tl.float32) + tl.load(coefficients).to(tl.float32)
    if interior:
        offset = top * src_stride + (x - ax) * CHANNELS + channel
        for iy in range(kh):
            for ix in range(kw):
                weight = tl.load(coefficients + 1 + iy * kw + ix).to(tl.float32)
                if weight != 0:
                    sample = tl.load(src + offset + iy * src_stride + ix * CHANNELS, valid, 0)
                    value += sample * weight
    else:
        for iy in range(kh):
            for ix in range(kw):
                weight = tl.load(coefficients + 1 + iy * kw + ix).to(tl.float32)
                if weight != 0:
                    sample = _load_pixel(src, x + ix - ax, top + iy, channel, width, height,
                                        src_stride, valid, CHANNELS, 3)
                    value += sample * weight
    tl.store(dst + y * dst_stride + x * CHANNELS + channel, value, valid)


@triton.jit
def filter2d_interior(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                      kernel_width, kernel_height, anchor_x, anchor_y,
                      CHANNELS: tl.constexpr, BORDER: tl.constexpr, FP64: tl.constexpr,
                      ZERO_FILTER: tl.constexpr, DEST_INTEGER: tl.constexpr,
                      DEST_MIN: tl.constexpr, DEST_MAX: tl.constexpr,
                      BLOCK: tl.constexpr, INTERIOR_TILE: tl.constexpr):
    if INTERIOR_TILE:
        _filter2d_interior_tile(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                               kernel_width, kernel_height, anchor_x, anchor_y, CHANNELS, BLOCK)
    else:
        filter2d_direct(src_ptr, dst_ptr, coeff_ptr, width, height, src_stride, dst_stride,
                        kernel_width, kernel_height, anchor_x, anchor_y, CHANNELS,
                        BORDER, FP64, ZERO_FILTER, DEST_INTEGER, DEST_MIN, DEST_MAX, BLOCK)


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
