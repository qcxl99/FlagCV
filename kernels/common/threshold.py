# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def _words64(lo, hi):
    return tl.cast(lo, tl.uint32).to(tl.uint64) | (tl.cast(hi, tl.uint32).to(tl.uint64) << 32)


@triton.jit
def threshold_apply(src_ptr, dst_ptr, cutoff_ptr, rows, row_values, src_rs, dst_rs,
                    cut_lo, cut_hi, max_lo, max_hi,
                    MODE: tl.constexpr, KIND: tl.constexpr, AUTO: tl.constexpr,
                    BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row = index // row_values
    col = index % row_values
    valid = row < rows
    value = tl.load(src_ptr + row * src_rs + col, mask=valid, other=0)
    if KIND == 3:
        cutoff = tl.cast(cut_lo, tl.uint32).to(tl.float32, bitcast=True)
        maximum = tl.cast(max_lo, tl.uint32).to(tl.float32, bitcast=True)
    elif KIND == 4:
        cutoff = _words64(cut_lo, cut_hi).to(tl.float64, bitcast=True)
        maximum = _words64(max_lo, max_hi).to(tl.float64, bitcast=True)
    else:
        value = value.to(tl.int32)
        cutoff = tl.load(cutoff_ptr) if AUTO else cut_lo
        maximum = max_lo
    if MODE == 0:
        output = tl.where(value > cutoff, maximum, 0)
    elif MODE == 1:
        output = tl.where(value <= cutoff, maximum, 0)
    elif MODE == 2:
        output = tl.where(cutoff < value, cutoff, value)
        if KIND == 3 or KIND == 4:
            lane_count: tl.constexpr = 4 if KIND == 3 else 2
            tight = (src_rs == row_values) & (dst_rs == row_values)
            span = tl.where(tight, tl.cast(rows, tl.int64) * row_values, row_values)
            lane = tl.where(tight, index, col)
            # SSE min returns its second operand for equality or a NaN input.
            vector = tl.where(value < cutoff, value, cutoff)
            output = tl.where(lane < span // lane_count * lane_count, vector, output)
        elif KIND == 0:
            output = tl.minimum(tl.maximum(output, 0), 255)
        elif KIND == 1:
            output = tl.minimum(tl.maximum(output, 0), 65535)
        else:
            # OpenCV fills zero for TRUNC below the signed pixel range.
            output = tl.where(cutoff < -32768, 0, output)
    elif MODE == 3:
        output = tl.where(value > cutoff, value, 0)
    else:
        output = tl.where(value <= cutoff, value, 0)
    tl.store(dst_ptr + row * dst_rs + col, output, mask=valid)


@triton.jit
def threshold_clear(workspace_ptr, bins, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(workspace_ptr + 1 + index, 0, mask=index < bins)


@triton.jit
def threshold_histogram(src_ptr, workspace_ptr, rows, row_values, src_rs,
                        BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row = index // row_values
    col = index % row_values
    valid = row < rows
    value = tl.load(src_ptr + row * src_rs + col, mask=valid, other=0)
    tl.atomic_add(workspace_ptr + 1 + value.to(tl.int32), 1, sem="relaxed", mask=valid)


@triton.jit
def threshold_histogram_u8_large(src_ptr, workspace_ptr, rows, row_values, src_rs,
                        BLOCK: tl.constexpr, PRIVATE_8U: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row = index // row_values
    col = index % row_values
    valid = row < rows
    value = tl.load(src_ptr + row * src_rs + col, mask=valid, other=0)
    if PRIVATE_8U:
        counts = tl.histogram(value.to(tl.int32), 256, valid)
        bins = tl.arange(0, 256)
        tl.atomic_add(workspace_ptr + 1 + bins, counts, sem="relaxed", mask=counts != 0)
    else:
        tl.atomic_add(workspace_ptr + 1 + value.to(tl.int32), 1, sem="relaxed", mask=valid)


@triton.jit
def _triangle_vector(hist):
    index = tl.arange(0, 256)
    counts = tl.load(hist + index)
    left = tl.maximum(tl.min(tl.where(counts > 0, index, 256), 0) - 1, 0)
    right = tl.minimum(tl.max(tl.where(counts > 0, index, 0), 0) + 1, 255)
    peak_count = tl.max(counts, 0)
    peak = tl.min(tl.where(counts == peak_count, index, 256), 0)
    flip = peak - left < right - peak
    left = tl.where(flip, 255 - right, left)
    peak = tl.where(flip, 255 - peak, peak)
    counts = tl.load(hist + tl.where(flip, 255 - index, index))
    scores = peak_count.to(tl.int64) * index + (left - peak).to(tl.int64) * counts
    scores = tl.where((index > left) & (index <= peak), scores, 0)
    best = tl.max(scores, 0)
    # The CPU scans from the left and advances only on a strict increase.
    first = tl.min(tl.where(scores == best, index, 256), 0)
    cutoff = tl.where(best > 0, first, left) - 1
    return tl.where(flip, 255 - cutoff, cutoff)


@triton.jit
def _otsu_score_tiles(hist, one, scale, mean, BINS: tl.constexpr, TILE: tl.constexpr):
    # Preserve the CPU's bin-by-bin probability and first-mean recurrence,
    # including multiplying the mean before rejecting an empty class. Only
    # the second mean and score are independent across iterations.
    first_mean = tl.full((), 0.0, tl.float64)
    first_prob = tl.full((), 0.0, tl.float64)
    best = tl.full((), 0.0, tl.float64)
    cutoff = tl.full((), 0, tl.int32)
    offsets = tl.arange(0, TILE)
    epsilon: tl.constexpr = 1.1920928955078125e-7
    base = 0
    # Histogram counts are nonnegative. Once the second class is below the
    # epsilon guard it cannot recover, so later tiles cannot change cutoff.
    while (base < BINS) & ((one - first_prob) >= epsilon) & (first_prob <= one - epsilon):
        counts = tl.load(hist + base + offsets)
        # Before a first class has qualified, first_mean is zero. An all-zero
        # tile changes neither state; empty tiles inside the scan still run.
        if (first_prob >= epsilon) | (tl.max(counts, 0) != 0):
            probability = tl.full((TILE,), 0.0, tl.float64)
            means = tl.full((TILE,), 0.0, tl.float64)
            for j in range(0, TILE):
                i = base + j
                count = tl.sum(tl.gather(counts, tl.full((1,), j, tl.int32), axis=0), 0)
                p = count.to(tl.float64) * scale
                first_mean *= first_prob
                first_prob += p
                other = one - first_prob
                if (tl.minimum(first_prob, other) >= epsilon) & (tl.maximum(first_prob, other) <= one - epsilon):
                    first_mean = (first_mean + i * p) / first_prob
                probability = tl.where(offsets == j, first_prob, probability)
                means = tl.where(offsets == j, first_mean, means)
            other = one - probability
            valid = (tl.minimum(probability, other) >= epsilon) & (tl.maximum(probability, other) <= one - epsilon)
            other_mean = (mean - probability * means) / tl.where(valid, other, one)
            delta = means - other_mean
            scores = tl.where(valid, ((probability * other) * delta) * delta, 0.0)
            tile_best = tl.max(scores, 0)
            if tile_best > best:
                best = tile_best
                cutoff = base + tl.min(tl.where(scores == tile_best, offsets, TILE), 0)
        base += TILE
    return cutoff


@triton.jit
def threshold_select(workspace_ptr, pixels, BINS: tl.constexpr, TRIANGLE: tl.constexpr,
                     OTSU_TILE: tl.constexpr = 0, TRIANGLE_VECTOR: tl.constexpr = False):
    hist = workspace_ptr + 1
    if TRIANGLE and TRIANGLE_VECTOR:
        cutoff = _triangle_vector(hist)
    elif TRIANGLE:
        left = tl.full((), 256, tl.int32)
        right = tl.full((), 0, tl.int32)
        peak = tl.full((), 0, tl.int32)
        count = tl.full((), 0, tl.int32)
        for i in range(0, 256):
            n = tl.load(hist + i)
            left = tl.where(n > 0, tl.minimum(left, i), left)
            right = tl.where(n > 0, i, right)
            peak = tl.where(n > count, i, peak)
            count = tl.maximum(n, count)
        left = tl.maximum(left - 1, 0)
        right = tl.minimum(right + 1, 255)
        flip = peak - left < right - peak
        left = tl.where(flip, 255 - right, left)
        peak = tl.where(flip, 255 - peak, peak)
        score = tl.full((), 0, tl.int64)
        cutoff = left
        for i in range(left + 1, peak + 1):
            n = tl.load(hist + tl.where(flip, 255 - i, i)).to(tl.int64)
            candidate = count.to(tl.int64) * i + (left - peak).to(tl.int64) * n
            cutoff = tl.where(candidate > score, i, cutoff)
            score = tl.maximum(score, candidate)
        cutoff -= 1
        cutoff = tl.where(flip, 255 - cutoff, cutoff)
    else:
        # With at most INT32_MAX pixels, the weighted count is below 2**47.
        # OpenCV's first scan and this integer reduction are both exact up to scale.
        lane = tl.arange(0, 256)
        moment = tl.full((256,), 0, tl.int64)
        for start in range(0, BINS, 256):
            index = start + lane
            moment += index.to(tl.int64) * tl.load(hist + index).to(tl.int64)
        one = tl.full((), 1.0, tl.float64)
        scale = one / tl.cast(pixels, tl.float64)
        mean = tl.sum(moment, 0).to(tl.float64) * scale
        if OTSU_TILE:
            cutoff = _otsu_score_tiles(hist, one, scale, mean, BINS, OTSU_TILE)
        else:
            first_mean = tl.full((), 0.0, tl.float64)
            first_prob = tl.full((), 0.0, tl.float64)
            best = tl.full((), 0.0, tl.float64)
            cutoff = tl.full((), 0, tl.int32)
            for i in range(0, BINS):
                probability = tl.load(hist + i).to(tl.float64) * scale
                first_mean *= first_prob
                first_prob += probability
                other_prob = one - first_prob
                epsilon: tl.constexpr = 1.1920928955078125e-7
                if (tl.minimum(first_prob, other_prob) >= epsilon) & (tl.maximum(first_prob, other_prob) <= one - epsilon):
                    first_mean = (first_mean + i * probability) / first_prob
                    other_mean = (mean - first_prob * first_mean) / other_prob
                    delta = first_mean - other_mean
                    score = ((first_prob * other_prob) * delta) * delta
                    if score > best:
                        best = score
                        cutoff = i
    tl.store(workspace_ptr, cutoff)
