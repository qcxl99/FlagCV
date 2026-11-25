# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright 2026, The FlagOS Contributors.
#
# Adapted from opencv_contrib 2f5682ba, modules/triton/kernels/warpaffine_cpu.py.
#
# GPU affine sampling with the selected OpenCV CPU coordinate and rounding rules.
# The cpu prefix describes the arithmetic, not where execution takes place.
# M maps destination pixels to source coordinates after host-side inversion if needed.
# Row strides count typed elements; border indices are source pixel indices.

import triton
import triton.language as tl


@triton.jit
def _round_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    odd = (lower.to(tl.int32) & 1) != 0
    return lower + ((fraction > 0.5) | ((fraction == 0.5) & odd)).to(tl.float32)


@triton.jit
def _pmod(p, n):
    """p mod n, always in [0, n). Triton's `//` and `%` truncate toward zero like
    C, so a negative index would otherwise come back negative; every border rule
    below is periodic and needs the mathematical modulo."""
    residue = p % n
    return tl.where(residue < 0, residue + n, residue)


@triton.jit
def _bm(p, n, BORDER: tl.constexpr):
    """borderInterpolate for one index, in closed form (borderInterpolate_fast).

    REPLICATE clips; WRAP is periodic with period n; REFLECT has period 2n and
    REFLECT_101 period 2(n-1) (the border pixel is not repeated). All three are
    full modulo forms rather than loops because the index range here is
    unbounded (an affine map can point anywhere).
    """
    if BORDER == 1:                                   # BORDER_REPLICATE
        return tl.minimum(tl.maximum(p, 0), n - 1).to(tl.int64)
    if BORDER == 3:                                   # BORDER_WRAP
        return _pmod(p, n).to(tl.int64)
    if BORDER == 2:                                   # BORDER_REFLECT, period 2n
        period = 2 * n.to(tl.int64)
        m = _pmod(p.to(tl.int64), period)
        return tl.where(m < n, m, period - m - 1)
    period = tl.maximum(2 * (n.to(tl.int64) - 1), 1)               # BORDER_REFLECT_101
    m = _pmod(p.to(tl.int64), period)
    return tl.where(m < n, m, period - m)


@triton.jit
def _fma(a, b, c, FUSED: tl.constexpr):
    """cv::v_fma: fused on an FMA3 build, multiply-then-add otherwise.

    core/hal/intrin_sse.hpp expands v_fma to _mm_add_ps(_mm_mul_ps(a,b), c) when CV_FMA3 is 0.
    OpenCV's warp kernels use v_fma for the coordinate term AND for the interpolation
    accumulation, so both must follow the same switch; reproducing only the coordinate one
    leaves the interpolation off by an ulp, which reaches the integer store.
    """
    if FUSED:
        return tl.fma(a, b, c)
    return a * b + c


@triton.jit
def _mapped_coord(xf, M00, M10, row_x, row_y, FUSED: tl.constexpr):
    """The column term of CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2.

    `v_fma(M0, dst_x, M_x)` on the CPU is only a fused operation when the build has FMA3:
    core/hal/intrin_sse.hpp implements v_fma as _mm_add_ps(_mm_mul_ps(a,b), c) when CV_FMA3 is
    0. An FMA3 build therefore rounds once and a baseline-only build rounds twice, and at a
    0.5 tie in the interpolation the two land on different source pixels.

    The C++ planner identifies the linked CPU arithmetic with fixed numerical probes
    and passes it as FUSED. A different rounding at a sampling tie can select an
    adjacent source pixel.
    """
    if FUSED:
        return (tl.fma(xf, M00.to(tl.float32), row_x),
                tl.fma(xf, M10.to(tl.float32), row_y))
    return (((xf * M00.to(tl.float32)) + row_x),
            ((xf * M10.to(tl.float32)) + row_y))


@triton.jit
def _warp_nearest(src_ptr, dst_ptr, M00, M01, M02, M10, M11, M12, src_h, src_w, dst_h, dst_w,
                  src_rs, dst_rs, bv0, bv1, bv2, bv3,
                  cn: tl.constexpr, BORDER: tl.constexpr, BORDER_X: tl.constexpr,
                  BORDER_Y: tl.constexpr, FUSED: tl.constexpr, TRANSPARENT: tl.constexpr,
                  BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, COL_GRID_LIMIT: tl.constexpr):
    src, dst = src_ptr, dst_ptr
    pid_h = tl.program_id(0).to(tl.int64)
    pid_w = tl.program_id(1).to(tl.int64)
    if COL_GRID_LIMIT:
        pid_w += tl.program_id(2).to(tl.int64) * COL_GRID_LIMIT
    ys = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    xs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    m = (ys < dst_h) & (xs < dst_w)

    # static_cast<float>(y*M1 + M2): OpenCV's M[] is FLOAT, so the row term is
    # float arithmetic rounded twice (multiply, add). Computing it in double and
    # narrowing once moves the result by an ulp on ties, which a 0.5 tie turns
    # into a whole output level.
    yf = ys.to(tl.float32)
    # The provider disables implicit FMA contraction for these entry points.
    row_x = yf * M01 + M02
    row_y = yf * M11 + M12
    xf = xs.to(tl.float32)
    sx, sy = _mapped_coord(xf, M00, M10, row_x, row_y, FUSED)

    ix = _round_even(sx).to(tl.int32)             # cvRound: half to even
    iy = _round_even(sy).to(tl.int32)

    if BORDER == 0 or TRANSPARENT:                   # BORDER_CONSTANT, or TRANSPARENT's range test
        inside = (ix >= 0) & (ix < src_w) & (iy >= 0) & (iy < src_h)
        cx = tl.minimum(tl.maximum(ix, 0), src_w - 1)
        cy = tl.minimum(tl.maximum(iy, 0), src_h - 1)
    else:
        inside = m
        # border_type_x/y: a 1-pixel axis is replicated whatever the mode, because
        # the reflection periods collapse there (imgwarp.cpp warpAffine)
        cx = _bm(ix, src_w, BORDER_X)
        cy = _bm(iy, src_h, BORDER_Y)

    # Row pitch in ELEMENTS, not src_w*cn: a GpuMat's step is padded for alignment, so
    # recomputing the pitch from the width reads the wrong row on every row but the first.
    base = cy.to(tl.int64) * src_rs + cx.to(tl.int64) * cn
    # BORDER_TRANSPARENT (NEAREST): write only where the rounded source point is inside, using
    # the CPU's own unsigned range test -- `(unsigned)cvRound(sx) < W && (unsigned)cvRound(sy) < H`.
    # The destination is never read, so "skip" and "write the old value" are the same thing for
    # the pixels that are skipped; the store mask is the whole rule.
    store_mask = m & inside if TRANSPARENT else m
    for c in tl.static_range(cn):
        v = tl.load(src + base + c, mask=m & inside, other=0)
        if BORDER == 0:
            # The border value is a Scalar carried per channel, NOT one value repeated:
            # cv2.warpAffine(..., borderValue=(10,20,30)) fills (10,20,30), and a scalar
            # fills (v,0,0). Collapsing it to one number is wrong for every channel but 0.
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            # bv is an fp32 runtime argument already rounded and clamped on the host into
            # the destination dtype (saturate_cast), so a uint8/uint16 store just narrows it.
            v = tl.where(m & ~inside, tl.full(v.shape, bv.to(v.dtype), v.dtype), v)
        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, v, mask=store_mask)


@triton.jit
def _warp_linear(src_ptr, dst_ptr, M00, M01, M02, M10, M11, M12, src_h, src_w, dst_h, dst_w,
                 src_rs, dst_rs, bv0, bv1, bv2, bv3,
                 cn: tl.constexpr, U8: tl.constexpr, U16: tl.constexpr,
                 BORDER: tl.constexpr, TRANSPARENT: tl.constexpr,
                 BORDER_X: tl.constexpr, BORDER_Y: tl.constexpr, FUSED: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, COL_GRID_LIMIT: tl.constexpr):
    src, dst = src_ptr, dst_ptr
    pid_h = tl.program_id(0).to(tl.int64)
    pid_w = tl.program_id(1).to(tl.int64)
    if COL_GRID_LIMIT:
        pid_w += tl.program_id(2).to(tl.int64) * COL_GRID_LIMIT
    ys = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    xs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    m = (ys < dst_h) & (xs < dst_w)

    # static_cast<float>(y*M1 + M2): OpenCV's M[] is FLOAT, so the row term is
    # float arithmetic rounded twice (multiply, add). Computing it in double and
    # narrowing once moves the result by an ulp on ties, which a 0.5 tie turns
    # into a whole output level.
    yf = ys.to(tl.float32)
    # The provider disables implicit FMA contraction for these entry points.
    row_x = yf * M01 + M02
    row_y = yf * M11 + M12
    xf = xs.to(tl.float32)
    sx, sy = _mapped_coord(xf, M00, M10, row_x, row_y, FUSED)

    fx = tl.floor(sx)
    fy = tl.floor(sy)
    ax = sx - fx
    ay = sy - fy
    ix = fx.to(tl.int32)
    iy = fy.to(tl.int32)
    ix1 = ix + 1
    iy1 = iy + 1

    if BORDER == 0 or TRANSPARENT:
        # whole neighbourhood outside -> border value; otherwise each tap that is
        # outside alone takes the border value (CV_WARP_SCALAR_FETCH_PIXEL). TRANSPARENT uses the
        # same in-range flags, but substitutes the OLD DESTINATION PIXEL for an out-of-range tap
        # and leaves the weights unchanged (see the lstore comment below).
        outside = (ix >= src_w) | (ix1 < 0) | (iy >= src_h) | (iy1 < 0)
        in0 = (ix >= 0) & (ix < src_w)
        in1 = (ix1 >= 0) & (ix1 < src_w)
        jy0 = (iy >= 0) & (iy < src_h)
        jy1 = (iy1 >= 0) & (iy1 < src_h)
    else:
        outside = m & False
        in0 = m
        in1 = m
        jy0 = m
        jy1 = m

    hw = src_rs
    for c in tl.static_range(cn):
        if BORDER == 0 or TRANSPARENT:
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            # `other` must have the pointer's element type; the border component is fp32.
            bvs = bv.to(src.dtype.element_ty)
            if TRANSPARENT:
                # Out-of-range taps take the destination pixel's CURRENT value, not a constant.
                bvs = tl.load(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c,
                              mask=m, other=0)
            p00 = tl.load(src + iy.to(tl.int64) * hw + ix.to(tl.int64) * cn + c,
                          mask=m & in0 & jy0, other=bvs)
            p01 = tl.load(src + iy.to(tl.int64) * hw + ix1.to(tl.int64) * cn + c,
                          mask=m & in1 & jy0, other=bvs)
            p10 = tl.load(src + iy1.to(tl.int64) * hw + ix.to(tl.int64) * cn + c,
                          mask=m & in0 & jy1, other=bvs)
            p11 = tl.load(src + iy1.to(tl.int64) * hw + ix1.to(tl.int64) * cn + c,
                          mask=m & in1 & jy1, other=bvs)
        else:
            cx0 = _bm(ix, src_w, BORDER_X)
            cx1 = _bm(ix1, src_w, BORDER_X)
            cy0 = _bm(iy, src_h, BORDER_Y)
            cy1 = _bm(iy1, src_h, BORDER_Y)
            p00 = tl.load(src + cy0.to(tl.int64) * hw + cx0.to(tl.int64) * cn + c, mask=m, other=0)
            p01 = tl.load(src + cy0.to(tl.int64) * hw + cx1.to(tl.int64) * cn + c, mask=m, other=0)
            p10 = tl.load(src + cy1.to(tl.int64) * hw + cx0.to(tl.int64) * cn + c, mask=m, other=0)
            p11 = tl.load(src + cy1.to(tl.int64) * hw + cx1.to(tl.int64) * cn + c, mask=m, other=0)

        if U8 or U16:
            f = tl.float32
            p00 = p00.to(f); p01 = p01.to(f); p10 = p10.to(f); p11 = p11.to(f)
        # v0 = fma(ax, p01 - p00, p00); v1 = fma(ax, p11 - p10, p10); out = fma(ay, v1 - v0, v0)
        v0 = _fma(ax, p01 - p00, p00, FUSED)
        v1 = _fma(ax, p11 - p10, p10, FUSED)
        v = _fma(ay, v1 - v0, v0, FUSED)
        if U8:
            v = tl.minimum(tl.maximum(_round_even(v), 0.0), 255.0).to(tl.uint8)
        elif U16:
            # v_pack_u(v_round(...)) = saturate_cast<ushort>(float): round, clamp, convert
            v = tl.minimum(tl.maximum(_round_even(v), 0.0), 65535.0).to(tl.uint16)
        if BORDER == 0:
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            v = tl.where(outside & m, tl.full(v.shape, bv.to(v.dtype), v.dtype), v)
        # BORDER_TRANSPARENT (LINEAR): each out-of-range tap is replaced by the OLD DESTINATION
        # pixel with its weight unchanged -- out = sum(w_k*S_k) + (1 - sum(w_k))*dst_old, so the
        # blend degenerates smoothly to "keep the old value" as the point leaves the image. No
        # renormalisation and no border value; the pixel is still written unless the whole 2x2
        # window is outside (ix < -1 || ix >= W || iy < -1 || iy >= H).
        # The destination must already hold its intended contents.
        lstore = m & ~outside if TRANSPARENT else m
        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, v, mask=lstore)


@triton.jit
def _bicubic_weights_t(a, FUSED: tl.constexpr):
    """bicubicWeights for one fractional coordinate, float32, in the C++ order.

    w0 = A*(a*b2) and w3 = A*(a2*b): the grouping is A*(a*b^2), not (A*a)*b^2 -- the
    two differ in the last bit and reach the integer store. w1 = fma(a2, fma(A+2, a,
    -(A+3)), 1) is the SIMD form, w2 = 1 - w0 - w1 - w3 in that order.
    """
    a2 = (a * a)
    b = (1.0 - a)
    b2 = (b * b)
    w0 = (-0.75 * (a * b2))
    w3 = (-0.75 * (a2 * b))
    w1 = _fma(a2, _fma(1.25, a, -2.25, FUSED), 1.0, FUSED)   # A+2 = 1.25, A+3 = 2.25
    w2 = (((1.0 - w0) - w1) - w3)
    return w0, w1, w2, w3


@triton.jit
def _cubic_outlier(ix0, iy0, src_w, src_h):
    # Match OpenCV bicubicCoeffs' unsigned bounds on origin + 4.
    # Compare the upper bound directly to preserve large source extents.
    return (ix0 >= src_w) | (ix0 + 4 < 0) | (iy0 >= src_h) | (iy0 + 4 < 0)


@triton.jit
def _cubic_constant_group(xs, dst_w, row_x, row_y, M00, M10, bigw, bigh, src_w, src_h):
    # The fixed SSE3 CPU processes four pixels per SIMD group in a
    # 128-column genericWarp chunk. Its last full group may overwrite
    # some pixels of the previous group. Chunks shorter than four pad
    # missing lanes with coordinates (0, 0), which prevents the shortcut.
    chunk_tail = dst_w % 128
    padded_tail = (chunk_tail > 0) & (chunk_tail < 4)
    aligned_start = xs // 4 * 4
    in_final_group = ~padded_tail & (xs >= dst_w - 4)
    group_start = tl.where(in_final_group, dst_w - 4, aligned_start)
    group_outside = group_start + 3 < dst_w
    for peer in tl.static_range(4):
        peer_x = (group_start + peer).to(tl.float32)
        sx = row_x + peer_x * M00
        sy = row_y + peer_x * M10
        sx = tl.minimum(tl.maximum(sx, -bigw), 2.0 * bigw)
        sy = tl.minimum(tl.maximum(sy, -bigh), 2.0 * bigh)
        ix0 = tl.floor(sx).to(tl.int64) - 1
        iy0 = tl.floor(sy).to(tl.int64) - 1
        group_outside = group_outside & _cubic_outlier(ix0, iy0, src_w, src_h)
    return group_outside


@triton.jit
def _warp_cubic(src_ptr, dst_ptr, M00, M01, M02, M10, M11, M12, src_h, src_w, dst_h, dst_w,
                src_rs, dst_rs, bv0, bv1, bv2, bv3,
                cn: tl.constexpr, U8: tl.constexpr, U16: tl.constexpr,
                BORDER: tl.constexpr, TRANSPARENT: tl.constexpr,
                BORDER_X: tl.constexpr, BORDER_Y: tl.constexpr,
                FUSED: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, COL_GRID_LIMIT: tl.constexpr):
    """INTER_CUBIC: genericWarp -> bicubicVec/bicubicRef (OpenCV 5.0.0).

    Different family from _warp_linear/_warp_nearest: the column coordinate is
    M_x + M_xx*x with NO fma, computed from the absolute x, so every block computes a
    column the same way and the result does not depend on the destination width.
    The cubic reference therefore does not need destination-width padding.

    Tap layout: ix0 = floor(sx) - 1, then ix0..ix0+3; sx and sy are clamped in float to
    [-max(dim,16), 2*max(dim,16)] first, as bicubicCoeffs does, so a wild matrix cannot
    index unpredictably.
    """
    src, dst = src_ptr, dst_ptr
    pid_h = tl.program_id(0).to(tl.int64)
    pid_w = tl.program_id(1).to(tl.int64)
    if COL_GRID_LIMIT:
        pid_w += tl.program_id(2).to(tl.int64) * COL_GRID_LIMIT
    ys = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    xs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    m = (ys < dst_h) & (xs < dst_w)

    # M_x = float(y)*M(0,1) + M(0,2): a float multiply then a float add, both rounded.
    yf = ys.to(tl.float32)
    row_x = yf * M01 + M02
    row_y = yf * M11 + M12
    xf = xs.to(tl.float32)
    # xbuf[dx] = M_x + M_xx*xf -- NOT fma, and this is what makes CUBIC width-invariant.
    sx = (row_x + (xf * M00.to(tl.float32)))
    sy = (row_y + (xf * M10.to(tl.float32)))

    # bicubicCoeffs' float clamp before the floor: bigwidth = max(size, 16)
    bigw = tl.maximum(src_w, 16).to(tl.float32)
    bigh = tl.maximum(src_h, 16).to(tl.float32)
    sx = tl.minimum(tl.maximum(sx, -bigw), 2.0 * bigw)
    sy = tl.minimum(tl.maximum(sy, -bigh), 2.0 * bigh)

    fx = tl.floor(sx)
    fy = tl.floor(sy)
    ax = (sx - fx)
    ay = (sy - fy)
    ix0 = fx.to(tl.int64) - 1
    iy0 = fy.to(tl.int64) - 1
    wx0, wx1, wx2, wx3 = _bicubic_weights_t(ax, FUSED)
    wy0, wy1, wy2, wy3 = _bicubic_weights_t(ay, FUSED)

    if BORDER == 0:
        all_out = _cubic_constant_group(xs, dst_w, row_x, row_y, M00, M10,
                                        bigw, bigh, src_w, src_h)
    elif TRANSPARENT:
        all_out = _cubic_outlier(ix0, iy0, src_w, src_h)

    wx = (wx0, wx1, wx2, wx3)
    wy = (wy0, wy1, wy2, wy3)

    for c in tl.static_range(cn):
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        for r in tl.static_range(4):
            wy_r = wy[r]
            ry = iy0 + r
            rowsum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            for k in tl.static_range(4):
                xx = ix0 + k
                if BORDER == 0 or TRANSPARENT:
                    ok = (xx >= 0) & (xx < src_w) & (ry >= 0) & (ry < src_h)
                    cx = tl.minimum(tl.maximum(xx, 0), src_w - 1)
                    cy = tl.minimum(tl.maximum(ry, 0), src_h - 1)
                else:
                    cx = _bm(xx, src_w, BORDER_X)
                    cy = _bm(ry, src_h, BORDER_Y)
                    ok = m
                idx = (cy.to(tl.int64) * src_rs + cx.to(tl.int64) * cn + c)
                if BORDER == 0 or TRANSPARENT:
                    if c == 0:
                        bv = bv0
                    elif c == 1:
                        bv = bv1
                    elif c == 2:
                        bv = bv2
                    else:
                        bv = bv3
                    bvs = bv.to(src.dtype.element_ty)
                    if TRANSPARENT:
                        # BORDER_TRANSPARENT (CUBIC): an out-of-range tap is the OLD DESTINATION
                        # pixel with unchanged weights, as in the LINEAR blend. The result is
                        # written unless the whole 4x4 window is outside.
                        bvs = tl.load(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c,
                                      mask=m, other=0)
                    v = tl.load(src + idx, mask=m & ok, other=bvs)
                else:
                    v = tl.load(src + idx, mask=m, other=0)
                if U8 or U16:
                    v = v.to(tl.float32)
                # BICUBIC_UPDATE_ACC_VEC: rowsum = fma(v, wx, rowsum)
                rowsum = _fma(v, wx[k], rowsum, FUSED)
            acc = _fma(rowsum, wy_r, acc, FUSED)
        if BORDER == 0:
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            acc = tl.where(m & all_out, tl.full(acc.shape, bv.to(tl.float32), tl.float32), acc)
        if U8:
            out = tl.minimum(tl.maximum(_round_even(acc), 0.0), 255.0).to(tl.uint8)
        elif U16:
            out = tl.minimum(tl.maximum(_round_even(acc), 0.0), 65535.0).to(tl.uint16)
        else:
            out = acc
        cstore = m & ~all_out if TRANSPARENT else m
        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, out, mask=cstore)
