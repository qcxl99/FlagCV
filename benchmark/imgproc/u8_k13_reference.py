# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent reference for padded U8C1 K13 on the fixed SSE3 OpenCV build."""

from fractions import Fraction

import cv2
import numpy as np


def _tile(length):
    extent = cv2.getOptimalDFTSize(min(length, 244) + 12)
    return extent, min(length, extent - 12)


def _round_u8(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _dft_operand(padded, weights, height, width):
    fft_h, block_h = _tile(height)
    fft_w, block_w = _tile(width)
    spectrum = np.zeros((fft_h, fft_w), np.float32)
    spectrum[:13, :13] = weights
    cv2.dft(spectrum, spectrum, nonzeroRows=13)
    result = np.empty((height, width), np.float32)
    for y in range(0, height, block_h):
        for x in range(0, width, block_w):
            h, w = min(height - y, block_h), min(width - x, block_w)
            tile = np.zeros_like(spectrum)
            tile[:h + 12, :w + 12] = padded[y:y + h + 12, x:x + w + 12]
            cv2.dft(tile, tile, nonzeroRows=h + 12)
            cv2.mulSpectrums(tile, spectrum, 0, tile, conjB=True)
            cv2.dft(tile, tile, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE, nonzeroRows=h)
            result[y:y + h, x:x + w] = tile[:h, :w] + np.float32(.125)
    return result


def ordered_k13_reference(image, weights, *, delta, border, anchor=(-1, -1)):
    """Return one complete expected image and the fixed CPU diagnostic.

    Call with an isolated U8C1 view and the effective binary32 weights.
    An unqualified case must use its ordinary fixed CPU reference.
    """
    if (image.ndim != 2 or image.dtype != np.uint8 or not image.size
            or weights.shape != (13, 13) or weights.dtype != np.float32
            or delta != .125 or border != cv2.BORDER_REFLECT_101
            or anchor not in ((-1, -1), (6, 6))):
        raise ValueError("outside padded K13 reference scope")
    if np.any(weights[[0, -1], :]) or np.any(weights[:, [0, -1]]):
        raise ValueError("K13 requires an all-zero outer ring")
    active = [(y, x, Fraction(float(weights[y, x])))
              for y, x in np.argwhere(weights) if np.isfinite(weights[y, x])]
    if (not np.isfinite(weights).all()
            or any(abs(value) < np.finfo(np.float32).tiny for _, _, value in active)):
        raise ValueError("K13 requires finite normal effective weights")
    l1 = sum(abs(value) for _, _, value in active)
    operations = 2 * len(active) + 1
    budget = Fraction(operations, 2**24 - operations) * (255 * l1 + Fraction(1, 8))
    if l1 > 8 or budget >= Fraction(3, 100):
        raise ValueError("outside the K13 arithmetic bound")
    if cv2.__version__ != "5.0.0" or cv2.getCPUFeaturesLine() != "SSE SSE2 SSE3":
        raise ValueError("K13 reference requires the fixed SSE3 configuration")

    h, w = image.shape
    edge = border | cv2.BORDER_ISOLATED
    pad = cv2.copyMakeBorder(image, 6, 6, 6, 6, edge)
    ordered = np.full(image.shape, delta, np.float32)
    for y, x, _ in active:
        product = pad[y:y + h, x:x + w].astype(np.float32) * weights[y, x]
        ordered += product
    expected = _round_u8(ordered)
    cpu = cv2.filter2D(image, -1, weights, anchor=anchor, delta=delta, borderType=edge)
    dft = _dft_operand(pad, weights, h, w)
    if not np.array_equal(_round_u8(dft), cpu):
        raise ValueError("K13 DFT reconstruction differs from fixed CPU")
    diagnostics = []
    for y, x in np.argwhere(expected != cpu):
        a, b = int(expected[y, x]), int(cpu[y, x])
        sequential, transformed = Fraction(float(ordered[y, x])), Fraction(float(dft[y, x]))
        term_values = [int(pad[y + j, x + i]) * weight for j, i, weight in active]
        exact = Fraction(1, 8) + sum(term_values)
        weighted_abs = Fraction(1, 8) + sum(abs(term) for term in term_values)
        midpoint = Fraction(a + b, 2)
        if (abs(a - b) != 1 or not 1 <= min(a, b) <= max(a, b) <= 254
                or weighted_abs > 64 * max(1, abs(exact))
                or not min(sequential, transformed) <= midpoint <= max(sequential, transformed)
                or max(abs(sequential - exact), abs(transformed - exact)) > budget
                or abs(sequential - transformed) > budget):
            raise ValueError(f"unqualified K13 CPU difference at ({y},{x})")
        diagnostics.append(dict(y=int(y), x=int(x), cpu=b, expected=a, exact=str(exact),
                                ordered=str(sequential), dft=str(transformed)))
    return expected, cpu, diagnostics
