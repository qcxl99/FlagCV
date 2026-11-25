"""OpenCV CPU reflection."""

from operator import index

from ._opencv import evaluate


def reference(src, dst, flip_code):
    code = index(flip_code)
    return evaluate("flip", (src,), dst, (code > 0) - (code < 0))
