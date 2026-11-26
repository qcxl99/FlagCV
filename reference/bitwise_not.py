"""OpenCV CPU unmasked bitwise complement."""

from ._opencv import evaluate


def reference(src, dst):
    return evaluate("bitwise_not", (src,), dst)
