"""OpenCV CPU unmasked bitwise AND."""

from ._opencv import evaluate


def reference(a, b, dst):
    return evaluate("bitwise_and", (a, b), dst)
