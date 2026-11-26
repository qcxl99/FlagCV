"""OpenCV CPU multiplication with unit scale."""

from ._opencv import evaluate


def reference(a, b, dst):
    return evaluate("multiply", (a, b), dst)
