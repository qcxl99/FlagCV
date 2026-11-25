"""OpenCV CPU saturating addition."""

from ._opencv import evaluate


def reference(a, b, dst):
    return evaluate("add", (a, b), dst)
