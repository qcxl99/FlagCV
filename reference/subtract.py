"""OpenCV CPU saturating subtraction."""

from ._opencv import evaluate


def reference(a, b, dst):
    return evaluate("subtract", (a, b), dst)
