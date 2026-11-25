"""Cross-check the pre-existing Torch references against OpenCV CPU."""

import pytest

from helpers.tolerance import (
    assert_bitwise_equal,
    assert_relative_close,
)
from reference import get_reference


@pytest.mark.parametrize(
    ("op", "dtype"),
    [
        (op, dtype)
        for op in ("add", "subtract", "multiply", "bitwise_and", "bitwise_not", "flip")
        for dtype in ("uint8", "float32")
        if dtype == "uint8" or not op.startswith("bitwise")
    ],
)
def test_legacy_aten_reference(op, dtype):
    import torch

    generator = torch.Generator().manual_seed(977)
    if dtype == "uint8":
        a = torch.randint(0, 256, (37, 83), dtype=torch.uint8, generator=generator)
        b = torch.randint(0, 256, a.shape, dtype=a.dtype, generator=generator)
        a[0, :4] = torch.tensor([0, 1, 254, 255], dtype=a.dtype)
        b[0, :4] = torch.tensor([255, 2, 192, 255], dtype=a.dtype)
    else:
        a = torch.randn((37, 83), generator=generator)
        b = torch.randn(a.shape, generator=generator)
        a[0, 0] = -0.0
    output = torch.empty_like(a)
    reference = get_reference(op, "cpu")
    if op == "flip":
        for code, axes in ((0, (0,)), (2, (1,)), (-7, (0, 1))):
            assert_bitwise_equal(torch.flip(a, axes), reference(a, output, code))
        return
    if op == "bitwise_and":
        assert_bitwise_equal(torch.bitwise_and(a, b), reference(a, b, output))
    elif op == "bitwise_not":
        assert_bitwise_equal(torch.bitwise_not(a), reference(a, output))
    else:
        aten = getattr(torch, op)
        expected = reference(a, b, output)
        if dtype == "uint8":
            assert_bitwise_equal(aten(a.int(), b.int()).clamp_(0, 255).byte(), expected)
        else:
            assert_relative_close(aten(a, b), expected)
