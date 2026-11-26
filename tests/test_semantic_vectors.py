"""Check the same small, fixed public semantics vectors as C++."""
import csv
from importlib import import_module
from pathlib import Path

import pytest

OPS = ("add", "subtract", "multiply", "bitwise_and", "bitwise_not", "flip")


@pytest.mark.parametrize("op", [pytest.param(op, marks=getattr(pytest.mark, op)) for op in OPS])
def test_common_u8_semantics(op, candidate, pytestconfig):
    import torch

    with Path(__file__).with_name("data").joinpath("semantic_u8.csv").open() as stream:
        vectors = list(csv.DictReader(stream))
    device = pytestconfig._flagcv_runtime.device

    def inputs():
        a = torch.tensor([[int(row["a"]) for row in vectors]], dtype=torch.uint8, device=device)
        dst = torch.empty_like(a)
        if op == "bitwise_not":
            return a, dst
        if op == "flip":
            return a, dst, 1
        b = torch.tensor([[int(row["b"]) for row in vectors]], dtype=torch.uint8, device=device)
        return a, b, dst

    expected = [[int(row[op]) for row in vectors]]
    assert candidate(*inputs()).cpu().tolist() == expected
    reference = import_module(f"reference.{op}").reference
    assert reference(*inputs()).cpu().tolist() == expected
