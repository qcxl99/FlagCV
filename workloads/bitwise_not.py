"""Correctness and timing workloads for bitwise_not."""

from helpers.inputs import generate_tensor_input
from workloads.core import PYTHON_SIZES, SMALL_CORRECTNESS_SIZES

OP_NAME = "bitwise_not"
_TIMING_SHAPES = PYTHON_SIZES
_SMALL_SHAPES = SMALL_CORRECTNESS_SIZES

TIMING_CASES = tuple((shape, "uint8", None) for shape in _TIMING_SHAPES)
CORRECTNESS_CASES = (
    *TIMING_CASES,
    ((1, 5), "uint8", [[0, 1, 170, 85, 255]]),
    *((shape, "uint8", None) for shape in _SMALL_SHAPES),
)


def case_id(case):
    (height, width), dtype, _ = case
    return f"{height}x{width}::{dtype}"


def describe(case):
    shape, dtype, _ = case
    shapes = dict.fromkeys(("src", "dst"), shape)
    params = {"channels": 1, "mask": None, "contiguous": True}
    return dtype, shapes, params


def make_inputs(case, device):
    import torch

    shape, dtype, fixed_input = case
    dtype = getattr(torch, dtype)
    if fixed_input is not None:
        src = torch.tensor(fixed_input, dtype=dtype, device=device)
    else:
        generator = torch.Generator(device="cpu").manual_seed(0)
        src = generate_tensor_input(shape, dtype, device, generator)
    return src, torch.empty_like(src)
